"""
Render short MP4 clips of a nominal PPO Hopper policy operating in perturbed environments,
for the website's "Idea Visually" section.

For each sampled dot we choose a perturbation type and an epsilon, convert epsilon to
the wrapper's noise_sigma using closed-form KL formulas, and roll out the policy for one
episode while recording frames. The resulting MP4s land under website/static/videos/.

KL closed forms (single-step, Gaussian-ish):
    action:       p = 1 - exp(-eps)
                  (mixture: p*Uniform + (1-p)*delta_a; KL(δ || mix) = -log(1-p) = eps)
    observation:  sigma = SIGMA_REF * sqrt(exp(2*eps/d) - 1)
                  (eps = (d/2) * log(1 + sigma^2 / SIGMA_REF^2))
    environment:  delta = sqrt(2*eps) / DELTA_SCALE
                  (Gaussian/Dirac limit of the uniform multiplicative perturbation)
"""
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import imageio_ffmpeg
import imageio.v2 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from utils import DisturbanceWrapper  # noqa: E402

CHECKPOINT = Path("/cmlscratch/anirudhs/RARL/rebuttal/runs/Hopper-v4__sac_continuous__0/sac_continuous.cleanrl_model")
NORM_STATS = CHECKPOINT.parent / "sac_continuous.norm_stats"
OUTPUT_DIR = REPO / "website" / "static" / "videos"
META_OUT = REPO / "website" / "static" / "js" / "hover_envs.json"

ENV_ID = "Hopper-v4"
NUM_DOTS = 25
SEED = 7
MAX_FRAMES = 1000      # full Hopper episode
FRAME_STRIDE = 1       # render every sim step
VIDEO_W, VIDEO_H = 240, 180
CLIP_FPS = 100         # playback fps -> ~3.3x real-time

# Mapping constants chosen so eps in [0.05, 0.5] gives the noise magnitudes
# used in the paper's evaluation grid.
SIGMA_REF = 0.1     # observation reference noise scale (per dim)
DELTA_SCALE = 1.5   # environment delta scaling; eps=0.5 -> delta≈0.67 (matches paper grid)


def kl_to_sigma(noise_type: str, eps: float, obs_dim: int) -> float:
    if noise_type == "action":
        p = 1.0 - np.exp(-eps)
        return float(np.clip(p, 0.0, 0.95))
    if noise_type == "observation":
        sigma = SIGMA_REF * np.sqrt(np.exp(2 * eps / obs_dim) - 1.0)
        return float(sigma)
    if noise_type == "environment":
        return float(np.sqrt(2 * eps) / DELTA_SCALE)
    raise ValueError(noise_type)


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Actor(nn.Module):
    """Matches sac/sac_continuous.py Actor exactly so the checkpoint loads."""
    def __init__(self, obs_dim: int, act_dim: int, action_low, action_high):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mean = nn.Linear(128, act_dim)
        self.fc_logstd = nn.Linear(128, act_dim)
        self.register_buffer(
            "action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer(
            "action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def set_obs_norm(self, mean, var, eps: float = 1e-8):
        self.obs_norm_mean = torch.as_tensor(mean, dtype=torch.float32)
        self.obs_norm_std = torch.sqrt(torch.as_tensor(var, dtype=torch.float32) + eps)

    def act(self, obs_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(obs_np).float().unsqueeze(0)
            if hasattr(self, "obs_norm_mean"):
                x = (x - self.obs_norm_mean) / self.obs_norm_std
            mean, _ = self(x)
            action = torch.tanh(mean) * self.action_scale + self.action_bias
            return action.squeeze(0).numpy()


def build_env(noise_type: str, noise_sigma: float, seed: int) -> gym.Env:
    env = gym.make(ENV_ID, render_mode="rgb_array",
                   width=VIDEO_W, height=VIDEO_H)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    env = DisturbanceWrapper(env, noise_type, noise_sigma)
    env.reset(seed=seed)
    return env


def roll_out(agent: Actor, env: gym.Env):
    frames = []
    total_reward = 0.0
    steps = 0
    obs, _ = env.reset()
    for t in range(MAX_FRAMES):
        action = agent.act(obs)
        obs, r, term, trunc, _info = env.step(action)
        total_reward += float(r)
        steps += 1
        if t % FRAME_STRIDE == 0:
            frames.append(env.unwrapped.render())
        if term or trunc:
            break
    return frames, total_reward, steps


def write_mp4(frames, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # imageio_ffmpeg backed: writes a small, browser-playable H.264 mp4.
    writer = iio.get_writer(
        str(out_path),
        fps=CLIP_FPS,
        codec="libx264",
        quality=9,
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    for f in frames:
        writer.append_data(f)
    writer.close()


def main():
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Inspect obs/action spaces with a throwaway env.
    probe = gym.make(ENV_ID)
    probe = gym.wrappers.FlattenObservation(probe)
    obs_dim = int(np.prod(probe.observation_space.shape))
    act_dim = int(np.prod(probe.action_space.shape))
    action_low = probe.action_space.low
    action_high = probe.action_space.high
    probe.close()

    agent = Actor(obs_dim, act_dim, action_low, action_high)
    state = torch.load(CHECKPOINT, map_location="cpu")
    agent.load_state_dict(state)
    if NORM_STATS.exists():
        norm = torch.load(NORM_STATS, map_location="cpu")
        agent.set_obs_norm(norm["obs_rms_mean"], norm["obs_rms_var"])
        print(f"loaded obs-norm stats from {NORM_STATS.name}")
    agent.eval()

    # Sample NUM_DOTS perturbations: ensure all three types appear at least once.
    perturb_types = ["action", "observation", "environment"]
    base = [perturb_types[i % 3] for i in range(NUM_DOTS)]
    rng.shuffle(base)
    epsilons = rng.uniform(0.05, 0.5, size=NUM_DOTS)

    # Nominal (no perturbation) clip first — used by the center P₀ marker.
    nominal_env = build_env("none", 0.0, seed=SEED - 1)
    nominal_frames, nominal_return, nominal_steps = roll_out(agent, nominal_env)
    nominal_env.close()
    nominal_clip = OUTPUT_DIR / "nominal.mp4"
    write_mp4(nominal_frames, nominal_clip)
    print(f"[nominal] return={nominal_return:.1f} steps={nominal_steps} "
          f"frames={len(nominal_frames)} -> {nominal_clip.name}")

    meta = []
    for i, (pt, eps) in enumerate(zip(base, epsilons)):
        sigma = kl_to_sigma(pt, float(eps), obs_dim)
        env = build_env(pt, sigma, seed=SEED + i)
        frames, ep_return, ep_steps = roll_out(agent, env)
        env.close()
        clip = OUTPUT_DIR / f"dot_{i:02d}_{pt}_eps{eps:.2f}.mp4"
        write_mp4(frames, clip)
        meta.append({
            "id": i,
            "perturbation": pt,
            "epsilon": float(eps),
            "sigma": float(sigma),
            "video": f"./static/videos/{clip.name}",
            "frames": len(frames),
            "episodic_return": float(ep_return),
            "episode_steps": int(ep_steps),
        })
        print(f"[{i+1}/{NUM_DOTS}] {pt:11s}  eps={eps:.2f}  sigma={sigma:.3f}  "
              f"return={ep_return:7.1f}  steps={ep_steps:4d}  -> {clip.name}")

    META_OUT.write_text(json.dumps({
        "env": ENV_ID,
        "checkpoint": str(CHECKPOINT),
        "nominal_video": f"./static/videos/{nominal_clip.name}",
        "nominal_return": float(nominal_return),
        "nominal_steps": int(nominal_steps),
        "kl_formulas": {
            "action":      "p = 1 - exp(-eps)",
            "observation": f"sigma = {SIGMA_REF} * sqrt(exp(2*eps/d) - 1), d={obs_dim}",
            "environment": f"delta = sqrt(2*eps) / {DELTA_SCALE}",
        },
        "dots": meta,
    }, indent=2))
    print(f"\nwrote {META_OUT}")


if __name__ == "__main__":
    main()
