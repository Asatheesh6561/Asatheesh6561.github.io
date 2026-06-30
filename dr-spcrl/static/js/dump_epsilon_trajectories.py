"""Dump per-(env, alg) epsilon and beta trajectories to JSON for the website's
interactive 2x6 grid. Reuses the loaders from plot_epsilon_progression.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from plot_epsilon_progression import (        # noqa: E402
    RESULTS_DIR, average_epsilon_over_seeds, pretty_names,
)

OUT = Path(__file__).resolve().parent / "epsilon_trajectories.json"

# Fixed 6-column layout: one column per (env, backbone) cell, two rows.
# Matches the existing PNG: rows = ε panel, β panel? Actually the PNG packs
# 12 plots into 2 rows x 6 cols of (env, algo) cells with ε + β on twin axes.
ENVS = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Humanoid-v4"]
BACKBONES = ["ppo", "sac", "ddpg"]


def downsample(steps, mean, std, target=120):
    """Reduce point count by stride sampling so the JSON stays compact."""
    n = len(steps)
    if n <= target:
        return steps.tolist(), mean.tolist(), std.tolist()
    idx = np.linspace(0, n - 1, target).round().astype(int)
    return steps[idx].tolist(), mean[idx].tolist(), std[idx].tolist()


def main():
    cells = []
    for env in ENVS:
        for backbone in BACKBONES:
            algo = f"dr{backbone}_self_paced_continuous"
            run_dirs = sorted(RESULTS_DIR.glob(f"{env}__{algo}__*"))
            if not run_dirs:
                continue
            steps, mean_eps, std_eps, mean_beta, std_beta, n_seeds = average_epsilon_over_seeds(run_dirs)
            if steps is None:
                continue
            env_label, algo_label = pretty_names(env, algo)
            s, m, sd = downsample(steps, mean_eps, std_eps)
            beta = None
            if mean_beta is not None:
                _, mb, sdb = downsample(steps, mean_beta, std_beta)
                # Clamp non-positive for log axis.
                mb = [max(1e-6, v) for v in mb]
                sdb = list(sdb)
                beta = {"mean": mb, "std": sdb}
            cells.append({
                "env": env_label,
                "algo": algo_label,
                "backbone": backbone.upper(),
                "n_seeds": n_seeds,
                "steps": s,
                "epsilon": {"mean": m, "std": sd},
                "beta": beta,
            })

    OUT.write_text(json.dumps({
        "envs": ENVS,
        "backbones": [b.upper() for b in BACKBONES],
        "cells": cells,
    }))
    sz = OUT.stat().st_size
    print(f"wrote {OUT} ({len(cells)} cells, {sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
