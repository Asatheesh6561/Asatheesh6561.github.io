"""Dump training-return curves (mean + std per (env, family, algo)) to JSON
for the website's interactive 3x4 grid. Mirrors plot_training_returns.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from plot_training_returns import (   # noqa: E402
    ALGORITHM_FAMILIES, ENVIRONMENTS, DISPLAY_NAMES,
    algo_color, average_over_seeds, find_run_dirs, get_subsample, smooth,
)

OUT = Path(__file__).resolve().parent / "training_returns.json"
TARGET_POINTS = 120  # downsample each curve to keep JSON compact


def downsample(steps, values, target=TARGET_POINTS):
    n = len(steps)
    if n <= target:
        return steps.tolist(), values.tolist()
    idx = np.linspace(0, n - 1, target).round().astype(int)
    return np.asarray(steps)[idx].tolist(), np.asarray(values)[idx].tolist()


def main():
    families = list(ALGORITHM_FAMILIES.keys())     # PPO, DDPG, SAC
    out = {
        "envs": [e.replace("-v4", "") for e in ENVIRONMENTS],
        "env_ids": ENVIRONMENTS,
        "families": families,
        "cells": [],
    }

    for family in families:
        for env in ENVIRONMENTS:
            cell_traces = []
            for algo in ALGORITHM_FAMILIES[family]:
                run_dirs = find_run_dirs(env, algo)
                if not run_dirs:
                    continue
                is_sp = "self_paced" in algo
                steps, mean, std, n_seeds = average_over_seeds(run_dirs, is_sp)
                if steps is None:
                    continue
                # Smooth + subsample the same way the matplotlib plot does.
                sub = get_subsample(env)
                steps_s, mean_s = smooth(steps, mean, subsample=sub)
                std_s = np.asarray(std)[::sub]
                # Downsample to TARGET_POINTS for JSON compactness.
                xs, ys = downsample(steps_s, mean_s)
                _, sds = downsample(steps_s, std_s)
                cell_traces.append({
                    "algo": algo,
                    "label": DISPLAY_NAMES.get(algo, algo),
                    "is_ours": "self_paced" in algo,
                    "color": algo_color(algo),
                    "n_seeds": n_seeds,
                    "x": xs,
                    "y": ys,
                    "std": sds,
                })
            out["cells"].append({
                "family": family,
                "env": env.replace("-v4", ""),
                "traces": cell_traces,
            })

    Path(OUT).write_text(json.dumps(out))
    sz = OUT.stat().st_size
    print(f"wrote {OUT} ({len(out['cells'])} cells, {sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
