"""Re-run plot_motivation's data loading and dump the curves as JSON for the website."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from plot_motivation import ALGOS, NOISE_TYPES, NOISE_LEVELS, load_robustness, compute_mean_ci

OUT = Path(__file__).resolve().parent / "motivation.json"


def main():
    series = {}
    for algo in ALGOS:
        data = load_robustness(algo)
        per_noise = {}
        for nt in NOISE_TYPES:
            means, cis = [], []
            for lvl in NOISE_LEVELS:
                m, ci = compute_mean_ci(data[nt][lvl])
                means.append(None if m is None else float(m))
                cis.append(None if ci is None else float(ci))
            per_noise[nt] = {"levels": NOISE_LEVELS, "mean": means, "ci": cis}
        series[algo["label"]] = {
            "color": algo["color"],
            "linestyle": algo["ls"],
            "noise": per_noise,
        }
    OUT.write_text(json.dumps({
        "noise_types": NOISE_TYPES,
        "levels": NOISE_LEVELS,
        "series": series,
    }, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
