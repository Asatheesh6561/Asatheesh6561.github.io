"""Parse per-env LaTeX robustness tables into a single JSON file consumed by the website."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TABLES_DIR = ROOT / "tables"
OUT = Path(__file__).resolve().parent / "results.json"

ALGS = ["PPO", "SAC", "DDPG"]
ENVS = ["HalfCheetah", "Hopper", "Walker2d", "Humanoid"]
PERTURBS = ["Action", "Observation", "Environment"]
EPSILONS = [0.1, 0.2, 0.3, 0.4, 0.5]

# Order the rows appear in inside each per-env table (matches every .tex file).
METHOD_ORDER = [
    "Nominal",
    "Fixed",
    "Space",
    "Naive",
    "Accel",
    "DomRand",
    "SelfPaced",
]

METHOD_LABELS = {
    "Nominal":   {"PPO": "PPO",            "SAC": "SAC",            "DDPG": "DDPG"},
    "Fixed":     "DR (Fixed)",
    "Space":     "DR (Space)",
    "Naive":     "DR (Naive)",
    "Accel":     "DR (Accel)",
    "DomRand":   "DR (Dom. Rand.)",
    "SelfPaced": "DR-SPCRL (Ours)",
}

VALUE_RE = re.compile(r"\$?\\mathbf\{?(-?[\d.]+)\s*\\pm\s*(-?[\d.]+)\}?\$?|\$(-?[\d.]+)\s*\\pm\s*(-?[\d.]+)\$")


def parse_value_cell(cell: str):
    """Return (mean, std, is_bold) parsed from a LaTeX numeric cell."""
    cell = cell.strip()
    is_bold = "mathbf" in cell
    m = VALUE_RE.search(cell)
    if not m:
        raise ValueError(f"could not parse cell: {cell!r}")
    if m.group(1) is not None:
        return float(m.group(1)), float(m.group(2)), is_bold
    return float(m.group(3)), float(m.group(4)), is_bold


def parse_table(tex: str):
    """Return {perturbation: {method_key: [{eps, mean, std, best}, ...]}}."""
    out = {p: {} for p in PERTURBS}
    body = tex.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    # Strip multirow control sequences.
    body = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", body)
    # Split into rows by `\\` at end of line.
    rows = [r for r in re.split(r"\\\\\s*", body) if r.strip()]

    current_perturb = None
    method_idx = 0
    for raw in rows:
        # Skip \midrule separators between perturbation blocks.
        raw = raw.replace(r"\midrule", "").strip()
        if not raw:
            continue
        cells = [c.strip() for c in raw.split("&")]
        # A row is either "Perturb & Method & vals..." (8 cells) or " & Method & vals..." (8 cells)
        # In either form: cells[0]=perturb-or-empty, cells[1]=method, cells[2..6]=epsilons
        if len(cells) < 7:
            continue
        first = cells[0]
        if first and first not in {""} and not first.startswith("$"):
            # New perturbation header (e.g. "Action", "Observation", "Environment")
            # Sometimes leading multirow label is followed by spaces.
            label = first.strip()
            # Map LaTeX-tex labels back to canonical names.
            for p in PERTURBS:
                if p.lower() in label.lower():
                    current_perturb = p
                    method_idx = 0
                    break

        method_key = METHOD_ORDER[method_idx]
        method_idx += 1

        series = []
        for j, eps in enumerate(EPSILONS):
            mean, std, bold = parse_value_cell(cells[2 + j])
            series.append({"eps": eps, "mean": mean, "std": std, "best": bold})
        out[current_perturb][method_key] = series

    return out


def label_for(alg: str, method_key: str) -> str:
    if method_key == "Nominal":
        return alg
    if method_key == "SelfPaced":
        return f"DR-{alg}-SPCRL (Ours)"
    suffix = METHOD_LABELS[method_key].replace("DR ", "")  # e.g. "(Fixed)"
    return f"DR-{alg} {suffix}"


def main():
    results = {}
    for alg in ALGS:
        results[alg] = {}
        for env in ENVS:
            path = TABLES_DIR / f"{alg}_{env}_robustness.tex"
            if not path.exists():
                continue
            tex = path.read_text()
            parsed = parse_table(tex)
            # Attach human-readable labels.
            labeled = {}
            for perturb, series_by_method in parsed.items():
                labeled[perturb] = []
                for mkey in METHOD_ORDER:
                    if mkey not in series_by_method:
                        continue
                    labeled[perturb].append({
                        "key": mkey,
                        "label": label_for(alg, mkey),
                        "is_ours": mkey == "SelfPaced",
                        "series": series_by_method[mkey],
                    })
            results[alg][env] = labeled

    OUT.write_text(json.dumps({
        "algorithms": ALGS,
        "environments": ENVS,
        "perturbations": PERTURBS,
        "epsilons": EPSILONS,
        "results": results,
    }, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
