"""Parse D4 convergence/variance training logs into pre-registered JSONs.

Implements master-plan section 7.3 metrics: E95, E73, best_acc, AUC, and
per-arm mean +/- 95% CI (t(2)=4.303 for n=3). Writes
`experiments/results/d4_convergence.json` and `d4_variance.json`.
"""
import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*\|[^\n]*?test_acc=([\d.]+)%")
LOG_RE = re.compile(r"d4_([a-z0-9_]+?)_s(\d+)\.txt$")

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
       7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def parse_log_text(text: str) -> Tuple[List[int], List[float]]:
    pairs = EPOCH_RE.findall(text)
    pairs = sorted((int(e), float(a)) for e, a in pairs)
    epochs = [e for e, _ in pairs]
    accs = [a for _, a in pairs]
    return epochs, accs


def reach_epoch(accs: List[float], frac: Optional[float] = None,
                threshold: Optional[float] = None) -> int:
    if not accs:
        return 0
    thr = threshold if threshold is not None else frac * max(accs)
    for i, a in enumerate(accs):
        if a >= thr:
            return i
    return len(accs)


def auc_of(accs: List[float]) -> float:
    if not accs:
        return 0.0
    if len(accs) == 1:
        return float(accs[0])
    return float(np.trapz(np.asarray(accs, dtype=np.float64), dx=1.0) / (len(accs) - 1))


def summarize_run(text: str) -> Dict:
    _, accs = parse_log_text(text)
    if not accs:
        return {"n_epochs": 0, "best_acc": 0.0, "e95": 0, "e73": 0, "auc": 0.0}
    return {
        "n_epochs": len(accs),
        "best_acc": float(max(accs)),
        "e95": reach_epoch(accs, frac=0.95),
        "e73": reach_epoch(accs, threshold=73.0),
        "auc": auc_of(accs),
    }


def summarize_arm(best_accs: List[float]) -> Dict:
    arr = np.asarray(best_accs, dtype=np.float64)
    n = int(arr.size)
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    t = T95.get(n, 1.96 if n > 10 else float("nan"))
    ci = float(t * std / np.sqrt(n)) if n > 1 else 0.0
    return {"n": n, "mean": float(arr.mean()), "std": std, "ci95": ci,
            "per_seed": [float(x) for x in arr]}


def welch_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    from scipy import stats
    res = stats.ttest_ind(np.asarray(a, dtype=np.float64),
                          np.asarray(b, dtype=np.float64), equal_var=False)
    return float(res.statistic), float(res.pvalue)


def build_report(logs: Dict[str, str]) -> Dict:
    runs: Dict[str, Dict] = {}
    by_arm: Dict[str, List[Dict]] = {}
    for fname, text in sorted(logs.items()):
        m = LOG_RE.search(fname)
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2))
        rec = summarize_run(text)
        rec["seed"] = seed
        runs[f"{arm}_s{seed}"] = rec
        by_arm.setdefault(arm, []).append(rec)

    arms: Dict[str, Dict] = {}
    for arm, recs in by_arm.items():
        recs = sorted(recs, key=lambda r: r["seed"])
        agg = summarize_arm([r["best_acc"] for r in recs])
        agg["e95_median"] = float(np.median([r["e95"] for r in recs]))
        agg["e73_median"] = float(np.median([r["e73"] for r in recs]))
        agg["auc_mean"] = float(np.mean([r["auc"] for r in recs]))
        agg["runs"] = [f"{arm}_s{r['seed']}" for r in recs]
        arms[arm] = agg

    gates: Dict[str, Optional[bool]] = {
        "C1_heat_vs_none": None,
        "C1_ideal_same_direction": None,
        "C2_variance": None,
    }
    if "baseline" in arms and "none" in arms:
        thr = 0.7 * arms["none"]["e95_median"]
        gates["C1_heat_vs_none"] = bool(arms["baseline"]["e95_median"] <= thr)
        if "ideal" in arms:
            gates["C1_ideal_same_direction"] = bool(arms["ideal"]["e95_median"] <= thr)

    variance: Dict[str, Dict] = {}
    if "baseline" in arms and "static_alpha" in arms:
        base_seeds = [r["best_acc"] for r in by_arm["baseline"]]
        stat_seeds = [r["best_acc"] for r in by_arm["static_alpha"]]
        t, p = welch_ttest(base_seeds, stat_seeds)
        variance = {
            "baseline": arms["baseline"],
            "static_alpha": arms["static_alpha"],
            "welch_t": t,
            "welch_p": p,
        }
        gates["C2_variance"] = bool(p < 0.05 and
                                    arms["baseline"]["ci95"] < arms["static_alpha"]["ci95"])

    return {"runs": runs, "arms": arms, "gates": gates, "variance": variance}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", default=".")
    ap.add_argument("--pattern", default="logs_heatab_d4_*.txt")
    ap.add_argument("--out-conv", default="experiments/results/d4_convergence.json")
    ap.add_argument("--out-var", default="experiments/results/d4_variance.json")
    args = ap.parse_args()

    logs = {}
    for path in sorted(glob.glob(os.path.join(args.logdir, args.pattern))):
        with open(path, errors="ignore") as f:
            logs[os.path.basename(path)] = f.read()
    report = build_report(logs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_conv)), exist_ok=True)
    conv = {"runs": report["runs"], "arms": report["arms"],
            "gates": {k: v for k, v in report["gates"].items() if k.startswith("C1")}}
    var = {"variance": report["variance"],
           "gates": {k: v for k, v in report["gates"].items() if k.startswith("C2")}}
    with open(args.out_conv, "w") as f:
        json.dump(conv, f, indent=2, sort_keys=True)
    with open(args.out_var, "w") as f:
        json.dump(var, f, indent=2, sort_keys=True)
    for arm, agg in sorted(report["arms"].items()):
        print(f"{arm:14s} best={agg['mean']:.2f}+/-{agg['ci95']:.2f} "
              f"E95_med={agg['e95_median']:.1f} E73_med={agg['e73_median']:.1f} "
              f"AUC={agg['auc_mean']:.2f}")
    print("gates:", report["gates"])
    print(f"[OK] {args.out_conv}, {args.out_var}")


if __name__ == "__main__":
    main()
