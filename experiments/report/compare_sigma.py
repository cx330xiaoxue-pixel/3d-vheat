"""Compare D3 sigma-dose robustness results (sigma=0.05 / 0.1 / 0.2).

Reads three eval_robustness JSONs and reports, per axis/level, the accuracy
trend across sigma plus a monotonicity count used by the D3 gate.
"""
import argparse
import json
from typing import Dict, List

DEFAULT_PATHS = {
    0.05: "experiments/results/d3_sigma0.05.json",
    0.1: "experiments/results/e4_final_baseline.json",
    0.2: "experiments/results/d3_sigma0.2.json",
}


def monotone_direction(values: List[float]) -> str:
    if all(b >= a for a, b in zip(values, values[1:])):
        return "increasing"
    if all(b <= a for a, b in zip(values, values[1:])):
        return "decreasing"
    return "non_monotone"


def build_trend(payloads: Dict[float, dict]) -> dict:
    sigmas = sorted(payloads)
    clean = [payloads[s]["clean"]["acc"] for s in sigmas]
    axes: Dict[str, Dict[str, dict]] = {}
    counts = {"increasing": 0, "decreasing": 0, "non_monotone": 0}
    for axis in sorted(payloads[sigmas[0]]["axes"]):
        axes[axis] = {}
        for level in sorted(payloads[sigmas[0]]["axes"][axis], key=float):
            accs = [payloads[s]["axes"][axis][level]["acc_mean"] for s in sigmas]
            direction = monotone_direction(accs)
            counts[direction] += 1
            axes[axis][level] = {"accs": accs, "direction": direction}
    return {"sigmas": sigmas, "clean": clean, "axes": axes, "counts": counts}


def format_markdown(trend: dict) -> str:
    sigmas = trend["sigmas"]
    header = "| axis | level | " + " | ".join(f"sigma={s}" for s in sigmas) + " | trend |"
    lines = [header, "|" + "---|" * (len(sigmas) + 3)]
    for axis, levels in trend["axes"].items():
        for level, rec in levels.items():
            vals = " | ".join(f"{a:.4f}" for a in rec["accs"])
            lines.append(f"| {axis} | {level} | {vals} | {rec['direction']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigma005", default=DEFAULT_PATHS[0.05])
    ap.add_argument("--sigma01", default=DEFAULT_PATHS[0.1])
    ap.add_argument("--sigma02", default=DEFAULT_PATHS[0.2])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    payloads = {}
    for sigma, path in ((0.05, args.sigma005), (0.1, args.sigma01), (0.2, args.sigma02)):
        with open(path) as f:
            payloads[sigma] = json.load(f)
    trend = build_trend(payloads)
    print("clean acc:", {s: round(c, 4) for s, c in zip(trend["sigmas"], trend["clean"])})
    print(format_markdown(trend))
    print("counts:", trend["counts"])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(trend, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
