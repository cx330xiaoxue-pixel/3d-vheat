"""Compare two or more eval_robustness JSONs (deltas vs a reference).

Usage:
  python -m experiments.report.compare_runs --ref A.json --runs B.json C.json
"""
import argparse
import json
from typing import Dict, List


def _acc(payload: dict, axis: str, level: str) -> float:
    return payload["axes"][axis][level]["acc_mean"]


def compare_payloads(payloads: Dict[str, dict], ref_name: str = "ref") -> dict:
    names = list(payloads)
    clean = {n: payloads[n]["clean"]["acc"] for n in names}
    axes: Dict[str, Dict[str, dict]] = {}
    ref = payloads[ref_name]
    for axis in sorted(ref["axes"]):
        axes[axis] = {}
        for level in sorted(ref["axes"][axis], key=float):
            accs = {n: _acc(payloads[n], axis, level) for n in names}
            axes[axis][level] = {
                "accs": accs,
                "deltas": {n: accs[n] - accs[ref_name] for n in names},
            }
    return {"ref": ref_name, "clean": clean, "axes": axes}


def format_markdown(out: dict) -> str:
    names = list(out["clean"])
    header = "| axis | level | " + " | ".join(names) + " | " + \
             " | ".join(f"d({n})" for n in names if n != out["ref"]) + " |"
    lines = [header, "|" + "---|" * (len(names) + len(names) - 1 + 3)]
    for axis, levels in out["axes"].items():
        for level, rec in levels.items():
            vals = " | ".join(f"{rec['accs'][n]:.4f}" for n in names)
            deltas = " | ".join(
                f"{rec['deltas'][n]*100:+.2f}pp" for n in names if n != out["ref"])
            lines.append(f"| {axis} | {level} | {vals} | {deltas} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref-name", default="ref")
    args = ap.parse_args()
    payloads = {args.ref_name: json.load(open(args.ref))}
    for path in args.runs:
        name = path.split("/")[-1].replace(".json", "")
        payloads[name] = json.load(open(path))
    out = compare_payloads(payloads, ref_name=args.ref_name)
    print("clean:", {k: round(v, 4) for k, v in out["clean"].items()})
    print(format_markdown(out))


if __name__ == "__main__":
    main()
