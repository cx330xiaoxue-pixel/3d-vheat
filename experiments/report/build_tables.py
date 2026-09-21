"""Render experiment JSON into Markdown tables (source of paper tables)."""
from typing import Dict, List


def robustness_table(payload: Dict) -> str:
    lines = ["| axis | level | acc_mean | acc_std | n_seeds |",
             "|---|---|---|---|---|"]
    for axis, levels in sorted(payload["axes"].items()):
        for level, rec in sorted(levels.items(), key=lambda kv: float(kv[0])):
            lines.append(
                f"| {axis} | {level} | {rec['acc_mean']:.4f} | "
                f"{rec['acc_std']:.4f} | {len(rec['per_seed'])} |"
            )
    lines.append(f"\nclean acc: {payload['clean']['acc']:.4f} (n={payload['clean']['n']})")
    return "\n".join(lines)


def pareto_table(rows: List[Dict]) -> str:
    lines = ["| model | params (M) | GFLOPs | top-1 |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['params_m']:.2f} | {r['gflops']:.2f} | {r['top1']:.2f} |")
    return "\n".join(lines)
