"""Gate-3 probe for the input-adaptive-k (IAK) pilot.

Acceptance criterion (pre-registered): the learned modulation delta
(``to_k_in`` output, in (-1, 1), scaling k by (1 + delta)) must separate
corruption types -- noise samples get larger k, rotation samples get smaller k.

Eval-only: hooks every ``HeatConduction3D.to_k_in`` and records per-sample
deltas for clean and corrupted inputs. Outputs group statistics per condition
plus a pre-registered noise-vs-rotation separation test (AUC via Mann-Whitney).

Outputs: experiments/results/iak_kprobe_<tag>.json
"""
import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (  # noqa: E402
    build_model, load_checkpoint, load_points_and_labels, save_json, seed_all, timestamp,
)
from experiments.robustness.axes import AXES  # noqa: E402

Condition = Tuple[str, float]

DEFAULT_CONDITIONS: List[Condition] = [
    ("gaussian_noise", 0.01), ("gaussian_noise", 0.05),
    ("dropout", 0.25), ("dropout", 0.875),
    ("coarse_quantize", 16), ("fps", 128),
    ("rotation_yaw", 30), ("rotation_yaw", 45),
    ("rotation_so3", 1),
]

NOISE_GROUP: List[Condition] = [("gaussian_noise", 0.01), ("gaussian_noise", 0.05)]
ROTATION_GROUP: List[Condition] = [
    ("rotation_yaw", 30), ("rotation_yaw", 45), ("rotation_so3", 1),
]


def cond_key(cond: Condition) -> str:
    return f"{cond[0]}:{cond[1]}"


def summarize(values: Sequence[float]) -> Dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "n": int(arr.size)}


def separation(a: Sequence[float], b: Sequence[float]) -> Dict:
    """One-sided Mann-Whitney: is `a` larger than `b`? AUC = P(a > b)."""
    from scipy.stats import mannwhitneyu

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.size == 0 or bb.size == 0:
        return {"auc": 0.0, "p_greater": 1.0, "mean_a": 0.0, "mean_b": 0.0,
                "n_a": int(aa.size), "n_b": int(bb.size)}
    u, p = mannwhitneyu(aa, bb, alternative="greater")
    return {
        "auc": float(u / (aa.size * bb.size)),
        "p_greater": float(p),
        "mean_a": float(aa.mean()),
        "mean_b": float(bb.mean()),
        "n_a": int(aa.size), "n_b": int(bb.size),
    }


class DeltaCollector:
    """Collects ``to_k_in`` outputs (B, C) from every heat-conduction block."""

    def __init__(self) -> None:
        self.values: List[torch.Tensor] = []
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def attach(self, model: torch.nn.Module) -> int:
        for m in model.modules():
            if hasattr(m, "to_k_in"):
                self.handles.append(m.to_k_in.register_forward_hook(self._collect))
        return len(self.handles)

    def _collect(self, _module, _inputs, output):
        self.values.append(output.detach().float().cpu())

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def per_sample(self) -> np.ndarray:
        """Mean over layers and channels -> (B,) per-sample delta."""
        if not self.values:
            return np.zeros(0)
        stacked = torch.stack([v.mean(dim=1) for v in self.values], dim=0)
        return stacked.mean(dim=0).numpy()

    def per_layer(self) -> List[float]:
        return [float(v.mean()) for v in self.values]

    def clear(self) -> None:
        self.values = []


def run_pass(model: torch.nn.Module, x: torch.Tensor, collector: DeltaCollector,
             batch_size: int) -> Tuple[np.ndarray, List[List[float]]]:
    per_sample: List[np.ndarray] = []
    per_layer: List[List[float]] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            collector.clear()
            model(x[i:i + batch_size])
            per_sample.append(collector.per_sample())
            per_layer.append(collector.per_layer())
    return np.concatenate(per_sample), per_layer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--backbone", type=str, default="tiny")
    p.add_argument("--n-samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data-pts", type=str, default="data/modelnet40_test_points.npy")
    p.add_argument("--data-lbl", type=str, default="data/modelnet40_test_labels.npy")
    p.add_argument("--out", type=str, required=True)
    return p


def run(argv: Optional[List[str]] = None) -> Dict:
    args = build_parser().parse_args(argv)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    seed_all(args.seed)
    pts, lbls = load_points_and_labels(args.data_pts, args.data_lbl)
    pts = pts[:args.n_samples]
    x = torch.from_numpy(pts).float().to(device)

    model = build_model(sigma=args.sigma, grid_size=args.grid, backbone=args.backbone,
                        device=device, input_adaptive_k=True)
    load_checkpoint(model, args.ckpt, device=device)
    model.eval()

    collector = DeltaCollector()
    n_layers = collector.attach(model)
    if n_layers == 0:
        raise SystemExit("no to_k_in modules found -- checkpoint is not an IAK model")

    per_sample_by_cond: Dict[str, np.ndarray] = {}
    clean_vals, clean_layers = run_pass(model, x, collector, args.batch_size)

    for cond in DEFAULT_CONDITIONS:
        axis, level = cond
        if axis not in AXES:
            raise SystemExit(f"unknown axis: {axis}")
        gen = torch.Generator(device=device).manual_seed(args.seed)
        xp = AXES[axis]["step"](x, level, gen)
        vals, layers = run_pass(model, xp, collector, args.batch_size)
        per_sample_by_cond[cond_key(cond)] = vals

    collector.detach()
    clean_layer_mean = np.asarray(clean_layers, dtype=np.float64).mean(axis=0).tolist()

    noise_vals = np.concatenate([per_sample_by_cond[cond_key(c)] for c in NOISE_GROUP])
    rot_vals = np.concatenate([per_sample_by_cond[cond_key(c)] for c in ROTATION_GROUP])
    clean_vs_noise = separation(noise_vals, clean_vals)
    clean_vs_rot = separation(rot_vals, clean_vals)
    noise_vs_rot = separation(noise_vals, rot_vals)

    payload = {
        "meta": {
            "ckpt": args.ckpt, "sigma": args.sigma, "grid": args.grid,
            "backbone": args.backbone, "n_samples": int(len(pts)),
            "batch_size": args.batch_size, "seed": args.seed,
            "n_to_k_in_layers": n_layers, "device": device,
            "timestamp": timestamp(),
        },
        "clean": {"summary": summarize(clean_vals), "per_layer": clean_layer_mean},
        "conditions": {
            cond_key(c): {
                "summary": summarize(per_sample_by_cond[cond_key(c)]),
            }
            for c in DEFAULT_CONDITIONS
        },
        "clean_per_sample": clean_vals.tolist(),
        "delta_per_sample": {k: v.tolist() for k, v in per_sample_by_cond.items()},
        "separation": {
            "noise_vs_clean": clean_vs_noise,
            "rotation_vs_clean": clean_vs_rot,
            "noise_vs_rotation": noise_vs_rot,
        },
        "gate": {
            "criterion": "noise delta > rotation delta (one-sided Mann-Whitney p<0.05)",
            "passed": bool(noise_vs_rot["auc"] > 0.5 and noise_vs_rot["p_greater"] < 0.05),
            "auc": noise_vs_rot["auc"],
            "p_greater": noise_vs_rot["p_greater"],
        },
    }
    save_json(args.out, payload)
    print(f"[OK] wrote {args.out} | layers={n_layers} | "
          f"noise_mean={noise_vs_rot['mean_a']:.4f} rot_mean={noise_vs_rot['mean_b']:.4f} "
          f"AUC={noise_vs_rot['auc']:.3f} p={noise_vs_rot['p_greater']:.2e} "
          f"gate={'PASS' if payload['gate']['passed'] else 'FAIL'}")
    return payload


if __name__ == "__main__":
    run()
