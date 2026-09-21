"""Test-time adaptive spectral filtering (eval-only).

Modes:
  clean     : no filtering (control)
  filtered  : always filter (control, equals --field-lowpass)
  avg       : average of clean and filtered logits (fixed 2x cost)
  gate      : run filtered branch only when the clean prediction's max
              softmax confidence < tau (early-exit; cost = 1 + usage_rate)

Writes JSON with accuracy per (mode, tau) plus filter usage rates.
"""
import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (  # noqa: E402
    build_model, load_checkpoint, load_points_and_labels, save_json, seed_all, timestamp,
)
from experiments.eval_robustness import attach_field_lowpass  # noqa: E402
from experiments.robustness.axes import AXES, run_axis  # noqa: E402


def select_logits(clean_logits: torch.Tensor, filtered_logits: torch.Tensor,
                  mode: str, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (chosen logits, boolean mask of samples using the filtered branch)."""
    if mode == "clean":
        return clean_logits, torch.zeros(clean_logits.shape[0], dtype=torch.bool,
                                         device=clean_logits.device)
    if mode == "filtered":
        return filtered_logits, torch.ones(clean_logits.shape[0], dtype=torch.bool,
                                           device=clean_logits.device)
    if mode == "avg":
        return (clean_logits + filtered_logits) / 2.0, torch.ones(
            clean_logits.shape[0], dtype=torch.bool, device=clean_logits.device)
    if mode == "gate":
        conf = torch.softmax(clean_logits.double(), dim=1).max(dim=1).values
        use = conf < tau
        return torch.where(use.unsqueeze(1), filtered_logits, clean_logits), use
    raise ValueError(f"unknown mode: {mode}")


def _pair_forwards(model, pts_np: np.ndarray, batch_size: int, device: str, r: float):
    """One clean pass + one filtered pass; returns CPU logits for both."""
    n = len(pts_np)
    clean_out, filt_out = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            x = torch.from_numpy(pts_np[i:i + batch_size]).float().to(device)
            clean_out.append(model(x).float().cpu())
    handle = attach_field_lowpass(model, r)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            x = torch.from_numpy(pts_np[i:i + batch_size]).float().to(device)
            filt_out.append(model(x).float().cpu())
    handle.remove()
    return torch.cat(clean_out), torch.cat(filt_out)


def _accuracy(logits: torch.Tensor, lbls: np.ndarray) -> float:
    y = torch.from_numpy(lbls).long()
    return float((logits.argmax(1) == y).float().mean())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-pts", default="data/modelnet40_test_points.npy")
    p.add_argument("--data-lbl", default="data/modelnet40_test_labels.npy")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--backbone", default="tiny")
    p.add_argument("--dct-augment", action="store_true")
    p.add_argument("--field-lowpass", type=float, default=0.3)
    p.add_argument("--modes", nargs="+", default=["clean", "filtered", "avg", "gate"])
    p.add_argument("--taus", nargs="+", type=float, default=[0.5, 0.7, 0.8, 0.9])
    p.add_argument("--axes", nargs="+", default=["gaussian_noise", "dropout",
                                                 "coarse_quantize", "fps"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    return p


def run(argv: Optional[List[str]] = None) -> Dict:
    args = build_parser().parse_args(argv)
    pts, lbls = load_points_and_labels(args.data_pts, args.data_lbl)
    if args.limit:
        pts, lbls = pts[:args.limit], lbls[:args.limit]
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = build_model(sigma=args.sigma, grid_size=args.grid, backbone=args.backbone,
                        dct_augment=args.dct_augment, device=device)
    load_checkpoint(model, args.ckpt, device=device)

    combos: List[Tuple[str, Optional[float]]] = []
    for mode in args.modes:
        if mode == "gate":
            combos.extend(("gate", t) for t in args.taus)
        else:
            combos.append((mode, None))

    acc: Dict[str, Dict[str, List[float]]] = {f"{m}@{t if t is not None else '-'}": {"clean": []}
                                              for m, t in combos}
    usage: Dict[str, List[float]] = {k: [] for k in acc}

    def eval_all(pts_np, lbls_np, tag):
        clean_logits, filt_logits = _pair_forwards(model, pts_np, args.batch_size,
                                                   device, args.field_lowpass)
        for m, t in combos:
            key = f"{m}@{t if t is not None else '-'}"
            out, use = select_logits(clean_logits, filt_logits, m,
                                     t if t is not None else 0.0)
            acc[key].setdefault(tag, []).append(_accuracy(out, lbls_np))
            usage[key].append(float(use.float().mean()))

    eval_all(pts, lbls, "clean")
    for axis in args.axes:
        if axis not in AXES:
            raise SystemExit(f"unknown axis: {axis}")
        for seed in args.seeds:
            seed_all(seed)
            x = torch.from_numpy(pts).float().to(device)
            gen = torch.Generator(device=device).manual_seed(seed)
            for lv, xp in run_axis(axis, x, generator=gen):
                eval_all(xp.detach().cpu().numpy(), lbls, f"{axis}/{lv}")

    results = {}
    for key in acc:
        results[key] = {
            "clean_acc_mean": float(np.mean(acc[key]["clean"])),
            "filter_usage_mean": float(np.mean(usage[key])),
            "axes": {t: float(np.mean(v)) for t, v in acc[key].items() if t != "clean"},
        }
    payload = {"meta": {"ckpt": args.ckpt, "sigma": args.sigma, "grid": args.grid,
                        "backbone": args.backbone, "dct_augment": args.dct_augment,
                        "field_lowpass": args.field_lowpass, "taus": args.taus,
                        "axes": args.axes, "seeds": args.seeds,
                        "n_samples": int(len(pts)), "timestamp": timestamp()},
               "results": results}
    save_json(args.out, payload)
    for key, rec in results.items():
        print(f"{key:14s} clean={rec['clean_acc_mean']:.4f} usage={rec['filter_usage_mean']:.2f}")
    return payload


if __name__ == "__main__":
    run()
