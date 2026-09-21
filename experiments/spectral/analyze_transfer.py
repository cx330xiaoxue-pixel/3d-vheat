"""Layer-level spectral transfer analysis (E5).

Measures how band-limited perturbations applied to a HeatBlock's frequency
input map to output-feature perturbations, and compares against the
analytical HCO decay mask.
"""
import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.common import build_model, load_checkpoint, save_json, timestamp  # noqa: E402
from experiments.spectral.spectral import band_mask, dct, idct  # noqa: E402


def _first_hco(model):
    for stage in model.stages:
        if hasattr(stage, "op"):
            return stage.op
    raise RuntimeError("no HeatBlock3D stage found")


@torch.no_grad()
def measure(model, points: torch.Tensor, n_bands: int = 6, sigma_noise: float = 0.1,
            seed: int = 0) -> dict:
    g = torch.Generator(device=points.device).manual_seed(seed)
    op = _first_hco(model)
    x = model.feature_proj(model.voxelization(points)[0])
    x_proj = op.proj_conv(x, torch.ones_like(x[:, :1]))
    x_freq = op.dct3d(x_proj, torch.ones_like(x[:, :1]))
    device = x_freq.device
    shape = x_freq.shape[-3:]
    results = {"bands": [], "decay_mask_stats": {}}
    decay = op.freq_decay
    results["decay_mask_stats"] = {
        "min": float(decay.min()), "max": float(decay.max()), "mean": float(decay.mean()),
    }
    output_ref = None
    bands = [(i / n_bands, (i + 1) / n_bands) for i in range(n_bands)]
    for r_lo, r_hi in bands:
        mask = band_mask(shape, r_lo, r_hi).to(device)
        delta_freq = torch.randn(x_freq.shape, generator=g, device=device) * sigma_noise * mask
        x_pert = idct(dct(x_proj) + delta_freq)
        out = op.spatial_conv(x_pert, torch.ones_like(x[:, :1]))
        if output_ref is None:
            output_ref = op.spatial_conv(x_proj, torch.ones_like(x[:, :1]))
        in_energy = float(delta_freq.pow(2).mean())
        out_energy = float((out - output_ref).pow(2).mean())
        gain = out_energy / max(in_energy, 1e-12)
        results["bands"].append({"r_lo": r_lo, "r_hi": r_hi,
                                 "in_energy": in_energy, "out_energy": out_energy,
                                 "gain": gain})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-pts", default="data/modelnet40_test_points.npy")
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--backbone", default="small")
    ap.add_argument("--n-bands", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = build_model(sigma=args.sigma, grid_size=args.grid,
                        backbone=args.backbone, device=device)
    load_checkpoint(model, args.ckpt, device=device)
    pts = np.load(args.data_pts)[:8].astype(np.float32)
    x = torch.from_numpy(pts).float().to(device)
    res = measure(model, x, n_bands=args.n_bands)
    res["meta"] = {"ckpt": args.ckpt, "sigma": args.sigma, "grid": args.grid,
                   "backbone": args.backbone, "timestamp": timestamp()}
    save_json(args.out, res)
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
