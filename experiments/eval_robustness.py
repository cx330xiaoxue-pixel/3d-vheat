"""Evaluate a checkpoint under the perturbation axes of E2/E4.

Supports the heat-diffusion ablation variants via --diffusion / --diffusion-steps /
--anisotropic / --static-alpha, so ablation checkpoints can be scored with the
exact architecture they were trained with.
"""
import argparse
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (  # noqa: E402
    build_model, evaluate, load_checkpoint, load_points_and_labels, save_json, seed_all, timestamp,
)
from experiments.robustness.axes import AXES, run_axis  # noqa: E402
from experiments.spectral.spectral import band_mask, dct, idct  # noqa: E402

# Two-sided 95% t multipliers for small seed counts (n = number of seeds).
_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def field_lowpass_tensor(f: torch.Tensor, r: float) -> torch.Tensor:
    """Extra DCT low-pass on the voxel field at test time (r<1 keeps low freqs)."""
    if r >= 1.0:
        return f
    orig_dtype = f.dtype
    fd = f.float()
    mask = band_mask(fd.shape[-3:], 0.0, r).to(fd.device)
    return idct(dct(fd) * mask).to(orig_dtype)


def field_hf_ratio(f: torch.Tensor, hf_floor: float = 0.5) -> torch.Tensor:
    """Per-(sample, channel) fraction of spectral power above `hf_floor`."""
    fd = f.float()
    power = dct(fd).pow(2)
    hf = band_mask(fd.shape[-3:], hf_floor, 1.0001).to(fd.device)
    total = power.sum(dim=(-3, -2, -1))
    high = (power * hf).sum(dim=(-3, -2, -1))
    return high / (total + 1e-12)


def field_lowpass_adaptive(f: torch.Tensor, r: float, thresh: float,
                           hf_floor: float = 0.5) -> torch.Tensor:
    """Low-pass only the samples whose high-frequency ratio exceeds `thresh`."""
    filtered = field_lowpass_tensor(f, r)
    ratio = field_hf_ratio(f, hf_floor).mean(dim=1)
    keep = (ratio > thresh).view(-1, *([1] * (f.dim() - 1)))
    return torch.where(keep, filtered, f)


def attach_field_lowpass(model: torch.nn.Module, r: float, thresh: float = 0.0,
                         hf_floor: float = 0.5):
    def hook(module, inputs, output):
        f, mask = output
        if thresh > 0:
            f = field_lowpass_adaptive(f, r, thresh, hf_floor)
        else:
            f = field_lowpass_tensor(f, r)
        return f, mask
    return model.voxelization.register_forward_hook(hook)


def summarize(accs: List[float]) -> Dict:
    arr = np.asarray(accs, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    t = _T95.get(len(arr), 1.96)
    ci95 = t * std / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return {"acc_mean": float(arr.mean()), "acc_std": std, "ci95": ci95,
            "per_seed": [float(a) for a in accs]}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--allow-random-weights", action="store_true")
    p.add_argument("--data-pts", type=str, default="data/modelnet40_test_points.npy")
    p.add_argument("--data-lbl", type=str, default="data/modelnet40_test_labels.npy")
    p.add_argument("--voxel-mode", type=str, default="gaussian")
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--backbone", type=str, default="small")
    p.add_argument("--dct-augment", action="store_true")
    p.add_argument("--diffusion", type=str, default="heat",
                   choices=["heat", "none", "ideal", "cosine", "fixed_heat"])
    p.add_argument("--diffusion-steps", type=int, default=1)
    p.add_argument("--anisotropic", action="store_true")
    p.add_argument("--diffusion-sharpness", type=float, default=1.0)
    p.add_argument("--static-alpha", action="store_true")
    p.add_argument("--input-adaptive-k", action="store_true")
    p.add_argument("--axes", nargs="+", default=["gaussian_noise", "dropout", "coarse_quantize"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--limit", type=int, default=0, help="0 = full test set")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--field-lowpass", type=float, default=0.0,
                   help="test-time DCT low-pass cutoff on the voxel field in [0,1]; 0 disables")
    p.add_argument("--field-lowpass-thresh", type=float, default=0.0,
                   help=">0 enables adaptive mode: filter only samples with HF ratio above it")
    p.add_argument("--field-hf-floor", type=float, default=0.5,
                   help="frequency radius above which power counts as high-frequency")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, required=True)
    return p


def run(argv: Optional[List[str]] = None) -> Dict:
    args = build_parser().parse_args(argv)
    for name in args.axes:
        if name not in AXES:
            raise SystemExit(f"unknown axis: {name} (available: {sorted(AXES)})")

    pts, lbls = load_points_and_labels(args.data_pts, args.data_lbl)
    if args.limit:
        pts, lbls = pts[:args.limit], lbls[:args.limit]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = build_model(voxel_mode=args.voxel_mode, sigma=args.sigma,
                        grid_size=args.grid, backbone=args.backbone,
                        dct_augment=args.dct_augment, device=device,
                        diffusion=args.diffusion,
                        diffusion_steps=args.diffusion_steps,
                        anisotropic=args.anisotropic,
                        decay_sharpness=args.diffusion_sharpness,
                        enable_dynamic_alpha=not args.static_alpha,
                        input_adaptive_k=args.input_adaptive_k)
    if args.ckpt:
        load_checkpoint(model, args.ckpt, device=device)
    elif not args.allow_random_weights:
        raise SystemExit("--ckpt required unless --allow-random-weights is set")

    if args.field_lowpass and args.field_lowpass < 1.0:
        attach_field_lowpass(model, args.field_lowpass,
                             thresh=args.field_lowpass_thresh,
                             hf_floor=args.field_hf_floor)

    clean = evaluate(model, pts, lbls, args.batch_size, device)

    axes_out: Dict[str, Dict] = {}
    for axis in args.axes:
        per_level: Dict[str, List[float]] = {}
        for seed in args.seeds:
            seed_all(seed)
            x = torch.from_numpy(pts).float().to(device)
            gen = torch.Generator(device=device).manual_seed(seed)
            for lv, xp in run_axis(axis, x, generator=gen):
                xp_np = xp.detach().cpu().numpy()
                acc = evaluate(model, xp_np, lbls, args.batch_size, device)["acc"]
                per_level.setdefault(str(lv), []).append(acc)
        axes_out[axis] = {lv: summarize(accs) for lv, accs in per_level.items()}

    payload = {
        "meta": {
            "ckpt": args.ckpt, "voxel_mode": args.voxel_mode, "sigma": args.sigma,
            "grid": args.grid, "backbone": args.backbone,
            "dct_augment": args.dct_augment, "device": device,
            "diffusion": args.diffusion, "diffusion_steps": args.diffusion_steps,
            "anisotropic": args.anisotropic, "static_alpha": args.static_alpha,
            "decay_sharpness": args.diffusion_sharpness,
            "field_lowpass": args.field_lowpass,
            "field_lowpass_thresh": args.field_lowpass_thresh,
            "field_hf_floor": args.field_hf_floor,
            "n_samples": int(len(pts)), "seeds": list(args.seeds),
            "axes": list(args.axes), "timestamp": timestamp(),
        },
        "clean": clean,
        "axes": axes_out,
    }
    save_json(args.out, payload)
    print(f"[OK] wrote {args.out} | clean_acc={clean['acc']:.4f} n={clean['n']}")
    return payload


if __name__ == "__main__":
    run()
