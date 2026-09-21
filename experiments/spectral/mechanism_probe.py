"""D2 mechanism probe at arbitrary resolution/backbone.

Re-tests the band-product hypothesis (accuracy drop ~ sum_b transfer_gain[b] *
corruption_energy[b]) with the same estimator as `mechanism_test.py`, but
parameterized so it can run at 32^3 with pre-existing checkpoints, e.g. the
heat/none pair (`checkpoints_fields32_{heat,none}_100`).

Also reports per-variant (within-model) R^2 over the 8 shared conditions, which
isolates "does the band model rank corruptions for one model?" from "does it
generalize across variants?".

Outputs: experiments/results/d2_mechanism_<tag>.json
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.common import (  # noqa: E402
    build_model, load_checkpoint, load_points_and_labels, save_json, timestamp,
)
from experiments.robustness.axes import AXES  # noqa: E402
from experiments.spectral.mechanism_test import (  # noqa: E402
    CONDITIONS, band_masks, transfer_gains,
)
from experiments.spectral.spectral import dct as dct3  # noqa: E402

DEFAULT_CKPTS_32 = {
    "heat": "checkpoints_fields32_heat_100/best_model.pth",
    "none": "checkpoints_fields32_none_100/best_model.pth",
}
DEFAULT_ACTUALS_32 = {
    "heat": "experiments/results/e7_32_heat.json",
    "none": "experiments/results/e7_32_none.json",
}
VARIANT_FLAGS = {"heat": [], "none": ["--diffusion", "none"]}


def parse_kv(items: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items or []:
        name, _, path = it.partition("=")
        if not path:
            raise SystemExit(f"expected name=path, got: {it}")
        out[name] = path
    return out


def flags_to_kwargs(flags: List[str]) -> Dict:
    kw: Dict = {}
    i = 0
    while i < len(flags):
        key = flags[i].lstrip("-").replace("-", "_")
        val = flags[i + 1]
        if key == "diffusion":
            kw[key] = val
        i += 2
    return kw


def corruption_spectra(pts: torch.Tensor, grid: int, sigma: float, device: str,
                       n_bands: int = 6) -> Dict[str, np.ndarray]:
    """s[cond][b] = mean DCT band power of (field_perturbed - field_clean)."""
    from vheat3d.modules.sparse_voxelization import SparseVoxelization
    vox = SparseVoxelization(grid_size=(grid, grid, grid), voxel_mode="gaussian",
                             sigma=sigma, bounds_mode="batch").to(device)
    with torch.no_grad():
        f_clean, _ = vox(pts)
    masks = band_masks((grid, grid, grid), n_bands, device)
    out = {}
    for axis, level in CONDITIONS:
        gen = torch.Generator(device=device).manual_seed(0)
        xp = AXES[axis]["step"](pts, level, gen)
        with torch.no_grad():
            f_p, _ = vox(xp)
            diff = dct3(f_p - f_clean)
        power = (diff ** 2).mean(dim=(0, 1))
        out[f"{axis}:{level}"] = np.array([float(power[m.bool()].mean()) for m in masks])
    return out


def r2_of(preds: List[float], acts: List[float]) -> Tuple[float, float]:
    if len(preds) < 3:
        return 0.0, 0.0
    r = float(np.corrcoef(preds, acts)[0, 1])
    return r, r * r


def run(argv: Optional[List[str]] = None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--backbone", type=str, default="small")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--bands", type=int, default=6)
    p.add_argument("--ckpt", action="append", default=None, help="name=path (repeatable)")
    p.add_argument("--actual", action="append", default=None, help="name=path (repeatable)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="experiments/results/d2_mechanism_32.json")
    args = p.parse_args(argv)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ckpts = parse_kv(args.ckpt) or dict(DEFAULT_CKPTS_32)
    actuals = parse_kv(args.actual) or dict(DEFAULT_ACTUALS_32)
    if set(ckpts) != set(actuals):
        raise SystemExit("--ckpt and --actual names must match")

    pts_np, _ = load_points_and_labels("data/modelnet40_test_points.npy",
                                       "data/modelnet40_test_labels.npy")
    pts = torch.from_numpy(pts_np[:args.n_samples]).float().to(device)
    spectra = corruption_spectra(pts, args.grid, args.sigma, device, args.bands)

    per_variant_points: Dict[str, List[Dict]] = {}
    gains_out: Dict[str, List[float]] = {}
    for name, ckpt in ckpts.items():
        d = json.load(open(actuals[name]))
        clean = d["clean"]["acc"]
        model = build_model(sigma=args.sigma, grid_size=args.grid, backbone=args.backbone,
                            device=device, **flags_to_kwargs(VARIANT_FLAGS.get(name, [])))
        load_checkpoint(model, ckpt, device=device)
        g = transfer_gains(model, pts, device, n_bands=args.bands)
        gains_out[name] = g.tolist()
        pts_v: List[Dict] = []
        for axis, level in CONDITIONS:
            key = f"{axis}:{level}"
            acc = d["axes"][axis][str(level)]["acc_mean"]
            pts_v.append({"variant": name, "cond": key, "actual": clean - acc,
                          "pred": float(np.sum(g * spectra[key]))})
        per_variant_points[name] = pts_v
        r, r2 = r2_of([q["pred"] for q in pts_v], [q["actual"] for q in pts_v])
        print(f"{name}: within-R^2={r2:.3f} (n={len(pts_v)}) gains={np.round(g, 3).tolist()}")

    points = [q for v in per_variant_points.values() for q in v]
    r, r2 = r2_of([q["pred"] for q in points], [q["actual"] for q in points])
    print(f"pooled N={len(points)} Pearson r={r:.3f} R^2={r2:.3f}")

    payload = {
        "meta": {
            "grid": args.grid, "sigma": args.sigma, "backbone": args.backbone,
            "n_samples": args.n_samples, "bands": args.bands,
            "ckpts": ckpts, "actuals": actuals, "timestamp": timestamp(),
        },
        "transfer_gains": gains_out,
        "corruption_spectra": {k: v.tolist() for k, v in spectra.items()},
        "points": points,
        "pearson_r": r, "r2": r2,
        "per_variant_r2": {
            name: r2_of([q["pred"] for q in v], [q["actual"] for q in v])[1]
            for name, v in per_variant_points.items()
        },
    }
    save_json(args.out, payload)
    print(f"[OK] wrote {args.out}")
    return payload


if __name__ == "__main__":
    run()
