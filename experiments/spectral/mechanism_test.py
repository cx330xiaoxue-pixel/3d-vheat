"""D2: spectral mechanism test.

Hypothesis: accuracy drop under a corruption ~ (model low-pass strength at band b)
x (corruption energy at band b). We measure both sides and correlate the product
with the actual drops from e4_final_*.json.

Outputs: experiments/results/d2_mechanism.json
"""
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.common import (  # noqa: E402
    build_model, load_checkpoint, load_points_and_labels, save_json, timestamp,
)
from experiments.robustness.axes import AXES  # noqa: E402
from experiments.spectral.spectral import dct as dct3, idct as idct3  # noqa: E402

VARIANTS = {
    "baseline": [],
    "none": ["--diffusion", "none"],
    "ideal": ["--diffusion", "ideal"],
    "steps2": ["--diffusion-steps", "2"],
    "sharp050": ["--diffusion-sharpness", "0.5"],
    "sharp200": ["--diffusion-sharpness", "2.0"],
    "sharp400": ["--diffusion-sharpness", "4.0"],
}
CONDITIONS = [  # (axis, level) pairs: low & high severity
    ("dropout", 0.25), ("dropout", 0.875),
    ("coarse_quantize", 32), ("coarse_quantize", 16),
    ("gaussian_noise", 0.01), ("gaussian_noise", 0.05),
    ("fps", 512), ("fps", 128),
]
N_BANDS = 6
N_SAMPLES = 32
INJECT_SEEDS = [0, 1]


def flags_to_kwargs(flags: List[str]) -> Dict:
    kw = {}
    i = 0
    while i < len(flags):
        key = flags[i].lstrip("-").replace("-", "_")
        val = flags[i + 1]
        if key == "diffusion_steps":
            kw[key] = int(val)
        elif key == "diffusion_sharpness":
            kw["decay_sharpness"] = float(val)
        else:
            kw[key] = val
        i += 2
    return kw



def band_masks(shape, n_bands=N_BANDS, device="cuda"):
    ds = [torch.arange(n, device=device, dtype=torch.float32) / n for n in shape]
    dd, hh, ww = torch.meshgrid(*ds, indexing="ij")
    r = torch.sqrt(dd ** 2 + hh ** 2 + ww ** 2) / np.sqrt(3.0)
    edges = torch.linspace(0, 1, n_bands + 1, device=device)
    return [((r >= edges[i]) & (r < edges[i + 1])).float() for i in range(n_bands)]


def transfer_gains(model, pts, device, n_bands=N_BANDS) -> np.ndarray:
    """g[b] = mean over samples/seeds of ||dlogits||^2 / ||delta||^2 per band."""
    model.eval()
    captured = {}

    def hook(_m, _inp, out):
        out = out + captured["delta"] if "delta" in captured else out
        return out

    h = model.feature_proj.register_forward_hook(hook)
    gains = np.zeros(n_bands)
    try:
        with torch.no_grad():
            logits0 = model(pts)
        feat_shape = None
        with torch.no_grad():
            f = model.feature_proj(model.voxelization(pts)[0])
        feat_shape = f.shape[-3:]
        masks = band_masks(feat_shape, n_bands, device)
        for seed in INJECT_SEEDS:
            g = torch.Generator(device=device).manual_seed(seed)
            for b in range(n_bands):
                with torch.no_grad():
                    f = model.feature_proj(model.voxelization(pts)[0])
                    noise = torch.randn(f.shape, generator=g, device=device) * f.std() * 0.1
                    delta = idct3(dct3(noise) * masks[b])
                    captured["delta"] = delta
                    logits1 = model(pts)
                    captured.clear()
            # accumulate after loop
                gains[b] += float(((logits1 - logits0) ** 2).mean()) / max(float(delta.pow(2).mean()), 1e-12)
        gains /= len(INJECT_SEEDS)
    finally:
        h.remove()
    return gains


def corruption_spectra(pts, device, n_bands=N_BANDS) -> Dict[str, np.ndarray]:
    """s[cond][b] = mean DCT band power of (field_perturbed - field_clean)."""
    from vheat3d.modules.sparse_voxelization import SparseVoxelization
    vox = SparseVoxelization(grid_size=(16, 16, 16), voxel_mode="gaussian",
                             sigma=0.1, bounds_mode="batch").to(device)
    with torch.no_grad():
        f_clean, _ = vox(pts)
    masks = band_masks((16, 16, 16), n_bands, device)
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts_np, _ = load_points_and_labels("data/modelnet40_test_points.npy",
                                       "data/modelnet40_test_labels.npy")
    pts = torch.from_numpy(pts_np[:N_SAMPLES]).float().to(device)

    spectra = corruption_spectra(pts, device)

    actual = {}
    for v in VARIANTS:
        d = json.load(open(f"experiments/results/e4_final_{v}.json"))
        clean = d["clean"]["acc"]
        for axis, level in CONDITIONS:
            key = f"{axis}:{level}"
            acc = d["axes"][axis][str(level)]["acc_mean"]
            actual[(v, key)] = clean - acc

    preds, acts, points = [], [], []
    gains_out = {}
    for v, flags in VARIANTS.items():
        ckpt = f"checkpoints_heatab_pilot_{v}_g16_tiny/best_model.pth"
        model = build_model(sigma=0.1, grid_size=16, backbone="tiny", device=device,
                            **flags_to_kwargs(flags))
        load_checkpoint(model, ckpt, device=device)
        g = transfer_gains(model, pts, device)
        gains_out[v] = g.tolist()
        for axis, level in CONDITIONS:
            key = f"{axis}:{level}"
            p = float(np.sum(g * spectra[key]))
            preds.append(p)
            acts.append(actual[(v, key)])
            points.append({"variant": v, "cond": key, "pred": p, "actual": actual[(v, key)]})
        print(f"{v}: gains={np.round(g, 3).tolist()}")

    preds = np.asarray(preds)
    acts = np.asarray(acts)
    r = float(np.corrcoef(preds, acts)[0, 1])
    r2 = r ** 2
    from scipy.stats import spearmanr
    rho = float(spearmanr(preds, acts).statistic)
    print(f"\nN points={len(preds)}  Pearson r={r:.3f}  R^2={r2:.3f}  Spearman rho={rho:.3f}")

    save_json("experiments/results/d2_mechanism.json", {
        "meta": {"n_samples": N_SAMPLES, "seeds": INJECT_SEEDS, "bands": N_BANDS,
                 "timestamp": timestamp()},
        "transfer_gains": gains_out,
        "corruption_spectra": {k: v.tolist() for k, v in spectra.items()},
        "points": points,
        "pearson_r": r, "r2": r2, "spearman_rho": rho,
    })
    print("[OK] wrote experiments/results/d2_mechanism.json")


if __name__ == "__main__":
    main()
