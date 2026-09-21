"""D1: batch-bounds vs per-sample-bounds A/B across the 7 heat-ablation checkpoints.

For each checkpoint, measure clean test accuracy under:
  bounds_mode in {batch, per_sample} x eval batch_size in {16, 64, 128}.
Writes experiments/results/e4_bounds_ab.json and prints a table.
"""
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (  # noqa: E402
    build_model, evaluate, load_checkpoint, load_points_and_labels, save_json, timestamp,
)

VARIANTS = {
    "baseline": [],
    "none": ["--diffusion", "none"],
    "ideal": ["--diffusion", "ideal"],
    "steps2": ["--diffusion-steps", "2"],
    "sharp050": ["--diffusion-sharpness", "0.5"],
    "sharp200": ["--diffusion-sharpness", "2.0"],
    "sharp400": ["--diffusion-sharpness", "4.0"],
}

BATCH_SIZES = [16, 64, 128]


def flags_to_kwargs(flags: List[str]) -> Dict:
    kw = {}
    i = 0
    while i < len(flags):
        key = flags[i].lstrip("-").replace("-", "_")
        val = flags[i + 1]
        if key == "diffusion_steps":
            kw["diffusion_steps"] = int(val)
        elif key == "diffusion_sharpness":
            kw["decay_sharpness"] = float(val)
        else:
            kw[key] = val
        i += 2
    return kw


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts, lbls = load_points_and_labels("data/modelnet40_test_points.npy",
                                       "data/modelnet40_test_labels.npy")
    out = {"meta": {"n_samples": len(pts), "timestamp": timestamp()}, "results": {}}
    for v, flags in VARIANTS.items():
        ckpt = f"checkpoints_heatab_pilot_{v}_g16_tiny/best_model.pth"
        kw = flags_to_kwargs(flags)
        rec = {"batch": {}, "per_sample": {}}
        for mode in ["batch", "per_sample"]:
            model = build_model(sigma=0.1, grid_size=16, backbone="tiny",
                                device=device, bounds_mode=mode, **kw)
            load_checkpoint(model, ckpt, device=device)
            for bs in BATCH_SIZES:
                acc = evaluate(model, pts, lbls, batch_size=bs, device=device)["acc"]
                rec[mode][str(bs)] = acc
                print(f"{v:<10s} {mode:<11s} bs={bs:<4d} acc={acc:.4f}", flush=True)
        out["results"][v] = rec
    save_json("experiments/results/e4_bounds_ab.json", out)
    print("[OK] wrote experiments/results/e4_bounds_ab.json")


if __name__ == "__main__":
    main()
