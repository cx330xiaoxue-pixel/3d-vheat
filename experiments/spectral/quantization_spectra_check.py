"""Recompute binary-vs-gaussian spectral HF ratios with the corrected DCT.

Writes experiments/results/quantization_spectra.json (theory prediction 2 check).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.common import load_points_and_labels, save_json, timestamp  # noqa: E402
from experiments.spectral.spectral import quantization_spectra  # noqa: E402


def main():
    pts, _ = load_points_and_labels("data/modelnet40_test_points.npy",
                                    "data/modelnet40_test_labels.npy")
    x = torch.from_numpy(pts[:256]).float()
    out = {}
    for sigma in (0.1, 0.2):
        out[f"sigma{sigma}"] = quantization_spectra(x, grid_size=16, sigma=sigma)
    payload = {"grid": 16, "n_samples": 256, "timestamp": timestamp(), "results": out}
    save_json("experiments/results/quantization_spectra.json", payload)
    for key, rec in out.items():
        print(f"{key}: hf_binary={rec['hf_ratio_binary']:.4f} "
              f"hf_gaussian={rec['hf_ratio_gaussian']:.4f}")


if __name__ == "__main__":
    main()
