"""Spectral utilities: DCT bands, radial power, quantization error spectra."""
import os
import sys
from typing import Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vheat3d.operators.dct3d import DCT3DFunction, IDCT3DFunction, _make_dct_matrix  # noqa: E402


def _dct_matrices(shape, device, dtype):
    return [_make_dct_matrix(int(n), device=device, dtype=dtype) for n in shape]


def dct(x: torch.Tensor) -> torch.Tensor:
    """Separable orthonormal DCT-II along the last three dims (no mask)."""
    d, h, w = x.shape[-3:]
    mats = _dct_matrices((d, h, w), x.device, x.dtype)
    out = x
    for axis, mat in zip((-3, -2, -1), mats):
        out = torch.tensordot(out, mat.t(), dims=([axis], [0]))
        out = out.movedim(-1, axis)
    return out


def idct(x: torch.Tensor) -> torch.Tensor:
    d, h, w = x.shape[-3:]
    mats = _dct_matrices((d, h, w), x.device, x.dtype)
    out = x
    for axis, mat in zip((-3, -2, -1), mats):
        out = torch.tensordot(out, mat, dims=([axis], [0]))
        out = out.movedim(-1, axis)
    return out


def _freq_grid(shape, device, dtype):
    ds = [torch.arange(n, device=device, dtype=dtype) / n for n in shape]
    dd, hh, ww = torch.meshgrid(*ds, indexing="ij")
    return torch.sqrt(dd ** 2 + hh ** 2 + ww ** 2) / np.sqrt(3.0)


def band_mask(shape, r_lo: float, r_hi: float) -> torch.Tensor:
    grid = _freq_grid(shape, torch.device("cpu"), torch.float32)
    return ((grid >= r_lo) & (grid < r_hi)).float()


def dct_roundtrip_error(x: torch.Tensor) -> float:
    return float((idct(dct(x)) - x).abs().max())


def radial_power(x: torch.Tensor, n_bins: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    spec = dct(x)
    power = spec.abs().mean(dim=1, keepdim=True) if spec.ndim == 5 else spec.abs()
    grid = _freq_grid(spec.shape[-3:], spec.device, spec.dtype)
    edges = torch.linspace(0, 1, n_bins + 1, device=spec.device)
    pflat = power.reshape(-1, grid.numel())
    mflat = grid.reshape(-1)
    centers, vals = [], []
    for i in range(n_bins):
        m = (mflat >= edges[i]) & (mflat < edges[i + 1])
        centers.append(float((edges[i] + edges[i + 1]) / 2))
        vals.append(float(pflat[:, m].mean()) if m.any() else 0.0)
    return np.asarray(centers), np.asarray(vals)


def quantization_spectra(points: torch.Tensor, grid_size: int, sigma: float) -> dict:
    from vheat3d.modules.sparse_voxelization import SparseVoxelization
    g = (grid_size, grid_size, grid_size)
    v_bin = SparseVoxelization(grid_size=g, voxel_mode="binary", sigma=sigma)
    v_gau = SparseVoxelization(grid_size=g, voxel_mode="gaussian", sigma=sigma)
    with torch.no_grad():
        f_bin, _ = v_bin(points)
        f_gau, _ = v_gau(points)
    _, p_bin = radial_power(f_bin, n_bins=8)
    _, p_gau = radial_power(f_gau, n_bins=8)
    half = len(p_bin) // 2
    return {
        "hf_ratio_binary": float(p_bin[half:].sum() / (p_bin.sum() + 1e-12)),
        "hf_ratio_gaussian": float(p_gau[half:].sum() / (p_gau.sum() + 1e-12)),
        "radial_binary": p_bin.tolist(),
        "radial_gaussian": p_gau.tolist(),
    }
