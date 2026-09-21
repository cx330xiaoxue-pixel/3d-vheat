"""Point-cloud perturbation primitives.

All functions operate on (B, N, 3) float tensors and take an optional
torch.Generator for determinism. Points are assumed to be normalized to
the unit ball (as produced by the training pipeline).
"""
import math
from typing import Optional

import torch


def gaussian_noise(xyz: torch.Tensor, sigma: float,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
    noise = torch.randn(xyz.shape, generator=generator, device=xyz.device, dtype=xyz.dtype)
    return xyz + sigma * noise


def random_dropout(xyz: torch.Tensor, keep_ratio: float,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
    B, N, _ = xyz.shape
    n_keep = max(1, int(round(N * keep_ratio)))
    idx = torch.stack([
        torch.randperm(N, generator=generator, device=xyz.device)[:n_keep]
        for _ in range(B)
    ])
    return torch.gather(xyz, 1, idx[..., None].expand(-1, -1, 3))


def fps_downsample(xyz: torch.Tensor, n_out: int,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
    B, N, _ = xyz.shape
    if n_out >= N:
        return xyz
    device = xyz.device
    idx = torch.zeros(B, n_out, dtype=torch.long, device=device)
    idx[:, 0] = torch.randint(0, N, (B,), generator=generator, device=device)
    dist = torch.full((B, N), float("inf"), device=device, dtype=xyz.dtype)
    rows = torch.arange(B, device=device)
    cur = xyz[rows, idx[:, 0]]
    for i in range(1, n_out):
        d = ((xyz - cur[:, None, :]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        idx[:, i] = dist.argmax(dim=1)
        cur = xyz[rows, idx[:, i]]
    return torch.gather(xyz, 1, idx[..., None].expand(-1, -1, 3))


def outlier_replace(xyz: torch.Tensor, ratio: float,
                    generator: Optional[torch.Generator] = None) -> torch.Tensor:
    B, N, _ = xyz.shape
    k = max(1, int(round(N * ratio)))
    out = xyz.clone()
    for b in range(B):
        pick = torch.randperm(N, generator=generator, device=xyz.device)[:k]
        uni = torch.rand(k, 3, generator=generator, device=xyz.device, dtype=xyz.dtype) * 2 - 1
        out[b, pick] = uni
    return out


def _random_rotation_matrices(B: int, generator=None, device="cpu", dtype=torch.float32):
    A = torch.randn(B, 3, 3, generator=generator, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    signs[signs == 0] = 1.0
    Q = Q * signs.unsqueeze(-2)
    neg = torch.det(Q) < 0
    if neg.any():
        Q[neg, :, -1] = -Q[neg, :, -1]
    return Q


def random_rotation(xyz: torch.Tensor, generator: Optional[torch.Generator] = None,
                    yaw_only: bool = False,
                    max_yaw_deg: Optional[float] = None) -> torch.Tensor:
    B = xyz.shape[0]
    device, dtype = xyz.device, xyz.dtype
    if yaw_only:
        if max_yaw_deg is None:
            angles = torch.rand(B, generator=generator, device=device, dtype=dtype) * 2 * math.pi
        else:
            lim = max_yaw_deg * math.pi / 180.0
            angles = (torch.rand(B, generator=generator, device=device, dtype=dtype) * 2 - 1) * lim
        c, s = torch.cos(angles), torch.sin(angles)
        z = torch.zeros_like(c)
        o = torch.ones_like(c)
        R = torch.stack([
            torch.stack([c, z, s], dim=-1),
            torch.stack([z, o, z], dim=-1),
            torch.stack([-s, z, c], dim=-1),
        ], dim=-2)
    else:
        R = _random_rotation_matrices(B, generator=generator, device=device, dtype=dtype)
    return torch.einsum("bij,bnj->bni", R, xyz)


def coarse_quantize(xyz: torch.Tensor, grid_size: int, extent: float = 2.0) -> torch.Tensor:
    cell = extent / grid_size
    shifted = (xyz + extent / 2.0) / cell
    snapped = torch.round(shifted) * cell - extent / 2.0
    return snapped.clamp(-extent / 2.0, extent / 2.0)
