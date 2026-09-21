import torch
import torch.nn as nn
import math
from typing import Optional
from ..config import CudaTileConfig


def _make_dct_matrix(N, device=None, dtype=None):
    """Create N×N orthogonal DCT-II matrix."""
    n = torch.arange(N, device=device, dtype=dtype).float().view(1, -1)
    k = torch.arange(N, device=device, dtype=dtype).float().view(-1, 1)
    W = torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
    W *= math.sqrt(2.0 / N)
    W[0, :] /= math.sqrt(2.0)
    return W


def _apply_dct_1d(x, dim, W):
    """Apply 1D DCT along given dimension using matrix W."""
    N = x.shape[dim]
    if N == 1:
        return x
    perm = list(range(x.dim()))
    perm[0], perm[dim] = perm[dim], perm[0]
    x_p = x.permute(perm).contiguous()
    flat = x_p.view(N, -1)
    result = W.to(x.dtype) @ flat
    result = result.view(x_p.shape).permute(perm)
    return result.contiguous()


def _apply_idct_1d(x, dim, W):
    """Apply 1D IDCT along given dimension using inverse matrix W^T."""
    N = x.shape[dim]
    if N == 1:
        return x
    perm = list(range(x.dim()))
    perm[0], perm[dim] = perm[dim], perm[0]
    x_p = x.permute(perm).contiguous()
    flat = x_p.view(N, -1)
    result = W.to(x.dtype).T @ flat
    result = result.view(x_p.shape).permute(perm)
    return result.contiguous()


class DCT3DFunction(torch.autograd.Function):
    """Proper 3D DCT using cosine basis functions."""

    _cache = {}

    @staticmethod
    def _get_dct(N, device):
        key = (N, str(device))
        if key not in DCT3DFunction._cache:
            DCT3DFunction._cache[key] = _make_dct_matrix(N, device=device)
        return DCT3DFunction._cache[key]

    @staticmethod
    def forward(ctx, x, mask, config):
        ctx.save_for_backward(x, mask)
        ctx.config = config
        B, C, D, H, W = x.shape
        device = x.device

        WD = DCT3DFunction._get_dct(D, device)
        WH = DCT3DFunction._get_dct(H, device)
        WW = DCT3DFunction._get_dct(W, device)

        x_m = x * mask
        out = _apply_dct_1d(x_m, 2, WD)
        out = _apply_dct_1d(out, 3, WH)
        out = _apply_dct_1d(out, 4, WW)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        config = ctx.config
        return IDCT3DFunction.apply(grad_output, mask, config), None, None


class IDCT3DFunction(torch.autograd.Function):
    """Proper 3D IDCT using cosine basis functions (DCT-III)."""

    _cache = {}

    @staticmethod
    def _get_dct(N, device):
        key = (N, str(device))
        if key not in IDCT3DFunction._cache:
            IDCT3DFunction._cache[key] = _make_dct_matrix(N, device=device)
        return IDCT3DFunction._cache[key]

    @staticmethod
    def forward(ctx, x, mask, config):
        ctx.save_for_backward(x, mask)
        ctx.config = config
        B, C, D, H, W = x.shape
        device = x.device

        WD = IDCT3DFunction._get_dct(D, device)
        WH = IDCT3DFunction._get_dct(H, device)
        WW = IDCT3DFunction._get_dct(W, device)

        out = _apply_idct_1d(x, 4, WW)
        out = _apply_idct_1d(out, 3, WH)
        out = _apply_idct_1d(out, 2, WD)
        out = out * mask
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        config = ctx.config
        return DCT3DFunction.apply(grad_output, mask, config), None, None


class DCT3D(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else CudaTileConfig()

    def forward(self, x, mask):
        return DCT3DFunction.apply(x, mask, self.config)


class IDCT3D(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else CudaTileConfig()

    def forward(self, x, mask):
        return IDCT3DFunction.apply(x, mask, self.config)
