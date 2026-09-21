import torch
import torch.nn as nn
from ..config import CudaTileConfig
from .dct3d import DCT3D, IDCT3D

DEFAULT_PROBS = {"hf": 0.2, "drop": 0.2, "jitter": 0.2, "band": 0.1}


def probs_from_skip(skip):
    """Component probabilities with the named components zeroed out."""
    probs = dict(DEFAULT_PROBS)
    for name in skip:
        if name not in probs:
            raise KeyError(f"unknown DCTAugment component: {name}")
        probs[name] = 0.0
    return probs


class DCTAugment(nn.Module):
    def __init__(self, config=None, probs=None):
        super().__init__()
        self.config = config if config is not None else CudaTileConfig()
        self.probs = dict(DEFAULT_PROBS if probs is None else probs)
        self.dct = DCT3D(self.config)
        self.idct = IDCT3D(self.config)

    def forward(self, x, mask):
        if not self.training:
            return x

        freq = self.dct(x, mask)
        B, C, D, H, W = freq.shape

        # 1. High-frequency masking (low-pass filter with random cutoff)
        if torch.rand(1).item() < self.probs["hf"]:
            cutoff = torch.rand(1).item() * 0.6 + 0.2
            di = torch.arange(D, device=freq.device, dtype=torch.float) / D
            hj = torch.arange(H, device=freq.device, dtype=torch.float) / H
            wk = torch.arange(W, device=freq.device, dtype=torch.float) / W
            di, hj, wk = torch.meshgrid(di, hj, wk, indexing='ij')
            freq_mask = (di + hj + wk) / 3.0 < cutoff
            freq = freq * freq_mask[None, None, ...]

        # 2. Frequency dropout
        if torch.rand(1).item() < self.probs["drop"]:
            drop_prob = torch.rand(1).item() * 0.15
            drop_mask = torch.rand_like(freq) > drop_prob
            freq = freq * drop_mask

        # 3. Frequency jitter
        if torch.rand(1).item() < self.probs["jitter"]:
            noise_std = torch.rand(1).item() * 0.05
            noise = torch.randn_like(freq) * noise_std
            freq = freq + noise

        # 4. Band masking (random continuous band)
        if torch.rand(1).item() < self.probs["band"]:
            band_dim = torch.randint(0, 3, (1,)).item()
            band_start = torch.rand(1).item() * 0.8
            band_end = band_start + torch.rand(1).item() * (1.0 - band_start)
            s_idx = int(band_start * [D, H, W][band_dim])
            e_idx = int(band_end * [D, H, W][band_dim])
            if band_dim == 0:
                freq[:, :, s_idx:e_idx, :, :] = 0
            elif band_dim == 1:
                freq[:, :, :, s_idx:e_idx, :] = 0
            else:
                freq[:, :, :, :, s_idx:e_idx] = 0

        x_aug = self.idct(freq, mask)
        return x_aug
