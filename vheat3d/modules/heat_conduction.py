import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from ..config import CudaTileConfig
from ..operators import DCT3D, IDCT3D, TileConv3d
from ..utils import sparse_mask_filter


def modulate_k(k: torch.Tensor, stats: torch.Tensor, delta_fn) -> torch.Tensor:
    """Scale the static diffusivity map k by an input-dependent factor.

    k: (..., C, D, H, W); stats: (B, C) pooled frequency statistics;
    delta_fn: maps stats -> per-(sample, channel) delta in (-1, 1).
    """
    delta = delta_fn(stats).view(stats.shape[0], stats.shape[1], 1, 1, 1)
    return k * (1.0 + delta)


class HeatConduction3D(nn.Module):
    """3D heat conduction with learnable frequency decay temperature."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 config: Optional[CudaTileConfig] = None,
                 enable_dynamic_alpha: bool = True,
                 enable_high_freq_boost: bool = True,
                 grid_size=(32, 32, 32),
                 use_bottleneck: bool = True,
                 post_norm: bool = True,
                 diffusion: str = "heat",
                 diffusion_steps: int = 1,
                 anisotropic: bool = False,
                 decay_sharpness: float = 1.0,
                 input_adaptive_k: bool = False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.config = config if config is not None else CudaTileConfig()
        self.grid_size = grid_size
        self.use_bottleneck = use_bottleneck
        self.hidden_dim = out_channels // 4 if use_bottleneck else out_channels
        self.enable_dynamic_alpha = enable_dynamic_alpha
        self.enable_high_freq_boost = enable_high_freq_boost
        self.input_adaptive_k = bool(input_adaptive_k)
        self.infer_mode = False
        assert diffusion in {"heat", "none", "ideal", "cosine", "fixed_heat"}, diffusion
        self.diffusion = diffusion
        self.diffusion_steps = int(diffusion_steps)
        self.anisotropic = anisotropic
        assert decay_sharpness > 0, decay_sharpness
        self.decay_sharpness = float(decay_sharpness)
        if anisotropic:
            self.aniso_logits = nn.Parameter(torch.zeros(3))

        self.proj_conv = TileConv3d(
            in_channels, self.hidden_dim, kernel_size=1,
            stride=1, padding=0, bias=False, config=self.config
        )

        D, H, W = grid_size
        decay = self._compute_decay_map((D, H, W), self.decay_sharpness)
        self.register_buffer('freq_decay_raw', decay, persistent=False)
        self.decay_temp = nn.Parameter(torch.tensor(1.0))

        if not enable_dynamic_alpha:
            self.alpha = nn.Parameter(torch.ones(1, self.hidden_dim, D, H, W))

        if enable_dynamic_alpha:
            self.to_k = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
                nn.ReLU(inplace=True),
            )
            if self.input_adaptive_k:
                self.to_k_in = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
                    nn.Tanh(),
                )

        self.output_scale = nn.Parameter(torch.tensor(10.0))

        if enable_high_freq_boost:
            self.high_freq_weight = nn.Parameter(torch.zeros(self.hidden_dim))

        self.dct3d = DCT3D(config=self.config)
        self.idct3d = IDCT3D(config=self.config)

        self.spatial_conv = TileConv3d(
            self.hidden_dim, self.hidden_dim, kernel_size=kernel_size,
            stride=1, padding=kernel_size // 2, groups=self.hidden_dim,
            bias=False, config=self.config
        )

        if use_bottleneck and self.hidden_dim != out_channels:
            self.out_proj = TileConv3d(
                self.hidden_dim, out_channels, kernel_size=1,
                stride=1, padding=0, bias=False, config=self.config
            )
        else:
            self.out_proj = nn.Identity()

        self.norm = nn.GroupNorm(1, out_channels) if post_norm else nn.BatchNorm3d(out_channels)
        self.activation = nn.GELU()

    @staticmethod
    def _compute_decay_map(grid_size, sharpness: float = 1.0):
        D, H, W = grid_size
        pi = math.pi
        n = torch.arange(D, dtype=torch.float).view(-1, 1, 1)
        m = torch.arange(H, dtype=torch.float).view(1, -1, 1)
        l = torch.arange(W, dtype=torch.float).view(1, 1, -1)
        r2 = (n * pi / D) ** 2 + (m * pi / H) ** 2 + (l * pi / W) ** 2
        decay = 0.1 + 0.9 * torch.exp(-sharpness * r2)
        return decay.unsqueeze(0).unsqueeze(0).float()

    def infer_init_heat3d(self, freq_embed: torch.Tensor):
        if self.input_adaptive_k:
            return
        device = freq_embed.device
        decay = self.freq_decay.to(device)
        fe = freq_embed.permute(3, 0, 1, 2).unsqueeze(0)
        k = self.to_k(fe.permute(0, 2, 3, 4, 1))
        k = k.permute(0, 4, 1, 2, 3)
        weight = torch.pow(decay + 1e-6, k)
        self.k_weight = nn.Parameter(weight, requires_grad=False)
        del self.to_k
        self.infer_mode = True

    @staticmethod
    def _alt_decay_map(grid_size, mode: str, sharpness: float = 1.0):
        D, H, W = grid_size
        n = torch.arange(D, dtype=torch.float).view(-1, 1, 1)
        m = torch.arange(H, dtype=torch.float).view(1, -1, 1)
        l = torch.arange(W, dtype=torch.float).view(1, 1, -1)
        r2 = (n * math.pi / D) ** 2 + (m * math.pi / H) ** 2 + (l * math.pi / W) ** 2
        r = torch.sqrt(sharpness * r2) / (math.pi * math.sqrt(3.0))
        if mode == "ideal":
            decay = (r <= 0.5).float() * 0.9 + 0.1
        elif mode == "cosine":
            decay = 0.1 + 0.45 * (1.0 + torch.cos(math.pi * r.clamp(0, 1)))
        else:
            raise ValueError(mode)
        return decay.unsqueeze(0).unsqueeze(0).float()

    def _diffusion_weight(self, x_freq: torch.Tensor, freq_embed) -> torch.Tensor:
        """Returns the multiplicative weight applied to x_freq (B, C, D, H, W)."""
        fd, fh, fw = x_freq.shape[-3:]
        if self.diffusion == "none":
            weight = torch.ones(1, 1, fd, fh, fw, device=x_freq.device, dtype=x_freq.dtype)
        elif self.diffusion in {"ideal", "cosine"}:
            decay = self._alt_decay_map(self.grid_size, self.diffusion,
                                        self.decay_sharpness).to(x_freq.device)
            weight = F.interpolate(decay, size=(fd, fh, fw), mode="trilinear", align_corners=False)
        elif self.diffusion == "fixed_heat":
            decay = F.interpolate(self.freq_decay_raw.to(x_freq.device), size=(fd, fh, fw),
                                  mode="trilinear", align_corners=False)
            weight = decay
        else:  # "heat": existing behaviour
            if self.infer_mode and not self.input_adaptive_k:
                weight = self.k_weight
            elif freq_embed is not None and self.enable_dynamic_alpha:
                decay = F.interpolate(self.freq_decay, size=(fd, fh, fw),
                                      mode="trilinear", align_corners=False)
                fe = freq_embed.permute(3, 0, 1, 2).unsqueeze(0)
                fe = F.interpolate(fe, size=(fd, fh, fw), mode="trilinear", align_corners=False)
                k = self.to_k(fe.permute(0, 2, 3, 4, 1))
                k = k.permute(0, 4, 1, 2, 3)
                if self.input_adaptive_k:
                    stats = x_freq.abs().mean(dim=(-3, -2, -1))
                    k = modulate_k(k, stats, self.to_k_in)
                weight = torch.pow(decay + 1e-6, k + 1e-6)
            else:
                weight = F.interpolate(self.alpha, size=(fd, fh, fw),
                                       mode="trilinear", align_corners=False)
        if self.anisotropic:
            n = torch.linspace(0, math.pi, fd, device=x_freq.device).view(-1, 1, 1)
            m = torch.linspace(0, math.pi, fh, device=x_freq.device).view(1, -1, 1)
            l = torch.linspace(0, math.pi, fw, device=x_freq.device).view(1, 1, -1)
            a = self.aniso_logits[0].exp() + 1e-6
            b = self.aniso_logits[1].exp() + 1e-6
            c = self.aniso_logits[2].exp() + 1e-6
            aniso = torch.exp(-(a * n ** 2 + b * m ** 2 + c * l ** 2))
            weight = weight * aniso.to(weight.dtype)
        if self.diffusion_steps > 1:
            weight = weight ** self.diffusion_steps
        return weight

    @property
    def freq_decay(self):
        return torch.pow(self.freq_decay_raw, self.decay_temp)

    def _apply_freq_weight(self, x_freq, freq_embed, device):
        weight = self._diffusion_weight(x_freq, freq_embed)
        x_freq = x_freq * weight

        if self.enable_high_freq_boost:
            hf = self.high_freq_weight.view(1, -1, 1, 1, 1).sigmoid()
            x_freq = x_freq * (1.0 + 0.1 * hf)

        return x_freq

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                freq_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.proj_conv(x, mask)

        x_freq = self.dct3d(x, mask)
        x_freq = self._apply_freq_weight(x_freq, freq_embed, x.device)
        x = self.idct3d(x_freq, mask)

        x = self.spatial_conv(x, mask)
        x = sparse_mask_filter(x, mask)

        if isinstance(self.out_proj, TileConv3d):
            x = self.out_proj(x, mask)
        else:
            x = self.out_proj(x)

        x = self.norm(x)
        x = self.activation(x)
        x = x * self.output_scale

        return x
