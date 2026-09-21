"""
vHeat算法完整迁移版 - 3D点云分类网络
核心特性迁移自 vHeat (CVPR 2025), https://github.com/MzeroMiko/vHeat

核心迁移特性：
1. Layer Scale (γ1, γ2): 层缩放因子，稳定深层网络训练
2. DropPath: 随机深度正则化，防止过拟合
3. Post Norm: 残差连接后归一化（与原始vHeat一致）
4. MLP/FFN分支: 增强特征变换能力
5. 频率嵌入 (freq_embed): 可学习的频率权重
6. 推理加速 (infer_mode): 预计算频率权重
7. 多阶段架构: patch embedding -> 多阶段HeatBlock -> 分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from typing import Optional, Tuple, List
from ..config import CudaTileConfig
from .sparse_voxelization import SparseVoxelization
from .heat_conduction import HeatConduction3D
from ..operators.dct_augment import DCTAugment, probs_from_skip
from ..utils import Timer, adaptive_avg_pool3d


class DropPath(nn.Module):
    """随机深度DropPath模块"""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}'


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """随机深度的drop path操作"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class LayerNorm3d(nn.Module):
    """3D Layer Normalization - 用于Post Norm"""
    def __init__(self, normalized_shape, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class Mlp3D(nn.Module):
    """3D MLP模块 - 迁移自vHeat的Mlp"""
    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, act_layer: nn.Module = nn.GELU,
                 drop: float = 0., channels_first: bool = True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if channels_first:
            self.fc1 = nn.Conv3d(in_features, hidden_features, kernel_size=1, padding=0)
            self.fc2 = nn.Conv3d(hidden_features, out_features, kernel_size=1, padding=0)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.channels_first = channels_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_first:
            B, C, D, H, W = x.shape
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
        return x


class HeatBlock3D(nn.Module):
    """
    3D热传导Block - 完整迁移自vHeat的HeatBlock

    核心特性：
    1. Post Norm: x = x + DropPath(norm(op(x, freq_embed)))
    2. Layer Scale: γ1, γ2 用于稳定训练
    3. MLP分支: FFN增强特征变换
    4. DropPath: 随机深度正则化
    """

    def __init__(self,
                 dim: int,
                 heat_layer: nn.Module,
                 drop_path: float = 0.,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 post_norm: bool = True,
                 layer_scale: float = None,
                 use_checkpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.post_norm = post_norm
        self.use_checkpoint = use_checkpoint

        self.norm1 = LayerNorm3d(dim)
        self.op = heat_layer
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = LayerNorm3d(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp3D(in_features=dim, hidden_features=mlp_hidden_dim,
                             act_layer=nn.GELU, drop=dropout, channels_first=True)

        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def _forward(self, x: torch.Tensor, mask: torch.Tensor, freq_embed: torch.Tensor = None) -> torch.Tensor:
        if not self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.op(x, mask, freq_embed)))
                if self.mlp_branch:
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.op(self.norm1(x), mask, freq_embed))
                if self.mlp_branch:
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        if self.post_norm:
            x = x + self.drop_path(self.gamma1[:, None, None, None] * self.norm1(self.op(x, mask, freq_embed)))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None, None] * self.norm2(self.mlp(x)))
        else:
            x = x + self.drop_path(self.gamma1[:, None, None, None] * self.op(self.norm1(x), mask, freq_embed))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None, None] * self.mlp(self.norm2(x)))
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freq_embed: torch.Tensor = None) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mask, freq_embed, use_reentrant=False)
        else:
            return self._forward(x, mask, freq_embed)


class VHeat3DMigrated(nn.Module):
    """
    vHeat算法完整迁移版 - 3D点云分类网络

    核心架构（迁移自vHeat）：
    1. Voxel Embedding: 点云 -> 体素特征
    2. 多阶段架构: [stage1, stage2, stage3, stage4]
    3. 每阶段多HeatBlock + freq_embed
    4. 分类器: LayerNorm3d + AdaptiveAvgPool3d + Linear

    与原始vHeat的对应关系：
    - patch_embed -> voxelization + feature_proj
    - HeatBlock -> HeatBlock3D
    - LayerNorm2d -> LayerNorm3d
    - freq_embed -> freq_embed (per stage)
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 40,
                 depths: List[int] = [2, 2, 6, 2],
                 dims: List[int] = [64, 128, 256, 512],
                 grid_size: Tuple[int, int, int] = (32, 32, 32),
                 config: Optional[CudaTileConfig] = None,
                 drop_path_rate: float = 0.2,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 post_norm: bool = True,
                 layer_scale: float = 1e-5,
                 use_checkpoint: bool = False,
                 enable_dynamic_alpha: bool = True,
                 enable_high_freq_boost: bool = True,
                 use_bottleneck: bool = True,
                 enable_dct_augment: bool = False,
                 dct_augment_skip=(),
                 voxel_mode: str = 'binary',
                 voxel_sigma: float = 0.05,
                 diffusion: str = 'heat',
                 diffusion_steps: int = 1,
                 anisotropic: bool = False,
                 decay_sharpness: float = 1.0,
                 input_adaptive_k: bool = False,
                 voxel_bounds_mode: str = 'batch'):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.depths = depths
        self.dims = dims
        self.grid_size = grid_size
        self.config = config if config is not None else CudaTileConfig()
        self.infer_mode = False

        if len(depths) != len(dims):
            raise ValueError(f"depths ({len(depths)}) must match dims ({len(dims)})")

        self.voxelization = SparseVoxelization(
            voxel_size=0.1,
            grid_size=grid_size,
            config=self.config,
            voxel_mode=voxel_mode,
            sigma=voxel_sigma,
            bounds_mode=voxel_bounds_mode,
        )

        self.dct_augment = (
            DCTAugment(self.config, probs=probs_from_skip(tuple(dct_augment_skip)))
            if enable_dct_augment else nn.Identity())

        first_dim = dims[0]
        self.feature_proj = nn.Sequential(
            nn.Conv3d(in_channels, first_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm3d(first_dim) if post_norm else nn.BatchNorm3d(first_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.freq_embed = nn.ParameterList()
        res_list = []
        for i in range(self.num_layers):
            grid_dim = grid_size[0] // (2 ** i)
            embed_dim = dims[i] // 4 if use_bottleneck else dims[i]
            self.freq_embed.append(nn.Parameter(
                torch.zeros(grid_dim, grid_dim, grid_dim, embed_dim), requires_grad=True))
            nn.init.trunc_normal_(self.freq_embed[i], std=.02)
            res_list.append(grid_dim)

        self.stages = nn.ModuleList()
        stage_idx = 0

        for i_layer in range(self.num_layers):
            for d in range(depths[i_layer]):
                heat_layer = HeatConduction3D(
                    dims[i_layer], dims[i_layer], kernel_size=3, config=self.config,
                    enable_dynamic_alpha=enable_dynamic_alpha,
                    enable_high_freq_boost=enable_high_freq_boost,
                    grid_size=(res_list[i_layer],) * 3,
                    use_bottleneck=use_bottleneck,
                    post_norm=post_norm,
                    diffusion=diffusion,
                    diffusion_steps=diffusion_steps,
                    anisotropic=anisotropic,
                    decay_sharpness=decay_sharpness,
                    input_adaptive_k=input_adaptive_k
                )

                block = HeatBlock3D(
                    dim=dims[i_layer],
                    heat_layer=heat_layer,
                    drop_path=dpr[stage_idx],
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    post_norm=post_norm,
                    layer_scale=layer_scale,
                    use_checkpoint=use_checkpoint
                )
                self.stages.append(block)
                stage_idx += 1

            if i_layer < self.num_layers - 1:
                downsample = nn.Sequential(
                    nn.Conv3d(dims[i_layer], dims[i_layer + 1], kernel_size=3, stride=2, padding=1, bias=False),
                    LayerNorm3d(dims[i_layer + 1]) if post_norm else nn.BatchNorm3d(dims[i_layer + 1])
                )
                self.stages.append(downsample)

        self.final_norm = LayerNorm3d(dims[-1]) if post_norm else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def infer_init(self):
        """推理初始化 - 预计算频率权重"""
        freq_idx = 0
        for stage in self.stages:
            if isinstance(stage, HeatBlock3D):
                stage.op.infer_init_heat3d(self.freq_embed[freq_idx])
                freq_idx = (freq_idx + 1) % self.num_layers
        self.infer_mode = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """特征提取"""
        voxel_features, mask = self.voxelization(x)
        if isinstance(self.dct_augment, DCTAugment):
            voxel_features = self.dct_augment(voxel_features, mask)
        voxel_features = self.feature_proj(voxel_features)

        freq_idx = 0
        for stage in self.stages:
            if isinstance(stage, HeatBlock3D):
                voxel_features = stage(voxel_features, mask, self.freq_embed[freq_idx])
            else:
                voxel_features = stage(voxel_features)
                if voxel_features.shape[2:] != mask.shape[2:]:
                    mask = torch.nn.functional.max_pool3d(
                        mask.float(), kernel_size=2, stride=2
                    ).bool()
                freq_idx += 1

        voxel_features = self.final_norm(voxel_features)
        return voxel_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.forward_features(x)
        pooled = self.avgpool(features).flatten(1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回特征和 logits"""
        features = self.forward_features(x)
        pooled = self.avgpool(features).flatten(1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits, pooled


def VHeat3D_Tiny_Migrated(num_classes=40, **kwargs):
    """轻量级vHeat迁移模型"""
    default_kwargs = {
        'depths': [2, 2, 2, 2],
        'dims': [32, 64, 128, 256],
        'drop_path_rate': 0.2,
        'layer_scale': 1e-5,
    }
    default_kwargs.update(kwargs)
    model = VHeat3DMigrated(
        num_classes=num_classes,
        **default_kwargs
    )
    return model


def VHeat3D_Small_Migrated(num_classes=40, **kwargs):
    """小规模vHeat迁移模型"""
    default_kwargs = {
        'depths': [2, 2, 6, 2],
        'dims': [64, 128, 256, 512],
        'drop_path_rate': 0.2,
        'layer_scale': 1e-5,
    }
    default_kwargs.update(kwargs)
    model = VHeat3DMigrated(
        num_classes=num_classes,
        **default_kwargs
    )
    return model


def VHeat3D_Base_Migrated(num_classes=40, **kwargs):
    """基础vHeat迁移模型"""
    default_kwargs = {
        'depths': [3, 4, 8, 3],
        'dims': [96, 192, 384, 768],
        'drop_path_rate': 0.3,
        'layer_scale': 1e-5,
    }
    default_kwargs.update(kwargs)
    model = VHeat3DMigrated(
        num_classes=num_classes,
        **default_kwargs
    )
    return model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VHeat3D_Small_Migrated(num_classes=40).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    x = torch.randn(4, 1024, 3).to(device)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
