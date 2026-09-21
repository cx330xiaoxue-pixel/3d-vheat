"""
3D-vHeat 骨干网络
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..config import CudaTileConfig
from .sparse_voxelization import SparseVoxelization
from .heat_conduction import HeatConduction3D
from ..utils import Timer, adaptive_avg_pool3d


class VHeat3D(nn.Module):
    """3D-vHeat 骨干网络
    
    输入：B×N×3 的稀疏点云
    输出：B×C×D×H×W 的体素级特征图、B×C 的全局特征
    """
    
    def __init__(self, in_channels: int = 1, hidden_channels: int = 64,
                 num_layers: int = 4, grid_size: Tuple[int, int, int] = (32, 32, 32),
                 config: Optional[CudaTileConfig] = None):
        """
        Args:
            in_channels: 输入通道数
            hidden_channels: 隐藏通道数
            num_layers: 热传导层数
            grid_size: 体素网格大小 (D, H, W)
            config: CUDA Tiles 配置
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.grid_size = grid_size
        self.config = config if config is not None else CudaTileConfig()
        
        # 稀疏点云体素化
        self.voxelization = SparseVoxelization(
            voxel_size=0.1,
            grid_size=grid_size,
            config=self.config
        )
        
        # 特征投影层
        self.feature_proj = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 多层热传导
        self.heat_conduction_layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_channels if i == 0 else hidden_channels
            out_ch = hidden_channels
            self.heat_conduction_layers.append(
                HeatConduction3D(in_ch, out_ch, kernel_size=3, config=self.config)
            )
        
        # 性能统计
        self.forward_time = 0.0
        self.call_count = 0
    
    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            points: 输入点云 (B, N, 3)
            
        Returns:
            voxel_features: 体素级特征 (B, C, D, H, W)
            global_features: 全局特征 (B, C)
        """
        if self.config.enable_profiling:
            with Timer("VHeat3D", enable_cuda=True) as timer:
                voxel_features, global_features = self._forward(points)
            self.forward_time += timer.get_elapsed()
            self.call_count += 1
        else:
            voxel_features, global_features = self._forward(points)
        
        return voxel_features, global_features
    
    def _forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行前向传播"""
        # 1. 稀疏点云体素化
        # print(f"VHeat3D._forward: 开始体素化...")
        voxel_features, mask = self.voxelization(points)
        # print(f"   体素化完成: voxel_features={voxel_features.shape}, mask={mask.shape}")
        # print(f"   体素化设备: voxel_features={voxel_features.device}, mask={mask.device}")
        
        # 2. 特征投影
        # print(f"VHeat3D._forward: 开始特征投影...")
        # print(f"   特征投影前: voxel_features={voxel_features.shape}")
        voxel_features = self.feature_proj(voxel_features)
        # print(f"   特征投影后: voxel_features={voxel_features.shape}")
        # print(f"   特征投影设备: {voxel_features.device}")
        
        # 3. 多层热传导
        # print(f"VHeat3D._forward: 开始热传导层处理...")
        for i, heat_layer in enumerate(self.heat_conduction_layers):
            # print(f"   热传导层 {i} 输入: voxel_features={voxel_features.shape}, mask={mask.shape}")
            voxel_features = heat_layer(voxel_features, mask)
            # print(f"   热传导层 {i} 输出: {voxel_features.shape}")
        
        # 4. 全局特征提取（自适应 3D 平均池化）
        # print(f"VHeat3D._forward: 开始全局特征提取...")
        global_features = adaptive_avg_pool3d(voxel_features, mask)
        # print(f"   全局特征提取完成: {global_features.shape}")
        
        return voxel_features, global_features
    
    def get_stats(self) -> dict:
        """获取性能统计"""
        avg_time = self.forward_time / self.call_count if self.call_count > 0 else 0.0
        
        layer_stats = []
        for i, layer in enumerate(self.heat_conduction_layers):
            layer_stats.append({
                f"layer_{i}": layer.get_stats()
            })
        
        return {
            "forward_time": self.forward_time,
            "call_count": self.call_count,
            "avg_time": avg_time,
            "voxelization_stats": self.voxelization.get_stats(),
            "heat_conduction_layers": layer_stats,
        }
    
    def reset_stats(self):
        """重置统计"""
        self.forward_time = 0.0
        self.call_count = 0
        self.voxelization.reset_stats()
        for layer in self.heat_conduction_layers:
            layer.reset_stats()
