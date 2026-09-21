"""
3D-vHeat 分类头
用于点云分类任务
"""

import torch
import torch.nn as nn
from typing import Optional
from ..config import CudaTileConfig
from .vheat3d import VHeat3D


class VHeat3DClassifier(nn.Module):
    """3D-vHeat 分类器
    
    输入：B×N×3 的稀疏点云
    输出：B×num_classes 的分类 logits
    """
    
    def __init__(self, num_classes: int, in_channels: int = 1, hidden_channels: int = 64,
                 num_layers: int = 4, grid_size: tuple = (32, 32, 32),
                 dropout: float = 0.5, config: Optional[CudaTileConfig] = None):
        """
        Args:
            num_classes: 分类类别数
            in_channels: 输入通道数
            hidden_channels: 隐藏通道数
            num_layers: 热传导层数
            grid_size: 体素网格大小 (D, H, W)
            dropout: Dropout 比例
            config: CUDA Tiles 配置
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.config = config if config is not None else CudaTileConfig()
        
        # 3D-vHeat 骨干网络
        self.backbone = VHeat3D(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            grid_size=grid_size,
            config=self.config
        )
        
        # 分类头（两层全连接层）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
        # 性能统计
        self.forward_time = 0.0
        self.call_count = 0
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            points: 输入点云 (B, N, 3)
            
        Returns:
            logits: 分类 logits (B, num_classes)
        """
        if self.config.enable_profiling:
            import time
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 提取全局特征
            _, global_features = self.backbone(points)
            
            # 分类
            logits = self.classifier(global_features)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            self.forward_time += end_time - start_time
            self.call_count += 1
        else:
            # 提取全局特征
            _, global_features = self.backbone(points)
            
            # 分类
            logits = self.classifier(global_features)
        
        return logits
    
    def get_stats(self) -> dict:
        """获取性能统计"""
        avg_time = self.forward_time / self.call_count if self.call_count > 0 else 0.0
        return {
            "forward_time": self.forward_time,
            "call_count": self.call_count,
            "avg_time": avg_time,
            "backbone_stats": self.backbone.get_stats(),
        }
    
    def reset_stats(self):
        """重置统计"""
        self.forward_time = 0.0
        self.call_count = 0
        self.backbone.reset_stats()
