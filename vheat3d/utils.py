"""
工具函数模块
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import time


class Timer:
    """性能计时器"""
    
    def __init__(self, name: str = "", enable_cuda: bool = True):
        self.name = name
        self.enable_cuda = enable_cuda
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        if self.enable_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        return self
    
    def __exit__(self, *args):
        if self.enable_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
    
    def get_elapsed(self) -> float:
        """获取耗时（秒）"""
        return self.elapsed


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage() -> Tuple[float, float]:
    """获取 GPU 显存使用情况（已分配，已缓存）"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return allocated, cached
    return 0.0, 0.0


def calculate_flops_3d_conv(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    depth: int,
    height: int,
    width: int,
    kernel_size: int
) -> int:
    """计算 3D 卷积的 FLOPs"""
    output_depth = depth
    output_height = height
    output_width = width
    
    flops_per_element = kernel_size ** 3 * in_channels
    total_elements = batch_size * out_channels * output_depth * output_height * output_width
    
    return flops_per_element * total_elements


def calculate_flops_3d_dct(
    batch_size: int,
    channels: int,
    depth: int,
    height: int,
    width: int
) -> int:
    """计算 3D DCT 的 FLOPs（近似）"""
    # 3D DCT 可以分解为三次 1D DCT
    # 每个 1D DCT 的复杂度为 O(N^2)
    total_elements = batch_size * channels * depth * height * width
    flops = total_elements * (depth + height + width)
    return flops


def sparse_mask_filter(
    features: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """使用稀疏掩码过滤特征"""
    return features * mask


def adaptive_avg_pool3d(
    x: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """自适应 3D 平均池化（考虑稀疏掩码）"""
    # print(f"adaptive_avg_pool3d 输入形状: x={x.shape}, mask={mask.shape}")
    
    # 检查输入形状
    if len(x.shape) != 5:
        raise ValueError(f"Expected x to have shape [B, C, D, H, W], got {x.shape}")
    
    batch_size, channels, depth, height, width = x.shape
    # print(f"  解析的维度: B={batch_size}, C={channels}, D={depth}, H={height}, W={width}")
    
    # 计算每个样本的有效体素数量
    valid_counts = mask.view(batch_size, -1).sum(dim=1, keepdim=True).clamp(min=1)
    # print(f"  valid_counts 形状: {valid_counts.shape}, 值: {valid_counts}")
    
    # 仅对有效体素求和
    x_masked = x * mask
    # print(f"  x_masked 形状: {x_masked.shape}")
    
    x_flat = x_masked.view(batch_size, channels, -1)
    # print(f"  x_flat 形状: {x_flat.shape}")
    
    pooled = x_flat.sum(dim=2)
    # print(f"  pooled 形状: {pooled.shape}")
    
    # 除以有效体素数量
    valid_counts_expanded = valid_counts.expand(-1, channels)
    # print(f"  valid_counts_expanded 形状: {valid_counts_expanded.shape}")
    
    pooled = pooled / valid_counts_expanded
    # print(f"  最终 pooled 形状: {pooled.shape}")
    
    return pooled


def generate_sparse_mask(
    points: torch.Tensor,
    voxel_size: float,
    grid_size: Tuple[int, int, int],
    bounds: Tuple[float, float, float, float, float, float]
) -> torch.Tensor:
    """生成稀疏掩码"""
    batch_size = points.shape[0]
    depth, height, width = grid_size
    
    # 初始化掩码
    mask = torch.zeros(batch_size, 1, depth, height, width, 
                       device=points.device, dtype=points.dtype)
    
    # 计算体素索引
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    
    for b in range(batch_size):
        # 归一化坐标到 [0, 1]
        x_norm = (points[b, :, 0] - min_x) / (max_x - min_x)
        y_norm = (points[b, :, 1] - min_y) / (max_y - min_y)
        z_norm = (points[b, :, 2] - min_z) / (max_z - min_z)
        
        # 转换为体素索引
        x_idx = (x_norm * width).long().clamp(0, width - 1)
        y_idx = (y_norm * height).long().clamp(0, height - 1)
        z_idx = (z_norm * depth).long().clamp(0, depth - 1)
        
        # 标记有效体素
        mask[b, 0, z_idx, y_idx, x_idx] = 1.0
    
    return mask


def print_model_summary(model: nn.Module, input_shape: Tuple[int, ...]):
    """打印模型摘要"""
    print("=" * 80)
    print("Model Summary")
    print("=" * 80)
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"Input Shape: {input_shape}")
    print("=" * 80)
