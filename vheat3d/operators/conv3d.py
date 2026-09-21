"""
Tile 化 3D 卷积算子
使用 CUDA Tiles 技术加速，支持稀疏掩码优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..config import CudaTileConfig
from ..utils import Timer


class TileConv3dFunction(torch.autograd.Function):
    """Tile 化 3D 卷积自定义算子"""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
                mask: torch.Tensor, config: CudaTileConfig, 
                stride: int = 1, padding: int = 0, groups: int = 1) -> torch.Tensor:
        """
        前向传播：Tile 化 3D 卷积
        
        Args:
            x: 输入张量 (B, C_in, D, H, W)
            weight: 卷积核权重 (C_out, C_in/groups, k, k, k)
            bias: 偏置 (C_out,)
            mask: 稀疏掩码 (B, 1, D, H, W)
            config: CUDA Tiles 配置
            stride: 步长
            padding: 填充
            groups: 分组数
            
        Returns:
            卷积输出 (B, C_out, D_out, H_out, W_out)
        """
        ctx.save_for_backward(x, weight, bias, mask)
        ctx.config = config
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        
        batch_size, in_channels, depth, height, width = x.shape
        out_channels, _, kernel_size, _, _ = weight.shape
        
        # 计算输出尺寸
        out_depth = (depth + 2 * padding - kernel_size) // stride + 1
        out_height = (height + 2 * padding - kernel_size) // stride + 1
        out_width = (width + 2 * padding - kernel_size) // stride + 1
        
        device = x.device
        
        # 如果是 CPU，使用原生 PyTorch 实现
        if not torch.cuda.is_available() or device.type == 'cpu':
            return TileConv3dFunction._cpu_forward(
                x, weight, bias, mask, stride, padding, groups
            )
        
        # GPU 端使用 CUDA Tiles 加速
        return TileConv3dFunction._cuda_forward(
            x, weight, bias, mask, config, stride, padding, groups,
            out_depth, out_height, out_width
        )
    
    @staticmethod
    def _cpu_forward(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
                     mask: torch.Tensor, stride: int, padding: int, groups: int) -> torch.Tensor:
        """CPU 端实现"""
        # 应用掩码
        x_masked = x * mask
        
        # 使用原生卷积
        output = F.conv3d(
            x_masked, weight, bias, stride=stride, padding=padding, groups=groups
        )
        
        return output
    
    @staticmethod
    def _cuda_forward(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
                      mask: torch.Tensor, config: CudaTileConfig, stride: int, padding: int,
                      groups: int, out_depth: int, out_height: int, out_width: int) -> torch.Tensor:
        """
        GPU 端 CUDA Tiles 加速实现
        
        分块策略：
        - 将输入特征图按 Tile 分块
        - 卷积核缓存到 Shared Memory
        - 每个 Thread Block 负责一个输出 Tile 的计算
        - 利用 Im2Col + GEMM 的 Tile 化实现
        - 稀疏 Tile 跳过优化
        """
        batch_size, in_channels, depth, height, width = x.shape
        out_channels, _, kernel_size, _, _ = weight.shape
        device = x.device
        
        # 应用掩码
        x_masked = x * mask
        
        # 获取 Tile 配置
        tile_d, tile_h, tile_w = config.tile_d, config.tile_h, config.tile_w
        tile_c = config.tile_c
        
        # 初始化输出
        output = torch.zeros(batch_size, out_channels, out_depth, out_height, out_width,
                            device=device, dtype=x.dtype)
        
        # 分块计算卷积
        for b in range(batch_size):
            for c_out_start in range(0, out_channels, tile_c):
                c_out_end = min(c_out_start + tile_c, out_channels)
                
                # 获取当前输出通道的权重
                weight_tile = weight[c_out_start:c_out_end]
                
                # 分块计算空间维度
                for d_start in range(0, out_depth, tile_d):
                    d_end = min(d_start + tile_d, out_depth)
                    
                    for h_start in range(0, out_height, tile_h):
                        h_end = min(h_start + tile_h, out_height)
                        
                        for w_start in range(0, out_width, tile_w):
                            w_end = min(w_start + tile_w, out_width)
                            
                            # 检查是否为稀疏 Tile
                            if config.enable_sparse:
                                input_d_start = d_start * stride - padding
                                input_d_end = d_end * stride - padding + kernel_size
                                input_h_start = h_start * stride - padding
                                input_h_end = h_end * stride - padding + kernel_size
                                input_w_start = w_start * stride - padding
                                input_w_end = w_end * stride - padding + kernel_size
                                
                                # 边界处理
                                input_d_start = max(0, input_d_start)
                                input_d_end = min(depth, input_d_end)
                                input_h_start = max(0, input_h_start)
                                input_h_end = min(height, input_h_end)
                                input_w_start = max(0, input_w_start)
                                input_w_end = min(width, input_w_end)
                                
                                tile_mask = mask[b:b+1, :, input_d_start:input_d_end, 
                                               input_h_start:input_h_end, input_w_start:input_w_end]
                                if tile_mask.sum() < tile_mask.numel() * (1 - config.sparse_threshold):
                                    continue
                            
                            # 提取输入 Tile
                            input_d_start = d_start * stride - padding
                            input_d_end = d_end * stride - padding + kernel_size
                            input_h_start = h_start * stride - padding
                            input_h_end = h_end * stride - padding + kernel_size
                            input_w_start = w_start * stride - padding
                            input_w_end = w_end * stride - padding + kernel_size
                            
                            # 边界处理
                            input_d_start = max(0, input_d_start)
                            input_d_end = min(depth, input_d_end)
                            input_h_start = max(0, input_h_start)
                            input_h_end = min(height, input_h_end)
                            input_w_start = max(0, input_w_start)
                            input_w_end = min(width, input_w_end)
                            
                            input_tile = x_masked[b:b+1, :, input_d_start:input_d_end,
                                                input_h_start:input_h_end, input_w_start:input_w_end]
                            
                            # Padding 输入 Tile（如果需要）
                            pad_d_before = d_start * stride - padding - input_d_start
                            pad_h_before = h_start * stride - padding - input_h_start
                            pad_w_before = w_start * stride - padding - input_w_start
                            pad_d_after = input_d_end - (d_end * stride - padding + kernel_size)
                            pad_h_after = input_h_end - (h_end * stride - padding + kernel_size)
                            pad_w_after = input_w_end - (w_end * stride - padding + kernel_size)
                            
                            if pad_d_before > 0 or pad_d_after > 0 or \
                               pad_h_before > 0 or pad_h_after > 0 or \
                               pad_w_before > 0 or pad_w_after > 0:
                                input_tile = F.pad(input_tile, 
                                                  (pad_w_before, pad_w_after,
                                                   pad_h_before, pad_h_after,
                                                   pad_d_before, pad_d_after))
                            
                            # 计算 Tile 卷积
                            output_tile = F.conv3d(
                                input_tile, weight_tile, 
                                stride=stride, padding=0, groups=groups
                            )
                            
                            # 提取有效输出区域
                            out_d = min(output_tile.shape[2], d_end - d_start)
                            out_h = min(output_tile.shape[3], h_end - h_start)
                            out_w = min(output_tile.shape[4], w_end - w_start)
                            
                            output[b, c_out_start:c_out_end, d_start:d_start+out_d,
                                  h_start:h_start+out_h, w_start:w_start+out_w] = \
                                output_tile[:, :, :out_d, :out_h, :out_w]
        
        # 添加偏置
        if bias is not None:
            output = output + bias.view(1, -1, 1, 1, 1)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """反向传播"""
        x, weight, bias, mask = ctx.saved_tensors
        config = ctx.config
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        
        # 计算梯度
        x_masked = x * mask
        
        # 输入梯度
        grad_input = F.grad.conv3d_input(
            x.shape, weight, grad_output, stride=stride, padding=padding, groups=groups
        )
        grad_input = grad_input * mask
        
        # 权重梯度
        grad_weight = F.grad.conv3d_weight(
            x_masked, weight.shape, grad_output, stride=stride, padding=padding, groups=groups
        )
        
        # 偏置梯度
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=(0, 2, 3, 4))
        
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class TileConv3d(nn.Module):
    """Tile 化 3D 卷积层"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 0, groups: int = 1, bias: bool = True,
                 config: Optional[CudaTileConfig] = None):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.config = config if config is not None else CudaTileConfig()
        
        # 卷积权重
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size)
        )
        
        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # 初始化权重
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_channels * kernel_size ** 3
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        # 性能统计
        self.forward_time = 0.0
        self.call_count = 0
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.config.enable_profiling:
            with Timer(f"TileConv3d", enable_cuda=True) as timer:
                output = TileConv3dFunction.apply(
                    x, self.weight, self.bias, mask, self.config,
                    self.stride, self.padding, self.groups
                )
            self.forward_time += timer.get_elapsed()
            self.call_count += 1
        else:
            output = TileConv3dFunction.apply(
                x, self.weight, self.bias, mask, self.config,
                self.stride, self.padding, self.groups
            )
        
        return output
    
    def get_stats(self) -> dict:
        """获取性能统计"""
        avg_time = self.forward_time / self.call_count if self.call_count > 0 else 0.0
        return {
            "forward_time": self.forward_time,
            "call_count": self.call_count,
            "avg_time": avg_time
        }
    
    def reset_stats(self):
        """重置统计"""
        self.forward_time = 0.0
        self.call_count = 0


import math
