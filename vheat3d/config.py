"""
CUDA Tiles 配置文件
定义分块策略、内存大小等可配置参数
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CudaTileConfig:
    """CUDA Tiles 配置类"""
    
    # 空间分块大小（D×H×W）
    tile_d: int = 8
    tile_h: int = 8
    tile_w: int = 8
    
    # 通道分块大小
    tile_c: int = 16
    
    # Shared Memory 大小限制（字节）
    max_shared_memory: int = 48 * 1024  # 48KB
    
    # 稀疏 Tile 跳过阈值（空白体素占比）
    sparse_threshold: float = 0.9
    
    # 是否启用稀疏优化
    enable_sparse: bool = True
    
    # GPU 架构配置
    gpu_arch: str = "auto"  # auto, ampere, hopper
    
    # 性能监控
    enable_profiling: bool = True
    
    def get_tile_shape(self) -> Tuple[int, int, int, int]:
        """获取 Tile 形状 (C, D, H, W)"""
        return (self.tile_c, self.tile_d, self.tile_h, self.tile_w)
    
    def calculate_shared_memory_per_tile(self, dtype_size: int = 4) -> int:
        """计算每个 Tile 需要的 Shared Memory 大小"""
        tile_size = self.tile_c * self.tile_d * self.tile_h * self.tile_w
        return tile_size * dtype_size
    
    def validate(self) -> bool:
        """验证配置是否有效"""
        shared_mem_needed = self.calculate_shared_memory_per_tile()
        if shared_mem_needed > self.max_shared_memory:
            raise ValueError(
                f"Tile 配置超出 Shared Memory 限制: "
                f"需要 {shared_mem_needed} 字节，限制 {self.max_shared_memory} 字节"
            )
        return True


# 预定义配置
class PresetConfigs:
    """预定义的 Tile 配置"""
    
    # RTX 3090 (Ampere) 最优配置
    RTX3090 = CudaTileConfig(
        tile_d=8,
        tile_h=8,
        tile_w=8,
        tile_c=16,
        max_shared_memory=48 * 1024,
        gpu_arch="ampere"
    )
    
    # A100 (Ampere) 最优配置
    A100 = CudaTileConfig(
        tile_d=8,
        tile_h=8,
        tile_w=8,
        tile_c=32,
        max_shared_memory=164 * 1024,  # A100 有更大的 Shared Memory
        gpu_arch="ampere"
    )
    
    # H100 (Hopper) 最优配置
    H100 = CudaTileConfig(
        tile_d=16,
        tile_h=16,
        tile_w=16,
        tile_c=32,
        max_shared_memory=228 * 1024,  # H100 有更大的 Shared Memory
        gpu_arch="hopper"
    )
    
    # CPU 降级配置
    CPU = CudaTileConfig(
        tile_d=32,
        tile_h=32,
        tile_w=32,
        tile_c=64,
        enable_sparse=False,
        enable_profiling=False,
        gpu_arch="cpu"
    )


def get_default_config() -> CudaTileConfig:
    """获取默认配置（根据 GPU 自动选择）"""
    import torch
    
    if not torch.cuda.is_available():
        return PresetConfigs.CPU
    
    gpu_name = torch.cuda.get_device_name(0)
    
    if "A100" in gpu_name:
        return PresetConfigs.A100
    elif "H100" in gpu_name:
        return PresetConfigs.H100
    elif "3090" in gpu_name or "3080" in gpu_name or "3070" in gpu_name:
        return PresetConfigs.RTX3090
    else:
        return PresetConfigs.RTX3090
