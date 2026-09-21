"""
3D-vHeat: 基于三维热扩散的稀疏点云处理骨干网络
使用真正的 CUDA Tiles 技术加速核心张量计算
"""

import os
import sys

# 设置库路径（解决 libc10.so 找不到的问题）
torch_lib_path = os.path.join(sys.prefix, 'lib/python3.8/site-packages/torch/lib')
if os.path.exists(torch_lib_path):
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ['LD_LIBRARY_PATH'] = torch_lib_path

from .config import CudaTileConfig, PresetConfigs, get_default_config
from .utils import count_parameters, get_memory_usage

# 导入 CUDA Tiles 加速算子（如果可用）
try:
    from .operators import DCT3D, IDCT3D, TileConv3d
    CUDA_TILES_AVAILABLE = True
    print("✓ CUDA Tiles 加速算子已加载")
except ImportError as e:
    from .operators import DCT3D as DCT3D_Fallback
    from .operators import IDCT3D as IDCT3D_Fallback
    from .operators import TileConv3d as TileConv3d_Fallback
    
    # 使用 fallback 版本
    DCT3D = DCT3D_Fallback
    IDCT3D = IDCT3D_Fallback
    TileConv3d = TileConv3d_Fallback
    CUDA_TILES_AVAILABLE = False
    print(f"警告: CUDA Tiles 算子未安装，使用 fallback 版本 ({e})")

# 导入网络模块
from .modules import SparseVoxelization, HeatConduction3D, VHeat3D, VHeat3DClassifier

__version__ = "2.0.0"
__all__ = [
    "CudaTileConfig",
    "PresetConfigs",
    "get_default_config",
    "DCT3D",
    "IDCT3D",
    "TileConv3d",
    "SparseVoxelization",
    "HeatConduction3D",
    "VHeat3D",
    "VHeat3DClassifier",
    "CUDA_TILES_AVAILABLE",
]
