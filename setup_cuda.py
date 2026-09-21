"""
PyTorch CUDA Extension Setup
编译真正的 CUDA Tiles 加速算子（禁用 CUDA 版本检查）
"""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import sys

# 禁用 CUDA 版本检查
import torch.utils.cpp_extension as cpp_ext
original_check = cpp_ext._check_cuda_version
def no_check(*args, **kwargs):
    pass
cpp_ext._check_cuda_version = no_check

# CUDA kernel 源文件
cuda_sources = [
    'vheat3d/cuda_kernels/dct3d.cu',
    'vheat3d/cuda_kernels/conv3d.cu',
]

# C++ 源文件
cpp_sources = [
    'vheat3d/cuda_kernels/pybind.cpp',
]

# 编译选项（仅 CUDA 13.1 支持的架构）
extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': [
        '-O3',
        '-gencode=arch=compute_75,code=sm_75',  # CUDA 11.0+ (Ampere)
        '-gencode=arch=compute_80,code=sm_80',  # CUDA 11.0+ (Ampere)
        '-gencode=arch=compute_86,code=sm_86',  # CUDA 11.1+ (Ampere)
        '-gencode=arch=compute_89,code=sm_89',  # CUDA 11.8+ (Hopper)
        '-gencode=arch=compute_90,code=sm_90',  # CUDA 12.0+ (Hopper)
        '-gencode=arch=compute_90a,code=sm_90a',  # CUDA 12.0+ (Hopper)
    ]
}

# 链接选项
extra_link_args = []

# 定义 CUDA Extension
ext_modules = [
    CUDAExtension(
        name='vheat3d_cuda',
        sources=cuda_sources + cpp_sources,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

# 设置包
setup(
    name='vheat3d_cuda',
    version='1.0.0',
    description='3D-vHeat CUDA Tiles 加速算子（CUDA 13.1）',
    author='3D-vHeat Team',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension,
    },
)
