"""
数据增强工具
"""

import numpy as np
import torch
from typing import Tuple


class PointCloudTransform:
    """点云数据增强"""
    
    def __init__(self, 
                 rotate: bool = True,
                 jitter: bool = True,
                 scale: bool = True,
                 shift: bool = True,
                 jitter_sigma: float = 0.01,
                 jitter_clip: float = 0.05,
                 scale_low: float = 0.8,
                 scale_high: float = 1.25,
                 shift_range: float = 0.1):
        """
        Args:
            rotate: 是否旋转
            jitter: 是否添加噪声
            scale: 是否缩放
            shift: 是否平移
            jitter_sigma: 噪声标准差
            jitter_clip: 噪声裁剪范围
            scale_low: 缩放下限
            scale_high: 缩放上限
            shift_range: 平移范围
        """
        self.rotate = rotate
        self.jitter = jitter
        self.scale = scale
        self.shift = shift
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.shift_range = shift_range
    
    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        应用数据增强
        
        Args:
            points: 点云 (num_points, 3)
            
        Returns:
            增强后的点云
        """
        if self.rotate:
            points = self._rotate_point_cloud(points)
        
        if self.jitter:
            points = self._jitter_point_cloud(points)
        
        if self.scale:
            points = self._scale_point_cloud(points)
        
        if self.shift:
            points = self._shift_point_cloud(points)
        
        return points
    
    def _rotate_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """随机旋转点云"""
        angle = np.random.uniform(0, 2 * np.pi)
        cosval = np.cos(angle)
        sinval = np.sin(angle)
        
        rotation_matrix = np.array([
            [cosval, 0, sinval],
            [0, 1, 0],
            [-sinval, 0, cosval]
        ])
        
        return np.dot(points, rotation_matrix)
    
    def _jitter_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        jitter = np.clip(
            np.random.normal(0, self.jitter_sigma, points.shape),
            -self.jitter_clip, self.jitter_clip
        )
        return points + jitter
    
    def _scale_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """随机缩放"""
        scale = np.random.uniform(self.scale_low, self.scale_high)
        return points * scale
    
    def _shift_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """随机平移"""
        shift = np.random.uniform(-self.shift_range, self.shift_range, 3)
        return points + shift


class PointCloudSegmentationTransform:
    """点云分割数据增强"""
    
    def __init__(self, 
                 rotate: bool = True,
                 jitter: bool = True,
                 scale: bool = True,
                 jitter_sigma: float = 0.01,
                 jitter_clip: float = 0.05,
                 scale_low: float = 0.8,
                 scale_high: float = 1.25):
        """
        Args:
            rotate: 是否旋转
            jitter: 是否添加噪声
            scale: 是否缩放
            jitter_sigma: 噪声标准差
            jitter_clip: 噪声裁剪范围
            scale_low: 缩放下限
            scale_high: 缩放上限
        """
        self.rotate = rotate
        self.jitter = jitter
        self.scale = scale
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.scale_low = scale_low
        self.scale_high = scale_high
    
    def __call__(self, points: np.ndarray, 
                 labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用数据增强
        
        Args:
            points: 点云 (num_points, 3)
            labels: 标签 (num_points,)
            
        Returns:
            增强后的点云和标签
        """
        if self.rotate:
            points = self._rotate_point_cloud(points)
        
        if self.jitter:
            points = self._jitter_point_cloud(points)
        
        if self.scale:
            points = self._scale_point_cloud(points)
        
        return points, labels
    
    def _rotate_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """随机旋转点云"""
        angle = np.random.uniform(0, 2 * np.pi)
        cosval = np.cos(angle)
        sinval = np.sin(angle)
        
        rotation_matrix = np.array([
            [cosval, 0, sinval],
            [0, 1, 0],
            [-sinval, 0, cosval]
        ])
        
        return np.dot(points, rotation_matrix)
    
    def _jitter_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        jitter = np.clip(
            np.random.normal(0, self.jitter_sigma, points.shape),
            -self.jitter_clip, self.jitter_clip
        )
        return points + jitter
    
    def _scale_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """随机缩放"""
        scale = np.random.uniform(self.scale_low, self.scale_high)
        return points * scale
