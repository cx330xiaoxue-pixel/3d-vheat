"""
稀疏点云体素化模块
将稀疏点云转换为三维体素表示（带掩码）
支持 binary / gaussian / sdf 三种模式
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from ..config import CudaTileConfig
from ..utils import Timer


class SparseVoxelization(nn.Module):
    """稀疏点云体素化模块
    
    输入：B×N×3 的稀疏点云张量（B=批次，N=点数量，3=xyz坐标）
    输出：B×1×D×H×W 的体素特征、B×1×D×H×W 的稀疏掩码
    
    支持四种 voxel_mode:
    - 'binary':  体素内有点=1，否则=0（默认）
    - 'gaussian': 体素值 = exp(-d²/σ²)，d=到最近点的距离，σ=sigma
    - 'sdf':      体素值 = 有符号距离，正=外部，负=内部，用法向量确定符号
    - 'trilinear': 点按三线性插值散布到 8 个邻近体素（硬分配的可微平滑对照）
    """
    
    def __init__(self, voxel_size: float = 0.1, grid_size: Tuple[int, int, int] = (32, 32, 32),
                 bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
                 config: Optional[CudaTileConfig] = None,
                 voxel_mode: str = 'binary',
                 sigma: float = 0.05,
                 knn: int = 10,
                 bounds_mode: str = 'batch'):
        """
        Args:
            voxel_size: 体素大小
            grid_size: 体素网格大小 (D, H, W)
            bounds: 点云边界 (min_x, min_y, min_z, max_x, max_y, max_z)，None 表示自动计算
            config: CUDA Tiles 配置
            voxel_mode: 体素化模式 ('binary', 'gaussian', 'sdf', 'trilinear')
            sigma: 高斯距离场参数（gaussian模式）
            knn: 法向量估计的邻域点数（sdf模式）
        """
        super().__init__()
        
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.bounds = bounds
        self.config = config if config is not None else CudaTileConfig()
        self.voxel_mode = voxel_mode
        self.sigma = sigma
        self.knn = knn
        assert bounds_mode in {'batch', 'per_sample'}, bounds_mode
        self.bounds_mode = bounds_mode
        
        # 预计算体素中心网格坐标（归一化 [0, 1] 空间）
        D, H, W = grid_size
        z = torch.linspace(0, 1, D)
        y = torch.linspace(0, 1, H)
        x = torch.linspace(0, 1, W)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        # (D*H*W, 3) — xx, yy, zz 的顺序
        self.register_buffer('grid_centers',
            torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1),
            persistent=False)
        
        # 性能统计
        self.forward_time = 0.0
        self.call_count = 0
    
    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            points: 输入点云 (B, N, 3)
            
        Returns:
            voxel_features: 体素特征 (B, 1, D, H, W)
            mask: 稀疏掩码 (B, 1, D, H, W)
        """
        if self.config.enable_profiling:
            with Timer("SparseVoxelization", enable_cuda=True) as timer:
                voxel_features, mask = self._voxelize(points)
            self.forward_time += timer.get_elapsed()
            self.call_count += 1
        else:
            voxel_features, mask = self._voxelize(points)
        
        return voxel_features, mask
    
    def _compute_normals(self, points: torch.Tensor) -> torch.Tensor:
        """通过局部PCA估计法向量（逐样本处理避免索引广播歧义）。
        Args:
            points: (B, N, 3)
        Returns:
            normals: (B, N, 3) 单位法向量
        """
        B, N, _ = points.shape
        K = min(self.knn, N)
        device = points.device
        normals_list = []
        for b in range(B):
            pts = points[b]  # (N, 3)
            dists = torch.cdist(pts, pts)  # (N, N)
            _, idx = dists.topk(K + 1, dim=-1, largest=False)
            idx = idx[:, 1:]  # (N, K) 排除自身
            neighbors = pts[idx]  # (N, K, 3)
            center = pts.unsqueeze(1)  # (N, 1, 3)
            centered = neighbors - center  # (N, K, 3)
            centered = centered / (centered.norm(dim=-1, keepdim=True) + 1e-8)
            cov = centered.transpose(-1, -2) @ centered  # (N, 3, 3)
            eigvec = torch.randn(N, 3, device=device)
            eigvec = eigvec / eigvec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            for _ in range(5):
                eigvec = cov @ eigvec.unsqueeze(-1)
                eigvec = eigvec.squeeze(-1)
                eigvec = eigvec / eigvec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            center_dir = -pts  # 指向原点
            flip = (eigvec * center_dir).sum(dim=-1, keepdim=True) < 0
            eigvec = torch.where(flip, -eigvec, eigvec)
            normals_list.append(eigvec)
        return torch.stack(normals_list, dim=0)
    
    def _compute_gaussian_field(self, points: torch.Tensor,
                                 bounds: Tuple[float, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """高斯距离场体素化"""
        B, N, _ = points.shape
        D, H, W = self.grid_size
        device = points.device
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        
        # 归一化点到 [0, 1]
        pts_norm = points.clone()
        pts_norm[:, :, 0] = (points[:, :, 0] - min_x) / (max_x - min_x + 1e-6)
        pts_norm[:, :, 1] = (points[:, :, 1] - min_y) / (max_y - min_y + 1e-6)
        pts_norm[:, :, 2] = (points[:, :, 2] - min_z) / (max_z - min_z + 1e-6)
        pts_norm = pts_norm.clamp(0, 1)
        
        # 体素中心网格
        grid = self.grid_centers.to(device).unsqueeze(0)  # (1, D*H*W, 3)
        
        field_list = []
        mask_list = []
        for b in range(B):
            # (D*H*W, N)
            dists = torch.cdist(grid, pts_norm[b:b+1])  # (1, D*H*W, N) -> squeeze
            dists = dists.squeeze(0)
            min_dists, _ = dists.min(dim=-1)  # (D*H*W)
            
            field = torch.exp(-(min_dists ** 2) / (2 * self.sigma ** 2))
            field = field.view(1, 1, D, H, W)
            mask = (min_dists < 3 * self.sigma).float().view(1, 1, D, H, W)
            field_list.append(field)
            mask_list.append(mask)
        
        return torch.cat(field_list, dim=0), torch.cat(mask_list, dim=0)
    
    def _compute_sdf_field(self, points: torch.Tensor,
                            bounds: Tuple[float, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """有符号距离场体素化"""
        B, N, _ = points.shape
        D, H, W = self.grid_size
        device = points.device
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        
        pts_norm = points.clone()
        pts_norm[:, :, 0] = (points[:, :, 0] - min_x) / (max_x - min_x + 1e-6)
        pts_norm[:, :, 1] = (points[:, :, 1] - min_y) / (max_y - min_y + 1e-6)
        pts_norm[:, :, 2] = (points[:, :, 2] - min_z) / (max_z - min_z + 1e-6)
        pts_norm = pts_norm.clamp(0, 1)
        
        normals = self._compute_normals(pts_norm)
        grid = self.grid_centers.to(device).unsqueeze(0)  # (1, D*H*W, 3)
        
        field_list = []
        mask_list = []
        for b in range(B):
            dists = torch.cdist(grid, pts_norm[b:b+1]).squeeze(0)  # (D*H*W, N)
            min_dists, min_idx = dists.min(dim=-1)  # (D*H*W)
            
            offset = grid[0] - pts_norm[b, min_idx]  # (D*H*W, 3)
            n = normals[b, min_idx]  # (D*H*W, 3)
            sign = (offset * n).sum(dim=-1).sign()
            
            sdf = sign * min_dists
            sdf = sdf.view(1, 1, D, H, W)
            mask = (min_dists < 3 * self.sigma).float().view(1, 1, D, H, W)
            field_list.append(sdf)
            mask_list.append(mask)
        
        return torch.cat(field_list, dim=0), torch.cat(mask_list, dim=0)
    
    def _compute_trilinear_field(self, points: torch.Tensor,
                                  bounds: Tuple[float, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """三线性散布体素化: 每个点按重心权重分配到 8 个邻近体素中心."""
        B, N, _ = points.shape
        D, H, W = self.grid_size
        device = points.device
        min_x, min_y, min_z, max_x, max_y, max_z = bounds

        pts_norm = points.clone()
        pts_norm[:, :, 0] = (points[:, :, 0] - min_x) / (max_x - min_x + 1e-6)
        pts_norm[:, :, 1] = (points[:, :, 1] - min_y) / (max_y - min_y + 1e-6)
        pts_norm[:, :, 2] = (points[:, :, 2] - min_z) / (max_z - min_z + 1e-6)
        pts_norm = pts_norm.clamp(0, 1)

        # 连续体素坐标 (B, N, 3)
        cont = pts_norm * torch.tensor([W - 1, H - 1, D - 1], device=device,
                                       dtype=pts_norm.dtype)
        base = cont.floor().long()
        frac = cont - base.float()

        field = torch.zeros(B, 1, D, H, W, device=device, dtype=points.dtype)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    wx = frac[..., 0] if dx else (1 - frac[..., 0])
                    wy = frac[..., 1] if dy else (1 - frac[..., 1])
                    wz = frac[..., 2] if dz else (1 - frac[..., 2])
                    weight = wx * wy * wz  # (B, N)
                    ix = (base[..., 0] + dx).clamp(0, W - 1)
                    iy = (base[..., 1] + dy).clamp(0, H - 1)
                    iz = (base[..., 2] + dz).clamp(0, D - 1)
                    field.view(B, -1).scatter_add_(
                        1, (iz * H + iy) * W + ix, weight.to(field.dtype))

        field = field.clamp(max=1.0)
        mask = (field > 0).to(field.dtype)
        return field, mask

    def _voxelize(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行体素化"""
        batch_size, num_points, _ = points.shape
        depth, height, width = self.grid_size
        device = points.device
        
        # 计算边界
        if self.bounds is None:
            min_coords = points.view(batch_size, -1, 3).min(dim=1)[0]
            max_coords = points.view(batch_size, -1, 3).max(dim=1)[0]
            min_x, min_y, min_z = min_coords.min(dim=0)[0].tolist()
            max_x, max_y, max_z = max_coords.max(dim=0)[0].tolist()
        else:
            min_x, min_y, min_z, max_x, max_y, max_z = self.bounds

        if self.bounds is None and self.bounds_mode == 'per_sample':
            outs, masks = [], []
            for b in range(batch_size):
                p_b = points[b:b + 1]
                mn = p_b.reshape(1, -1, 3).min(dim=1)[0][0]
                mx = p_b.reshape(1, -1, 3).max(dim=1)[0][0]
                bd = (float(mn[0]), float(mn[1]), float(mn[2]),
                      float(mx[0]), float(mx[1]), float(mx[2]))
                if self.voxel_mode == 'gaussian':
                    v, m = self._compute_gaussian_field(p_b, bd)
                elif self.voxel_mode == 'sdf':
                    v, m = self._compute_sdf_field(p_b, bd)
                elif self.voxel_mode == 'trilinear':
                    v, m = self._compute_trilinear_field(p_b, bd)
                else:
                    v = torch.zeros(1, 1, depth, height, width, device=device, dtype=points.dtype)
                    m = torch.zeros_like(v)
                    if not torch.cuda.is_available() or device.type == 'cpu':
                        v, m = self._cpu_voxelize(p_b, v, m, bd)
                    else:
                        v, m = self._cuda_voxelize(p_b, v, m, bd)
                outs.append(v)
                masks.append(m)
            return torch.cat(outs, 0), torch.cat(masks, 0)

        if self.voxel_mode == 'gaussian':
            return self._compute_gaussian_field(points, (min_x, min_y, min_z, max_x, max_y, max_z))
        elif self.voxel_mode == 'sdf':
            return self._compute_sdf_field(points, (min_x, min_y, min_z, max_x, max_y, max_z))
        elif self.voxel_mode == 'trilinear':
            return self._compute_trilinear_field(points, (min_x, min_y, min_z, max_x, max_y, max_z))
        
        # binary 模式（原有逻辑）
        # 初始化体素特征和掩码
        voxel_features = torch.zeros(batch_size, 1, depth, height, width,
                                    device=device, dtype=points.dtype)
        mask = torch.zeros(batch_size, 1, depth, height, width,
                          device=device, dtype=points.dtype)
        
        # 如果是 CPU，使用原生 PyTorch 实现
        if not torch.cuda.is_available() or device.type == 'cpu':
            return self._cpu_voxelize(points, voxel_features, mask,
                                     (min_x, min_y, min_z, max_x, max_y, max_z))
        
        # GPU 端使用 CUDA Tiles 加速
        return self._cuda_voxelize(points, voxel_features, mask,
                                   (min_x, min_y, min_z, max_x, max_y, max_z))
    
    def _cpu_voxelize(self, points: torch.Tensor, voxel_features: torch.Tensor,
                      mask: torch.Tensor, bounds: Tuple[float, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """CPU 端实现"""
        batch_size, num_points, _ = points.shape
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        
        for b in range(batch_size):
            # 归一化坐标到 [0, 1]
            x_norm = (points[b, :, 0] - min_x) / (max_x - min_x + 1e-6)
            y_norm = (points[b, :, 1] - min_y) / (max_y - min_y + 1e-6)
            z_norm = (points[b, :, 2] - min_z) / (max_z - min_z + 1e-6)
            
            # 转换为体素索引
            x_idx = (x_norm * (self.grid_size[2] - 1)).long().clamp(0, self.grid_size[2] - 1)
            y_idx = (y_norm * (self.grid_size[1] - 1)).long().clamp(0, self.grid_size[1] - 1)
            z_idx = (z_norm * (self.grid_size[0] - 1)).long().clamp(0, self.grid_size[0] - 1)
            
            # 标记有效体素
            mask[b, 0, z_idx, y_idx, x_idx] = 1.0
            voxel_features[b, 0, z_idx, y_idx, x_idx] = 1.0
        
        return voxel_features, mask
    
    def _cuda_voxelize(self, points: torch.Tensor, voxel_features: torch.Tensor,
                       mask: torch.Tensor, bounds: Tuple[float, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU 端实现（简化版，避免分块处理）
        """
        batch_size, num_points, _ = points.shape
        depth, height, width = self.grid_size
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        device = points.device
        
        # print(f"_cuda_voxelize 输入: points={points.shape}, bounds={bounds}")
        # print(f"_cuda_voxelize 设备: points={points.device}, voxel_features={voxel_features.device}, mask={mask.device}")
        
        for b in range(batch_size):
            # 归一化坐标到 [0, 1]
            x_norm = (points[b, :, 0] - min_x) / (max_x - min_x + 1e-6)
            y_norm = (points[b, :, 1] - min_y) / (max_y - min_y + 1e-6)
            z_norm = (points[b, :, 2] - min_z) / (max_z - min_z + 1e-6)
            
            # print(f"  Batch {b}: x_norm范围={x_norm.min().item():.4f}~{x_norm.max().item():.4f}")
            # print(f"  Batch {b}: y_norm范围={y_norm.min().item():.4f}~{y_norm.max().item():.4f}")
            # print(f"  Batch {b}: z_norm范围={z_norm.min().item():.4f}~{z_norm.max().item():.4f}")
            
            # 转换为体素索引
            x_idx = (x_norm * (width - 1)).long().clamp(0, width - 1)
            y_idx = (y_norm * (height - 1)).long().clamp(0, height - 1)
            z_idx = (z_norm * (depth - 1)).long().clamp(0, depth - 1)
            
            # print(f"  Batch {b}: x_idx范围={x_idx.min().item()}~{x_idx.max().item()}")
            # print(f"  Batch {b}: y_idx范围={y_idx.min().item()}~{y_idx.max().item()}")
            # print(f"  Batch {b}: z_idx范围={z_idx.min().item()}~{z_idx.max().item()}")
            
            # 确保索引在有效范围内
            x_idx = x_idx.clamp(0, width - 1)
            y_idx = y_idx.clamp(0, height - 1)
            z_idx = z_idx.clamp(0, depth - 1)
            
            # print(f"  Batch {b}: 修正后索引范围: x={x_idx.min().item()}~{x_idx.max().item()}, y={y_idx.min().item()}~{y_idx.max().item()}, z={z_idx.min().item()}~{z_idx.max().item()}")
            
            # 标记有效体素
            try:
                # print("  标记有效体素...")
                mask[b, 0, z_idx, y_idx, x_idx] = 1.0
                voxel_features[b, 0, z_idx, y_idx, x_idx] = 1.0
                # print("  体素标记完成")
            except Exception as e:
                # print(f"  体素标记错误: {e}")
                import traceback
                traceback.print_exc()
                return voxel_features, mask
        
        # print(f"_cuda_voxelize 输出形状: voxel_features={voxel_features.shape}, mask={mask.shape}")
        # print(f"_cuda_voxelize 输出非零值: voxel_features={voxel_features.sum().item()}, mask={mask.sum().item()}")
        
        return voxel_features, mask
    
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
