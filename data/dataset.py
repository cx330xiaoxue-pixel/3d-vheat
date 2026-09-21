"""
点云数据集加载器
支持常见点云数据集：ModelNet40、ShapeNet、ScanNet、S3DIS 等
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import h5py
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')


class PointCloudDataset(Dataset):
    """通用点云数据集基类"""

    def __init__(self, points: np.ndarray, labels: np.ndarray,
                 num_points: int = 1024, transform=None):
        """
        Args:
            points: 点云数据 (N, 3) 或 (N, num_points, 3)
            labels: 标签 (N,) 或 (N, num_classes)
            num_points: 每个样本的点数
            transform: 数据增强
        """
        self.points = points
        self.labels = labels
        self.num_points = num_points
        self.transform = transform

        if len(points.shape) == 2:
            self.points = self._resample_points(points, num_points)

    def _resample_points(self, points: np.ndarray, num_points: int) -> np.ndarray:
        """重新采样点云到固定点数"""
        N = len(points)
        resampled = np.zeros((N, num_points, 3), dtype=np.float32)

        for i in range(N):
            if len(points[i]) >= num_points:
                idx = np.random.choice(len(points[i]), num_points, replace=False)
                resampled[i] = points[i][idx]
            else:
                idx = np.random.choice(len(points[i]), num_points, replace=True)
                resampled[i] = points[i][idx]

        return resampled

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        point = self.points[idx]
        label = self.labels[idx]

        if self.transform:
            point = self.transform(point)

        return torch.from_numpy(point).float(), torch.tensor(label).long()


class ModelNet40(Dataset):
    """ModelNet40 数据集（点云分类）"""

    def __init__(self, root: str, train: bool = True,
                 num_points: int = 1024, transform=None):
        """
        Args:
            root: 数据集根目录
            train: 是否为训练集
            num_points: 每个样本的点数
            transform: 数据增强
        """
        self.root = root
        self.train = train
        self.num_points = num_points
        self.transform = transform

        self.points, self.labels = self._load_data()

    @staticmethod
    def _sample_surface_uniform(vertices, faces, num_points):
        """从三角形网格表面均匀采样点"""
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        v0_64 = v0.astype(np.float64)
        v1_64 = v1.astype(np.float64)
        v2_64 = v2.astype(np.float64)
        cross = np.cross(v1_64 - v0_64, v2_64 - v0_64)
        face_areas = 0.5 * np.sqrt(np.sum(cross * cross, axis=1))
        face_areas = np.maximum(face_areas, 1e-12)
        face_probs = face_areas / face_areas.sum()

        face_indices = np.random.choice(len(faces), size=num_points, p=face_probs)
        chosen_faces = faces[face_indices]

        r1 = np.random.random(num_points)
        r2 = np.random.random(num_points)
        sqrt_r1 = np.sqrt(r1)
        u = 1.0 - sqrt_r1
        v = sqrt_r1 * (1.0 - r2)
        w = sqrt_r1 * r2

        v0 = vertices[chosen_faces[:, 0]]
        v1 = vertices[chosen_faces[:, 1]]
        v2 = vertices[chosen_faces[:, 2]]
        return (u[:, None] * v0 + v[:, None] * v1 + w[:, None] * v2).astype(np.float32)

    def _read_off_file(self, file_path):
        """
        读取OFF文件并返回顶点和面数据

        Args:
            file_path: OFF文件路径

        Returns:
            tuple: (vertices, faces) or None
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            lines = content.strip().split('\n')
            lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]

            if not lines:
                return None

            if lines[0] == 'OFF':
                lines = lines[1:]
            elif len(lines[0].split()) == 4 and lines[0].split()[0] == 'OFF':
                lines = lines[1:]

            if not lines:
                return None

            try:
                header = lines[0].split()
                n_vertices = int(header[0])
                n_faces = int(header[1])
            except (ValueError, IndexError):
                return None

            # 用索引方式遍历，避免 lines = lines[1:] 的 O(n²) 开销
            idx = 1  # 跳过计数行

            # 读取顶点
            vertices = []
            for _ in range(n_vertices):
                while idx < len(lines) and lines[idx].startswith('#'):
                    idx += 1
                if idx >= len(lines):
                    break
                try:
                    parts = lines[idx].split()
                    vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except (ValueError, IndexError):
                    pass
                idx += 1

            if not vertices:
                return None
            vertices = np.array(vertices, dtype=np.float32)

            # 读取面（三角形）
            faces = []
            for _ in range(n_faces):
                while idx < len(lines) and lines[idx].startswith('#'):
                    idx += 1
                if idx >= len(lines):
                    break
                try:
                    parts = list(map(int, lines[idx].split()))
                    n_sides = parts[0]
                    indices = parts[1:1 + n_sides]
                    if n_sides == 3:
                        faces.append(indices)
                    elif n_sides == 4:
                        faces.append([indices[0], indices[1], indices[2]])
                        faces.append([indices[0], indices[2], indices[3]])
                    elif n_sides > 4:
                        for j in range(1, n_sides - 1):
                            faces.append([indices[0], indices[j], indices[j + 1]])
                except (ValueError, IndexError):
                    pass
                idx += 1

            if not faces:
                return None
            faces = np.array(faces, dtype=np.int64)

            return vertices, faces
        except Exception:
            return None

    def _process_file(self, file_path, cat_label, num_points):
        """
        处理单个OFF文件（表面均匀采样）

        Returns:
            tuple: (sampled_points, label) 或 None
        """
        try:
            result = self._read_off_file(file_path)
            if result is None:
                return None
            vertices, faces = result

            if len(vertices) < 3 or len(faces) < 1:
                return None

            sampled_points = self._sample_surface_uniform(vertices, faces, num_points)

            centroid = np.mean(sampled_points, axis=0)
            sampled_points -= centroid
            max_dist = np.max(np.linalg.norm(sampled_points, axis=1))
            if max_dist > 0:
                sampled_points /= max_dist

            return sampled_points, cat_label
        except Exception:
            return None

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载 ModelNet40 数据"""
        split = 'train' if self.train else 'test'

        modelnet_dir = os.path.join(self.root, 'ModelNet40')
        if not os.path.exists(modelnet_dir):
            if os.path.exists(self.root) and 'airplane' in os.listdir(self.root):
                modelnet_dir = self.root
            else:
                raise FileNotFoundError(f"ModelNet40目录不存在: {modelnet_dir}")

        categories = sorted([d for d in os.listdir(modelnet_dir) if os.path.isdir(os.path.join(modelnet_dir, d))])
        category_to_label = {cat: i for i, cat in enumerate(categories)}

        print(f"Found {len(categories)} categories in ModelNet40")

        file_label_pairs = []
        print("Collecting file paths...")
        for cat_idx, cat in enumerate(categories):
            cat_dir = os.path.join(modelnet_dir, cat, split)
            if os.path.exists(cat_dir):
                files = [f for f in os.listdir(cat_dir) if f.endswith('.off')]
                print(f"  [{cat_idx+1}/{len(categories)}] {cat}: {len(files)} files")
                for file in files:
                    file_path = os.path.join(cat_dir, file)
                    file_label_pairs.append((file_path, category_to_label[cat]))

        total_files = len(file_label_pairs)
        print(f"Total files to process: {total_files}")
        print("Processing files with multi-threading...")

        points_list = []
        labels_list = []
        errors = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_file = {executor.submit(self._process_file, file_path, label, self.num_points): (file_path, label)
                             for file_path, label in file_label_pairs}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                file_path, label = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        points_list.append(result[0])
                        labels_list.append(result[1])
                    else:
                        errors += 1

                    if (i + 1) % 50 == 0 or (i + 1) == total_files:
                        print(f"  Progress: {i+1}/{total_files} files ({len(points_list)} valid, {errors} errors)")
                except Exception:
                    errors += 1

        if not points_list:
            raise FileNotFoundError(f"No data found in {modelnet_dir}")

        print(f"Loaded {len(points_list)} samples for {split} set")
        print(f"Processing completed: {total_files} files processed, {errors} errors")
        return np.array(points_list, dtype=np.float32), np.array(labels_list, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        point = self.points[idx]
        label = self.labels[idx]

        if len(point) > self.num_points:
            choice = np.random.choice(len(point), self.num_points, replace=False)
            point = point[choice]
        elif len(point) < self.num_points:
            choice = np.random.choice(len(point), self.num_points, replace=True)
            point = point[choice]

        if self.transform:
            point = self.transform(point)

        return torch.from_numpy(point).float(), torch.tensor(label).long()


class SyntheticPointCloudDataset(Dataset):
    """合成点云数据集（用于测试）"""

    def __init__(self, num_samples: int = 1000, num_points: int = 1024,
                 num_classes: int = 10, transform=None):
        """
        Args:
            num_samples: 样本数量
            num_points: 每个样本的点数
            num_classes: 类别数量
            transform: 数据增强
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_classes = num_classes
        self.transform = transform

        self.points, self.labels = self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成合成点云数据"""
        points = np.zeros((self.num_samples, self.num_points, 3), dtype=np.float32)
        labels = np.zeros(self.num_samples, dtype=np.int64)

        for i in range(self.num_samples):
            label = np.random.randint(0, self.num_classes)
            labels[i] = label

            if label == 0:  # 球体
                phi = np.random.uniform(0, 2 * np.pi, self.num_points)
                theta = np.random.uniform(0, np.pi, self.num_points)
                r = np.random.uniform(0.8, 1.0, self.num_points)
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
            elif label == 1:  # 立方体
                x = np.random.uniform(-1, 1, self.num_points)
                y = np.random.uniform(-1, 1, self.num_points)
                z = np.random.uniform(-1, 1, self.num_points)
            elif label == 2:  # 圆柱体
                theta = np.random.uniform(0, 2 * np.pi, self.num_points)
                r = np.random.uniform(0.8, 1.0, self.num_points)
                h = np.random.uniform(-1, 1, self.num_points)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = h
            elif label == 3:  # 圆锥体
                theta = np.random.uniform(0, 2 * np.pi, self.num_points)
                r = np.random.uniform(0, 1.0, self.num_points)
                h = np.random.uniform(0, 1.0, self.num_points)
                x = r * (1 - h) * np.cos(theta)
                y = r * (1 - h) * np.sin(theta)
                z = h * 2 - 1
            elif label == 4:  # 环面
                theta = np.random.uniform(0, 2 * np.pi, self.num_points)
                phi = np.random.uniform(0, 2 * np.pi, self.num_points)
                R = 1.0
                r = 0.3
                x = (R + r * np.cos(theta)) * np.cos(phi)
                y = (R + r * np.cos(theta)) * np.sin(phi)
                z = r * np.sin(theta)
            elif label == 5:  # 双锥体
                theta = np.random.uniform(0, 2 * np.pi, self.num_points)
                r = np.random.uniform(0, 1.0, self.num_points)
                h = np.random.uniform(-1, 1, self.num_points)
                x = r * (1 - abs(h)) * np.cos(theta)
                y = r * (1 - abs(h)) * np.sin(theta)
                z = h
            elif label == 6:  # 椭球体
                phi = np.random.uniform(0, 2 * np.pi, self.num_points)
                theta = np.random.uniform(0, np.pi, self.num_points)
                x = 1.5 * np.sin(theta) * np.cos(phi)
                y = 1.0 * np.sin(theta) * np.sin(phi)
                z = 0.8 * np.cos(theta)
            elif label == 7:  # 金字塔
                h = np.random.uniform(0, 1, self.num_points)
                r = (1 - h) * 1.5
                theta = np.random.uniform(0, 2 * np.pi, self.num_points)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = h * 2 - 1
            elif label == 8:  # 八面体
                u = np.random.uniform(0, 1, self.num_points)
                v = np.random.uniform(0, 1, self.num_points)
                mask = (u + v > 1)
                u = np.where(mask, 1 - u, u)
                v = np.where(mask, 1 - v, v)
                x = 2 * u - 1
                y = 2 * v - 1
                z = 1 - 2 * u - 2 * v
            else:  # 随机点云
                x = np.random.uniform(-1, 1, self.num_points)
                y = np.random.uniform(-1, 1, self.num_points)
                z = np.random.uniform(-1, 1, self.num_points)

            noise = np.random.normal(0, 0.02, (self.num_points, 3))
            points[i] = np.stack([x, y, z], axis=1) + noise
            points[i] = (points[i] - points[i].mean(axis=0)) / (points[i].std(axis=0) + 1e-8)

        return points, labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        point = self.points[idx]
        label = self.labels[idx]

        if self.transform:
            point = self.transform(point)

        return torch.from_numpy(point).float(), torch.tensor(label).long()


class PointCloudSegmentationDataset(Dataset):
    """点云分割数据集"""

    def __init__(self, points: np.ndarray, labels: np.ndarray,
                 num_points: int = 2048, transform=None):
        self.points = points
        self.labels = labels
        self.num_points = num_points
        self.transform = transform

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        point = self.points[idx]
        label = self.labels[idx]

        if len(point) > self.num_points:
            choice = np.random.choice(len(point), self.num_points, replace=False)
            point = point[choice]
            label = label[choice]

        if self.transform:
            point, label = self.transform(point, label)

        return torch.from_numpy(point).float(), torch.from_numpy(label).long()


def get_data_loaders(dataset_name: str, data_root: str, batch_size: int = 32,
                    num_workers: int = 4, num_points: int = 1024,
                    num_classes: int = 10) -> Tuple[DataLoader, DataLoader]:
    """
    获取数据加载器

    Args:
        dataset_name: 数据集名称 ('modelnet40', 'synthetic')
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        num_points: 每个样本的点数
        num_classes: 类别数量

    Returns:
        train_loader, test_loader
    """
    from .transforms import PointCloudTransform
    transform = PointCloudTransform(
        rotate=True,
        jitter=True,
        scale=True,
        shift=True
    )

    if dataset_name == 'modelnet40':
        train_dataset = ModelNet40(data_root, train=True, num_points=num_points, transform=transform)
        test_dataset = ModelNet40(data_root, train=False, num_points=num_points)
    elif dataset_name == 'synthetic':
        train_dataset = SyntheticPointCloudDataset(
            num_samples=10000, num_points=num_points, num_classes=num_classes
        )
        test_dataset = SyntheticPointCloudDataset(
            num_samples=2000, num_points=num_points, num_classes=num_classes
        )
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
