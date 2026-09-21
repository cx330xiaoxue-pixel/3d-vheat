#!/usr/bin/env python3
"""
ModelNet40数据集转换脚本
将原始ModelNet40 OFF格式文件转换为点云数据（表面均匀采样）
"""

import os
import sys
import concurrent.futures
import numpy as np
import h5py
import argparse


def read_off_file(file_path):
    """
    读取OFF文件并返回顶点和面数据

    Args:
        file_path: OFF文件路径

    Returns:
        tuple: (vertices, faces) or None
            vertices: (N, 3) float32
            faces: (F, 3) int64
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        lines = content.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]

        if not lines:
            raise ValueError(f"Empty OFF file: {file_path}")

        # 跳过OFF头
        if lines[0] == 'OFF':
            lines = lines[1:]
        elif len(lines[0].split()) == 4 and lines[0].split()[0] == 'OFF':
            lines = lines[1:]
        else:
            raise ValueError(f"Invalid OFF header: {file_path}")

        # 跳过注释
        while lines and lines[0].startswith('#'):
            lines = lines[1:]

        if not lines:
            raise ValueError(f"Empty OFF file after header: {file_path}")

        # 读取顶点和面数
        try:
            parts = lines[0].split()
            n_vertices = int(parts[0])
            n_faces = int(parts[1])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid header format: {lines[0]}")

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
            raise ValueError(f"No valid vertices found in: {file_path}")

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
            raise ValueError(f"No valid faces found in: {file_path}")

        faces = np.array(faces, dtype=np.int64)

        return vertices, faces

    except Exception as e:
        raise ValueError(f"Error reading OFF file {file_path}: {e}")


def sample_surface_uniform(vertices, faces, num_points):
    """
    从三角形网格表面均匀采样点

    Args:
        vertices: (N, 3) float32
        faces: (F, 3) int64
        num_points: 采样点数

    Returns:
        sampled_points: (num_points, 3)
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # 用 float64 计算面面积防止大坐标溢出
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


def process_file(file_path, cat_label, num_points):
    """
    处理单个OFF文件

    Returns:
        tuple: (sampled_points, label) 或 None
    """
    try:
        result = read_off_file(file_path)
        if result is None:
            return None
        vertices, faces = result

        if len(vertices) < 3 or len(faces) < 1:
            return None

        sampled_points = sample_surface_uniform(vertices, faces, num_points)

        centroid = np.mean(sampled_points, axis=0)
        sampled_points -= centroid
        max_dist = np.max(np.linalg.norm(sampled_points, axis=1))
        if max_dist > 0:
            sampled_points /= max_dist

        return sampled_points, cat_label

    except Exception:
        return None


def convert_modelnet40(input_dir, output_dir, num_points=1024, save_h5=False, save_npy=False):
    """
    转换ModelNet40数据集

    Args:
        input_dir: 输入ModelNet40目录
        output_dir: 输出目录
        num_points: 每个点云的点数
        save_h5: 是否保存HDF5格式
        save_npy: 是否保存NPY格式
    """
    os.makedirs(output_dir, exist_ok=True)

    categories = sorted([d for d in os.listdir(input_dir)
                         if os.path.isdir(os.path.join(input_dir, d))])
    category_to_label = {cat: i for i, cat in enumerate(categories)}

    print(f"Found {len(categories)} categories")
    print(f"Categories: {categories}")

    for split in ['train', 'test']:
        points_list = []
        labels_list = []
        errors = 0

        print(f"\n{'='*50}")
        print(f"Processing {split} set...")

        # 收集所有文件
        file_label_pairs = []
        for cat in categories:
            cat_dir = os.path.join(input_dir, cat, split)
            if not os.path.exists(cat_dir):
                continue
            cat_label = category_to_label[cat]
            for fname in os.listdir(cat_dir):
                if fname.endswith('.off'):
                    file_label_pairs.append((os.path.join(cat_dir, fname), cat_label))

        print(f"  Total files: {len(file_label_pairs)}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_file, fp, lbl, num_points): (fp, lbl)
                       for fp, lbl in file_label_pairs}

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    if result:
                        points_list.append(result[0])
                        labels_list.append(result[1])
                    else:
                        errors += 1
                except Exception:
                    errors += 1

                if (i + 1) % 500 == 0 or (i + 1) == len(file_label_pairs):
                    print(f"\r  Progress: {i+1}/{len(file_label_pairs)} "
                          f"({len(points_list)} valid, {errors} errors)", end='', flush=True)
            print()

        if not points_list:
            print(f"  No valid data for {split} set")
            continue

        points_array = np.array(points_list, dtype=np.float32)
        labels_array = np.array(labels_list, dtype=np.int64)

        print(f"  {split}: {len(points_array)} samples, "
              f"points={points_array.shape}, labels={labels_array.shape}")

        if save_h5:
            h5_path = os.path.join(output_dir, f'modelnet40_{split}.h5')
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('data', data=points_array)
                f.create_dataset('label', data=labels_array)
            print(f"  Saved HDF5: {h5_path}")

        if save_npy:
            npy_points_path = os.path.join(output_dir, f'modelnet40_{split}_points.npy')
            npy_labels_path = os.path.join(output_dir, f'modelnet40_{split}_labels.npy')
            np.save(npy_points_path, points_array)
            np.save(npy_labels_path, labels_array)
            print(f"  Saved NPY: {npy_points_path}")
            print(f"  Saved NPY: {npy_labels_path}")

    print(f"\n{'='*50}")
    print("Conversion completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ModelNet40数据集转换')
    parser.add_argument('--input_dir', type=str,
                        default='./ModelNet40',
                        help='ModelNet40 OFF文件目录')
    parser.add_argument('--output_dir', type=str,
                        default='./data',
                        help='输出目录')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='每个点云的采样点数')
    parser.add_argument('--save_h5', action='store_true',
                        help='保存HDF5格式')
    parser.add_argument('--save_npy', action='store_true',
                        help='保存NPY格式')
    args = parser.parse_args()

    convert_modelnet40(args.input_dir, args.output_dir,
                       num_points=args.num_points,
                       save_h5=args.save_h5,
                       save_npy=args.save_npy)
