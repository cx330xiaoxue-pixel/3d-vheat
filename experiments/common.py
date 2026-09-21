"""Shared helpers: seeding, point normalization, model build/load, evaluation."""
import json
import os
import random
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vheat3d.modules.vheat3d_migrated import VHeat3DMigrated  # noqa: E402

BACKBONE_CONFIGS = {
    "tiny": {"depths": [2, 2, 2, 2], "dims": [32, 64, 128, 256]},
    "small": {"depths": [2, 2, 6, 2], "dims": [64, 128, 256, 512]},
    "medium": {"depths": [2, 2, 6, 2], "dims": [96, 192, 384, 768]},
    "large": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
    "xlarge": {"depths": [3, 4, 12, 3], "dims": [128, 256, 512, 1024]},
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    centroid = pts.mean(axis=1, keepdims=True)
    pts = pts - centroid
    d = np.linalg.norm(pts, axis=2, keepdims=True).max(axis=1, keepdims=True)
    d = np.maximum(d, 1e-8)
    return (pts / d).astype(np.float32)


def load_points_and_labels(pts_path: str, lbl_path: str,
                           normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.load(pts_path).astype(np.float32)
    lbls = np.load(lbl_path).astype(np.int64)
    if normalize:
        pts = normalize_points(pts)
    return pts, lbls


def build_model(voxel_mode: str = "gaussian", sigma: float = 0.05,
                grid_size: int = 32, backbone: str = "small",
                dct_augment: bool = False, device: str = "cuda",
                diffusion: str = "heat", diffusion_steps: int = 1,
                anisotropic: bool = False, decay_sharpness: float = 1.0,
                enable_dynamic_alpha: bool = True, bounds_mode: str = "batch",
                input_adaptive_k: bool = False,
                **extra) -> VHeat3DMigrated:
    cfg = BACKBONE_CONFIGS[backbone]
    model = VHeat3DMigrated(
        num_classes=40,
        depths=cfg["depths"], dims=cfg["dims"],
        post_norm=False, layer_scale=None, use_bottleneck=False,
        enable_dynamic_alpha=enable_dynamic_alpha, enable_high_freq_boost=False,
        drop_path_rate=0.2, mlp_ratio=4.0, dropout=0.1,
        grid_size=(grid_size, grid_size, grid_size),
        use_checkpoint=False, enable_dct_augment=dct_augment,
        voxel_mode=voxel_mode, voxel_sigma=sigma,
        diffusion=diffusion, diffusion_steps=diffusion_steps, anisotropic=anisotropic,
        decay_sharpness=decay_sharpness,
        input_adaptive_k=input_adaptive_k,
        voxel_bounds_mode=bounds_mode,
        **extra,
    )
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str,
                    device: str = "cuda") -> torch.nn.Module:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model.eval()


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, int]:
    correct = (logits.argmax(dim=1) == labels).sum().item()
    return correct / labels.numel(), labels.numel()


@torch.no_grad()
def evaluate(model: torch.nn.Module, pts: np.ndarray, lbls: np.ndarray,
             batch_size: int = 64, device: str = "cuda",
             amp: bool = True) -> Dict:
    model.eval()
    n = len(pts)
    correct = 0
    for i in range(0, n, batch_size):
        x = torch.from_numpy(pts[i:i + batch_size]).float().to(device)
        y = torch.from_numpy(lbls[i:i + batch_size]).long().to(device)
        with torch.autocast("cuda", enabled=amp and device.startswith("cuda")):
            logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
    return {"acc": correct / n, "n": n}


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
