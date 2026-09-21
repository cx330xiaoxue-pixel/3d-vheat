"""Perturbation axes for E2.

`category` follows the spec's frequency partition:
  - "info_loss": high-frequency information loss (low-pass friendly)
  - "broadband": broadband additive perturbations
  - "transformation": global geometric transforms
`official` maps the axis onto the ModelNet40-C taxonomy when applicable.
"""
from typing import Dict, List, Tuple

import torch

from . import perturbations as P

CATEGORIES = {"info_loss", "broadband", "transformation"}

AXES: Dict[str, dict] = {
    "gaussian_noise": {
        "step": lambda x, lv, g: P.gaussian_noise(x, lv, generator=g),
        "levels": [0.005, 0.01, 0.02, 0.03, 0.05],
        "category": "broadband",
        "official": {"uniform", "gaussian"},
    },
    "dropout": {
        "step": lambda x, lv, g: P.random_dropout(x, 1.0 - lv, generator=g),
        "levels": [0.25, 0.50, 0.75, 0.875],
        "category": "info_loss",
        "official": {"cutout", "occlusion", "density_dec"},
    },
    "fps": {
        "step": lambda x, lv, g: P.fps_downsample(x, int(lv), generator=g),
        "levels": [768, 512, 256, 128],
        "category": "info_loss",
        "official": {"density_dec"},
    },
    "outlier_replace": {
        "step": lambda x, lv, g: P.outlier_replace(x, lv, generator=g),
        "levels": [0.01, 0.05],
        "category": "broadband",
        "official": {"impulse", "background"},
    },
    "rotation_yaw": {
        "step": lambda x, lv, g: P.random_rotation(x, generator=g, yaw_only=True, max_yaw_deg=lv),
        "levels": [15, 30, 45],
        "category": "transformation",
        "official": {"rotation"},
    },
    "rotation_so3": {
        "step": lambda x, lv, g: P.random_rotation(x, generator=g),
        "levels": [1],
        "category": "transformation",
        "official": {"rotation"},
    },
    "coarse_quantize": {
        "step": lambda x, lv, g: P.coarse_quantize(x, int(lv)),
        "levels": [16, 24, 32],
        "category": "info_loss",
        "official": None,
    },
}


def get_axis(name: str) -> dict:
    return AXES[name]


def run_axis(name: str, xyz: torch.Tensor, generator=None
             ) -> List[Tuple[float, torch.Tensor]]:
    axis = get_axis(name)
    return [(lv, axis["step"](xyz, lv, generator)) for lv in axis["levels"]]
