"""Run FLOPs counting over a grid of model configs (T5 support).

GFLOPs convention: 1 MAC = 2 FLOPs, so gflops = 2 * macs / 1e9.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import build_model, save_json, timestamp  # noqa: E402
from experiments.efficiency.flops import count_flops  # noqa: E402
from experiments.efficiency.measure_efficiency import count_parameters  # noqa: E402


def config_name(voxel_mode: str, sigma: float, grid: int, backbone: str) -> str:
    return f"{voxel_mode}-s{sigma}-{grid}-{backbone}"


def to_row(payload: Dict, name: str) -> Dict:
    return {
        "name": name,
        "params": payload["params"],
        "params_m": payload["params"] / 1e6,
        "macs": payload["macs"],
        "gflops": 2.0 * payload["macs"] / 1e9,
        "by_type": payload.get("by_type", {}),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--voxel-mode", default="gaussian")
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--grids", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--backbones", nargs="+", default=["tiny", "small"])
    p.add_argument("--n-points", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    return p


def run(argv: Optional[List[str]] = None) -> Dict:
    args = build_parser().parse_args(argv)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    rows = []
    for grid in args.grids:
        for backbone in args.backbones:
            model = build_model(voxel_mode=args.voxel_mode, sigma=args.sigma,
                                grid_size=grid, backbone=backbone, device=device)
            x = torch.randn(1, args.n_points, 3, device=device)
            payload = count_flops(model, x)
            payload["params"] = count_parameters(model)
            name = config_name(args.voxel_mode, args.sigma, grid, backbone)
            row = to_row(payload, name)
            rows.append(row)
            print(f"{name:28s} params={row['params_m']:.2f}M gflops={row['gflops']:.3f}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    result = {"meta": {"voxel_mode": args.voxel_mode, "sigma": args.sigma,
                       "n_points": args.n_points, "device": device,
                       "timestamp": timestamp()},
              "rows": rows}
    save_json(args.out, result)
    print(f"[OK] -> {args.out}")
    return result


if __name__ == "__main__":
    run()
