"""Measure params / latency / peak memory / throughput for a model config."""
import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import build_model, save_json, timestamp  # noqa: E402


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def latency_stats(times_ms: List[float]) -> Dict:
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "p95_ms": float(np.percentile(arr, 95)),
        "n_iters": int(arr.size),
    }


def measure_cpu_smoke(model: torch.nn.Module, input_dim: int = 8,
                      iters: int = 5) -> Dict:
    x = torch.randn(4, input_dim)
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    return {"params": count_parameters(model), "latency_ms": latency_stats(times)}


def _measure_gpu(model: torch.nn.Module, batch_size: int, n_points: int,
                 iters: int, warmup: int) -> Dict:
    device = "cuda"
    x = torch.randn(batch_size, n_points, 3, device=device)
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    per_sample = np.asarray(times) / batch_size
    return {
        "batch_size": batch_size,
        "latency_ms": latency_stats(times),
        "per_sample_ms": latency_stats(per_sample.tolist()),
        "peak_mem_mb": float(peak_mb),
        "throughput_samples_per_s": float(1000.0 / np.median(per_sample)),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--voxel-mode", default="gaussian")
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--backbone", default="small")
    p.add_argument("--dct-augment", action="store_true")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 64])
    p.add_argument("--n-points", type=int, default=1024)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out", required=True)
    return p


def run(argv: Optional[List[str]] = None) -> Dict:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("GPU required for measure_efficiency")
    model = build_model(voxel_mode=args.voxel_mode, sigma=args.sigma,
                        grid_size=args.grid, backbone=args.backbone,
                        dct_augment=args.dct_augment, device="cuda")
    payload = {
        "meta": {"voxel_mode": args.voxel_mode, "sigma": args.sigma,
                 "grid": args.grid, "backbone": args.backbone,
                 "dct_augment": args.dct_augment, "timestamp": timestamp(),
                 "gpu": torch.cuda.get_device_name(0)},
        "params": count_parameters(model),
        "per_batch": [],
    }
    for bs in args.batch_sizes:
        rec = _measure_gpu(model, bs, args.n_points, args.iters, args.warmup)
        payload["per_batch"].append(rec)
        print(f"bs={bs}: median={rec['latency_ms']['median_ms']:.2f} ms "
              f"per_sample={rec['per_sample_ms']['median_ms']:.2f} ms "
              f"peak={rec['peak_mem_mb']:.0f} MB")
    save_json(args.out, payload)
    return payload


if __name__ == "__main__":
    run()
