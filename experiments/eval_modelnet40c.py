"""Evaluate a checkpoint on ModelNet40-C using an index built by inspect_mn40c."""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (  # noqa: E402
    build_model, evaluate, load_checkpoint, normalize_points, save_json, timestamp,
)
from experiments.robustness.mn40c import BASELINE_ER, compute_er_report  # noqa: E402


def _load_array(path: str) -> np.ndarray:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        key = data.files[0]
        data = data[key]
    return np.asarray(data, dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--index", required=True, help="JSON from inspect_mn40c")
    p.add_argument("--labels", default=None, help="optional .npy test labels (default: unique labels per file are ignored)")
    p.add_argument("--voxel-mode", default="gaussian")
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--backbone", default="small")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dct-augment", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    return p


def run(argv: Optional[List[str]] = None) -> Dict:
    args = build_parser().parse_args(argv)
    with open(args.index) as f:
        index_meta = json.load(f)
    root, index = index_meta["root"], index_meta["files"]

    lbls = None
    if args.labels:
        lbls = np.load(args.labels).astype(np.int64)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = build_model(voxel_mode=args.voxel_mode, sigma=args.sigma,
                        grid_size=args.grid, backbone=args.backbone, device=device,
                        dct_augment=args.dct_augment)
    load_checkpoint(model, args.ckpt, device=device)

    accs: Dict[str, float] = {}
    for rel, meta in sorted(index.items()):
        pts = _load_array(os.path.join(root, rel))
        if pts.ndim == 2:
            pts = pts[None]
        if pts.shape[-1] != 3 and pts.shape[1] == 3:
            pts = pts.transpose(0, 2, 1)
        if lbls is not None and len(lbls) != len(pts):
            lbls_use = np.tile(lbls, (len(pts) // len(lbls) + 1))[:len(pts)]
        elif lbls is not None:
            lbls_use = lbls
        else:
            raise SystemExit("--labels is required: corrupted files carry no labels")
        if args.limit:
            pts, lbls_use = pts[:args.limit], lbls_use[:args.limit]
        pts = normalize_points(pts)
        acc = evaluate(model, pts, lbls_use, args.batch_size, device)["acc"]
        accs[rel] = acc
        print(f"{rel}: acc={acc:.4f}")

    report = compute_er_report(index, accs)
    report["meta"] = {"ckpt": args.ckpt, "sigma": args.sigma, "grid": args.grid,
                      "backbone": args.backbone, "device": device,
                      "timestamp": timestamp()}
    report["baselines"] = BASELINE_ER
    save_json(args.out, report)
    print(f"[OK] ER_cor={report['er_cor']:.4f} -> {args.out}")
    return report


if __name__ == "__main__":
    run()
