"""Dump learned decay temperatures (tau) per heat block for all checkpoints.

tau = decay_temp in HeatConduction3D; the effective weight is
decay_raw^(tau * k). tau -> 0 disables the spectral filter entirely.

Usage: python -m experiments.inspect_heat_tau --out experiments/results/heat_tau.json
"""
import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import save_json, timestamp  # noqa: E402

DEFAULT_CKPTS = {
    "16-baseline-learned-heat": "checkpoints_heatab_pilot_baseline_g16_tiny/best_model.pth",
    "16-none": "checkpoints_heatab_pilot_none_g16_tiny/best_model.pth",
    "16-ideal": "checkpoints_heatab_pilot_ideal_g16_tiny/best_model.pth",
    "16-steps2": "checkpoints_heatab_pilot_steps2_g16_tiny/best_model.pth",
    "16-sharp050": "checkpoints_heatab_pilot_sharp050_g16_tiny/best_model.pth",
    "16-sharp200": "checkpoints_heatab_pilot_sharp200_g16_tiny/best_model.pth",
    "16-sharp400": "checkpoints_heatab_pilot_sharp400_g16_tiny/best_model.pth",
    "16-fixed_heat": "checkpoints_heatab_fixed_heat_s0/best_model.pth",
    "16-iak": "checkpoints_heatab_iak_s0/best_model.pth",
    "32-heat": "checkpoints_fields32_heat_100/best_model.pth",
    "32-none": "checkpoints_fields32_none_100/best_model.pth",
    "32-iak": "checkpoints_fields32_iak_100/best_model.pth",
}


def taus_of(ckpt: str) -> List[float]:
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return [float(v) for k, v in state.items() if k.endswith("decay_temp")]


def run() -> Dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="experiments/results/heat_tau.json")
    ap.add_argument("--ckpt", action="append", default=None, help="name=path (repeatable)")
    args = ap.parse_args()

    ckpts = dict(DEFAULT_CKPTS)
    for item in args.ckpt or []:
        name, _, path = item.partition("=")
        if path:
            ckpts[name] = path

    out: Dict[str, Dict] = {}
    for name, path in ckpts.items():
        if not os.path.exists(path):
            out[name] = {"ckpt": path, "missing": True}
            continue
        taus = taus_of(path)
        out[name] = {"ckpt": path, "n_blocks": len(taus), "tau": taus}
        print(f"{name:28s} n={len(taus):2d} tau={[round(t, 3) for t in taus]}")

    save_json(args.out, {"timestamp": timestamp(), "checkpoints": out})
    print(f"[OK] wrote {args.out}")
    return out


if __name__ == "__main__":
    run()
