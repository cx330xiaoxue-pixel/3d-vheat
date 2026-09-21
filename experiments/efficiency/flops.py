"""Hook-based MACs counting for standard + custom 3D modules.

- nn.Conv3d / nn.Linear: exact dense MACs.
- TileConv3d (custom): same formula as dense conv (weights are dense).
- DCT3D / IDCT3D: analytic separable DCT-II estimate; counted via the
  module's config (grid size) when available, else skipped and reported.
"""
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.nn as nn


def conv3d_macs(out_shape, weight_shape) -> int:
    out_numel = 1
    for s in out_shape:
        out_numel *= int(s)
    kernel_vol = 1
    for s in weight_shape[2:]:
        kernel_vol *= int(s)
    return out_numel * int(weight_shape[1]) * kernel_vol


def linear_macs(out_shape, weight_shape) -> int:
    out_numel = 1
    for s in out_shape:
        out_numel *= int(s)
    return out_numel * int(weight_shape[1])


def dct_macs(x_shape, steps: int = 2) -> int:
    """Separable 3D DCT-II MACs: steps * V * (d + h + w) with V = d*h*w."""
    _, _, d, h, w = (int(s) for s in x_shape)
    v = d * h * w
    return steps * v * (d + h + w)


class FlopCounter:
    def __init__(self):
        self.macs = 0
        self.by_type: Dict[str, int] = defaultdict(int)
        self._handles = []

    def attach(self, model: nn.Module) -> "FlopCounter":
        for name, mod in model.named_modules():
            if isinstance(mod, (nn.Conv3d, nn.Linear)):
                self._handles.append(mod.register_forward_hook(self._hook_std))
            elif mod.__class__.__name__ in {"TileConv3d"}:
                self._handles.append(mod.register_forward_hook(self._hook_conv))
            elif mod.__class__.__name__ in {"DCT3D", "IDCT3D"}:
                self._handles.append(mod.register_forward_hook(self._hook_dct))
        return self

    def _record(self, key: str, macs: int):
        self.macs += macs
        self.by_type[key] += macs

    def _hook_std(self, module, inputs, output):
        if isinstance(module, nn.Linear):
            self._record("linear", linear_macs(output.shape, module.weight.shape))
        else:
            self._record("conv3d", conv3d_macs(output.shape, module.weight.shape))

    def _hook_conv(self, module, inputs, output):
        weight = getattr(module, "weight", None)
        if weight is not None and weight.dim() == 5:
            self._record("tile_conv3d", conv3d_macs(output.shape, weight.shape))
        else:
            self._record("tile_conv3d", 0)

    def _hook_dct(self, module, inputs, output):
        self._record("dct", dct_macs(inputs[0].shape, steps=1))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def count_flops(model: nn.Module, example_input: torch.Tensor) -> Dict:
    counter = FlopCounter().attach(model)
    model.eval()
    with torch.no_grad():
        model(example_input)
    counter.remove()
    return {"macs": counter.macs, "by_type": dict(counter.by_type)}
