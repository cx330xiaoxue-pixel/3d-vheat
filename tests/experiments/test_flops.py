import pytest
import torch
import torch.nn as nn

from experiments.efficiency.flops import (
    count_flops, conv3d_macs, dct_macs,
)


def test_conv3d_macs_formula():
    out_shape = (2, 8, 4, 5, 6)
    weight_shape = (8, 4, 3, 3, 3)
    expect = 2 * 8 * 4 * 5 * 6 * (4 * 27)
    assert conv3d_macs(out_shape, weight_shape) == expect


def test_dct_macs_formula():
    n = 16
    macs = dct_macs((1, 1, n, n, n), steps=2)
    expect = (n ** 3) * (3 * n) * 2
    assert macs == expect


def test_count_flops_plain_conv_matches_fvcore():
    pytest.importorskip("fvcore")
    from fvcore.nn import FlopCountAnalysis
    model = nn.Conv3d(4, 8, 3, padding=1)
    x = torch.randn(2, 4, 6, 6, 6)
    ours = count_flops(model, x)["macs"]
    fv = FlopCountAnalysis(model, x).total()
    ratio = ours / max(fv, 1)
    # fvcore counts MACs either as 1 or 2 FLOPs depending on version.
    assert min(abs(ratio - 1.0), abs(ratio - 2.0)) < 1e-3


def test_count_flops_counts_linear():
    model = nn.Sequential(nn.Linear(10, 5))
    x = torch.randn(3, 10)
    assert count_flops(model, x)["macs"] == 3 * 5 * 10
