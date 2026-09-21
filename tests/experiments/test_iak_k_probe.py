import numpy as np
import torch

from experiments.iak_k_probe import (
    DEFAULT_CONDITIONS,
    NOISE_GROUP,
    DeltaCollector,
    ROTATION_GROUP,
    cond_key,
    separation,
    summarize,
)


def test_cond_key_formats_levels():
    assert cond_key(("gaussian_noise", 0.05)) == "gaussian_noise:0.05"
    assert cond_key(("rotation_so3", 1)) == "rotation_so3:1"


def test_groups_are_disjoint_and_in_defaults():
    defaults = {cond_key(c) for c in DEFAULT_CONDITIONS}
    noise = {cond_key(c) for c in NOISE_GROUP}
    rot = {cond_key(c) for c in ROTATION_GROUP}
    assert noise.isdisjoint(rot)
    assert noise | rot <= defaults


def test_summarize_basic_stats():
    out = summarize([1.0, 2.0, 3.0])
    assert out["n"] == 3
    assert abs(out["mean"] - 2.0) < 1e-9
    assert abs(out["std"] - 1.0) < 1e-9
    assert summarize([])["n"] == 0


def test_separation_auc_direction():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [-3.0, -2.0, -1.0, 0.0]
    out = separation(a, b)
    assert out["auc"] == 1.0
    assert out["p_greater"] < 0.05
    assert out["mean_a"] > out["mean_b"]


def test_separation_symmetric_auc_half():
    a = [0.0, 1.0]
    b = [0.0, 1.0]
    out = separation(a, b)
    assert abs(out["auc"] - 0.5) < 1e-9


def test_delta_collector_per_sample_and_layer():
    c = DeltaCollector()
    c.values = [
        torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
        torch.tensor([[0.5, 0.5], [1.0, 1.0]]),
    ]
    per_sample = c.per_sample()
    assert np.allclose(per_sample, [1.25, 2.0])
    assert np.allclose(c.per_layer(), [2.5, 0.75])
    c.clear()
    assert c.per_sample().size == 0
