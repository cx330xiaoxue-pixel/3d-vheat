import numpy as np
import torch

from experiments.common import accuracy, normalize_points


def test_normalize_points_zero_mean_unit_maxnorm():
    g = np.random.RandomState(0)
    pts = g.randn(4, 128, 3).astype(np.float32) * 5 + 3
    out = normalize_points(pts)
    assert out.shape == pts.shape
    for b in range(4):
        assert abs(out[b].mean(0)).max() < 1e-5
        assert abs(np.linalg.norm(out[b], axis=1).max() - 1.0) < 1e-5


def test_normalize_points_handles_degenerate():
    pts = np.zeros((1, 8, 3), dtype=np.float32)
    out = normalize_points(pts)
    assert np.isfinite(out).all()


def test_accuracy_counts_matches():
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    labels = torch.tensor([0, 1, 1])
    acc, n = accuracy(logits, labels)
    assert n == 3
    assert abs(acc - 2 / 3) < 1e-6
