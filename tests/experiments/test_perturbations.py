import math

import torch

from experiments.robustness.perturbations import (
    coarse_quantize,
    fps_downsample,
    gaussian_noise,
    outlier_replace,
    random_dropout,
    random_rotation,
)


def make_points(B=2, N=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, N, 3, generator=g) * 2 - 1


def test_gaussian_noise_shape_and_determinism():
    x = make_points()
    a = gaussian_noise(x, 0.02, generator=torch.Generator().manual_seed(1))
    b = gaussian_noise(x, 0.02, generator=torch.Generator().manual_seed(1))
    c = gaussian_noise(x, 0.02, generator=torch.Generator().manual_seed(2))
    assert a.shape == x.shape
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)
    assert (a - x).abs().mean() > 0


def test_random_dropout_keeps_exact_unique_count():
    x = make_points()
    out = random_dropout(x, keep_ratio=0.5, generator=torch.Generator().manual_seed(0))
    assert out.shape == (2, 32, 3)
    for b in range(2):
        uniq = torch.unique(out[b], dim=0)
        assert uniq.shape[0] == 32


def test_fps_downsample_returns_subset_and_spreads():
    x = make_points(B=1, N=32)
    out = fps_downsample(x, 8, generator=torch.Generator().manual_seed(0))
    assert out.shape == (1, 8, 3)
    for p in out[0]:
        assert (x[0] - p).abs().sum(dim=1).min() < 1e-6


def test_outlier_replace_count_and_bounds():
    x = make_points(B=2, N=100)
    out = outlier_replace(x, ratio=0.05, generator=torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    changed = ((out - x).abs().sum(-1) > 1e-6).float().sum(1)
    assert torch.equal(changed, torch.full((2,), 5.))
    assert out.abs().max() <= 1.0 + 1e-6


def test_rotation_is_orthogonal_det_one_and_norm_preserving():
    x = make_points()
    g = torch.Generator().manual_seed(0)
    out = random_rotation(x, generator=g)
    assert out.shape == x.shape
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    yaw = random_rotation(x, generator=g, yaw_only=True)
    assert torch.allclose(yaw.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_coarse_quantize_error_bound():
    x = make_points()
    n = 16
    out = coarse_quantize(x, grid_size=n, extent=2.0)
    cell = 2.0 / n
    assert (out - x).abs().max() <= cell / 2 + 1e-6
    for v in torch.unique(out):
        snapped = (v + 1.0) / cell
        assert abs(snapped - snapped.round()) < 1e-5
