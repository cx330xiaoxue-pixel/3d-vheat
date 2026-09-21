import torch

from experiments.robustness.axes import AXES, CATEGORIES, get_axis, run_axis


def test_axes_have_required_fields():
    for name, axis in AXES.items():
        assert axis["category"] in CATEGORIES, name
        assert len(axis["levels"]) >= 1, name
        assert callable(axis["step"]), name


def test_levels_are_unique_and_monotonic():
    for name, axis in AXES.items():
        levels = axis["levels"]
        ascending = levels == sorted(levels)
        descending = levels == sorted(levels, reverse=True)
        assert ascending or descending, name
        assert len(levels) == len(set(levels)), name


def test_run_axis_returns_all_levels():
    x = torch.rand(2, 64, 3) * 2 - 1
    for name in ["gaussian_noise", "dropout", "coarse_quantize"]:
        out = run_axis(name, x, generator=torch.Generator().manual_seed(0))
        assert len(out) == len(AXES[name]["levels"])
        for _, xp in out:
            assert xp.shape[0] == 2 and xp.shape[-1] == 3


def test_get_axis_unknown_raises():
    try:
        get_axis("nope")
    except KeyError:
        return
    raise AssertionError("expected KeyError")
