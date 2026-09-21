import copy

import pytest
import torch

torch.manual_seed(0)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="HeatConduction3D uses TileConv3d (CUDA kernel)",
)


def make_hco(**kw):
    from vheat3d.modules.heat_conduction import HeatConduction3D
    return HeatConduction3D(8, 8, grid_size=(8, 8, 8), use_bottleneck=False, **kw).cuda()


def _rand_inputs():
    x = torch.randn(2, 8, 8, 8, 8, device="cuda")
    mask = torch.ones(2, 1, 8, 8, 8, device="cuda")
    freq_embed = torch.randn(8, 8, 8, 8, device="cuda")
    return x, mask, freq_embed


def test_default_state_dict_keys_unchanged():
    a = make_hco()
    b = make_hco()
    sd_a = set(a.state_dict().keys())
    sd_b = set(b.state_dict().keys())
    assert sd_a == sd_b


def test_default_outputs_equal_when_loading_same_state():
    a = make_hco().eval()
    b = make_hco().eval()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x, mask, fe = _rand_inputs()
    with torch.no_grad():
        ya = a(x, mask, fe)
        yb = b(x, mask, fe)
    assert torch.equal(ya, yb)


def test_diffusion_none_changes_output_but_keeps_keys():
    a = make_hco().eval()
    b = make_hco(diffusion="none").eval()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x, mask, fe = _rand_inputs()
    with torch.no_grad():
        ya = a(x, mask, fe)
        yb = b(x, mask, fe)
    assert not torch.allclose(ya, yb)


def test_diffusion_steps_two_equals_squared_weight():
    a = make_hco().eval()
    b = make_hco(diffusion_steps=2).eval()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x, mask, fe = _rand_inputs()
    with torch.no_grad():
        f = a.dct3d(a.proj_conv(x, mask), mask)
        wa = a._diffusion_weight(f, fe)
        wb = b._diffusion_weight(f, fe)
    assert torch.allclose(wb, wa ** 2, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("mode", ["ideal", "cosine", "fixed_heat"])
def test_alternative_diffusion_modes_run(mode):
    m = make_hco(diffusion=mode).eval()
    x, mask, fe = _rand_inputs()
    with torch.no_grad():
        y = m(x, mask, fe)
    assert y.shape == (2, 8, 8, 8, 8)


def test_decay_sharpness_default_is_identity():
    a = make_hco().eval()
    b = make_hco(decay_sharpness=1.0).eval()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x, mask, fe = _rand_inputs()
    with torch.no_grad():
        f = a.dct3d(a.proj_conv(x, mask), mask)
        assert torch.equal(a._diffusion_weight(f, fe), b._diffusion_weight(f, fe))


def test_decay_sharpness_narrows_band():
    from vheat3d.modules.heat_conduction import HeatConduction3D
    d1 = HeatConduction3D._compute_decay_map((8, 8, 8), 1.0)
    d2 = HeatConduction3D._compute_decay_map((8, 8, 8), 2.0)
    assert float(d2.mean()) < float(d1.mean())
    assert float(d2.max()) <= float(d1.max()) + 1e-6


def test_decay_sharpness_changes_output():
    a = make_hco().eval()
    b = make_hco(decay_sharpness=2.0).eval()
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x, mask, fe = _rand_inputs()
    with torch.no_grad():
        assert not torch.allclose(a(x, mask, fe), b(x, mask, fe))


def test_anisotropic_flag_runs_and_is_off_by_default():
    m = make_hco(anisotropic=True).eval()
    x, mask, fe = _rand_inputs()
    with torch.no_grad():
        y = m(x, mask, fe)
    assert y.shape == (2, 8, 8, 8, 8)
    d = make_hco().eval()
    assert "aniso_logits" not in d.state_dict()
