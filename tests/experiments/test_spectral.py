import torch

from experiments.spectral.spectral import (
    band_mask, dct, dct_roundtrip_error, idct, radial_power, quantization_spectra,
)


def test_band_mask_fraction_monotone():
    m_lo = band_mask((16, 16, 16), r_lo=0.0, r_hi=0.25)
    m_hi = band_mask((16, 16, 16), r_lo=0.0, r_hi=0.5)
    assert 0 < m_lo.mean() < m_hi.mean() < 1.01
    assert set(torch.unique(m_lo).tolist()) <= {0.0, 1.0}


def test_dct_roundtrip_small_error():
    x = torch.rand(1, 2, 8, 8, 8)
    err = dct_roundtrip_error(x)
    assert err < 1e-5


def test_radial_power_shape_and_dc_dominance():
    x = torch.ones(1, 1, 8, 8, 8)
    centers, power = radial_power(x, n_bins=4)
    assert len(centers) == len(power) == 4
    assert power[0] == power.max()


def test_quantization_spectra_hf_ratio():
    pts = torch.rand(1, 256, 3) * 2 - 1
    out = quantization_spectra(pts, grid_size=16, sigma=0.1)
    assert "hf_ratio_binary" in out and "hf_ratio_gaussian" in out
    assert out["hf_ratio_binary"] >= 0 and out["hf_ratio_gaussian"] >= 0


def test_dct_matches_module_forward_transform():
    from vheat3d.operators import DCT3D
    x = torch.randn(1, 2, 8, 8, 8)
    mask = torch.ones(1, 1, 8, 8, 8)
    mod = DCT3D()
    assert torch.allclose(dct(x), mod(x, mask), atol=1e-5)


def test_idct_is_exact_inverse_of_dct():
    from vheat3d.operators import DCT3D, IDCT3D
    x = torch.randn(1, 2, 8, 8, 8)
    mask = torch.ones(1, 1, 8, 8, 8)
    assert torch.allclose(idct(dct(x)), x, atol=1e-5)
    assert torch.allclose(DCT3D()(IDCT3D()(x, mask), mask), x, atol=1e-5)
