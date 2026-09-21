import torch

from vheat3d.operators.dct_augment import DCTAugment, probs_from_skip


def test_probs_from_skip_defaults():
    p = probs_from_skip(())
    assert p["hf"] == 0.2 and p["drop"] == 0.2 and p["jitter"] == 0.2 and p["band"] == 0.1


def test_probs_from_skip_zeroes_selected():
    p = probs_from_skip(("hf", "band"))
    assert p["hf"] == 0.0 and p["band"] == 0.0
    assert p["drop"] == 0.2 and p["jitter"] == 0.2


def test_dct_augment_all_disabled_is_identity():
    aug = DCTAugment(probs=probs_from_skip(("hf", "drop", "jitter", "band")))
    aug.train()
    x = torch.randn(2, 4, 8, 8, 8)
    mask = torch.ones(2, 1, 8, 8, 8)
    assert torch.allclose(aug(x, mask), x, atol=1e-5)
