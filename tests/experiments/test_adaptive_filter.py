import torch

from experiments.eval_adaptive_filter import select_logits


def test_select_logits_modes():
    clean = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    filt = torch.tensor([[0.0, 3.0], [1.0, 0.0]], dtype=torch.float64)

    out, use = select_logits(clean, filt, "clean", tau=0.9)
    assert float(use.sum()) == 0.0
    assert torch.equal(out, clean)

    out, use = select_logits(clean, filt, "filtered", tau=0.9)
    assert float(use.sum()) == 2.0
    assert torch.equal(out, filt)

    out, use = select_logits(clean, filt, "avg", tau=0.9)
    assert torch.allclose(out, (clean + filt) / 2)
    assert float(use.sum()) == 2.0


def test_select_logits_gate_uses_clean_confidence():
    clean = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    filt = torch.tensor([[0.0, 3.0], [1.0, 0.0]], dtype=torch.float64)
    out, use = select_logits(clean, filt, "gate", tau=0.8)
    assert use.tolist() == [False, True]
    assert torch.equal(out[0], clean[0])
    assert torch.equal(out[1], filt[1])
