import torch

from vheat3d.modules.heat_conduction import modulate_k


def test_modulate_k_identity_when_stats_zero():
    k = torch.ones(1, 3, 2, 2, 2)
    lin = torch.nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(3))
    delta_fn = torch.nn.Sequential(lin, torch.nn.Tanh())
    out = modulate_k(k, torch.zeros(2, 3), delta_fn)
    assert torch.allclose(out, k.expand_as(out), atol=1e-6)


def test_modulate_k_varies_with_input_stats():
    k = torch.ones(1, 2, 2, 2, 2)
    lin = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(2))
    delta_fn = torch.nn.Sequential(lin, torch.nn.Tanh())
    out_a = modulate_k(k, torch.zeros(1, 2), delta_fn)
    out_b = modulate_k(k, torch.ones(1, 2), delta_fn)
    assert not torch.allclose(out_a, out_b)
    assert float(out_b.mean()) > float(out_a.mean())
