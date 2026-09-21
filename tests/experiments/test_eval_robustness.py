import json

import numpy as np
import pytest
import torch

from experiments.eval_robustness import build_parser, summarize


def test_summarize_mean_std():
    out = summarize([0.9, 0.8, 0.85])
    assert abs(out["acc_mean"] - 0.85) < 1e-9
    assert out["acc_std"] > 0
    assert out["per_seed"] == [0.9, 0.8, 0.85]


def test_summarize_ci95_with_three_seeds():
    out = summarize([0.90, 0.80, 0.85])
    assert out["ci95"] == pytest.approx(4.303 * out["acc_std"] / (3 ** 0.5), rel=1e-3)


def test_summarize_single_seed_std_is_zero():
    out = summarize([0.7])
    assert out["acc_std"] == 0.0
    assert out["ci95"] == 0.0


def test_parser_defaults():
    args = build_parser().parse_args(["--out", "/tmp/unused.json"])
    assert args.grid == 32 and args.backbone == "small" and args.seeds == [0, 1, 2]
    assert args.diffusion == "heat" and args.diffusion_steps == 1
    assert args.anisotropic is False and args.static_alpha is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TileConv3d requires CUDA")
def test_end_to_end_gpu_stub(tmp_path):
    from experiments.eval_robustness import run
    pts = (np.random.RandomState(0).rand(16, 256, 3).astype(np.float32) * 2 - 1)
    np.save(tmp_path / "pts.npy", pts)
    np.save(tmp_path / "lbl.npy", np.zeros(16, dtype=np.int64))
    out = tmp_path / "res.json"
    run([
        "--data-pts", str(tmp_path / "pts.npy"),
        "--data-lbl", str(tmp_path / "lbl.npy"),
        "--device", "cuda", "--grid", "8", "--backbone", "tiny",
        "--axes", "gaussian_noise", "--seeds", "0", "1",
        "--limit", "8", "--batch-size", "4",
        "--allow-random-weights", "--out", str(out),
    ])
    payload = json.loads(out.read_text())
    assert payload["meta"]["n_samples"] == 8
    assert "gaussian_noise" in payload["axes"]
    assert "clean" in payload


def test_field_lowpass_tensor_identity_and_attenuation():
    from experiments.eval_robustness import field_lowpass_tensor
    x = torch.randn(1, 2, 8, 8, 8)
    assert torch.allclose(field_lowpass_tensor(x, 1.0), x)
    checker = torch.zeros(1, 1, 8, 8, 8)
    checker[..., ::2] = 1.0
    checker[..., 1::2] = -1.0
    out = field_lowpass_tensor(checker, 0.25)
    assert out.abs().mean() < 0.3 * checker.abs().mean()
    assert out.std() < 0.3 * checker.std()


def test_field_lowpass_flag_applies_hook():
    from experiments.eval_robustness import build_parser, attach_field_lowpass
    args = build_parser().parse_args(["--out", "/tmp/x.json"])
    assert args.field_lowpass == 0.0
    import torch.nn as nn

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.voxelization = nn.Module()

        def forward(self, x):
            return self.voxelization(x)

    m = Dummy()
    m.voxelization.forward = lambda x: (torch.ones(1, 1, 8, 8, 8), None)
    attach_field_lowpass(m, 0.5)
    f, _ = m.voxelization(torch.zeros(1, 8, 3))
    assert torch.isfinite(f).all()


def test_field_lowpass_adaptive_gates_by_hf_ratio():
    from experiments.eval_robustness import field_hf_ratio, field_lowpass_adaptive
    zero = torch.zeros(1, 1, 8, 8, 8)
    checker = torch.zeros(2, 1, 8, 8, 8)
    checker[..., ::2] = 1.0
    checker[..., 1::2] = -1.0
    x = torch.cat([zero, checker], dim=0)
    ratio = field_hf_ratio(x, hf_floor=0.5)
    assert float(ratio[0].mean()) < 1e-9
    assert float(ratio[1].mean()) > 0.5
    out = field_lowpass_adaptive(x, r=0.25, thresh=0.5, hf_floor=0.5)
    assert torch.allclose(out[0], zero[0])
    assert out[1].abs().mean() < 0.3 * checker[1].abs().mean()
