import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_model_kwargs_mapping_defaults():
    from train_fields_scaled import build_arg_parser, model_kwargs_from_args
    args = build_arg_parser().parse_args([])
    kw = model_kwargs_from_args(args)
    assert kw["voxel_mode"] == "gaussian"
    assert kw["sigma"] == 0.05
    assert kw["grid_size"] == 32
    assert kw["backbone"] == "small"
    assert kw["diffusion"] == "heat"
    assert kw["diffusion_steps"] == 1
    assert kw["anisotropic"] is False
    assert kw["decay_sharpness"] == 1.0


def test_new_flags_parse():
    from train_fields_scaled import build_arg_parser, model_kwargs_from_args
    args = build_arg_parser().parse_args([
        "--diffusion", "ideal", "--diffusion-steps", "2", "--anisotropic",
    ])
    kw = model_kwargs_from_args(args)
    assert kw["diffusion"] == "ideal"
    assert kw["diffusion_steps"] == 2
    assert kw["anisotropic"] is True
    assert kw["enable_dynamic_alpha"] is True


def test_static_alpha_flag():
    from train_fields_scaled import build_arg_parser, model_kwargs_from_args
    args = build_arg_parser().parse_args(["--static-alpha"])
    assert model_kwargs_from_args(args)["enable_dynamic_alpha"] is False


def test_tau_invert_flag_parses():
    from train_fields_scaled import build_arg_parser
    assert build_arg_parser().parse_args(["--tau-invert"]).tau_invert is True


def test_diffusion_sharpness_flag():
    from train_fields_scaled import build_arg_parser, model_kwargs_from_args
    args = build_arg_parser().parse_args(["--diffusion-sharpness", "4.0"])
    assert model_kwargs_from_args(args)["decay_sharpness"] == 4.0


def test_seed_flag_default_disabled():
    from train_fields_scaled import build_arg_parser
    assert build_arg_parser().parse_args([]).seed == -1


def test_seed_flag_reproducible_rngs():
    import random

    import numpy as np
    import torch

    from train_fields_scaled import apply_seed, build_arg_parser
    args = build_arg_parser().parse_args(["--seed", "123"])
    apply_seed(args.seed)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    apply_seed(args.seed)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second


def test_apply_seed_disabled_does_not_reseed():
    import random

    import numpy as np
    import torch

    from train_fields_scaled import apply_seed
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    apply_seed(-1)
    assert random.random() == random.Random(7).random()
    assert float(np.random.rand()) == float(np.random.RandomState(7).rand())
    torch.manual_seed(7)
    expected_torch = float(torch.rand(1))
    torch.manual_seed(7)
    apply_seed(-1)
    assert float(torch.rand(1)) == expected_torch


def test_dct_augment_flag():
    from train_fields_scaled import build_arg_parser, model_kwargs_from_args
    args = build_arg_parser().parse_args(["--dct-augment"])
    assert model_kwargs_from_args(args)["dct_augment"] is True
    args0 = build_arg_parser().parse_args([])
    assert model_kwargs_from_args(args0)["dct_augment"] is False


def test_dct_augment_skip_flag():
    from train_fields_scaled import build_arg_parser, model_kwargs_from_args
    args = build_arg_parser().parse_args(["--dct-augment", "--dct-augment-skip", "hf", "band"])
    kw = model_kwargs_from_args(args)
    assert kw["dct_augment_skip"] == ["hf", "band"]
    args0 = build_arg_parser().parse_args([])
    assert model_kwargs_from_args(args0)["dct_augment_skip"] == []


def test_input_adaptive_k_flag():
    from train_fields_scaled import build_arg_parser, model_kwargs_from_args
    args = build_arg_parser().parse_args(["--input-adaptive-k"])
    assert model_kwargs_from_args(args)["input_adaptive_k"] is True
    args0 = build_arg_parser().parse_args([])
    assert model_kwargs_from_args(args0)["input_adaptive_k"] is False
