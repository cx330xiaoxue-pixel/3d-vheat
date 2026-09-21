from experiments.efficiency.flops_matrix import config_name, to_row


def test_config_name():
    assert config_name("gaussian", 0.2, 16, "tiny") == "gaussian-s0.2-16-tiny"


def test_to_row_converts_params_and_gflops():
    row = to_row({"params": 1_000_000, "macs": 500_000_000,
                  "by_type": {"conv3d": 500_000_000}}, "x")
    assert row["name"] == "x"
    assert abs(row["params_m"] - 1.0) < 1e-9
    assert abs(row["gflops"] - 1.0) < 1e-9
    assert row["by_type"]["conv3d"] == 500_000_000
