import json

from experiments.report.build_tables import robustness_table, pareto_table


def test_robustness_table_markdown():
    payload = {
        "clean": {"acc": 0.9, "n": 100},
        "axes": {
            "gaussian_noise": {
                "0.01": {"acc_mean": 0.88, "acc_std": 0.01, "per_seed": [0.87, 0.88, 0.89]},
                "0.05": {"acc_mean": 0.60, "acc_std": 0.02, "per_seed": [0.58, 0.60, 0.62]},
            }
        },
    }
    md = robustness_table(payload)
    assert "| axis |" in md
    assert "gaussian_noise" in md and "0.60" in md


def test_pareto_table_markdown():
    rows = [
        {"name": "ours-tiny-16", "params_m": 1.2, "gflops": 0.5, "top1": 88.0},
        {"name": "dgcnn", "params_m": 1.81, "gflops": 2.68, "top1": 92.82},
    ]
    md = pareto_table(rows)
    assert "ours-tiny-16" in md and "dgcnn" in md
    assert "params" in md.lower()
