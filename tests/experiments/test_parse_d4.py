import json

from experiments.report.parse_d4 import (
    parse_log_text, reach_epoch, auc_of, summarize_run, summarize_arm,
    welch_ttest, build_report,
)

SAMPLE_LOG = """Epoch   0 | train_loss=2.1 train_acc=10.00% test_acc=20.00% lr=1e-3 time=10s
★ New best: 20.00% at epoch 0
Epoch   1 | train_loss=1.9 train_acc=40.00% test_acc=50.00% lr=9e-4 time=20s
★ New best: 50.00% at epoch 1
Epoch   2 | train_loss=1.5 train_acc=60.00% test_acc=80.00% lr=8e-4 time=30s
"""


def test_parse_log_text_extracts_epochs_and_accs():
    epochs, accs = parse_log_text(SAMPLE_LOG)
    assert epochs == [0, 1, 2]
    assert accs == [20.0, 50.0, 80.0]


def test_reach_epoch_uses_fraction_of_final_best():
    accs = [20.0, 50.0, 80.0, 76.0]
    assert reach_epoch(accs, frac=0.95) == 2
    assert reach_epoch([10.0, 20.0], frac=0.95) == 1


def test_reach_epoch_threshold_mode():
    assert reach_epoch([10.0, 73.0, 74.0], threshold=73.0) == 1
    assert reach_epoch([10.0, 20.0], threshold=73.0) == 2


def test_auc_of_mean_curve():
    assert auc_of([100.0] * 10) == 100.0
    assert auc_of([0.0, 100.0]) == 50.0


def test_summarize_run():
    s = summarize_run(SAMPLE_LOG)
    assert s["n_epochs"] == 3
    assert s["best_acc"] == 80.0
    assert s["e95"] == 2
    assert s["e73"] == 2
    assert abs(s["auc"] - (20 + 50 + 80) / 3) < 1e-6


def test_summarize_arm_ci_and_welch():
    arm = summarize_arm([80.0, 82.0, 84.0])
    assert arm["n"] == 3
    assert abs(arm["mean"] - 82.0) < 1e-9
    assert abs(arm["ci95"] - 4.303 * 2.0 / (3 ** 0.5)) < 1e-3
    t, p = welch_ttest([80.0, 82.0, 84.0], [60.0, 62.0, 64.0])
    assert t > 0 and p < 0.01


def test_build_report_gates(tmp_path):
    curves = {
        "baseline": [78.0, 80.0, 80.0],
        "none": [20.0, 50.0, 60.0],
        "ideal": [75.0, 76.0, 76.0],
    }
    logs = {}
    for arm, curve in curves.items():
        for s in range(3):
            lines = [
                f"Epoch {e:3d} | train_loss=2.0 train_acc=10.00% "
                f"test_acc={a:.2f}% lr=1e-3 time=1s"
                for e, a in enumerate(curve)
            ]
            logs[f"logs_heatab_d4_{arm}_s{s}.txt"] = "\n".join(lines) + "\n"
    report = build_report(logs)
    assert set(report["arms"]) == {"baseline", "none", "ideal"}
    assert report["gates"]["C1_heat_vs_none"] is True
    assert report["gates"]["C1_ideal_same_direction"] is True
