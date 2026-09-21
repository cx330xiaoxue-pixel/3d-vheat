import json

import numpy as np

from experiments.robustness.mn40c import (
    BASELINE_ER, FREQUENCY_CLASS, compute_er_report, taxonomy_of,
)


def test_taxonomy_known_corruptions():
    assert taxonomy_of("occlusion") == "density"
    assert taxonomy_of("gaussian") == "noise"
    assert taxonomy_of("rotation") == "transformation"


def test_frequency_class_values():
    for v in FREQUENCY_CLASS.values():
        assert v in {"info_loss", "broadband", "transformation"}


def test_compute_er_report_math():
    index = {
        "a.npy": {"corruption": "gaussian", "severity": 1},
        "b.npy": {"corruption": "gaussian", "severity": 2},
        "c.npy": {"corruption": "rotation", "severity": 1},
    }
    accs = {"a.npy": 0.8, "b.npy": 0.6, "c.npy": 0.9}
    report = compute_er_report(index, accs)
    assert abs(report["per_corruption"]["gaussian"]["mean_acc"] - 0.7) < 1e-9
    assert abs(report["per_corruption"]["gaussian"]["er"] - 0.3) < 1e-9
    assert abs(report["by_taxonomy"]["noise"] - 0.3) < 1e-9
    assert abs(report["by_taxonomy"]["transformation"] - 0.1) < 1e-9
    assert "er_cor" in report


def test_baseline_table_has_key_models():
    for m in ["pointnet++", "dgcnn", "pointmlp", "curvenet", "pct"]:
        assert m in BASELINE_ER


def test_compute_mce_formula():
    from experiments.robustness.mn40c import compute_mce
    ours = {"jitter": {1: 0.9, 2: 0.8}, "drop_g": {1: 0.7, 2: 0.5}}
    base = {"jitter": {1: 0.95, 2: 0.9}, "drop_g": {1: 0.9, 2: 0.8}}
    mce = compute_mce(ours, base)
    ce_jitter = (0.1 + 0.2) / (0.05 + 0.1)
    ce_drop = (0.3 + 0.5) / (0.1 + 0.2)
    assert abs(mce - (ce_jitter + ce_drop) / 2) < 1e-9


def test_compute_mce_missing_severity_raises():
    import pytest
    from experiments.robustness.mn40c import compute_mce
    with pytest.raises(KeyError):
        compute_mce({"jitter": {1: 0.9}}, {"jitter": {2: 0.9}})
