from experiments.report.compare_sigma import build_trend, monotone_direction


def _payload(clean, axes):
    return {"clean": {"acc": clean, "n": 1860}, "axes": axes}


def test_monotone_direction():
    assert monotone_direction([0.5, 0.6, 0.7]) == "increasing"
    assert monotone_direction([0.7, 0.6, 0.5]) == "decreasing"
    assert monotone_direction([0.5, 0.7, 0.6]) == "non_monotone"


def test_build_trend_orders_by_sigma():
    payloads = {
        0.2: _payload(0.80, {"gaussian_noise": {"0.05": {"acc_mean": 0.5, "acc_std": 0.01, "per_seed": [0.5]}}}),
        0.05: _payload(0.70, {"gaussian_noise": {"0.05": {"acc_mean": 0.4, "acc_std": 0.01, "per_seed": [0.4]}}}),
        0.1: _payload(0.75, {"gaussian_noise": {"0.05": {"acc_mean": 0.45, "acc_std": 0.01, "per_seed": [0.45]}}}),
    }
    trend = build_trend(payloads)
    assert trend["sigmas"] == [0.05, 0.1, 0.2]
    assert trend["clean"] == [0.70, 0.75, 0.80]
    rec = trend["axes"]["gaussian_noise"]["0.05"]
    assert rec["accs"] == [0.4, 0.45, 0.5]
    assert rec["direction"] == "increasing"
    assert trend["counts"] == {"increasing": 1, "decreasing": 0, "non_monotone": 0}
