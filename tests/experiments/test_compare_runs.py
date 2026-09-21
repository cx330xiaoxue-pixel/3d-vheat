from experiments.report.compare_runs import compare_payloads, format_markdown


def _payload(clean, axes):
    return {"clean": {"acc": clean, "n": 100}, "axes": axes}


def test_compare_payloads_delta_and_ci():
    ref = _payload(0.75, {"gaussian_noise": {"0.05": {"acc_mean": 0.40, "acc_std": 0.01, "ci95": 0.02, "per_seed": [0.4]}}})
    run = _payload(0.72, {"gaussian_noise": {"0.05": {"acc_mean": 0.55, "acc_std": 0.02, "ci95": 0.03, "per_seed": [0.55]}}})
    out = compare_payloads({"ref": ref, "run": run})
    assert out["clean"]["ref"] == 0.75 and out["clean"]["run"] == 0.72
    rec = out["axes"]["gaussian_noise"]["0.05"]
    assert abs(rec["deltas"]["run"] - 0.15) < 1e-9
    md = format_markdown(out)
    assert "gaussian_noise" in md and "+15.00" in md
