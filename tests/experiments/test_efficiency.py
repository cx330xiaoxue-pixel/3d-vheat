import json

import numpy as np
import pytest
import torch

from experiments.efficiency.measure_efficiency import (
    latency_stats, count_parameters, measure_cpu_smoke,
)


def test_latency_stats_basic():
    times = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = latency_stats(times)
    assert out["median_ms"] == 3.0
    assert out["mean_ms"] == 3.0
    assert out["p95_ms"] >= 4.0


def test_count_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.Linear(5, 2))
    assert count_parameters(model) == 10 * 5 + 5 + 5 * 2 + 2


def test_measure_cpu_smoke():
    model = torch.nn.Linear(8, 4)
    payload = measure_cpu_smoke(model, input_dim=8, iters=3)
    assert payload["params"] == 8 * 4 + 4
    assert payload["latency_ms"]["median_ms"] > 0
