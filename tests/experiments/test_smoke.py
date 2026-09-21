import importlib


def test_experiment_packages_import():
    for name in [
        "experiments",
        "experiments.robustness",
        "experiments.efficiency",
        "experiments.spectral",
        "experiments.report",
    ]:
        assert importlib.import_module(name) is not None
