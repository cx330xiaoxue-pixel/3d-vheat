"""ModelNet40-C index handling and error-rate metrics.

BASELINE_ER: corruption-error-rate (percent) from Sun et al., ICLR 2022,
Table 2 (standard training), used for relative comparisons only.
"""
from typing import Dict

CORRUPTIONS = {
    "occlusion": "density", "lidar": "density", "density_inc": "density",
    "density_dec": "density", "cutout": "density",
    "uniform": "noise", "gaussian": "noise", "impulse": "noise",
    "upsampling": "noise", "background": "noise",
    "rotation": "transformation", "shear": "transformation",
    "ffd": "transformation", "rbf_inv": "transformation", "rbf": "transformation",
}

FREQUENCY_CLASS = {
    "occlusion": "info_loss", "lidar": "info_loss", "density_dec": "info_loss",
    "cutout": "info_loss", "upsampling": "info_loss",
    "density_inc": "info_loss",
    "uniform": "broadband", "gaussian": "broadband", "impulse": "broadband",
    "background": "broadband",
    "rotation": "transformation", "shear": "transformation",
    "ffd": "transformation", "rbf_inv": "transformation", "rbf": "transformation",
}

BASELINE_ER = {
    "pointnet": 28.3, "pointnet++": 23.6, "dgcnn": 25.9, "rscnn": 26.2,
    "pct": 25.5, "simpleview": 27.2, "curvenet": 22.7,
    "gdanet": 25.6, "pointmlp": 31.9, "pointmlp_elite": 32.4,
}

# Ren et al. (ICML 2022) ModelNet-C atomic corruptions (mCE protocol).
REN_CORRUPTIONS = ["scale", "rotate", "jitter", "drop_g", "drop_l", "add_g", "add_l"]


def taxonomy_of(corruption: str) -> str:
    return CORRUPTIONS[corruption]


def frequency_class_of(corruption: str) -> str:
    return FREQUENCY_CLASS[corruption]


def compute_er_report(index: Dict[str, dict], accs: Dict[str, float]) -> Dict:
    per_corruption: Dict[str, Dict] = {}
    for fname, meta in index.items():
        corr = meta["corruption"]
        per_corruption.setdefault(corr, []).append(accs[fname])
    out_per: Dict[str, Dict] = {}
    for corr, vals in per_corruption.items():
        mean_acc = sum(vals) / len(vals)
        out_per[corr] = {"mean_acc": mean_acc, "er": 1.0 - mean_acc,
                         "n_files": len(vals)}
    by_tax: Dict[str, float] = {}
    by_freq: Dict[str, float] = {}
    for corr, rec in out_per.items():
        by_tax.setdefault(taxonomy_of(corr), []).append(rec["er"])
        by_freq.setdefault(frequency_class_of(corr), []).append(rec["er"])
    er_cor = sum(r["er"] for r in out_per.values()) / len(out_per)
    return {
        "per_corruption": out_per,
        "by_taxonomy": {k: sum(v) / len(v) for k, v in by_tax.items()},
        "by_frequency": {k: sum(v) / len(v) for k, v in by_freq.items()},
        "er_cor": er_cor,
    }


def compute_mce(per_corruption_acc: Dict[str, Dict], baseline_acc: Dict[str, Dict],
                corruptions=None) -> float:
    """Ren et al. ModelNet-C mCE (DGCNN-normalized).

    per_corruption_acc / baseline_acc: {corruption: {severity: accuracy}}.
    """
    corrs = corruptions if corruptions is not None else sorted(per_corruption_acc)
    ces = []
    for corr in corrs:
        num = sum(1.0 - acc for acc in per_corruption_acc[corr].values())
        den = sum(1.0 - baseline_acc[corr][sev] for sev in per_corruption_acc[corr])
        ces.append(num / den)
    return sum(ces) / len(ces)
