# Experiment notes

This document records the protocols behind the numbers in `README.md` and what the
files under `experiments/results/` and `docs/` contain.

## Scope and caveats

- **Dataset**: a 33-class subset of ModelNet40 (7,589 train / 1,860 test, 1,024 points).
  It was produced from the raw OFF files with `data/convert_modelnet40.py`. Seven of the
  40 classes were excluded due to malformed OFF headers in the local copy.
  Full ModelNet40 experiments (9,843 / 2,468 / 40 classes) are in progress and are **not**
  reported here.
- **Batch size is part of the protocol.** Voxelization bounds are computed per batch, so
  the same checkpoint evaluates very differently at different batch sizes
  (`e4_bounds_ab.json`: 75.9% @ bs16 vs 88.0% @ bs128). Training and evaluation batch
  sizes must match.
- All local runs used a single consumer GPU (16 GB) and small batches (8–16).
  The AutoDL runs used large batches (180–220) on an 80 GB-class GPU.

## Training logs

### `docs/logs/` — local (small batch)

| File | Config | Batch | Best |
|---|---|---|---|
| `logs_fields32_heat_100.txt` | 32³, σ=0.05, learned heat, 100 ep | 8 | 83.33% @ ep92 |
| `logs_fields32_none_100.txt` | 32³, σ=0.05, no spectral filter, 100 ep | 8 | 82.20% @ ep97 |
| `logs_fields32_iak_100.txt` | 32³, σ=0.05, input-adaptive k, 100 ep | 8 | 81.88% @ ep97 |
| `logs_fields32_fixed_heat_100.txt` | 32³, σ=0.05, fixed decay, 100 ep | 8 | 81.88% @ ep93 |
| `logs_fields_gaussian_surface.txt` | 32³, σ=0.2, surface-uniform field, 200 ep | ≤16 | 82.47% @ ep193 |
| `logs_fields_gaussian.txt` | early vertex-sampling run (superseded) | — | 57.85% |

### `docs/autodl_logs/` — AutoDL (large batch, 33-class subset)

These are the raw training logs of the large-batch runs (an 80 GB-class GPU,
batch 180–220). They correspond to the 91.40–92.63% numbers reported in `README.md`.

| File | Command (batch) | Epochs | Notes |
|---|---|---|---|
| `autodl_exp2_sigma0.05_bs180.md` | σ=0.05, bs 180 (43 batches/epoch) | 200 | early fragment |
| `autodl_exp3_bs24.md` | bs 24 | — | small-batch control |
| `autodl_exp4_sigma0.1_bs180.md` | σ=0.1, bs 180 | 200 | |
| `autodl_exp5_bs200.md` | bs 200 (38 batches/epoch) | 200 | |
| `autodl_exp6_bs200.md` | bs 200 | 200 | |
| `autodl_exp7_bs200.md` | bs 200 | 200 | |
| `autodl_exp8_bs200_200ep.md` | σ=0.2 surface, bs 200 | 200 | best 92.63% @ ep97; 91.40% reached at several epochs |
| `autodl_exp9_bs220_100ep.md` | σ=0.2 surface, bs 220 (35 batches/epoch) | 100 | |

Note: `38 batches/epoch × 200 = 7,589` training samples — i.e., these runs also use the
33-class subset, not the full 40-class ModelNet40.

## `experiments/results/` (JSON)

Selected artifacts:

| File | Contents |
|---|---|
| `e4_bounds_ab.json` | Accuracy vs evaluation batch size (same checkpoint) |
| `e4_final_*.json` | 16³ robustness axes, 4 arms × 3 seeds |
| `e5_dctaug_*.json` | DCTAugment main comparison and leave-one-out ablation |
| `e6_*.json`, `e7_32_*.json`, `e8_32_iak*.json` | Heat-diffusion and IAK pilots / 32³ runs |
| `e9_*.json`, `e10_*.json` | Resolution / cross-backbone controls |
| `heat_tau.json`, `heat_tau_16small.json` | Per-layer learned decay temperature (τ) dumps |
| `d3_*.json`, `d4_*.json` | σ dose–response and 4-arm convergence/variance matrix |
| `efficiency_*.json`, `flops_matrix.json` | Params / FLOPs / throughput |
| `spectral_filter_*.json`, `quantization_spectra.json` | Spectral mechanism probes |
| `mn40c` index/metrics | ModelNet40-C ER/mCE (computed with `experiments/robustness/mn40c.py`) |
