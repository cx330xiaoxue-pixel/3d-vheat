# 3D-vHeat

**A heat-conduction (DCT) backbone for 3D point clouds — a frequency-domain design space for robust recognition at low compute.**

3D-vHeat is a 3D extension of **vHeat: Building Vision Models upon Heat Conduction** (CVPR 2025, [code](https://github.com/MzeroMiko/vHeat)). Self-attention is replaced by a **Heat Conduction Operator (HCO)** implemented with 3D DCT/IDCT over sparse voxels, and the resulting frequency-domain design space is studied for corruption robustness.

> **Status — research code, paper in progress.**
> All experiments in this repository use a **33-class subset of ModelNet40** (7,589 train / 1,860 test, 1,024 points per shape), unless stated otherwise. Full ModelNet40 experiments (9,843 / 2,468 / 40 classes) are in progress.
> Accuracy depends strongly on the evaluation batch size (per-batch voxelization bounds) — see [Protocol note](#protocol-note).

---

## Highlights

| Component | What it is | Cost |
|---|---|---|
| **3D HCO** | Heat-conduction operator in the 3D DCT domain (O(N^1.5) vs O(N^2) attention) | — |
| **Gaussian field voxelization** | Points → smooth voxel field; σ acts as a free robustness knob (dose–response measured) | free |
| **DCTAugment** | Frequency-domain training augmentation (low-pass / dropout / jitter / band masking) | zero inference cost |
| **IAK** | Input-adaptive thermal diffusivity predicted from frequency statistics | +5.6% params |
| **Spectral TTA / gating** | Test-time spectral filtering and confidence gating | zero retraining |

## Results (33-class ModelNet40 subset)

Clean accuracy, 32³ grid, `small` backbone (17.15M params):

| Variant | Batch | Test acc | Log |
|---|---|---|---|
| Learned heat, σ=0.05, 100 ep | 8 | **83.33%** | `docs/logs/logs_fields32_heat_100.txt` |
| No filter, σ=0.05, 100 ep | 8 | 82.20% | `docs/logs/logs_fields32_none_100.txt` |
| IAK, σ=0.05, 100 ep | 8 | 81.88% | `docs/logs/logs_fields32_iak_100.txt` |
| Fixed heat, σ=0.05, 100 ep | 8 | 81.88% | `docs/logs/logs_fields32_fixed_heat_100.txt` |
| Surface-uniform field, σ=0.2, 200 ep | 8–16 | 82.47% | `docs/logs/logs_fields_gaussian_surface.txt` |
| Surface-uniform field, σ=0.2, 200 ep (AutoDL) | 200 | **91.40–92.63%** | `docs/autodl_logs/` |

Corruption robustness (subset, 16³ grid, n=3 seeds): DCTAugment with σ=0.2 improves corrupted-input accuracy by **+14–20 pp** over no augmentation, at zero inference cost (`experiments/results/e5_dctaug_s02_s*.json`).

## Protocol note

Voxelization bounds are computed **per batch**, so the input representation — and the reported accuracy — depends on the batch size. The same checkpoint scores 75.9% at batch 16 and 88.0% at batch 128 (`experiments/results/e4_bounds_ab.json`); the same subset configuration scores 82.47% at small batch and 91.40–92.63% at batch 180–220. Training and evaluation batch sizes must match, and must be reported.

---

## Installation

```bash
git clone https://github.com/<your-account>/3dvheat.git
cd 3dvheat
pip install -r requirements.txt

# Optional: build the CUDA acceleration kernels (a pure-PyTorch fallback is built in)
python setup_cuda.py build_ext --inplace
```

Requirements: Python 3.8+, PyTorch 2.0+, CUDA 11.x+ (optional), numpy, scipy, h5py, tqdm, matplotlib.

## Data preparation

**Full ModelNet40 (40 classes, official 2048-point HDF5):**

```bash
wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
python data/prepare_modelnet40_full.py --source hdf5 \
  --hdf5-zip modelnet40_ply_hdf5_2048.zip --out-dir data/full
# Expected: 9,843 train / 2,468 test / 40 classes
```

**33-class subset from raw OFF files (as used in the logs here):**

```bash
python data/convert_modelnet40.py --input_dir /path/to/ModelNet40 \
  --output_dir data --num_points 1024 --save_npy
# Produces data/modelnet40_{train,test}_{points,labels}.npy
```

## Training

```bash
# 32³ Gaussian field, 4-stage `small` backbone
python3 train_fields_scaled.py --voxel_mode gaussian --sigma 0.05 \
  --grid_size 32 --backbone small --epochs 100 --batch_size 8 --seed 0 \
  --tag my_run

# With frequency-domain augmentation
python3 train_fields_scaled.py --voxel_mode gaussian --sigma 0.2 \
  --grid_size 32 --backbone small --epochs 200 --batch_size 8 \
  --dct-augment --tag my_run_dctaug
```

Checkpoints and logs are written to `checkpoints_<tag>/` and `logs_<tag>.txt`.

## Evaluation

```bash
# Corruption-axis evaluation (noise / dropout / FPS / rotation / quantization ...)
python -m experiments.eval_robustness \
  --ckpt checkpoints_my_run/best_model.pth \
  --sigma 0.05 --grid 32 --backbone small \
  --axes gaussian_noise dropout fps rotation_yaw coarse_quantize \
  --seeds 0 1 2 --batch-size 8 --out experiments/results/my_eval.json

# ModelNet40-C (ER / mCE vs published baselines)
python -m experiments.inspect_mn40c --root /path/to/ModelNet40-C \
  --out experiments/data/mn40c_index.json
python -m experiments.eval_modelnet40c --ckpt checkpoints_my_run/best_model.pth \
  --index experiments/data/mn40c_index.json --out experiments/results/my_mn40c.json

# FLOPs / params / throughput
python -m experiments.efficiency.flops_matrix --grids 16 32 64 \
  --backbones tiny small --out experiments/results/my_flops.json

# Tests
python -m pytest tests/experiments -q
```

## Repository layout

```
vheat3d/                  core package
  modules/                VHeat3DMigrated backbone, heat-conduction block,
                          Gaussian/binary/SDF/trilinear voxelization
  operators/              3D DCT/IDCT, DCT augmentation, tiled conv3d
  cuda_kernels/           optional CUDA kernels
train_fields.py           single-scale trainer (32³)
train_fields_scaled.py    multi-scale trainer (16/32/64 grids, tiny→xlarge)
inference.py              inference script
data/                     dataset loaders and ModelNet40 preparation scripts
experiments/
  robustness/             corruption axes, ModelNet40-C index/metrics (ER, mCE)
  efficiency/             FLOPs / params / throughput measurement
  spectral/               spectral mechanism probes
  report/                 result parsing and figure utilities
  results/                experiment outputs (JSON)
tests/                    test suite
docs/logs/                local training logs
docs/autodl_logs/         AutoDL (large-batch) training logs
```

## Direction

The backbone is a drop-in 3D vision encoder for efficient and robust recognition: classification and part-segmentation heads plug in directly, and the design targets compute-constrained settings (robotics, embodied AI, multimodal 3D perception). As a modular encoder it is compatible with downstream vision-language pipelines (e.g., as a 3D vision tower for 3D-VLM), which is **not implemented in this repository** and remains future work.

## Attribution

The Heat Conduction Operator and the vHeat architecture are from:

> Zhaozhi Wang, Yue Liu, Yunjie Tian, Yunfan Liu, Yaowei Wang, Qixiang Ye. *Building Vision Models upon Heat Conduction.* CVPR 2025.

This repository adapts those ideas to 3D sparse voxels and adds Gaussian field voxelization, DCTAugment, input-adaptive diffusivity, and the robustness/efficiency evaluation suite.

## License

MIT (see `LICENSE`). Note: the upstream vHeat repository does not carry an explicit license; if you plan to use the original 2D implementation, please contact the vHeat authors.
