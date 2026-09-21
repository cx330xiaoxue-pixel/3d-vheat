3D-vHeat Gaussian Field:  grid_size=(32, 32, 32)，sigma=0.05  ★** New best: 85.91% at epoch 32**

```plain
def create_model(voxel_mode='gaussian', sigma=0.05):
    return VHeat3DMigrated(
        num_classes=40, depths=[2, 2, 6, 2], dims=[64, 128, 256, 512],
        post_norm=False, layer_scale=None, use_bottleneck=False,
        enable_dynamic_alpha=True, enable_high_freq_boost=False,
        drop_path_rate=0.2, mlp_ratio=4.0, dropout=0.1,
        grid_size=(32, 32, 32), use_checkpoint=False, enable_dct_augment=False,
        voxel_mode=voxel_mode, voxel_sigma=sigma,
    )
```

root@autodl-container-tjudth3qst-d627f841:~/autodl-tmp/3dvheat-full# python3 train_fields.py --voxel_mode gaussian --sigma 0.05 --epochs 200 --batch_size 180


[image omitted]


[image omitted]

```plain
root@autodl-container-tjudth3qst-d627f841:~/autodl-tmp/3dvheat-full# python3 train_fields.py --voxel_mode gaussian --sigma 0.05 --epochs 200 --batch_size 180
✓ CUDA Tiles 加速算子已加载
Params: 17,153,792
Epoch 0: 100%|████████████████████████████████████████████████████████| 43/43 [01:12<00:00,  1.69s/it]
Epoch   0 | train_loss=3.3317 train_acc=39.79% test_acc=57.63% lr=1.00e-03 time=80s
  ★ New best: 57.63% at epoch 0
Epoch 1: 100%|████████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch   1 | train_loss=2.3345 train_acc=61.29% test_acc=62.26% lr=1.00e-03 time=159s
  ★ New best: 62.26% at epoch 1
Epoch 2: 100%|████████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch   2 | train_loss=2.0844 train_acc=70.27% test_acc=73.98% lr=9.99e-04 time=239s
  ★ New best: 73.98% at epoch 2
Epoch 3: 100%|████████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch   3 | train_loss=1.9099 train_acc=76.02% test_acc=79.73% lr=9.99e-04 time=318s
  ★ New best: 79.73% at epoch 3
Epoch 4: 100%|████████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch   4 | train_loss=1.8053 train_acc=81.01% test_acc=76.29% lr=9.98e-04 time=397s
Epoch 5: 100%|████████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch   5 | train_loss=1.7055 train_acc=84.15% test_acc=80.91% lr=9.98e-04 time=476s
  ★ New best: 80.91% at epoch 5
Epoch 6: 100%|████████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch   6 | train_loss=1.6277 train_acc=88.29% test_acc=80.27% lr=9.97e-04 time=556s
Epoch 7: 100%|████████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch   7 | train_loss=1.5791 train_acc=90.08% test_acc=81.13% lr=9.96e-04 time=635s
  ★ New best: 81.13% at epoch 7
Epoch 8: 100%|████████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch   8 | train_loss=1.5451 train_acc=91.87% test_acc=80.16% lr=9.95e-04 time=714s
Epoch 9: 100%|████████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch   9 | train_loss=1.5053 train_acc=93.71% test_acc=82.96% lr=9.94e-04 time=793s
  ★ New best: 82.96% at epoch 9
Epoch 10: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  10 | train_loss=1.4842 train_acc=93.74% test_acc=82.74% lr=9.93e-04 time=873s
Epoch 11: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  11 | train_loss=1.4412 train_acc=95.59% test_acc=81.67% lr=9.91e-04 time=952s
Epoch 12: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  12 | train_loss=1.4258 train_acc=95.80% test_acc=84.19% lr=9.90e-04 time=1031s
  ★ New best: 84.19% at epoch 12
Epoch 13: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  13 | train_loss=1.3966 train_acc=97.05% test_acc=83.66% lr=9.88e-04 time=1110s
Epoch 14: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  14 | train_loss=1.3749 train_acc=97.65% test_acc=83.23% lr=9.86e-04 time=1190s
Epoch 15: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  15 | train_loss=1.3702 train_acc=98.09% test_acc=83.71% lr=9.84e-04 time=1269s
Epoch 16: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  16 | train_loss=1.3541 train_acc=98.29% test_acc=83.44% lr=9.82e-04 time=1348s
Epoch 17: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  17 | train_loss=1.3474 train_acc=98.45% test_acc=83.55% lr=9.80e-04 time=1427s
Epoch 18: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  18 | train_loss=1.3365 train_acc=98.58% test_acc=83.60% lr=9.78e-04 time=1506s
Epoch 19: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  19 | train_loss=1.3210 train_acc=98.92% test_acc=84.19% lr=9.76e-04 time=1586s
Epoch 20: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  20 | train_loss=1.3256 train_acc=98.85% test_acc=85.54% lr=9.73e-04 time=1665s
  ★ New best: 85.54% at epoch 20
Epoch 21: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  21 | train_loss=1.3079 train_acc=99.09% test_acc=84.46% lr=9.70e-04 time=1744s
Epoch 22: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  22 | train_loss=1.3013 train_acc=99.43% test_acc=82.96% lr=9.68e-04 time=1823s
Epoch 23: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  23 | train_loss=1.2989 train_acc=99.28% test_acc=84.30% lr=9.65e-04 time=1903s
Epoch 24: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  24 | train_loss=1.2939 train_acc=99.51% test_acc=84.19% lr=9.62e-04 time=1982s
Epoch 25: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  25 | train_loss=1.2881 train_acc=99.53% test_acc=85.86% lr=9.59e-04 time=2061s
  ★ New best: 85.86% at epoch 25
Epoch 26: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  26 | train_loss=1.2818 train_acc=99.59% test_acc=85.05% lr=9.56e-04 time=2140s
Epoch 27: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  27 | train_loss=1.2813 train_acc=99.51% test_acc=84.19% lr=9.52e-04 time=2219s
Epoch 28: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  28 | train_loss=1.2784 train_acc=99.51% test_acc=83.28% lr=9.49e-04 time=2299s
Epoch 29: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  29 | train_loss=1.2774 train_acc=99.59% test_acc=84.09% lr=9.46e-04 time=2378s
Epoch 30: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  30 | train_loss=1.2766 train_acc=99.63% test_acc=84.19% lr=9.42e-04 time=2457s
Epoch 31: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  31 | train_loss=1.2693 train_acc=99.70% test_acc=85.00% lr=9.38e-04 time=2536s
Epoch 32: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  32 | train_loss=1.2732 train_acc=99.68% test_acc=85.91% lr=9.34e-04 time=2616s
  ★ New best: 85.91% at epoch 32
Epoch 33: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  33 | train_loss=1.2679 train_acc=99.75% test_acc=84.46% lr=9.30e-04 time=2695s
Epoch 34: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  34 | train_loss=1.2667 train_acc=99.68% test_acc=85.38% lr=9.26e-04 time=2774s
Epoch 35: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  35 | train_loss=1.2617 train_acc=99.80% test_acc=85.97% lr=9.22e-04 time=2853s
  ★ New best: 85.97% at epoch 35
Epoch 36: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  36 | train_loss=1.2616 train_acc=99.83% test_acc=84.41% lr=9.18e-04 time=2933s
Epoch 37: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  37 | train_loss=1.2640 train_acc=99.76% test_acc=84.09% lr=9.14e-04 time=3012s
Epoch 38: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  38 | train_loss=1.2649 train_acc=99.74% test_acc=84.62% lr=9.09e-04 time=3091s
Epoch 39: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  39 | train_loss=1.2593 train_acc=99.86% test_acc=85.05% lr=9.05e-04 time=3170s
Epoch 40: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  40 | train_loss=1.2564 train_acc=99.80% test_acc=85.48% lr=9.00e-04 time=3250s
Epoch 41: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  41 | train_loss=1.2602 train_acc=99.80% test_acc=85.59% lr=8.95e-04 time=3329s
Epoch 42: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  42 | train_loss=1.2590 train_acc=99.74% test_acc=85.59% lr=8.90e-04 time=3408s
Epoch 43: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  43 | train_loss=1.2559 train_acc=99.84% test_acc=84.68% lr=8.85e-04 time=3487s
Epoch 44: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  44 | train_loss=1.2544 train_acc=99.86% test_acc=84.73% lr=8.80e-04 time=3566s
Epoch 45: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  45 | train_loss=1.2522 train_acc=99.83% test_acc=84.78% lr=8.75e-04 time=3646s
Epoch 46: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  46 | train_loss=1.2528 train_acc=99.83% test_acc=84.46% lr=8.70e-04 time=3725s
Epoch 47: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  47 | train_loss=1.2585 train_acc=99.83% test_acc=84.62% lr=8.64e-04 time=3804s
Epoch 48: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  48 | train_loss=1.2493 train_acc=99.92% test_acc=84.84% lr=8.59e-04 time=3883s
Epoch 49: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  49 | train_loss=1.2545 train_acc=99.86% test_acc=85.16% lr=8.54e-04 time=3963s
Epoch 50: 100%|███████████████████████████████████████████████████████| 43/43 [01:11<00:00,  1.67s/it]
Epoch  50 | train_loss=1.2472 train_acc=99.87% test_acc=85.00% lr=8.48e-04 time=4042s
```

