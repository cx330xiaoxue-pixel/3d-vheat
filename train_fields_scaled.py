import os, sys, time, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vheat3d.modules.vheat3d_migrated import VHeat3DMigrated

BACKBONE_CONFIGS = {
    'tiny': {'depths': [2, 2, 2, 2], 'dims': [32, 64, 128, 256]},
    'small': {'depths': [2, 2, 6, 2], 'dims': [64, 128, 256, 512]},
    'medium': {'depths': [2, 2, 6, 2], 'dims': [96, 192, 384, 768]},
    'large': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
    'xlarge': {'depths': [3, 4, 12, 3], 'dims': [128, 256, 512, 1024]},
}


def apply_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(voxel_mode='gaussian', sigma=0.05, grid_size=32,
                 backbone='small', post_norm=False,
                 diffusion='heat', diffusion_steps=1, anisotropic=False,
                 decay_sharpness=1.0, enable_dynamic_alpha=True,
                 voxel_bounds_mode='batch', dct_augment=False, dct_augment_skip=(),
                 input_adaptive_k=False):
    cfg = BACKBONE_CONFIGS[backbone]
    return VHeat3DMigrated(
        num_classes=40,
        depths=cfg['depths'], dims=cfg['dims'],
        post_norm=post_norm, layer_scale=None, use_bottleneck=False,
        enable_dynamic_alpha=enable_dynamic_alpha, enable_high_freq_boost=False,
        drop_path_rate=0.2, mlp_ratio=4.0, dropout=0.1,
        grid_size=(grid_size, grid_size, grid_size),
        use_checkpoint=False, enable_dct_augment=dct_augment,
        dct_augment_skip=tuple(dct_augment_skip),
        voxel_mode=voxel_mode, voxel_sigma=sigma,
        diffusion=diffusion, diffusion_steps=diffusion_steps, anisotropic=anisotropic,
        decay_sharpness=decay_sharpness,
        input_adaptive_k=input_adaptive_k,
        voxel_bounds_mode=voxel_bounds_mode,
    )

class MN40(Dataset):
    def __init__(self, pts, lbls):
        self.pts = np.load(pts).astype(np.float32)
        self.lbls = np.load(lbls).astype(np.int64)
    def __len__(self):
        return len(self.pts)
    def __getitem__(self, idx):
        p = self.pts[idx].copy()
        l = self.lbls[idx]
        centroid = p.mean(0)
        p = p - centroid
        d = np.max(np.sqrt(np.sum(p ** 2, axis=1)))
        if d > 1e-8: p = p / d
        return torch.from_numpy(p).float(), torch.tensor(l).long()

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxel_mode', type=str, default='gaussian', choices=['binary', 'gaussian', 'sdf', 'trilinear'])
    parser.add_argument('--sigma', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1.5e-3)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--grid_size', type=int, default=32, choices=[16, 32, 64])
    parser.add_argument('--backbone', type=str, default='small', choices=list(BACKBONE_CONFIGS.keys()))
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--diffusion', type=str, default='heat',
                        choices=['heat', 'none', 'ideal', 'cosine', 'fixed_heat'])
    parser.add_argument('--diffusion-steps', type=int, default=1)
    parser.add_argument('--anisotropic', action='store_true')
    parser.add_argument('--freeze-decay-temp', action='store_true')
    parser.add_argument('--static-alpha', action='store_true')
    parser.add_argument('--tau-invert', action='store_true')
    parser.add_argument('--diffusion-sharpness', type=float, default=1.0)
    parser.add_argument('--bounds-mode', type=str, default='batch',
                        choices=['batch', 'per_sample'])
    parser.add_argument('--seed', type=int, default=-1,
                        help='RNG seed for random/numpy/torch/cuda; -1 disables seeding')
    parser.add_argument('--dct-augment', action='store_true',
                        help='enable DCT-domain training augmentation')
    parser.add_argument('--dct-augment-skip', nargs='+', default=[],
                        choices=['hf', 'drop', 'jitter', 'band'],
                        help='disable DCTAugment components (leave-one-out ablation)')
    parser.add_argument('--input-adaptive-k', action='store_true',
                        help='predict heat diffusivity k from input statistics (beyond 2D vHeat)')
    return parser


def model_kwargs_from_args(args) -> dict:
    return {
        'voxel_mode': args.voxel_mode,
        'sigma': args.sigma,
        'grid_size': args.grid_size,
        'backbone': args.backbone,
        'post_norm': args.post_norm,
        'diffusion': args.diffusion,
        'diffusion_steps': args.diffusion_steps,
        'anisotropic': args.anisotropic,
        'decay_sharpness': args.diffusion_sharpness,
        'enable_dynamic_alpha': not args.static_alpha,
        'voxel_bounds_mode': args.bounds_mode,
        'dct_augment': args.dct_augment,
        'dct_augment_skip': list(getattr(args, 'dct_augment_skip', [])),
        'input_adaptive_k': bool(getattr(args, 'input_adaptive_k', False)),
    }


def main():
    args = build_arg_parser().parse_args()
    apply_seed(args.seed)

    if args.batch_size is None:
        args.batch_size = {16: 16, 32: 8, 64: 4}[args.grid_size]

    voxel_size = 0.1 * (32 / args.grid_size)

    tag_parts = [args.tag or f'grid{args.grid_size}_{args.backbone}']
    tag = '_'.join(filter(None, tag_parts))
    ckpt_dir = f'./checkpoints_{tag}'
    log_file = f'logs_{tag}.txt'
    os.makedirs(ckpt_dir, exist_ok=True)

    model = create_model(**model_kwargs_from_args(args)).cuda()
    nparams = sum(p.numel() for p in model.parameters())
    print(f'Params: {nparams:,}', file=sys.stderr)

    if args.freeze_decay_temp:
        for m in model.modules():
            if hasattr(m, 'decay_temp'):
                m.decay_temp.requires_grad_(False)
    if args.tau_invert:
        blocks = [m for m in model.modules() if hasattr(m, 'decay_temp')]
        with torch.no_grad():
            for i, m in enumerate(blocks):
                m.decay_temp.fill_(0.5 + 1.5 * i / max(len(blocks) - 1, 1))

    train_ds = MN40('./data/modelnet40_train_points.npy', './data/modelnet40_train_labels.npy')
    test_ds = MN40('./data/modelnet40_test_points.npy', './data/modelnet40_test_labels.npy')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    try:
        scaler = torch.amp.GradScaler('cuda')
    except AttributeError:  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    best_acc = 0.0
    best_epoch = -1
    start = time.time()
    log_lines = []

    for epoch in range(args.epochs):
        model.train()
        train_correct, train_total, train_loss = 0, 0, 0.0
        for pts, lbls in tqdm(train_loader, desc=f'Epoch {epoch}'):
            pts = pts.cuda(non_blocking=True)
            lbls = lbls.cuda(non_blocking=True)
            with autocast('cuda'):
                out = model(pts)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_loss += loss.item()
            _, pred = out.max(1)
            train_total += lbls.size(0)
            train_correct += pred.eq(lbls).sum().item()
        scheduler.step()
        train_acc = 100.0 * train_correct / train_total

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for pts, lbls in test_loader:
                pts = pts.cuda(non_blocking=True)
                lbls = lbls.cuda(non_blocking=True)
                with autocast('cuda'):
                    out = model(pts)
                _, pred = out.max(1)
                test_total += lbls.size(0)
                test_correct += pred.eq(lbls).sum().item()
        test_acc = 100.0 * test_correct / test_total

        elapsed = time.time() - start
        msg = (f'Epoch {epoch:3d} | train_loss={train_loss/len(train_loader):.4f} '
               f'train_acc={train_acc:.2f}% test_acc={test_acc:.2f}% '
               f'lr={scheduler.get_last_lr()[0]:.2e} time={elapsed:.0f}s')
        print(msg)
        log_lines.append(msg)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), f'{ckpt_dir}/best_model.pth')
            print(f'  ★ New best: {best_acc:.2f}% at epoch {epoch}')

    total_time = time.time() - start
    summary = (f'\n===== {tag} =====\n'
               f'Best: {best_acc:.2f}% at epoch {best_epoch}\n'
               f'Params: {nparams:,}\n'
               f'Grid: {args.grid_size}³\n'
               f'Backbone: {args.backbone}\n'
               f'Batch: {args.batch_size}\n'
               f'Time: {total_time:.0f}s\n')
    print(summary)
    log_lines.append(summary)

    with open(log_file, 'w') as f:
        f.write('\n'.join(log_lines))

if __name__ == '__main__':
    main()
