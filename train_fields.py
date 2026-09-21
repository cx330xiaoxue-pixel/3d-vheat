import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vheat3d.modules.vheat3d_migrated import VHeat3DMigrated

def create_model(voxel_mode='gaussian', sigma=0.05):
    return VHeat3DMigrated(
        num_classes=40, depths=[2, 2, 6, 2], dims=[64, 128, 256, 512],
        post_norm=False, layer_scale=None, use_bottleneck=False,
        enable_dynamic_alpha=True, enable_high_freq_boost=False,
        drop_path_rate=0.2, mlp_ratio=4.0, dropout=0.1,
        grid_size=(32, 32, 32), use_checkpoint=False, enable_dct_augment=True,
        voxel_mode=voxel_mode, voxel_sigma=sigma,
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxel_mode', type=str, default='gaussian', choices=['binary', 'gaussian', 'sdf', 'trilinear'])
    parser.add_argument('--sigma', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    tag = args.tag or args.voxel_mode
    ckpt_dir = f'./checkpoints_fields_{tag}'
    log_file = f'logs_fields_{tag}.txt'
    os.makedirs(ckpt_dir, exist_ok=True)

    model = create_model(voxel_mode=args.voxel_mode, sigma=args.sigma).cuda()
    nparams = sum(p.numel() for p in model.parameters())
    print(f'Params: {nparams:,}', file=sys.stderr)

    train_ds = MN40('./data/modelnet40_train_points.npy', './data/modelnet40_train_labels.npy')
    test_ds = MN40('./data/modelnet40_test_points.npy', './data/modelnet40_test_labels.npy')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')
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
    summary = (f'\n===== {args.voxel_mode} (sigma={args.sigma}) =====\n'
               f'Best: {best_acc:.2f}% at epoch {best_epoch}\n'
               f'Params: {nparams:,}\n'
               f'Time: {total_time:.0f}s\n')
    print(summary)
    log_lines.append(summary)

    with open(log_file, 'w') as f:
        f.write('\n'.join(log_lines))

if __name__ == '__main__':
    main()
