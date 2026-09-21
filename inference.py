import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vheat3d.modules.vheat3d_migrated import VHeat3DMigrated


def create_model(checkpoint=None, voxel_mode='gaussian', sigma=0.05, num_classes=40):
    model = VHeat3DMigrated(
        num_classes=num_classes, depths=[2, 2, 6, 2], dims=[64, 128, 256, 512],
        post_norm=False, layer_scale=None, use_bottleneck=False,
        enable_dynamic_alpha=True, enable_high_freq_boost=False,
        drop_path_rate=0.2, mlp_ratio=4.0, dropout=0.1,
        grid_size=(32, 32, 32), use_checkpoint=False, enable_dct_augment=False,
        voxel_mode=voxel_mode, voxel_sigma=sigma,
    )
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(state)
    return model


def main():
    parser = argparse.ArgumentParser(description='3D-vHeat Inference (Gaussian Field)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--input', type=str, required=True, help='Input point cloud (.npy, shape [N,3] or [B,N,3])')
    parser.add_argument('--voxel_mode', type=str, default='gaussian', choices=['binary', 'gaussian', 'sdf', 'trilinear'])
    parser.add_argument('--sigma', type=float, default=0.05)
    parser.add_argument('--num_classes', type=int, default=40)
    parser.add_argument('--output', type=str, default=None, help='Output file for predictions (.npz)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(args.checkpoint, args.voxel_mode, args.sigma, args.num_classes)
    model = model.to(device).eval()

    points = np.load(args.input).astype(np.float32)
    if points.ndim == 2:
        points = points[None]  # (N,3) -> (1,N,3)
    points_t = torch.from_numpy(points).to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(points_t)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    for i in range(len(points)):
        print(f'Sample {i}: class={preds[i].item()}, prob={probs[i, preds[i]].item():.4f}')

    if args.output:
        np.savez(args.output, predictions=preds.cpu().numpy(), probabilities=probs.cpu().numpy())


if __name__ == '__main__':
    main()
