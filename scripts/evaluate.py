"""
evaluate.py — Quantitative comparison between baseline and depth-supervised NeRF.

Evaluates both checkpoints on the held-out val frames and reports:
  - Rendering quality: PSNR, SSIM, LPIPS
  - Depth accuracy: RMSE, Abs Rel, δ < 1.25 (vs LiDAR ground truth)

Outputs both a per-frame breakdown and aggregate means, written to a JSON file
that's easy to paste into the README results table.
"""
import os
import sys
import argparse
import yaml
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.kitti_raw_loader import KittiRawSequence
from src.lidar_projection import project_lidar_to_image
from src.nerf_model import NeRFMLP
from src.metrics import psnr, ssim, lpips_score, depth_rmse, abs_rel, delta_threshold

# Reuse render_image from render.py
sys.path.insert(0, os.path.dirname(__file__))
from render import render_image


def evaluate_checkpoint(checkpoint_path: str, device: str = 'cuda:0') -> dict:
    """Run full evaluation on val frames for a single checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['config']

    # Load sequence + identify val frames
    seq = KittiRawSequence(cfg['data']['sequence_dir'], cfg['data']['calib_dir'])
    val_stride = cfg['data'].get('val_stride', 8)
    val_idx = [i for i in range(len(seq)) if i % val_stride == 0]

    # Build models
    nerf_coarse = NeRFMLP(**cfg['model']).to(device)
    nerf_fine = NeRFMLP(**cfg['model']).to(device)
    nerf_coarse.load_state_dict(ckpt['coarse'])
    nerf_fine.load_state_dict(ckpt['fine'])
    nerf_coarse.eval()
    nerf_fine.eval()

    K = torch.from_numpy(seq.calib['K']).float().to(device)

    per_frame_metrics = []

    for vi in tqdm(val_idx, desc='Evaluating'):
        sample = seq[vi]
        H, W = sample['image'].shape[:2]
        pose = torch.from_numpy(sample['pose']).float().to(device)

        # Render
        pred_rgb, pred_depth = render_image(
            nerf_coarse, nerf_fine, pose, K, H, W, cfg, device=device,
        )

        # Ground truth RGB (normalize to [0, 1])
        true_rgb = torch.from_numpy(sample['image']).float().to(device) / 255.0

        # Ground truth depth (project LiDAR)
        true_depth_np, _, _ = project_lidar_to_image(
            lidar_points=sample['lidar'],
            P_rect_02=sample['P_rect_02'],
            R_rect=sample['R_rect'],
            T_velo_cam=sample['T_velo_cam'],
            image_height=H,
            image_width=W,
        )
        true_depth = torch.from_numpy(true_depth_np).float().to(device)

        # Compute metrics
        valid_mask = (true_depth > cfg['rendering']['near']) & \
                     (true_depth < cfg['rendering']['far'])

        m = {
            'frame_idx': vi,
            'frame_id': sample['frame_id'],
            'psnr': psnr(pred_rgb, true_rgb),
            'ssim': ssim(pred_rgb, true_rgb),
        }

        # Depth metrics (only where LiDAR is valid)
        if valid_mask.sum() > 100:
            m['depth_rmse'] = depth_rmse(pred_depth, true_depth, valid_mask)
            m['depth_abs_rel'] = abs_rel(pred_depth, true_depth, valid_mask)
            m['depth_delta_1_25'] = delta_threshold(
                pred_depth, true_depth, 1.25, valid_mask,
            )

        per_frame_metrics.append(m)

    # Aggregate
    means = {}
    keys = ['psnr', 'ssim', 'depth_rmse', 'depth_abs_rel', 'depth_delta_1_25']
    for k in keys:
        vals = [m[k] for m in per_frame_metrics if k in m]
        if vals:
            means[k] = float(np.mean(vals))

    return {
        'checkpoint': checkpoint_path,
        'num_val_frames': len(val_idx),
        'per_frame': per_frame_metrics,
        'means': means,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, help='Baseline checkpoint path')
    parser.add_argument('--supervised', required=True, help='Depth-supervised checkpoint path')
    parser.add_argument('--output', default='outputs/comparison.json')
    args = parser.parse_args()

    print('=== Evaluating BASELINE checkpoint ===')
    baseline_results = evaluate_checkpoint(args.baseline)

    print('\n=== Evaluating DEPTH-SUPERVISED checkpoint ===')
    supervised_results = evaluate_checkpoint(args.supervised)

    # Build comparison report
    report = {
        'baseline': baseline_results,
        'depth_supervised': supervised_results,
    }

    # Side-by-side delta
    print('\n=== Summary ===\n')
    print(f"{'Metric':<20s} {'Baseline':>12s} {'Supervised':>12s} {'Δ':>10s}")
    print('-' * 60)
    for k in ['psnr', 'ssim', 'depth_rmse', 'depth_abs_rel', 'depth_delta_1_25']:
        b = baseline_results['means'].get(k)
        s = supervised_results['means'].get(k)
        if b is not None and s is not None:
            delta = s - b
            # For RMSE and Abs Rel, lower is better; for others, higher is better
            arrow = '↑' if (k in ['psnr', 'ssim', 'depth_delta_1_25'] and delta > 0) \
                else '↓' if (k in ['depth_rmse', 'depth_abs_rel'] and delta < 0) \
                else ' '
            print(f"{k:<20s} {b:>12.4f} {s:>12.4f} {delta:>+10.4f} {arrow}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nFull report saved to {args.output}')


if __name__ == '__main__':
    main()
