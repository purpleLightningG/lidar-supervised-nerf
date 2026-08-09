"""
render.py — Render images from a trained NeRF checkpoint.

Loads a checkpoint and renders either:
  - A single specific frame (--frame-idx)
  - All val frames (--mode val)
  - A smooth interpolated camera path through the scene (--mode interp)

Saves RGB + depth maps for each rendered view.
"""
import os
import sys
import argparse
import yaml
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.kitti_raw_loader import KittiRawSequence
from src.nerf_model import NeRFMLP
from src.nerf_renderer import render_rays


@torch.no_grad()
def render_image(nerf_coarse, nerf_fine, pose, K, H, W, cfg, chunk_size=4096, device='cuda'):
    """Render a full RGB + depth image one ray-chunk at a time to avoid OOM."""
    # Generate all rays for this pose
    i, j = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        indexing='xy',
    )
    dirs_cam = torch.stack([
        (i - K[0, 2]) / K[0, 0],
        (j - K[1, 2]) / K[1, 1],
        torch.ones_like(i),
    ], dim=-1)

    R = pose[:3, :3]
    t = pose[:3, 3]
    ray_dirs = (dirs_cam @ R.T).reshape(-1, 3)
    ray_origins = t.expand(H * W, 3)

    # Render in chunks to manage GPU memory
    rgb_list, depth_list = [], []
    for chunk_start in range(0, ray_origins.shape[0], chunk_size):
        chunk_o = ray_origins[chunk_start:chunk_start + chunk_size]
        chunk_d = ray_dirs[chunk_start:chunk_start + chunk_size]

        out = render_rays(
            nerf_coarse, nerf_fine,
            chunk_o, chunk_d,
            near=cfg['rendering']['near'],
            far=cfg['rendering']['far'],
            num_coarse=cfg['rendering']['num_coarse'],
            num_fine=cfg['rendering']['num_fine'],
            perturb=False,
        )

        rgb_list.append(out['rgb_fine'])
        depth_list.append(out['depth_fine'])

    rgb = torch.cat(rgb_list, dim=0).reshape(H, W, 3)
    depth = torch.cat(depth_list, dim=0).reshape(H, W)
    return rgb, depth


def interpolate_poses(poses: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Build a smooth interpolated camera path between consecutive train poses.

    Uses spherical linear interpolation (slerp) for rotation and linear for translation.
    """
    from scipy.spatial.transform import Rotation, Slerp

    n_keys = poses.shape[0]
    times = np.linspace(0, n_keys - 1, n_keys)
    sample_times = np.linspace(0, n_keys - 1, num_frames)

    # Slerp rotations
    rots = Rotation.from_matrix(poses[:, :3, :3].cpu().numpy())
    slerp = Slerp(times, rots)
    interp_rots = slerp(sample_times).as_matrix()

    # Linear interp translations
    interp_trans = np.zeros((num_frames, 3))
    for i, t in enumerate(sample_times):
        i_low = int(np.floor(t))
        i_high = min(i_low + 1, n_keys - 1)
        alpha = t - i_low
        interp_trans[i] = (1 - alpha) * poses[i_low, :3, 3].cpu().numpy() + \
                          alpha * poses[i_high, :3, 3].cpu().numpy()

    # Build (num_frames, 4, 4) homogeneous transforms
    interp = np.zeros((num_frames, 4, 4))
    interp[:, :3, :3] = interp_rots
    interp[:, :3, 3] = interp_trans
    interp[:, 3, 3] = 1.0
    return torch.from_numpy(interp).float().to(poses.device)


def colorize_depth(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Colorize a depth map for visualization (H, W) → (H, W, 3) uint8."""
    import matplotlib.cm as cm
    valid = (depth > 0) & np.isfinite(depth)
    normalized = np.clip((depth - near) / (far - near), 0, 1)
    cmap = cm.get_cmap('turbo')
    colored = cmap(normalized)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    # Mask invalid regions to gray
    colored[~valid] = 128
    return colored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    parser.add_argument('--mode', choices=['val', 'interp', 'single'], default='val',
                        help='val: all val frames; interp: smooth path; single: one frame')
    parser.add_argument('--frame-idx', type=int, default=0,
                        help='For --mode single, which train frame index')
    parser.add_argument('--num-frames', type=int, default=60,
                        help='For --mode interp, how many frames to render')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--save-depth', action='store_true', help='Also save depth visualizations')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint and reconstruct config
    print(f'Loading checkpoint from {args.checkpoint}...')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']

    # Load sequence (for poses and intrinsics)
    seq = KittiRawSequence(cfg['data']['sequence_dir'], cfg['data']['calib_dir'])
    val_stride = cfg['data'].get('val_stride', 8)
    train_idx = [i for i in range(len(seq)) if i % val_stride != 0]
    val_idx = [i for i in range(len(seq)) if i % val_stride == 0]

    train_poses = torch.stack(
        [torch.from_numpy(seq[i]['pose']).float() for i in train_idx]
    ).to(device)
    K = torch.from_numpy(seq.calib['K']).float().to(device)
    sample_img = seq[0]['image']
    H, W = sample_img.shape[:2]

    # Build NeRF models and load weights
    nerf_coarse = NeRFMLP(**cfg['model']).to(device)
    nerf_fine = NeRFMLP(**cfg['model']).to(device)
    nerf_coarse.load_state_dict(ckpt['coarse'])
    nerf_fine.load_state_dict(ckpt['fine'])
    nerf_coarse.eval()
    nerf_fine.eval()

    os.makedirs(args.output, exist_ok=True)
    if args.save_depth:
        os.makedirs(os.path.join(args.output, 'depth'), exist_ok=True)

    # Pick which poses to render based on mode
    if args.mode == 'single':
        poses_to_render = train_poses[args.frame_idx:args.frame_idx + 1]
    elif args.mode == 'val':
        poses_to_render = torch.stack(
            [torch.from_numpy(seq[i]['pose']).float() for i in val_idx]
        ).to(device)
    elif args.mode == 'interp':
        poses_to_render = interpolate_poses(train_poses, args.num_frames)

    # Render each pose
    print(f'Rendering {poses_to_render.shape[0]} frames at {W}x{H}...')
    for i, pose in enumerate(tqdm(poses_to_render, desc='Rendering')):
        rgb, depth = render_image(
            nerf_coarse, nerf_fine, pose, K, H, W, cfg,
            chunk_size=4096, device=device,
        )

        # Save RGB
        rgb_np = (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb_np).save(os.path.join(args.output, f'{i:04d}_rgb.png'))

        # Save depth visualization
        if args.save_depth:
            depth_np = depth.cpu().numpy()
            depth_vis = colorize_depth(
                depth_np,
                near=cfg['rendering']['near'],
                far=cfg['rendering']['far'],
            )
            Image.fromarray(depth_vis).save(
                os.path.join(args.output, 'depth', f'{i:04d}_depth.png')
            )
            # Also save raw depth as npy for evaluation
            np.save(os.path.join(args.output, 'depth', f'{i:04d}_depth.npy'),
                    depth_np.astype(np.float16))

    print(f'Done. Output: {args.output}')


if __name__ == '__main__':
    main()
