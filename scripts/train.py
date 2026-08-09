"""
train.py — Main NeRF training loop.

For each iteration:
  1. Sample a random batch of pixels (rays) across the training images
  2. Render those rays via volume rendering through coarse + fine MLPs
  3. Compute RGB loss (and optionally LiDAR depth loss)
  4. Backpropagate and step optimizer

Same script trains both baseline and depth-supervised models — configure via
the YAML config file (see `configs/baseline.yaml` vs `configs/depth_supervised.yaml`).
"""
import os
import sys
import argparse
import yaml
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.kitti_raw_loader import KittiRawSequence
from src.nerf_model import NeRFMLP
from src.nerf_renderer import render_rays
from src.losses import NeRFLoss
from src.metrics import psnr


def build_rays(pose: torch.Tensor, K: torch.Tensor, H: int, W: int):
    """Generate ray origins and directions for an entire image.

    Args:
        pose: (4, 4) camera-to-world transform
        K: (3, 3) camera intrinsics

    Returns:
        ray_origins:    (H * W, 3)
        ray_directions: (H * W, 3) unnormalized — caller may normalize as needed
    """
    device = pose.device
    # Pixel grid
    i, j = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        indexing='xy',
    )
    # Pinhole back-projection: (x, y, 1) → camera-frame direction
    dirs_cam = torch.stack([
        (i - K[0, 2]) / K[0, 0],
        (j - K[1, 2]) / K[1, 1],
        torch.ones_like(i),
    ], dim=-1)  # (H, W, 3)

    # Transform directions to world frame
    R = pose[:3, :3]
    t = pose[:3, 3]
    ray_directions = (dirs_cam @ R.T).reshape(-1, 3)
    ray_origins = t.expand(H * W, 3)

    return ray_origins, ray_directions


def sample_random_rays(images, depths, poses, K, num_rays, H, W, device):
    """Sample `num_rays` random pixels across all training images."""
    N_images = images.shape[0]

    # Random image and pixel indices
    img_idx = torch.randint(0, N_images, (num_rays,), device=device)
    pix_y = torch.randint(0, H, (num_rays,), device=device)
    pix_x = torch.randint(0, W, (num_rays,), device=device)

    # Gather target colors and depths
    true_rgb = images[img_idx, pix_y, pix_x]                # (num_rays, 3)
    true_depth = depths[img_idx, pix_y, pix_x] if depths is not None else None

    # Build rays — different camera origin per image
    pose_per_ray = poses[img_idx]  # (num_rays, 4, 4)

    # Camera-frame ray direction for this pixel
    dirs_cam = torch.stack([
        (pix_x.float() - K[0, 2]) / K[0, 0],
        (pix_y.float() - K[1, 2]) / K[1, 1],
        torch.ones_like(pix_x, dtype=torch.float32),
    ], dim=-1)

    # Rotate to world frame
    R = pose_per_ray[:, :3, :3]
    t = pose_per_ray[:, :3, 3]
    ray_directions = torch.bmm(R, dirs_cam.unsqueeze(-1)).squeeze(-1)
    ray_origins = t

    return ray_origins, ray_directions, true_rgb, true_depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    output_dir = cfg['output']['dir']
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)

    # ----- Load sequence -----
    print('Loading KITTI Raw sequence...')
    seq = KittiRawSequence(
        cfg['data']['sequence_dir'],
        cfg['data']['calib_dir'],
    )
    print(f'  {len(seq)} frames available')

    # Train / val split (every Nth frame held out)
    val_stride = cfg['data'].get('val_stride', 8)
    train_indices = [i for i in range(len(seq)) if i % val_stride != 0]
    val_indices = [i for i in range(len(seq)) if i % val_stride == 0]
    print(f'  {len(train_indices)} train frames, {len(val_indices)} val frames')

    # Pre-load training images, depths, poses into memory
    print('Pre-loading training data...')
    images_list, depths_list, poses_list = [], [], []
    for idx in tqdm(train_indices, desc='Loading'):
        sample = seq[idx]
        images_list.append(torch.from_numpy(sample['image']).float() / 255.0)
        poses_list.append(torch.from_numpy(sample['pose']).float())

        # Load pre-computed depth if it exists
        if cfg['loss']['use_depth_supervision']:
            depth_path = os.path.join(
                cfg['data']['depth_cache_dir'],
                f"{sample['frame_id']}.npy",
            )
            depth = np.load(depth_path).astype(np.float32)
            depths_list.append(torch.from_numpy(depth))

    images = torch.stack(images_list).to(device)  # (N, H, W, 3) in [0, 1]
    poses = torch.stack(poses_list).to(device)
    depths = torch.stack(depths_list).to(device) if depths_list else None

    H, W = images.shape[1], images.shape[2]
    K = torch.from_numpy(seq.calib['K']).float().to(device)

    # ----- Build models -----
    print('Building NeRF models...')
    nerf_coarse = NeRFMLP(**cfg['model']).to(device)
    nerf_fine = NeRFMLP(**cfg['model']).to(device)
    print(f'  Coarse params: {sum(p.numel() for p in nerf_coarse.parameters()):,}')

    optimizer = torch.optim.Adam(
        list(nerf_coarse.parameters()) + list(nerf_fine.parameters()),
        lr=cfg['training']['learning_rate'],
    )

    # ----- Optionally resume from checkpoint -----
    start_iter = 0
    if args.resume:
        print(f'Resuming from {args.resume}...')
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        nerf_coarse.load_state_dict(ckpt['coarse'])
        nerf_fine.load_state_dict(ckpt['fine'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt['iteration']
        print(f'  Resumed at iteration {start_iter}')

    # Exponential LR decay
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cfg['training']['lr_decay'],
    )

    loss_fn = NeRFLoss(
        use_depth_supervision=cfg['loss']['use_depth_supervision'],
        depth_weight=cfg['loss']['depth_weight'],
        coarse_weight=cfg['loss']['coarse_weight'],
        near=cfg['rendering']['near'],
        far=cfg['rendering']['far'],
    )

    # ----- Training loop -----
    num_iters = cfg['training']['num_iterations']
    rays_per_iter = cfg['training']['rays_per_iter']

    print(f'\nTraining for {num_iters} iterations, {rays_per_iter} rays per iter')
    start = time.time()

    nerf_coarse.train()
    nerf_fine.train()

    pbar = tqdm(range(start_iter, num_iters), desc='Training')
    for it in pbar:
        # Sample rays
        ray_o, ray_d, true_rgb, true_depth = sample_random_rays(
            images, depths, poses, K, rays_per_iter, H, W, device,
        )

        # Render
        render_out = render_rays(
            nerf_coarse, nerf_fine,
            ray_o, ray_d,
            near=cfg['rendering']['near'],
            far=cfg['rendering']['far'],
            num_coarse=cfg['rendering']['num_coarse'],
            num_fine=cfg['rendering']['num_fine'],
        )

        # Loss
        loss_dict = loss_fn(render_out, true_rgb, true_depth)
        loss = loss_dict['total']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % cfg['training']['lr_decay_every'] == 0 and it > 0:
            scheduler.step()

        # Logging
        if it % 100 == 0:
            with torch.no_grad():
                train_psnr = psnr(render_out['rgb_fine'], true_rgb)
            writer.add_scalar('train/loss_total', loss.item(), it)
            writer.add_scalar('train/loss_rgb_fine', loss_dict['rgb_fine'].item(), it)
            if 'depth_fine' in loss_dict:
                writer.add_scalar('train/loss_depth', loss_dict['depth_fine'].item(), it)
            writer.add_scalar('train/psnr', train_psnr, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'psnr': f'{train_psnr:.2f}'})

        # Save checkpoint
        if it > 0 and it % cfg['training']['save_every'] == 0:
            ckpt_path = os.path.join(output_dir, f'checkpoint_{it:06d}.pt')
            torch.save({
                'iteration': it,
                'coarse': nerf_coarse.state_dict(),
                'fine': nerf_fine.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': cfg,
            }, ckpt_path)

    # Final checkpoint
    final_path = os.path.join(output_dir, f'checkpoint_{num_iters:06d}.pt')
    torch.save({
        'iteration': num_iters,
        'coarse': nerf_coarse.state_dict(),
        'fine': nerf_fine.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': cfg,
    }, final_path)

    elapsed = (time.time() - start) / 3600
    print(f'\nTraining complete in {elapsed:.2f} hours')
    print(f'Final checkpoint: {final_path}')


if __name__ == '__main__':
    main()
