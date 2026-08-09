"""
precompute_depth.py — Project LiDAR to sparse depth maps for all frames.

Runs once per sequence. For each frame, projects the velodyne point cloud onto
the image_02 camera plane and saves the result as a numpy array on disk.

This way the training loop doesn't have to do the projection on every iteration.
"""
import os
import sys
import argparse
import yaml
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.kitti_raw_loader import KittiRawSequence
from src.lidar_projection import project_lidar_to_image, compute_coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='YAML config file path')
    parser.add_argument('--visualize', action='store_true',
                        help='Also save depth overlay PNGs for visual sanity check')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sequence_dir = cfg['data']['sequence_dir']
    calib_dir = cfg['data']['calib_dir']
    output_dir = cfg['data']['depth_cache_dir']
    os.makedirs(output_dir, exist_ok=True)

    seq = KittiRawSequence(sequence_dir, calib_dir)
    print(f'Loaded sequence with {len(seq)} frames')

    # Optional visualization output
    if args.visualize:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

    coverages = []
    for i in tqdm(range(len(seq)), desc='Projecting LiDAR'):
        sample = seq[i]
        H, W = sample['image'].shape[:2]

        depth_map, _, _ = project_lidar_to_image(
            lidar_points=sample['lidar'],
            P_rect_02=sample['P_rect_02'],
            R_rect=sample['R_rect'],
            T_velo_cam=sample['T_velo_cam'],
            image_height=H,
            image_width=W,
        )

        # Save as .npy
        out_path = os.path.join(output_dir, f"{sample['frame_id']}.npy")
        np.save(out_path, depth_map.astype(np.float16))  # half precision saves space

        # Track stats
        coverages.append(compute_coverage(depth_map))

        # Save visualization for first 5 frames
        if args.visualize and i < 5:
            from src.lidar_projection import visualize_depth_on_image
            from PIL import Image
            overlay = visualize_depth_on_image(sample['image'], depth_map)
            Image.fromarray(overlay).save(
                os.path.join(vis_dir, f"{sample['frame_id']}_overlay.png")
            )

    # Summary statistics
    mean_coverage = np.mean([c['coverage_percent'] for c in coverages])
    mean_valid = np.mean([c['num_valid_pixels'] for c in coverages])
    print(f'\nDone. Saved {len(seq)} depth maps to {output_dir}/')
    print(f'  Average coverage: {mean_coverage:.1f}% of image pixels')
    print(f'  Average valid pixels per frame: {mean_valid:.0f}')


if __name__ == '__main__':
    main()
