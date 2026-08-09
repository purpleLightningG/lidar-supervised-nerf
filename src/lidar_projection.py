"""
lidar_projection.py — Project LiDAR points onto camera image to generate sparse depth maps.

This is the core component that enables LiDAR-supervised NeRF training. For each
RGB frame, we:
  1. Take the corresponding LiDAR scan (in velodyne frame)
  2. Transform it into the camera frame using calibration
  3. Project onto the image plane using camera intrinsics
  4. Keep only points that land inside the image and in front of the camera
  5. Write each surviving point's depth into a sparse depth map

The resulting depth map has shape (H, W) with values in meters at LiDAR-covered
pixels and 0 elsewhere. Coverage is roughly 25-30% of image pixels for KITTI.
"""
import numpy as np
import os
from typing import Dict, Tuple


def project_lidar_to_image(
    lidar_points: np.ndarray,    # (N, 4) — [x, y, z, reflectance] in velodyne frame
    P_rect_02: np.ndarray,       # (3, 4) — left color rectified projection
    R_rect: np.ndarray,          # (4, 4) — rectification rotation
    T_velo_cam: np.ndarray,      # (4, 4) — velodyne → cam0 transform
    image_height: int,
    image_width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project velodyne points onto a camera image.

    Returns:
        depth_map:    (H, W) float32 — depth in meters, 0 where no LiDAR hit
        valid_uv:     (M, 2) int32 — pixel coordinates of valid projections
        valid_depths: (M,) float32 — depths at those pixel coordinates
    """
    # Get XYZ (drop reflectance)
    xyz = lidar_points[:, :3]

    # Convert to homogeneous coords (N, 4)
    xyz_hom = np.hstack([xyz, np.ones((xyz.shape[0], 1))])

    # Filter: drop points behind the velodyne (negative x in velo frame)
    # These can never be visible to the front-facing camera.
    front_mask = xyz[:, 0] > 0
    xyz_hom = xyz_hom[front_mask]

    # Transform velodyne → rectified camera frame
    # cam_pts = R_rect @ T_velo_cam @ velo_pts
    velo_to_cam_rect = R_rect @ T_velo_cam  # (4, 4)
    cam_pts = (velo_to_cam_rect @ xyz_hom.T).T  # (N, 4)

    # Keep only points in front of the camera (positive Z in camera frame)
    front_cam_mask = cam_pts[:, 2] > 0.5  # 0.5m minimum, avoids divide-by-zero artifacts
    cam_pts = cam_pts[front_cam_mask]

    # Project to image plane: pixel_uv = P_rect_02 @ cam_pts
    # P_rect_02 is (3, 4) so output is (3, N)
    pixels_hom = (P_rect_02 @ cam_pts.T).T  # (N, 3)

    # Perspective divide: (u, v) = (x/z, y/z)
    pixels_uv = pixels_hom[:, :2] / pixels_hom[:, 2:3]
    depths = pixels_hom[:, 2]  # depth in camera Z

    # Keep only points inside image boundary
    u = pixels_uv[:, 0].astype(np.int32)
    v = pixels_uv[:, 1].astype(np.int32)
    inside_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)

    valid_u = u[inside_mask]
    valid_v = v[inside_mask]
    valid_depths = depths[inside_mask]

    # Build sparse depth map
    # Multiple LiDAR points can project to the same pixel — keep the closest
    depth_map = np.zeros((image_height, image_width), dtype=np.float32)
    for ui, vi, d in zip(valid_u, valid_v, valid_depths):
        if depth_map[vi, ui] == 0 or d < depth_map[vi, ui]:
            depth_map[vi, ui] = d

    valid_uv = np.stack([valid_u, valid_v], axis=1)
    return depth_map, valid_uv, valid_depths


def compute_coverage(depth_map: np.ndarray) -> Dict[str, float]:
    """Compute statistics on a sparse depth map."""
    H, W = depth_map.shape
    total_px = H * W
    valid_mask = depth_map > 0
    num_valid = int(valid_mask.sum())

    return {
        'coverage_percent': 100.0 * num_valid / total_px,
        'num_valid_pixels': num_valid,
        'min_depth_m': float(depth_map[valid_mask].min()) if num_valid > 0 else 0.0,
        'max_depth_m': float(depth_map[valid_mask].max()) if num_valid > 0 else 0.0,
        'mean_depth_m': float(depth_map[valid_mask].mean()) if num_valid > 0 else 0.0,
    }


def visualize_depth_on_image(
    image: np.ndarray,       # (H, W, 3) uint8
    depth_map: np.ndarray,   # (H, W) float32 in meters
    max_depth: float = 80.0,
    alpha: float = 0.8,
) -> np.ndarray:
    """Overlay a colored depth map on top of an RGB image.

    Returns:
        (H, W, 3) uint8 image with depth points drawn over RGB.
    """
    import matplotlib.cm as cm

    overlay = image.copy()
    valid_mask = depth_map > 0
    if not valid_mask.any():
        return overlay

    # Normalize depths to [0, 1] for colormap lookup
    depths_normalized = np.clip(depth_map / max_depth, 0, 1)

    # Apply colormap (jet: blue=near, red=far)
    cmap = cm.get_cmap('jet_r')  # reversed so warm = near, cool = far
    colored_depth = cmap(depths_normalized)[:, :, :3]  # (H, W, 3) in [0, 1]
    colored_depth = (colored_depth * 255).astype(np.uint8)

    # Alpha blend only where depth is valid
    overlay[valid_mask] = (
        alpha * colored_depth[valid_mask]
        + (1 - alpha) * overlay[valid_mask]
    ).astype(np.uint8)

    return overlay


if __name__ == '__main__':
    # Smoke test: project LiDAR for one frame and report statistics
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from kitti_raw_loader import KittiRawSequence

    if len(sys.argv) < 2:
        print('Usage: python lidar_projection.py <path_to_sequence_dir>')
        sys.exit(1)

    seq_dir = sys.argv[1]
    calib_dir = os.path.dirname(seq_dir)

    seq = KittiRawSequence(seq_dir, calib_dir)
    sample = seq[0]

    H, W = sample['image'].shape[:2]
    print(f'Image size: {W} x {H}')

    depth_map, uv, depths = project_lidar_to_image(
        lidar_points=sample['lidar'],
        P_rect_02=sample['P_rect_02'],
        R_rect=sample['R_rect'],
        T_velo_cam=sample['T_velo_cam'],
        image_height=H,
        image_width=W,
    )

    stats = compute_coverage(depth_map)
    print(f"Coverage: {stats['coverage_percent']:.1f}% of pixels")
    print(f"  Valid pixels: {stats['num_valid_pixels']}")
    print(f"  Depth range: {stats['min_depth_m']:.2f}m -- {stats['max_depth_m']:.2f}m")
    print(f"  Mean depth: {stats['mean_depth_m']:.2f}m")

    # Save visualization
    out_path = 'depth_overlay_smoketest.png'
    overlay = visualize_depth_on_image(sample['image'], depth_map)
    from PIL import Image
    Image.fromarray(overlay).save(out_path)
    print(f'\nVisualization saved to {out_path}')
