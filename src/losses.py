"""
losses.py — RGB photometric loss and LiDAR depth supervision loss.

The single conceptual difference between baseline NeRF and LiDAR-supervised NeRF
lives here: the depth-supervised variant adds an L1 loss on predicted ray depth
versus LiDAR-projected depth, but only for pixels where LiDAR provides a valid
measurement.
"""
import torch
import torch.nn.functional as F


def photometric_loss(
    pred_rgb: torch.Tensor,    # (N, 3) — rendered colors
    true_rgb: torch.Tensor,    # (N, 3) — ground truth colors
) -> torch.Tensor:
    """Standard NeRF RGB loss — pixel-wise MSE."""
    return F.mse_loss(pred_rgb, true_rgb)


def depth_loss(
    pred_depth: torch.Tensor,    # (N,) — predicted ray depths
    lidar_depth: torch.Tensor,   # (N,) — LiDAR depths (0 = invalid)
    valid_mask: torch.Tensor = None,  # (N,) bool — optional pre-computed mask
    near: float = 0.5,
    far: float = 80.0,
) -> torch.Tensor:
    """L1 loss between predicted depth and LiDAR depth, masked to valid pixels.

    Args:
        pred_depth: NeRF's expected ray termination depth
        lidar_depth: depth from LiDAR projection (0 where no LiDAR hit)
        valid_mask: optional explicit mask. If None, mask = (lidar_depth > 0)
        near, far: depth range to consider valid

    Returns:
        L1 loss averaged over valid pixels, or 0 tensor if no valid pixels.
    """
    if valid_mask is None:
        valid_mask = (lidar_depth > near) & (lidar_depth < far)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_depth.device)

    pred_masked = pred_depth[valid_mask]
    true_masked = lidar_depth[valid_mask]
    return F.l1_loss(pred_masked, true_masked)


class NeRFLoss:
    """Combined RGB + (optional) depth loss for training NeRF.

    Configure with `use_depth_supervision=True` to enable the LiDAR depth term.
    The `depth_weight` controls how much the depth loss contributes:
        L_total = L_rgb + depth_weight * L_depth
    """

    def __init__(
        self,
        use_depth_supervision: bool = False,
        depth_weight: float = 0.1,
        coarse_weight: float = 1.0,
        near: float = 0.5,
        far: float = 80.0,
    ):
        self.use_depth_supervision = use_depth_supervision
        self.depth_weight = depth_weight
        self.coarse_weight = coarse_weight
        self.near = near
        self.far = far

    def __call__(
        self,
        render_dict: dict,         # output from render_rays()
        true_rgb: torch.Tensor,    # (N, 3) ground truth pixel colors
        lidar_depth: torch.Tensor = None,  # (N,) optional LiDAR depths
    ) -> dict:
        """Compute total loss and return all components for logging."""
        # RGB loss on both coarse and fine outputs
        rgb_fine_loss = photometric_loss(render_dict['rgb_fine'], true_rgb)
        rgb_coarse_loss = photometric_loss(render_dict['rgb_coarse'], true_rgb)
        rgb_total = rgb_fine_loss + self.coarse_weight * rgb_coarse_loss

        components = {
            'rgb_fine': rgb_fine_loss,
            'rgb_coarse': rgb_coarse_loss,
            'rgb_total': rgb_total,
        }

        total = rgb_total

        # Optional: depth supervision on fine output
        if self.use_depth_supervision and lidar_depth is not None:
            depth_fine_loss = depth_loss(
                render_dict['depth_fine'], lidar_depth,
                near=self.near, far=self.far,
            )
            components['depth_fine'] = depth_fine_loss
            total = total + self.depth_weight * depth_fine_loss

        components['total'] = total
        return components
