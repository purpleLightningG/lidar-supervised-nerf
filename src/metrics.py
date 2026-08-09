"""
metrics.py — Standard NeRF + depth evaluation metrics.

For rendering quality:
  - PSNR: peak signal-to-noise ratio (higher = better)
  - SSIM: structural similarity (higher = better, in [-1, 1])
  - LPIPS: perceptual similarity (lower = better)

For depth accuracy (vs LiDAR ground truth):
  - RMSE: root mean squared error in meters (lower = better)
  - Abs Rel: |pred - gt| / gt (lower = better)
  - δ < 1.25: % pixels where max(pred/gt, gt/pred) < 1.25 (higher = better)
"""
import torch
import torch.nn.functional as F
import numpy as np


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """Structural Similarity Index.

    Implementation uses a small Gaussian window — for full-image SSIM we recommend
    using `pytorch_msssim` or `scikit-image` for production accuracy.

    Args:
        pred, target: (H, W, 3) or (B, 3, H, W) in [0, 1]
    """
    try:
        from skimage.metrics import structural_similarity as ski_ssim
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        if pred_np.ndim == 3:  # (H, W, 3)
            return float(ski_ssim(pred_np, target_np, channel_axis=-1, data_range=1.0))
        else:  # batched
            scores = []
            for i in range(pred_np.shape[0]):
                p = pred_np[i].transpose(1, 2, 0) if pred_np.ndim == 4 else pred_np[i]
                t = target_np[i].transpose(1, 2, 0) if target_np.ndim == 4 else target_np[i]
                scores.append(ski_ssim(p, t, channel_axis=-1, data_range=1.0))
            return float(np.mean(scores))
    except ImportError:
        # Fallback: simple windowed variance-correlation approximation
        pred_mean = pred.mean()
        target_mean = target.mean()
        pred_var = pred.var()
        target_var = target.var()
        cov = ((pred - pred_mean) * (target - target_mean)).mean()
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        num = (2 * pred_mean * target_mean + c1) * (2 * cov + c2)
        denom = (pred_mean ** 2 + target_mean ** 2 + c1) * (pred_var + target_var + c2)
        return float((num / denom).item())


def lpips_score(pred: torch.Tensor, target: torch.Tensor, net: str = 'alex') -> float:
    """Learned Perceptual Image Patch Similarity (Zhang et al., 2018).

    Requires `pip install lpips`. Returns 0.0 if not installed.

    Args:
        pred, target: (H, W, 3) or (3, H, W) in [0, 1]
    """
    try:
        import lpips
    except ImportError:
        print('Warning: lpips not installed. Install: pip install lpips')
        return 0.0

    # Cache the LPIPS model per call (slow but works)
    loss_fn = lpips.LPIPS(net=net, verbose=False)

    # LPIPS expects (B, 3, H, W) in [-1, 1]
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    if pred.shape[-1] == 3:  # (B, H, W, 3) → (B, 3, H, W)
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

    pred_lpips = (pred * 2.0) - 1.0
    target_lpips = (target * 2.0) - 1.0
    return float(loss_fn(pred_lpips, target_lpips).mean().item())


def depth_rmse(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor = None, near: float = 0.5, far: float = 80.0) -> float:
    """RMSE in meters between predicted and ground truth depth maps.

    `target` is the LiDAR depth map (0 where no valid measurement).
    """
    if mask is None:
        mask = (target > near) & (target < far)
    if mask.sum() == 0:
        return 0.0
    diff = (pred[mask] - target[mask]) ** 2
    return float(torch.sqrt(diff.mean()).item())


def abs_rel(pred: torch.Tensor, target: torch.Tensor,
            mask: torch.Tensor = None, near: float = 0.5, far: float = 80.0) -> float:
    """Absolute relative depth error."""
    if mask is None:
        mask = (target > near) & (target < far)
    if mask.sum() == 0:
        return 0.0
    return float((torch.abs(pred[mask] - target[mask]) / target[mask]).mean().item())


def delta_threshold(pred: torch.Tensor, target: torch.Tensor,
                     threshold: float = 1.25, mask: torch.Tensor = None,
                     near: float = 0.5, far: float = 80.0) -> float:
    """Fraction of pixels where max(pred/gt, gt/pred) < threshold."""
    if mask is None:
        mask = (target > near) & (target < far)
    if mask.sum() == 0:
        return 0.0
    ratio = torch.max(pred[mask] / target[mask], target[mask] / pred[mask])
    return float((ratio < threshold).float().mean().item())


def evaluate_image(pred_rgb: torch.Tensor, true_rgb: torch.Tensor,
                   pred_depth: torch.Tensor = None,
                   true_depth: torch.Tensor = None) -> dict:
    """Compute all metrics on a single image pair."""
    metrics = {
        'psnr': psnr(pred_rgb, true_rgb),
        'ssim': ssim(pred_rgb, true_rgb),
    }

    # LPIPS is expensive (loads a network), skip if not needed
    if pred_rgb.numel() < 1_000_000:  # only for small images
        metrics['lpips'] = lpips_score(pred_rgb, true_rgb)

    if pred_depth is not None and true_depth is not None:
        metrics['depth_rmse'] = depth_rmse(pred_depth, true_depth)
        metrics['depth_abs_rel'] = abs_rel(pred_depth, true_depth)
        metrics['depth_delta_1_25'] = delta_threshold(pred_depth, true_depth, 1.25)

    return metrics
