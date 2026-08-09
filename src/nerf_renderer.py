"""
nerf_renderer.py — Volume rendering for NeRF.

Given a NeRF MLP and a batch of rays, render each ray's:
  - Final color (alpha-composited along the ray)
  - Expected depth (Σ w_i * t_i)
  - Per-sample weights (for fine sampling refinement)

We follow the standard 2-stage rendering: coarse uniform sampling → importance
sampling → fine rendering with combined sample set.
"""
import torch
import torch.nn.functional as F


def sample_along_rays(
    ray_origins: torch.Tensor,    # (N, 3)
    ray_directions: torch.Tensor, # (N, 3)
    near: float,
    far: float,
    num_samples: int,
    perturb: bool = True,
) -> tuple:
    """Stratified sampling of points along each ray between near and far planes.

    Returns:
        positions:    (N, num_samples, 3) sample points in world space
        z_vals:       (N, num_samples) depth values along the ray
    """
    device = ray_origins.device
    N = ray_origins.shape[0]

    # Linearly spaced t values from near to far
    t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
    z_vals = near + (far - near) * t_vals  # (num_samples,)
    z_vals = z_vals.expand(N, num_samples)  # (N, num_samples)

    if perturb:
        # Add stratified noise within each bin so the network sees different
        # sample positions each forward pass
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)

    # Compute sample positions: o + t*d
    positions = (
        ray_origins.unsqueeze(1)
        + ray_directions.unsqueeze(1) * z_vals.unsqueeze(-1)
    )  # (N, num_samples, 3)

    return positions, z_vals


def importance_sample(
    z_vals: torch.Tensor,       # (N, num_coarse)
    weights: torch.Tensor,      # (N, num_coarse)
    num_samples: int,
    perturb: bool = True,
) -> torch.Tensor:
    """Importance sample additional points where the coarse network found density.

    Uses inverse-CDF sampling proportional to weight values.
    Bins are defined between consecutive z_vals, so we use midpoints as bin edges.
    """
    eps = 1e-5
    # Strip edge weights — these correspond to the boundary samples
    weights = weights[..., 1:-1] + eps  # (N, num_coarse - 2)

    # Bin midpoints between consecutive z_vals — these become our sample positions
    bins = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])  # (N, num_coarse - 1)

    # Normalize to a PDF
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (N, num_coarse - 1)

    # Sample uniformly in [0, 1)
    if perturb:
        u = torch.rand(*cdf.shape[:-1], num_samples, device=cdf.device)
    else:
        u = torch.linspace(0, 1, num_samples, device=cdf.device).expand(
            *cdf.shape[:-1], num_samples
        )
    u = u.contiguous()

    # Invert the CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)  # (N, num_samples, 2)

    # Gather CDF values and bin positions at the indices
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)

    # Linear interpolation between bin edges
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def volume_render(
    densities: torch.Tensor,      # (N, num_samples) raw density values
    colors: torch.Tensor,         # (N, num_samples, 3) RGB
    z_vals: torch.Tensor,         # (N, num_samples) sample depths
    ray_directions: torch.Tensor, # (N, 3)
    white_background: bool = False,
) -> dict:
    """Standard alpha compositing along each ray.

    Returns:
        rgb:    (N, 3) rendered color
        depth:  (N,)   expected depth (Σ w_i * t_i)
        acc:    (N,)   total accumulated opacity
        weights:(N, num_samples) per-sample contribution weights
    """
    # Distances between consecutive samples (in physical units along the ray)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # Pad final bin with a large value (represents continuing to infinity)
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

    # Multiply by ray direction norm to convert to true 3D distances
    dists = dists * ray_directions.norm(dim=-1, keepdim=True)

    # Alpha: 1 - exp(-σ * δ)
    alpha = 1.0 - torch.exp(-F.relu(densities) * dists)

    # Transmittance: cumulative product of (1 - alpha_i) along ray
    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[..., :-1]

    # Per-sample weight: alpha * cumulative transmittance
    weights = alpha * trans

    # Composite color and depth
    rgb = (weights.unsqueeze(-1) * colors).sum(dim=-2)        # (N, 3)
    depth = (weights * z_vals).sum(dim=-1)                     # (N,)
    acc = weights.sum(dim=-1)                                  # (N,)

    if white_background:
        rgb = rgb + (1.0 - acc.unsqueeze(-1))

    return {
        'rgb': rgb,
        'depth': depth,
        'acc': acc,
        'weights': weights,
    }


def render_rays(
    nerf_coarse,
    nerf_fine,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near: float,
    far: float,
    num_coarse: int = 64,
    num_fine: int = 128,
    perturb: bool = True,
    white_background: bool = False,
) -> dict:
    """Full coarse + fine NeRF rendering pipeline for a batch of rays.

    Returns a dict with:
        rgb_coarse, depth_coarse, rgb_fine, depth_fine, weights_fine, ...
    """
    # ----- Coarse pass: uniform sampling -----
    coarse_pos, coarse_z = sample_along_rays(
        ray_origins, ray_directions, near, far, num_coarse, perturb=perturb
    )
    # Reshape for batched MLP call
    flat_pos = coarse_pos.reshape(-1, 3)
    flat_dir = ray_directions.unsqueeze(1).expand(-1, num_coarse, -1).reshape(-1, 3)
    flat_dir = flat_dir / (flat_dir.norm(dim=-1, keepdim=True) + 1e-8)

    coarse_density, coarse_color = nerf_coarse(flat_pos, flat_dir)
    coarse_density = coarse_density.reshape(-1, num_coarse)
    coarse_color = coarse_color.reshape(-1, num_coarse, 3)

    coarse_out = volume_render(
        coarse_density, coarse_color, coarse_z, ray_directions, white_background
    )

    # ----- Fine pass: importance sampling -----
    fine_z = importance_sample(coarse_z, coarse_out['weights'], num_fine, perturb=perturb)

    # Combine coarse + fine samples and re-sort by depth
    combined_z, _ = torch.sort(torch.cat([coarse_z, fine_z], dim=-1), dim=-1)
    n_combined = combined_z.shape[-1]

    combined_pos = (
        ray_origins.unsqueeze(1)
        + ray_directions.unsqueeze(1) * combined_z.unsqueeze(-1)
    )

    flat_pos = combined_pos.reshape(-1, 3)
    flat_dir = ray_directions.unsqueeze(1).expand(-1, n_combined, -1).reshape(-1, 3)
    flat_dir = flat_dir / (flat_dir.norm(dim=-1, keepdim=True) + 1e-8)

    fine_density, fine_color = nerf_fine(flat_pos, flat_dir)
    fine_density = fine_density.reshape(-1, n_combined)
    fine_color = fine_color.reshape(-1, n_combined, 3)

    fine_out = volume_render(
        fine_density, fine_color, combined_z, ray_directions, white_background
    )

    return {
        'rgb_coarse': coarse_out['rgb'],
        'depth_coarse': coarse_out['depth'],
        'rgb_fine': fine_out['rgb'],
        'depth_fine': fine_out['depth'],
        'acc_fine': fine_out['acc'],
        'weights_fine': fine_out['weights'],
        'z_vals_fine': combined_z,
    }
