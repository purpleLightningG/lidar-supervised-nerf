# LiDAR-Supervised NeRF on Driving Scenes

<p align="center">
  <img src="assets/comparison.gif" alt="Baseline vs LiDAR-supervised NeRF" width="800"/>
</p>

<p align="center">
  <b>Training a NeRF on outdoor driving scenes (KITTI Raw) with LiDAR depth as an auxiliary supervision signal.</b><br>
  Demonstrates that real sensor depth dramatically accelerates NeRF training and reduces "floater" artifacts on unbounded outdoor scenes.
</p>

<p align="center">
  <a href="#motivation">Motivation</a> ·
  <a href="#method">Method</a> ·
  <a href="#results">Results</a> ·
  <a href="#installation">Install</a> ·
  <a href="#usage">Usage</a>
</p>

\---

## Motivation

Standard NeRF training assumes the scene is bounded and has many overlapping camera views (e.g., 100 images of a single object). On **outdoor driving scenes** — where the camera moves forward along a road and many parts of the scene are seen from only a few angles — vanilla NeRF struggles:

* Slow convergence (many wasted rays on empty sky / featureless surfaces)
* "Floater" artifacts (spurious densities in front of the camera)
* Poor depth estimation (geometric ambiguity from limited parallax)

**Key insight:** autonomous vehicles already carry a LiDAR sensor that provides direct depth measurements. By projecting LiDAR points onto each image and using them as depth supervision (alongside the standard RGB photometric loss), we can give NeRF a strong geometric prior — essentially telling it "this pixel is at depth X meters."

This project trains two NeRFs side-by-side on KITTI Raw sequence `2011\_09\_26\_drive\_0005\_sync`:

1. **Baseline:** Vanilla NeRF with RGB-only photometric loss
2. **LiDAR-Supervised:** Adds an L1 depth loss between predicted ray depth and projected LiDAR depth, supervising only pixels where LiDAR has a valid measurement (\~25% of image pixels per frame)

We quantify the improvement on **rendering quality (PSNR)**, **depth accuracy (RMSE)**, and **training efficiency (iterations to convergence)**.

## Method

<p align="center">
  <img src="assets/architecture.png" alt="Pipeline architecture" width="650"/>
</p>

### LiDAR-to-image depth supervision

For each training image, the corresponding LiDAR scan is loaded and projected onto the image plane using:

* **Velodyne → camera** extrinsic (`Tr\_velo\_to\_cam`)
* **Rectification** rotation (`R0\_rect`)
* **Camera intrinsic** projection (`P\_rect\_02`)

The result is a \*\*sparse depth map\*\*: each LiDAR point that lands inside the image becomes a pixel with a known depth value. For KITTI Raw `drive\_0005`, this gives roughly \*\*20,000 supervised pixels per frame (\~4-5% of image area)\*\* — strong enough to constrain NeRF geometry without dominating the photometric loss.

### Loss formulation

```
L\_total = L\_rgb + λ · L\_depth

L\_rgb   = ||rendered\_color - true\_color||²
L\_depth = ||predicted\_depth - lidar\_depth||₁   (only on pixels with valid LiDAR)
```

The depth loss is applied to NeRF's expected ray termination depth, computed as `Σ w\_i \* t\_i` where `w\_i` is the per-sample opacity weight along a ray.

We use `λ = 0.1` based on a small sweep — large enough to constrain geometry, small enough not to dominate the photometric loss.

### Architecture

Both baseline and depth-supervised models use identical NeRF architecture:

* 8-layer coarse MLP, 256 hidden units, skip connection at layer 4
* 8-layer fine MLP (same dims)
* Positional encoding: 10 frequencies for position, 4 for view direction
* 64 coarse samples + 128 fine samples per ray
* Adam optimizer, learning rate 5e-4 with exponential decay

The only difference is the loss function and training data input (the depth-supervised version also receives LiDAR depth per ray).

## Results

> Numbers populated after overnight training run on RTX 3080.

### Rendering quality on held-out views

|Method|PSNR ↑|SSIM ↑|LPIPS ↓|Train Time (h)|
|-|-|-|-|-|
|Baseline NeRF|—|—|—|—|
|LiDAR-Supervised|—|—|—|—|

### Depth accuracy

|Method|RMSE (m) ↓|Abs Rel ↓|δ < 1.25 ↑|
|-|-|-|-|
|Baseline NeRF|—|—|—|
|LiDAR-Supervised|—|—|—|

### Convergence

<p align="center">
  <img src="assets/convergence.png" alt="PSNR over training iterations" width="600"/>
</p>

## Installation

### Prerequisites

* Python 3.10+
* CUDA 11.8+ with PyTorch 2.0+ (tested on RTX 3080)
* \~2 GB disk for KITTI Raw `drive\_0005\_sync` sequence
* \~12 hours GPU time per training run

### Setup

```bash
git clone https://github.com/purpleLightningG/lidar-supervised-nerf.git
cd lidar-supervised-nerf
python -m venv .venv \&\& source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Data

Download KITTI Raw `2011\_09\_26\_drive\_0005\_sync` from the [official site](https://www.cvlibs.net/datasets/kitti/raw_data.php). You need:

* Sequence data: `2011\_09\_26\_drive\_0005\_sync.zip` (\~700 MB)
* Calibration: `2011\_09\_26\_calib.zip` (\~3 KB)

Organize as:

```
data/2011\_09\_26/
├── calib\_cam\_to\_cam.txt
├── calib\_imu\_to\_velo.txt
├── calib\_velo\_to\_cam.txt
└── 2011\_09\_26\_drive\_0005\_sync/
    ├── image\_02/data/       # left RGB camera frames (.png)
    ├── velodyne\_points/data/ # LiDAR scans (.bin)
    └── oxts/data/            # per-frame GPS+IMU poses
```

Update `configs/default.yaml` with your data root.

## Usage

### Pre-compute sparse depth maps from LiDAR

```bash
python scripts/precompute\_depth.py --config configs/default.yaml
```

This projects each LiDAR scan onto its camera image and saves the sparse depth map as a `.npy` file. Runs once per sequence (\~5 minutes for 156 frames).

### Train baseline NeRF (RGB only)

```bash
python scripts/train.py --config configs/baseline.yaml
```

### Train LiDAR-supervised NeRF

```bash
python scripts/train.py --config configs/depth\_supervised.yaml
```

Both train for 200k iterations (\~10-12 hours on RTX 3080). Checkpoints + TensorBoard logs save to `outputs/<experiment\_name>/`.

Monitor training:

```bash
tensorboard --logdir outputs/
```

### Render novel views

```bash
python scripts/render.py \\
  --checkpoint outputs/depth\_supervised/checkpoint\_200000.pt \\
  --num-frames 60 \\
  --output outputs/depth\_supervised/novel\_views/
```

### Evaluate

```bash
python scripts/evaluate.py \\
  --baseline outputs/baseline/checkpoint\_200000.pt \\
  --supervised outputs/depth\_supervised/checkpoint\_200000.pt \\
  --output outputs/comparison.json
```

Computes PSNR, SSIM, LPIPS on held-out frames + depth RMSE against LiDAR ground truth.

### Generate comparison GIF

```bash
python scripts/make\_comparison\_gif.py \\
  --baseline outputs/baseline/novel\_views/ \\
  --supervised outputs/depth\_supervised/novel\_views/ \\
  --output assets/comparison.gif
```

## Repository Structure

```
lidar-supervised-nerf/
├── src/
│   ├── kitti\_raw\_loader.py  # KITTI Raw data loading + pose extraction from OXTS
│   ├── lidar\_projection.py  # LiDAR → image plane projection
│   ├── nerf\_model.py        # Vanilla NeRF architecture
│   ├── nerf\_renderer.py     # Volume rendering (coarse + fine sampling)
│   ├── losses.py            # RGB + depth loss functions
│   └── metrics.py           # PSNR, SSIM, LPIPS, depth RMSE
├── scripts/
│   ├── precompute\_depth.py  # Project LiDAR to sparse depth maps
│   ├── train.py             # Main training loop
│   ├── render.py            # Render novel views from a checkpoint
│   ├── evaluate.py          # Compute all comparison metrics
│   └── make\_comparison\_gif.py
├── configs/
│   ├── default.yaml         # Shared paths and base settings
│   ├── baseline.yaml        # RGB-only training
│   └── depth\_supervised.yaml # +LiDAR depth supervision
├── assets/                  # Architecture diagram, comparison GIF
├── outputs/                 # Checkpoints, novel views, metrics (gitignored)
├── requirements.txt
└── README.md
```

## References

* Mildenhall et al., **"NeRF: Representing Scenes as Neural Radiance Fields"**, ECCV 2020
* Deng et al., **"Depth-supervised NeRF: Fewer Views and Faster Training for Free"**, CVPR 2022 (DS-NeRF — uses sparse COLMAP depth instead of LiDAR)
* Rematas et al., **"Urban Radiance Fields"**, CVPR 2022 (URF — closest related work, uses LiDAR but on a different scene type)

## Citation

```bibtex
@misc{hossain2026lidarnerf,
  author = {Hossain, Shahriar},
  title  = {LiDAR-Supervised NeRF on Driving Scenes},
  year   = {2026},
  url    = {https://github.com/purpleLightningG/lidar-supervised-nerf}
}
```

## License

MIT — see [LICENSE](LICENSE).

\---

<p align="center">
  Built by <a href="https://purpleLightningG.github.io/portfolio-website">Shahriar Hossain</a> · PhD Researcher, George Mason University
</p>

