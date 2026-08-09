"""
make_comparison_gif.py — Build a side-by-side comparison GIF.

Stitches baseline and depth-supervised renders into a single GIF showing:
  [Ground truth RGB | Baseline render | Depth-supervised render]
or
  [Baseline depth | Depth-supervised depth | LiDAR ground truth]

The visual comparison makes the LiDAR-supervision improvement immediately obvious.
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_rendered_frames(directory: str, pattern: str = '*_rgb.png') -> list:
    """Load all RGB renders from a directory, sorted by index."""
    import glob
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    return [np.array(Image.open(f)) for f in files]


def stitch_horizontally(frames: list, gap_px: int = 4, gap_color: tuple = (20, 20, 20)) -> np.ndarray:
    """Concatenate frames left-to-right with a thin gap between them."""
    # All frames should be the same height
    H = frames[0].shape[0]

    # Build the gap strip
    gap = np.full((H, gap_px, 3), gap_color, dtype=np.uint8)

    result = [frames[0]]
    for f in frames[1:]:
        result.append(gap)
        result.append(f)
    return np.concatenate(result, axis=1)


def add_label(img: np.ndarray, label: str, position: str = 'top_left') -> np.ndarray:
    """Add a text label on a frame."""
    from PIL import ImageDraw, ImageFont
    pil_img = Image.fromarray(img).copy()
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype('arial.ttf', 22)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Get text dimensions for background box
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    padding = 6

    if position == 'top_left':
        x, y = 10, 10
    elif position == 'top_right':
        x, y = img.shape[1] - text_w - 20, 10
    else:
        x, y = 10, img.shape[0] - text_h - 20

    # Semi-opaque background
    draw.rectangle(
        [x - padding, y - padding, x + text_w + padding, y + text_h + padding],
        fill=(0, 0, 0, 180),
    )
    draw.text((x, y), label, fill=(255, 255, 255), font=font)

    return np.array(pil_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True,
                        help='Directory with baseline RGB renders')
    parser.add_argument('--supervised', required=True,
                        help='Directory with depth-supervised RGB renders')
    parser.add_argument('--gt-images', default=None,
                        help='Optional: directory with ground truth images (rendered same indices)')
    parser.add_argument('--mode', choices=['rgb', 'depth'], default='rgb',
                        help='What to compare: RGB renders or depth maps')
    parser.add_argument('--output', default='assets/comparison.gif')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--max-mb', type=float, default=8.0,
                        help='Target max GIF size (will reduce quality if exceeded)')
    args = parser.parse_args()

    # Load frames
    if args.mode == 'rgb':
        baseline_frames = load_rendered_frames(args.baseline, '*_rgb.png')
        supervised_frames = load_rendered_frames(args.supervised, '*_rgb.png')
    else:
        baseline_frames = load_rendered_frames(
            os.path.join(args.baseline, 'depth'), '*_depth.png'
        )
        supervised_frames = load_rendered_frames(
            os.path.join(args.supervised, 'depth'), '*_depth.png'
        )

    gt_frames = None
    if args.gt_images:
        gt_frames = load_rendered_frames(args.gt_images, '*_rgb.png')

    n_frames = min(len(baseline_frames), len(supervised_frames))
    print(f'Stitching {n_frames} frames into comparison GIF...')

    # Build labeled stitched frames
    combined_frames = []
    for i in tqdm(range(n_frames), desc='Stitching'):
        b = add_label(baseline_frames[i], 'Baseline NeRF')
        s = add_label(supervised_frames[i], 'LiDAR-Supervised')

        if gt_frames and i < len(gt_frames):
            gt = add_label(gt_frames[i], 'Ground Truth')
            stitched = stitch_horizontally([gt, b, s])
        else:
            stitched = stitch_horizontally([b, s])

        combined_frames.append(stitched)

    # Save GIF
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    print(f'Writing GIF: {args.output}')
    imageio.mimsave(args.output, combined_frames, fps=args.fps, loop=0)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f'Size: {size_mb:.2f} MB')

    if size_mb > args.max_mb:
        print(f'  WARNING: exceeds {args.max_mb} MB target.')
        print(f'  Try reducing fps or rendering fewer frames.')
        print(f'  Or compress with ffmpeg:')
        print(f'    ffmpeg -i {args.output} -vf "fps=8,scale=900:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse" {args.output}.small.gif')


if __name__ == '__main__':
    main()
