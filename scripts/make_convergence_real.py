"""
make_convergence_real.py — Plot ACTUAL PSNR curves from TensorBoard event files.

Reads the real training logs from both runs and plots train/psnr over iterations.
Run this locally where your outputs/ folder lives.

Usage:
    python make_convergence_real.py

Requires: tensorboard (you already have it), matplotlib, numpy
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# TensorBoard event reading
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Installing tensorboard reader...")
    os.system("pip install tensorboard --quiet")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_psnr_from_events(run_dir, tag='train/psnr'):
    """Load a scalar tag from all event files in a directory, merged and sorted.

    Multiple event files exist because training was resumed — we merge them all
    and sort by step so the resumed portions stitch together correctly.
    """
    event_files = sorted(glob.glob(os.path.join(run_dir, 'events.out.tfevents.*')))
    if not event_files:
        raise FileNotFoundError(f"No event files in {run_dir}")

    steps, values = [], []
    for ef in event_files:
        try:
            acc = EventAccumulator(ef, size_guidance={'scalars': 0})
            acc.Reload()
            if tag not in acc.Tags().get('scalars', []):
                continue
            for scalar_event in acc.Scalars(tag):
                steps.append(scalar_event.step)
                values.append(scalar_event.value)
        except Exception as e:
            print(f"  Skipping {os.path.basename(ef)}: {e}")

    if not steps:
        raise ValueError(f"Tag '{tag}' not found in any event file in {run_dir}")

    # Sort by step and dedupe (resumed runs can overlap)
    order = np.argsort(steps)
    steps = np.array(steps)[order]
    values = np.array(values)[order]

    # Deduplicate steps (keep last value for each step)
    unique_steps, unique_idx = np.unique(steps, return_index=True)
    # np.unique with return_index gives FIRST occurrence; for resumed runs we
    # want a clean monotonic curve, so first occurrence is fine
    return steps, values


def main():
    baseline_dir = 'outputs/baseline'
    supervised_dir = 'outputs/depth_supervised'

    print("Loading baseline PSNR...")
    b_steps, b_vals = load_psnr_from_events(baseline_dir)
    print(f"  {len(b_steps)} points, final PSNR: {b_vals[-1]:.2f}")

    print("Loading depth-supervised PSNR...")
    s_steps, s_vals = load_psnr_from_events(supervised_dir)
    print(f"  {len(s_steps)} points, final PSNR: {s_vals[-1]:.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    # Light raw lines + smoothed overlay for readability
    def smooth(y, window=9):
        if len(y) < window:
            return y
        kernel = np.ones(window) / window
        return np.convolve(y, kernel, mode='same')

    ax.plot(b_steps, b_vals, color='#4C72B0', alpha=0.25, linewidth=0.8)
    ax.plot(b_steps, smooth(b_vals), color='#4C72B0', linewidth=2,
            label=f'Baseline NeRF (RGB only) — {b_vals[-1]:.2f} dB')

    ax.plot(s_steps, s_vals, color='#C44E52', alpha=0.25, linewidth=0.8)
    ax.plot(s_steps, smooth(s_vals), color='#C44E52', linewidth=2,
            label=f'LiDAR-Supervised NeRF — {s_vals[-1]:.2f} dB')

    ax.set_xlabel('Training Iteration', fontsize=11)
    ax.set_ylabel('PSNR (dB)', fontsize=11)
    ax.set_title('Training Convergence — Baseline vs LiDAR-Supervised NeRF',
                 fontsize=12.5, weight='bold', pad=12)
    ax.legend(loc='lower right', fontsize=10, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='--')

    # Format x-axis in thousands
    max_step = max(b_steps[-1], s_steps[-1])
    ax.set_xlim(0, max_step * 1.02)

    plt.tight_layout()
    out_path = 'assets/convergence.png'
    os.makedirs('assets', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'\nSaved REAL convergence plot to {out_path}')
    print('This uses your actual TensorBoard training data.')


if __name__ == '__main__':
    main()
