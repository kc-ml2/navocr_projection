#!/usr/bin/env python3
"""
Visualize NavOCR reprojection error on actual images
Shows detected bounding box centers and projected landmark centers
"""

import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def parse_reprojection_file(filename):
    """Parse NavOCR_reprojection_error.txt file

    Format: landmark_id,text,timestamp,bbox_u,bbox_v,proj_u,proj_v,error_px
    """
    observations = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse CSV line: landmark_id,"text",timestamp,bbox_u,bbox_v,proj_u,proj_v,error_px
            match = re.match(
                r'(\d+),"([^"]*)",([\d.]+),([\d.-]+),([\d.-]+),([\d.-]+),([\d.-]+),([\d.]+)',
                line
            )
            if match:
                obs = {
                    'landmark_id': int(match.group(1)),
                    'text': match.group(2),
                    'timestamp': float(match.group(3)),
                    'bbox_u': float(match.group(4)),
                    'bbox_v': float(match.group(5)),
                    'proj_u': float(match.group(6)),
                    'proj_v': float(match.group(7)),
                    'error': float(match.group(8))
                }
                observations.append(obs)

    return observations


def group_by_timestamp(observations, tolerance=0.05):
    """Group observations by timestamp (within tolerance seconds)"""
    if not observations:
        return {}

    # Sort by timestamp
    sorted_obs = sorted(observations, key=lambda x: x['timestamp'])

    groups = {}
    current_group = [sorted_obs[0]]
    current_ts = sorted_obs[0]['timestamp']

    for obs in sorted_obs[1:]:
        if abs(obs['timestamp'] - current_ts) <= tolerance:
            current_group.append(obs)
        else:
            # Save current group
            avg_ts = sum(o['timestamp'] for o in current_group) / len(current_group)
            groups[avg_ts] = current_group
            # Start new group
            current_group = [obs]
            current_ts = obs['timestamp']

    # Save last group
    if current_group:
        avg_ts = sum(o['timestamp'] for o in current_group) / len(current_group)
        groups[avg_ts] = current_group

    return groups


def find_closest_image(timestamp, image_dir):
    """Find image closest to timestamp"""
    image_dir = Path(image_dir)
    images = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))

    if not images:
        return None

    closest_img = None
    min_dt = float('inf')

    for img_path in images:
        try:
            img_timestamp = float(img_path.stem)
            dt = abs(img_timestamp - timestamp)

            if dt < min_dt:
                min_dt = dt
                closest_img = img_path
        except:
            continue

    # Only return if within reasonable time (1 second)
    if min_dt < 1.0:
        return closest_img
    return None


def visualize_frame(observations, image_path, output_path, frame_idx, total_frames, show=False):
    """Visualize reprojection on a single frame"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    timestamp = observations[0]['timestamp']

    # === Left subplot: Detected text centers ===
    ax1.imshow(img_rgb)
    ax1.set_title(f'Detected Text Centers\nFrame {frame_idx}/{total_frames}, t={timestamp:.3f}s',
                  fontsize=14, fontweight='bold')
    ax1.axis('off')

    for obs in observations:
        bbox_u, bbox_v = obs['bbox_u'], obs['bbox_v']
        text = obs['text']

        # Draw detected center (GREEN circle)
        ax1.plot(bbox_u, bbox_v, 'go', markersize=8, markeredgewidth=2, markeredgecolor='white')

        # Draw text label
        ax1.text(bbox_u + 10, bbox_v - 10, f"{text}\n({bbox_u:.0f}, {bbox_v:.0f})",
                color='lime', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # === Right subplot: Reprojection comparison ===
    ax2.imshow(img_rgb)
    ax2.set_title(f'Reprojection Error Visualization\n(Green=Detected, Red=Projected)',
                  fontsize=14, fontweight='bold')
    ax2.axis('off')

    for i, obs in enumerate(observations):
        bbox_u, bbox_v = obs['bbox_u'], obs['bbox_v']
        proj_u, proj_v = obs['proj_u'], obs['proj_v']
        text = obs['text']
        error = obs['error']

        # Skip if projected coordinates are invalid
        if proj_u < 0 or proj_v < 0:
            continue

        # Draw detected center (GREEN)
        label_det = 'Detected' if i == 0 else ''
        ax2.plot(bbox_u, bbox_v, 'go', markersize=10, markeredgewidth=2,
                 markeredgecolor='white', label=label_det)

        # Draw projected center (RED)
        label_proj = 'Projected' if i == 0 else ''
        ax2.plot(proj_u, proj_v, 'rx', markersize=12, markeredgewidth=3, label=label_proj)

        # Draw error vector (line connecting detected to projected)
        ax2.plot([bbox_u, proj_u], [bbox_v, proj_v], 'y-', linewidth=2, alpha=0.7)

        # Draw text label
        mid_u = (bbox_u + proj_u) / 2
        mid_v = (bbox_v + proj_v) / 2

        ax2.text(mid_u + 15, mid_v, f"{text}\nError: {error:.2f}px",
                color='yellow', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

    # Add legend
    ax2.legend(loc='upper right', fontsize=12, framealpha=0.9)

    # Add statistics box
    errors = [obs['error'] for obs in observations]
    stats_text = f"Observations: {len(errors)}\n"
    stats_text += f"Mean error: {np.mean(errors):.2f}px\n"
    stats_text += f"Max error: {np.max(errors):.2f}px\n"
    stats_text += f"Min error: {np.min(errors):.2f}px"

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            fontsize=11, family='monospace', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return True


def main():
    # Configuration
    reprojection_file = '/home/sehyeon/ros2_ws/src/navocr_projection/results_cpp/NavOCR_reprojection_error.txt'
    image_dir = '/home/sehyeon/ros2_ws/src/TextSLAM/custom_dataset/coex_0108_1/images'
    output_dir = '/home/sehyeon/ros2_ws/src/navocr_projection/results_cpp/reprojection_results'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Parse reprojection data
    print("Parsing reprojection error data...")
    observations = parse_reprojection_file(reprojection_file)

    if not observations:
        print("Error: No observation data found!")
        print("Make sure the reprojection file contains coordinate data.")
        print("Expected format: landmark_id,text,timestamp,bbox_u,bbox_v,proj_u,proj_v,error_px")
        return

    print(f"Found {len(observations)} observations")

    # Group by timestamp
    print("Grouping observations by timestamp...")
    grouped = group_by_timestamp(observations)
    print(f"Found {len(grouped)} unique frames")

    if not grouped:
        print("Error: Could not group observations by timestamp!")
        return

    print(f"Output directory: {output_dir}/")
    print("=" * 80)

    # Process all frames
    processed = 0
    failed = 0
    total_obs = 0

    sorted_timestamps = sorted(grouped.keys())
    total_frames = len(sorted_timestamps)

    for idx, timestamp in enumerate(sorted_timestamps, 1):
        frame_obs = grouped[timestamp]
        num_obs = len(frame_obs)

        print(f"[{idx}/{total_frames}] t={timestamp:.3f}s ({num_obs} obs)...", end=" ")

        # Find closest image
        image_path = find_closest_image(timestamp, image_dir)

        if image_path is None:
            print("SKIP (no image)")
            failed += 1
            continue

        # Visualize and save
        output_path = os.path.join(output_dir, f'frame_{idx:04d}.png')
        success = visualize_frame(frame_obs, image_path, output_path, idx, total_frames, show=False)

        if success:
            print("OK")
            processed += 1
            total_obs += num_obs
        else:
            print("FAILED")
            failed += 1

    # Summary
    print("=" * 80)
    print(f"Completed: {processed}/{total_frames} frames")
    print(f"Total observations visualized: {total_obs}")
    print(f"Output saved to: {output_dir}/")


if __name__ == '__main__':
    main()
