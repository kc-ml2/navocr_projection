#!/usr/bin/env python3
"""
Compare reprojection errors between TextSLAM and NavOCR

This script:
1. Loads per-landmark summaries from both systems
2. Matches landmarks based on spatial proximity and text similarity
3. Computes error ratio statistics
4. Generates visualization plots
5. Outputs comparison summary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys


def levenshtein_distance(s1, s2):
    """Compute Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def text_similarity(s1, s2):
    """Compute text similarity (0-1) based on Levenshtein distance"""
    if not s1 or not s2:
        return 0.0
    dist = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    return 1.0 - (dist / max_len)


def load_textslam_landmarks(summary_file):
    """Load TextSLAM landmark summary"""
    print(f"Loading TextSLAM landmarks from {summary_file}")

    # Skip header lines and load CSV
    data = []
    with open(summary_file, 'r') as f:
        for line in f:
            if line.startswith('landmark_id,'):
                # This is the header
                continue
            if line.strip() and not line.startswith('===') and not line.startswith('Total') and not line.startswith('Overall') and not line.startswith('Weighted'):
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    try:
                        landmark_id = int(parts[0])
                        text = parts[1].strip('"')
                        num_obs = int(parts[2])
                        mean_error = float(parts[3])
                        data.append({
                            'landmark_id': landmark_id,
                            'text': text,
                            'num_observations': num_obs,
                            'mean_error': mean_error
                        })
                    except:
                        continue

    df = pd.DataFrame(data)
    print(f"  Loaded {len(df)} TextSLAM landmarks")
    return df


def load_navocr_landmarks(summary_file, landmarks_csv):
    """Load NavOCR landmark summary with 3D positions"""
    print(f"Loading NavOCR landmarks from {summary_file}")

    # Load error summary
    data = []
    with open(summary_file, 'r') as f:
        for line in f:
            if line.startswith('landmark_id,'):
                continue
            if line.strip() and not line.startswith('===') and not line.startswith('Total') and not line.startswith('Overall') and not line.startswith('Weighted') and not line.startswith('Cumulative'):
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    try:
                        landmark_id = int(parts[0])
                        text = parts[1].strip('"')
                        num_obs = int(parts[2])
                        mean_error = float(parts[3])
                        data.append({
                            'landmark_id': landmark_id,
                            'text': text,
                            'num_observations': num_obs,
                            'mean_error': mean_error
                        })
                    except:
                        continue

    df_errors = pd.DataFrame(data)

    # Load 3D positions from landmarks CSV
    print(f"Loading NavOCR 3D positions from {landmarks_csv}")
    df_positions = pd.read_csv(landmarks_csv)

    # Merge
    df = pd.merge(df_errors, df_positions[['landmark_id', 'x', 'y', 'z']], on='landmark_id', how='left')

    print(f"  Loaded {len(df)} NavOCR landmarks")
    return df


def load_textslam_3d_positions(text_info_file):
    """
    Load TextSLAM 3D positions from Text_info.txt
    Format: landmark_id STATE n1 n2 n3 x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4
    We compute center as average of 4 corners
    """
    print(f"Loading TextSLAM 3D positions from {text_info_file}")

    positions = {}
    with open(text_info_file, 'r') as f:
        for line in f:
            if line.startswith('--') or not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) >= 18:
                try:
                    landmark_id = int(parts[0])
                    # 4 corner positions at indices 5-16
                    x1, y1, z1 = float(parts[5]), float(parts[6]), float(parts[7])
                    x2, y2, z2 = float(parts[8]), float(parts[9]), float(parts[10])
                    x3, y3, z3 = float(parts[11]), float(parts[12]), float(parts[13])
                    x4, y4, z4 = float(parts[14]), float(parts[15]), float(parts[16])

                    # Compute center
                    x_center = (x1 + x2 + x3 + x4) / 4.0
                    y_center = (y1 + y2 + y3 + y4) / 4.0
                    z_center = (z1 + z2 + z3 + z4) / 4.0

                    positions[landmark_id] = {'x': x_center, 'y': y_center, 'z': z_center}
                except:
                    continue

    print(f"  Loaded {len(positions)} TextSLAM 3D positions")
    return positions


def match_landmarks(df_textslam, df_navocr, textslam_positions,
                   spatial_threshold=1.5, text_sim_threshold=0.6):
    """
    Match NavOCR landmarks to TextSLAM landmarks
    Based on spatial proximity + text similarity
    """
    print("\nMatching landmarks...")
    print(f"  Spatial threshold: {spatial_threshold}m")
    print(f"  Text similarity threshold: {text_sim_threshold}")

    # Add 3D positions to TextSLAM dataframe
    df_ts = df_textslam.copy()
    df_ts['x'] = df_ts['landmark_id'].map(lambda lid: textslam_positions.get(lid, {}).get('x', np.nan))
    df_ts['y'] = df_ts['landmark_id'].map(lambda lid: textslam_positions.get(lid, {}).get('y', np.nan))
    df_ts['z'] = df_ts['landmark_id'].map(lambda lid: textslam_positions.get(lid, {}).get('z', np.nan))
    df_ts = df_ts.dropna(subset=['x', 'y', 'z'])

    matches = []

    for idx_nav, row_nav in df_navocr.iterrows():
        nav_id = row_nav['landmark_id']
        nav_text = row_nav['text']
        nav_pos = np.array([row_nav['x'], row_nav['y'], row_nav['z']])

        best_match = None
        best_score = 0.0

        for idx_ts, row_ts in df_ts.iterrows():
            ts_id = row_ts['landmark_id']
            ts_text = row_ts['text']
            ts_pos = np.array([row_ts['x'], row_ts['y'], row_ts['z']])

            # Spatial distance
            distance = np.linalg.norm(nav_pos - ts_pos)

            if distance < spatial_threshold:
                # Text similarity
                text_sim = text_similarity(nav_text, ts_text)

                if text_sim > text_sim_threshold:
                    # Combined score: 70% text, 30% spatial
                    spatial_score = np.exp(-distance / spatial_threshold)
                    combined_score = 0.7 * text_sim + 0.3 * spatial_score

                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = {
                            'nav_id': nav_id,
                            'ts_id': ts_id,
                            'nav_text': nav_text,
                            'ts_text': ts_text,
                            'distance': distance,
                            'text_similarity': text_sim,
                            'score': combined_score,
                            'nav_error': row_nav['mean_error'],
                            'ts_error': row_ts['mean_error'],
                            'nav_num_obs': row_nav['num_observations'],
                            'ts_num_obs': row_ts['num_observations']
                        }

        if best_match and best_score > 0.5:
            matches.append(best_match)

    df_matches = pd.DataFrame(matches)
    print(f"  Matched: {len(df_matches)} / {len(df_navocr)} NavOCR landmarks ({100*len(df_matches)/len(df_navocr):.1f}%)")

    return df_matches


def compute_statistics(df_matches):
    """Compute comparison statistics"""
    if len(df_matches) == 0:
        print("No matches found!")
        return None

    # Error ratio
    df_matches['error_ratio'] = df_matches['nav_error'] / df_matches['ts_error']

    # Weighted statistics (weight by sqrt of min observations)
    df_matches['weight'] = np.sqrt(np.minimum(df_matches['nav_num_obs'], df_matches['ts_num_obs']))

    stats = {
        'num_matches': len(df_matches),
        'mean_ratio': df_matches['error_ratio'].mean(),
        'median_ratio': df_matches['error_ratio'].median(),
        'std_ratio': df_matches['error_ratio'].std(),
        'weighted_mean_ratio': (df_matches['error_ratio'] * df_matches['weight']).sum() / df_matches['weight'].sum(),
        'mean_nav_error': df_matches['nav_error'].mean(),
        'mean_ts_error': df_matches['ts_error'].mean(),
        'weighted_mean_nav_error': (df_matches['nav_error'] * df_matches['weight']).sum() / df_matches['weight'].sum(),
        'weighted_mean_ts_error': (df_matches['ts_error'] * df_matches['weight']).sum() / df_matches['weight'].sum(),
        'within_1_4x': (df_matches['error_ratio'] <= 1.4).sum(),
        'within_1_5x': (df_matches['error_ratio'] <= 1.5).sum()
    }

    return stats


def generate_plots(df_matches, output_dir):
    """Generate comparison plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(df_matches['ts_error'], df_matches['nav_error'],
               s=50, alpha=0.6, c=df_matches['weight'], cmap='viridis')
    plt.colorbar(label='Weight (sqrt of min obs)')

    max_error = max(df_matches['ts_error'].max(), df_matches['nav_error'].max())
    plt.plot([0, max_error], [0, max_error], 'r--', label='1:1 line', linewidth=2)
    plt.plot([0, max_error], [0, 1.4*max_error], 'g--', label='1.4:1 line (acceptable)', linewidth=2)

    plt.xlabel('TextSLAM Reprojection Error (pixels)', fontsize=12)
    plt.ylabel('NavOCR Reprojection Error (pixels)', fontsize=12)
    plt.title('Reprojection Error Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_plot.png', dpi=300)
    print(f"  Saved scatter plot: {output_dir / 'scatter_plot.png'}")
    plt.close()

    # 2. Error ratio histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df_matches['error_ratio'], bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(df_matches['error_ratio'].mean(), color='r', linestyle='--',
               linewidth=2, label=f'Mean: {df_matches["error_ratio"].mean():.2f}')
    plt.axvline(df_matches['error_ratio'].median(), color='g', linestyle='--',
               linewidth=2, label=f'Median: {df_matches["error_ratio"].median():.2f}')
    plt.axvline(1.4, color='orange', linestyle='--',
               linewidth=2, label='Target: 1.4x')
    plt.xlabel('Error Ratio (NavOCR / TextSLAM)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Error Ratios', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'ratio_histogram.png', dpi=300)
    print(f"  Saved histogram: {output_dir / 'ratio_histogram.png'}")
    plt.close()


def save_summary(stats, df_matches, output_file):
    """Save comparison summary to text file"""
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Reprojection Error Comparison: NavOCR vs. TextSLAM\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Matched landmarks: {stats['num_matches']}\n\n")

        f.write("--- Error Ratio Statistics ---\n")
        f.write(f"Mean ratio:           {stats['mean_ratio']:.3f}x\n")
        f.write(f"Median ratio:         {stats['median_ratio']:.3f}x\n")
        f.write(f"Std ratio:            {stats['std_ratio']:.3f}\n")
        f.write(f"Weighted mean ratio:  {stats['weighted_mean_ratio']:.3f}x\n\n")

        f.write("--- Absolute Errors ---\n")
        f.write(f"NavOCR mean error:    {stats['mean_nav_error']:.2f} pixels\n")
        f.write(f"TextSLAM mean error:  {stats['mean_ts_error']:.2f} pixels\n")
        f.write(f"NavOCR weighted:      {stats['weighted_mean_nav_error']:.2f} pixels\n")
        f.write(f"TextSLAM weighted:    {stats['weighted_mean_ts_error']:.2f} pixels\n\n")

        f.write("--- Success Criteria ---\n")
        f.write(f"Within 1.4x:  {stats['within_1_4x']}/{stats['num_matches']} ({100*stats['within_1_4x']/stats['num_matches']:.1f}%)\n")
        f.write(f"Within 1.5x:  {stats['within_1_5x']}/{stats['num_matches']} ({100*stats['within_1_5x']/stats['num_matches']:.1f}%)\n\n")

        if stats['weighted_mean_ratio'] <= 1.4:
            f.write("✅ SUCCESS: NavOCR achieves >70% of TextSLAM quality\n")
            f.write("   Modular structure is viable for practical use!\n")
        else:
            f.write("❌ WARNING: NavOCR error ratio exceeds 1.4x target\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"\n  Saved summary: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare TextSLAM and NavOCR reprojection errors')
    parser.add_argument('--textslam-summary', required=True, help='TextSLAM Text_reprojection_summary.txt')
    parser.add_argument('--textslam-3d', required=True, help='TextSLAM Text_info.txt (for 3D positions)')
    parser.add_argument('--navocr-summary', required=True, help='NavOCR NavOCR_reprojection_summary.txt')
    parser.add_argument('--navocr-landmarks', required=True, help='NavOCR landmarks CSV file')
    parser.add_argument('--output-dir', default='./comparison_results', help='Output directory')

    args = parser.parse_args()

    # Load data
    df_textslam = load_textslam_landmarks(args.textslam_summary)
    df_navocr = load_navocr_landmarks(args.navocr_summary, args.navocr_landmarks)
    textslam_positions = load_textslam_3d_positions(args.textslam_3d)

    # Match landmarks
    df_matches = match_landmarks(df_textslam, df_navocr, textslam_positions)

    if len(df_matches) == 0:
        print("\nERROR: No landmark matches found!")
        print("Check spatial/text similarity thresholds or input data")
        return 1

    # Compute statistics
    stats = compute_statistics(df_matches)

    # Print results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"Matched landmarks: {stats['num_matches']}")
    print(f"Weighted mean error ratio: {stats['weighted_mean_ratio']:.3f}x")
    print(f"  NavOCR:   {stats['weighted_mean_nav_error']:.2f} pixels")
    print(f"  TextSLAM: {stats['weighted_mean_ts_error']:.2f} pixels")
    print(f"Within 1.4x target: {stats['within_1_4x']}/{stats['num_matches']} ({100*stats['within_1_4x']/stats['num_matches']:.1f}%)")
    print("=" * 70)

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(df_matches, args.output_dir)

    # Save summary
    summary_file = Path(args.output_dir) / 'comparison_summary.txt'
    save_summary(stats, df_matches, summary_file)

    # Save matched pairs CSV
    matched_csv = Path(args.output_dir) / 'matched_landmarks.csv'
    df_matches.to_csv(matched_csv, index=False)
    print(f"  Saved matched pairs: {matched_csv}")

    print("\n✅ Comparison complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
