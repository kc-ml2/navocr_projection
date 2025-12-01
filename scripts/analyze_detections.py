#!/usr/bin/env python3
"""
Analyze NavOCR SLAM detection results from CSV.
Usage: python3 analyze_detections.py <csv_file>
"""

import sys
import pandas as pd
import numpy as np


def analyze_detections(csv_path):
    """Analyze detection CSV file and print statistics."""
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"📊 Total detections: {len(df)}")
    
    # Filter valid detections (has_world_pos == 1)
    valid_df = df[df['has_world_pos'] == 1]
    invalid_df = df[df['has_world_pos'] == 0]
    
    print(f"✅ Valid detections (has_world_pos=1): {len(valid_df)}")
    print(f"❌ Invalid detections (has_world_pos=0): {len(invalid_df)}")
    
    if len(invalid_df) > 0:
        print(f"\n⚠️  Invalid frames: {invalid_df['frame'].tolist()}")
        print("   (rtabmap map not yet initialized)")
    
    if len(valid_df) == 0:
        print("\n⚠️  No valid detections found!")
        return
    
    # Statistics for valid detections
    print("\n📍 Valid Detection Statistics:")
    print(f"   Frames: {valid_df['frame'].min()} to {valid_df['frame'].max()}")
    print(f"   Average confidence: {valid_df['confidence'].mean():.3f}")
    print(f"   Depth range: {valid_df['depth_m'].min():.2f}m to {valid_df['depth_m'].max():.2f}m")
    
    # World coordinates range
    print("\n🌍 World Coordinates Range:")
    print(f"   X: {valid_df['world_x'].min():.2f} to {valid_df['world_x'].max():.2f}")
    print(f"   Y: {valid_df['world_y'].min():.2f} to {valid_df['world_y'].max():.2f}")
    print(f"   Z: {valid_df['world_z'].min():.2f} to {valid_df['world_z'].max():.2f}")
    
    # Group by spatial clusters (simple binning)
    print("\n📦 Detection Groups (spatial clustering):")
    
    # Use simple K-means-like grouping by world coordinates
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    
    world_coords = valid_df[['world_x', 'world_y', 'world_z']].values
    
    if len(world_coords) > 1:
        # Hierarchical clustering with 0.5m threshold
        distances = pdist(world_coords)
        if len(distances) > 0:
            linkage_matrix = linkage(world_coords, method='average')
            clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')
            
            for cluster_id in np.unique(clusters):
                cluster_df = valid_df[clusters == cluster_id]
                avg_pos = cluster_df[['world_x', 'world_y', 'world_z']].mean()
                print(f"\n   Group {cluster_id}: {len(cluster_df)} detections")
                print(f"      Center: ({avg_pos['world_x']:.2f}, {avg_pos['world_y']:.2f}, {avg_pos['world_z']:.2f})")
                print(f"      Frames: {cluster_df['frame'].tolist()}")
                print(f"      Avg confidence: {cluster_df['confidence'].mean():.3f}")
    
    # Save valid detections only
    output_path = csv_path.replace('.csv', '_valid.csv')
    valid_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved valid detections to: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_detections.py <csv_file>")
        print("\nExample:")
        print("  python3 analyze_detections.py results_cpp/detections_20251201_175709.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        analyze_detections(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
