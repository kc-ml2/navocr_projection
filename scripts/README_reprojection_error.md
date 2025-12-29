# Reprojection Error Evaluation Guide

This guide explains how to compute and compare reprojection errors between TextSLAM and NavOCR.

## Overview

The reprojection error evaluation consists of three parts:
1. **TextSLAM**: Computes reprojection error for bundle-adjusted text landmarks
2. **NavOCR**: Computes reprojection error for simple averaged text landmarks
3. **Comparison Script**: Matches landmarks and computes error ratios

## Part 1: Running TextSLAM

### Prerequisites
- TextSLAM dataset with text detections
- Compiled TextSLAM binary

### Steps

1. **Navigate to TextSLAM directory**:
```bash
cd /home/sehyeon/ros2_ws/src/TextSLAM/build
```

2. **Run TextSLAM on your dataset**:
```bash
./TextSLAM ../yaml/GeneralMotion.yaml
```

### Output Files

After running, you will find:
- `Text_reprojection_error.txt`: Per-observation errors
- `Text_reprojection_summary.txt`: Per-landmark statistics and global summary
- `Text_info.txt`: 3D landmark positions (needed for matching)
- `keyframe.txt`: Camera poses

### Example Output
```
=== TextSLAM Reprojection Error Summary ===

landmark_id,text,num_observations,mean_error,std_error,min_error,max_error
5,"화장실",12,2.34,0.85,1.12,4.23
12,"EXIT",8,1.87,0.62,0.95,3.11
...

=== Global Statistics ===
Total observations: 156
Total landmarks: 18
Overall mean error: 2.51 pixels
Weighted mean error: 2.36 pixels (weighted by sqrt(num_obs))

=== Error Distribution ===
0-1 px:   12 (7.7%)
1-2 px:   45 (28.8%)
2-3 px:   62 (39.7%)
3-4 px:   28 (17.9%)
4-5 px:   7 (4.5%)
...
Cumulative (<=3px): 76.2%
```

## Part 2: Running NavOCR

### Prerequisites
- ROS 2 workspace with navocr_projection
- NavOCR detection node running
- Depth camera and odometry data

### Steps

1. **Launch NavOCR SLAM system**:
```bash
cd /home/sehyeon/ros2_ws
ros2 launch navocr_projection navocr_slam_full.launch.py
```

2. **Play your rosbag or run live**:
```bash
ros2 bag play your_dataset.bag
```

3. **Stop the node (Ctrl+C)**: Reprojection errors are computed automatically on shutdown

### Output Files

Located in `/home/sehyeon/ros2_ws/src/navocr_projection/results_cpp/`:
- `NavOCR_reprojection_error.txt`: Per-observation errors
- `NavOCR_reprojection_summary.txt`: Per-landmark statistics and global summary
- `landmarks_YYYYMMDD_HHMMSS.csv`: 3D landmark positions with text labels

### Example Output
```
=== NavOCR Reprojection Error Summary ===

landmark_id,text,num_observations,mean_error,std_error,min_error,max_error
1,"화장실",10,3.01,1.12,1.45,5.23
2,"EXIT",7,2.35,0.89,1.12,3.98
...

=== Global Statistics ===
Total observations: 142
Total landmarks: 16
Overall mean error: 3.25 pixels
Weighted mean error: 3.12 pixels (weighted by sqrt(num_obs))

=== Error Distribution ===
0-1 px:   8 (5.6%)
1-2 px:   34 (23.9%)
2-3 px:   52 (36.6%)
3-4 px:   32 (22.5%)
...
Cumulative (<=3px): 66.2%
```

## Part 3: Comparison

### Prerequisites
- Completed TextSLAM and NavOCR runs on the **same dataset**
- Python 3 with pandas, numpy, matplotlib

### Installation
```bash
pip3 install pandas numpy matplotlib
```

### Running the Comparison

```bash
cd /home/sehyeon/ros2_ws/src/navocr_projection/scripts

python3 compare_reprojection_errors.py \
  --textslam-summary /path/to/TextSLAM/build/Text_reprojection_summary.txt \
  --textslam-3d /path/to/TextSLAM/build/Text_info.txt \
  --navocr-summary /path/to/navocr_projection/results_cpp/NavOCR_reprojection_summary.txt \
  --navocr-landmarks /path/to/navocr_projection/results_cpp/landmarks_YYYYMMDD_HHMMSS.csv \
  --output-dir ./comparison_results
```

### Example Command
```bash
python3 compare_reprojection_errors.py \
  --textslam-summary ../../TextSLAM/build/Text_reprojection_summary.txt \
  --textslam-3d ../../TextSLAM/build/Text_info.txt \
  --navocr-summary ../results_cpp/NavOCR_reprojection_summary.txt \
  --navocr-landmarks ../results_cpp/landmarks_20251229_153045.csv \
  --output-dir ./comparison_results
```

### Output Files

Located in `./comparison_results/`:
- `comparison_summary.txt`: Overall statistics and verdict
- `matched_landmarks.csv`: Matched landmark pairs with error ratios
- `scatter_plot.png`: Scatter plot of errors (NavOCR vs. TextSLAM)
- `ratio_histogram.png`: Distribution of error ratios

### Example Comparison Summary
```
======================================================================
Reprojection Error Comparison: NavOCR vs. TextSLAM
======================================================================

Matched landmarks: 16

--- Error Ratio Statistics ---
Mean ratio:           1.342x
Median ratio:         1.287x
Std ratio:            0.285
Weighted mean ratio:  1.324x

--- Absolute Errors ---
NavOCR mean error:    3.12 pixels
TextSLAM mean error:  2.36 pixels
NavOCR weighted:      3.12 pixels
TextSLAM weighted:    2.36 pixels

--- Success Criteria ---
Within 1.4x:  14/16 (87.5%)
Within 1.5x:  16/16 (100.0%)

✅ SUCCESS: NavOCR achieves >70% of TextSLAM quality
   Modular structure is viable for practical use!

======================================================================
```

## Interpretation

### Success Criteria

**Goal**: NavOCR should achieve 70-80% of TextSLAM's accuracy (error ratio 1.2-1.4x)

**Verdict**:
- ✅ **Success**: Weighted ratio ≤ 1.4x
- ⚠️ **Borderline**: Weighted ratio 1.4-1.6x
- ❌ **Needs improvement**: Weighted ratio > 1.6x

### What the Results Mean

**Error Ratio = 1.32x** means:
- NavOCR's reprojection error is 32% higher than TextSLAM's
- NavOCR achieves **76% of TextSLAM's accuracy** (1/1.32 ≈ 0.76)
- This is **acceptable** because:
  - Modular structure is simpler to develop/maintain
  - No bundle adjustment needed
  - 1-2 pixel difference is negligible for navigation

**Cumulative distributions**:
- TextSLAM: 76% under 3 pixels
- NavOCR: 66% under 3 pixels
- → Small practical difference for robot navigation

## Troubleshooting

### No Matches Found

**Possible causes**:
1. Different datasets used for TextSLAM and NavOCR
2. Coordinate frame mismatch
3. Text labels differ significantly

**Solutions**:
- Ensure both systems ran on the **exact same image sequence**
- Check that world frames are aligned
- Verify text recognition is consistent

### Too Few Matches (<50%)

**Possible causes**:
1. NavOCR detected much fewer landmarks (expected)
2. Spatial threshold too strict (default: 1.5m)
3. Text similarity threshold too strict (default: 0.6)

**Solutions**:
- This is expected! NavOCR should detect 20-30% of TextSLAM landmarks
- Adjust thresholds in the script if needed:
  ```python
  df_matches = match_landmarks(df_textslam, df_navocr, textslam_positions,
                               spatial_threshold=2.0,  # Increase if needed
                               text_sim_threshold=0.5) # Decrease if needed
  ```

### High Error Ratio (>1.6x)

**Possible causes**:
1. Poor visual odometry or TF tree
2. Insufficient observations per landmark
3. OCR confidence too low

**Solutions**:
- Check odometry quality
- Increase `min_observations` parameter in navocr_slam_node
- Improve NavOCR detection quality

## Advanced Usage

### Custom Analysis

You can load the CSV files for custom analysis:
```python
import pandas as pd

# Load matched pairs
df = pd.read_csv('comparison_results/matched_landmarks.csv')

# Filter by text type
korean_only = df[df['nav_text'].str.contains('[가-힣]', regex=True)]
print(f"Korean text error ratio: {korean_only['error_ratio'].mean():.2f}x")

# Filter by observation count
high_obs = df[df['nav_num_obs'] >= 10]
print(f"Well-observed landmarks ratio: {high_obs['error_ratio'].mean():.2f}x")
```

### Batch Processing

For multiple datasets:
```bash
for dataset in dataset1 dataset2 dataset3; do
  python3 compare_reprojection_errors.py \
    --textslam-summary ~/textslam_results/${dataset}/Text_reprojection_summary.txt \
    --textslam-3d ~/textslam_results/${dataset}/Text_info.txt \
    --navocr-summary ~/navocr_results/${dataset}/NavOCR_reprojection_summary.txt \
    --navocr-landmarks ~/navocr_results/${dataset}/landmarks.csv \
    --output-dir ./comparison_results/${dataset}
done
```

## Citation

If you use this reprojection error evaluation in your research, please cite:

```bibtex
@article{your_paper,
  title={NavOCR-Based Text Mapping with Modular SLAM Architecture},
  author={Your Name},
  journal={TBD},
  year={2025}
}
```

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: your_email@example.com
