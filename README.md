# navocr_projection

Standalone tools to project NavOCR detections to 3D world coordinates, without modifying the original NavOCR code.

## Components

### C++ Nodes (Recommended)
- **navocr_slam_cpp**: Integrated C++ node that subscribes to NavOCR detections, images, depth, and camera info to compute 3D world coordinates via rtabmap SLAM. Publishes RViz markers and saves CSV + annotated images.

### Python Nodes (Legacy)
- projector (rclpy): Subscribes to IR image, depth, CameraInfo, and /navocr/detections (JSON) to compute 3D points in map frame; publishes markers and saves outputs.
- run_navocr_ros_publisher (rclpy): Separate YOLO inference node that subscribes to the same IR image and publishes /navocr/detections. Original NavOCR remains untouched. (Alias: `navocr_publisher`)
- detector_bridge (optional): Placeholder if you want to bridge from files; not required when using run_navocr_ros_publisher.

## Inputs and Outputs

### C++ Node (navocr_slam_cpp)

Inputs:
- Image: `/camera/infra1/image_rect_raw` (sensor_msgs/Image)
- Depth: `/camera/depth/image_rect_raw` (sensor_msgs/Image, 16UC1 in mm)
- CameraInfo: `/camera/infra1/camera_info`
- Detections: `/navocr/detections` (vision_msgs/Detection2DArray)
- Odometry: `/odom` (nav_msgs/Odometry)
- TF: camera_infra1_optical_frame → camera_link → odom → map (via rtabmap)

Outputs:
- `/navocr/markers` (visualization_msgs/MarkerArray) - Green cubes and text labels in RViz
- CSV file: `results_cpp/detections_YYYYMMDD_HHMMSS.csv` with columns:
  - frame, bbox (x,y,w,h), confidence, timestamp (sec, nsec)
  - depth_m, camera (x,y,z), world (x,y,z), has_world_pos
- Images: `results_cpp/images/detection_XXXXXX.jpg` with drawn bounding boxes

**Important**: Only use detections where `has_world_pos=1` for valid 3D world coordinates:
```python
import pandas as pd
df = pd.read_csv('detections_YYYYMMDD_HHMMSS.csv')
valid_detections = df[df['has_world_pos'] == 1]
```

### Python Nodes (Legacy)

Publisher (run_navocr_ros_publisher):
- Inputs:
  - Image: `/camera/infra1/image_rect_raw` (sensor_msgs/Image, configurable `image_topic`)
  - Params: `model_path` (YOLO .pt), `publish_annotated` (bool), `save_dir` (str)
- Outputs:
  - `/navocr/detections` (std_msgs/String JSON): { stamp, frame_id, image_width, image_height, boxes[{x1,y1,x2,y2,conf,cls}] }
  - `/navocr/annotated` (sensor_msgs/Image, optional)
  - Optional saved 2D images under `save_dir`

Projector:
- Inputs:
  - IR image: `/camera/infra1/image_rect_raw`
  - Depth: `/camera/depth/image_rect_raw` (aligned)
  - CameraInfo: `/camera/infra1/camera_info`
  - Detections: `/navocr/detections` (from publisher)
  - TF: camera optical -> `map` at message timestamps; `/clock` when using rosbag
- Outputs:
  - `/navocr/points_markers` (visualization_msgs/MarkerArray) in `map`
  - `/navocr/annotated_world` (sensor_msgs/Image)
  - Files under `results/navocr_world/` (PNG + TXT)

## Install dependencies

Python deps (example):

```bash
pip install ultralytics opencv-python numpy
```

Make sure your ROS 2 workspace has cv_bridge and image_geometry installed (standard on ROS 2 Humble desktop). If using system Python, prefer a virtualenv/conda and make sure `ros2 run` resolves to the same environment.

## Build

From your workspace root (e.g., `/home/sehyeon/ros2_ws`):

```bash
colcon build --packages-select navocr_projection
source install/setup.bash
```

## Usage

### C++ Node with rtabmap SLAM (Recommended)

Full integrated system with rtabmap for accurate world coordinates:

**Terminal 1: Play rosbag with clock**
```bash
ros2 bag play ~/Downloads/rosbag2_2025_11_18-10_05_53 --clock
```

**Terminal 2: Run NavOCR detector (Python)**
```bash
cd ~/ros2_ws/src/NavOCR
conda activate navocr  # or your NavOCR environment
python3 run_navocr_ros.py
```

**Terminal 3: Launch integrated rtabmap + projection**
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch navocr_projection navocr_slam_full.launch.py
```

**Terminal 4: RViz visualization (optional)**
```bash
rviz2 --ros-args -p use_sim_time:=true
```

Results:
- CSV: `results_cpp/detections_YYYYMMDD_HHMMSS.csv`
- Images: `results_cpp/images/detection_XXXXXX.jpg`
- RViz markers: `/navocr/markers` topic

### Python Nodes (Legacy)

1) Start your RTAB-Map + sensors launch and play rosbag with `--clock`.

2) Run the standalone detector publisher (no changes to NavOCR):

```bash
ros2 run navocr_projection run_navocr_ros_publisher \
  --ros-args \
  -p image_topic:=/camera/infra1/image_rect_raw \
  -p model_path:=/path/to/nav_ocr_weight.pt \
  -p publish_annotated:=true \
  -p save_dir:=/home/sehyeon/ros2_ws/results/navocr_2d
```

3) Run the projector to compute world 3D coordinates and save annotated outputs:

```bash
ros2 run navocr_projection projector \
  --ros-args \
  -p image_topic:=/camera/infra1/image_rect_raw \
  -p depth_topic:=/camera/depth/image_rect_raw \
  -p camera_info_topic:=/camera/infra1/camera_info \
  -p detections_topic:=/navocr/detections \
  -p target_frame:=map \
  -p save_dir:=/home/sehyeon/ros2_ws/results/navocr_world
```

Behavior summary:
- Publisher runs YOLO on IR frames and publishes JSON detections (+ annotated 2D image if enabled).
- Projector consumes detections + depth, back-projects bbox centers, transforms to `map`, publishes markers and saves PNG/TXT.

## Notes

### C++ Node
- **use_sim_time**: All nodes configured with `use_sim_time:=true` for rosbag replay
- **Timestamp synchronization**: Uses image/depth buffering to match detections with correct frames (dt=0.0ms)
- **Data validation**: Filter CSV with `has_world_pos==1` to get only valid world coordinates
  - `has_world_pos=0`: rtabmap map not yet initialized (usually first few frames)
  - `has_world_pos=1`: Valid 3D world coordinates in map frame
- **World coordinate origin**: rtabmap's map frame origin is where SLAM first initialized (NOT the first detection frame)
- **Depth encoding**: 16UC1 in millimeters (RealSense D455)

### Python Nodes (Legacy)
- Ensure depth is aligned to IR. Use `use_sim_time=true` and rosbag `--clock` so TF lookups succeed at timestamps.
- If GPU is available, ultralytics will use it automatically; otherwise it will run on CPU.
