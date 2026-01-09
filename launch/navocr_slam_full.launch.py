#!/usr/bin/env python3
"""
NavOCR-SLAM 통합 Launch 파일
rtabmap SLAM + NavOCR detection + Landmark clustering을 실행
"""

from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    
    # Get directories
    home_dir = os.path.expanduser('~')
    output_dir = os.path.join(home_dir, 'ros2_ws', 'src', 'navocr_projection', 'results_cpp')
    navocr_script = os.path.join(home_dir, 'ros2_ws', 'src', 'NavOCR', 'run_navocr_ros_with_ocr.py')
    
    # rtabmap 파라미터
    rtabmap_parameters = [{
        'frame_id': 'camera_link',
        'subscribe_depth': True,
        'subscribe_odom_info': True,
        'approx_sync': True,  # rosbag 재생 시 필수 (메시지 동기화)
        'wait_imu_to_init': True,
        'use_sim_time': True,
    }]
    
    rtabmap_remappings = [
        ('imu', '/imu/data'),
        ('rgb/image', '/camera/infra1/image_rect_raw'),
        ('rgb/camera_info', '/camera/infra1/camera_info'),
        ('depth/image', '/camera/depth/image_rect_raw')
    ]
    
    return LaunchDescription([
        # Global parameter: use simulation time for rosbag playback
        SetParameter(name='use_sim_time', value=True),
        
        # 1. IMU 필터 (quaternion 계산)
        # RealSense IMU는 camera_gyro_frame에서 데이터 출력
        # fixed_frame을 camera_link로 설정하여 올바른 좌표계 변환 적용
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_madgwick_node',
            output='screen',
            parameters=[{
                'use_mag': False,
                'world_frame': 'enu',
                'fixed_frame': 'camera_link',  # IMU 데이터를 camera_link 기준으로 변환
                'publish_tf': False,
                'use_sim_time': True,
            }],
            remappings=[('imu/data_raw', '/camera/imu')]
        ),
        
        # 2. RGBD Odometry
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            output='screen',
            parameters=rtabmap_parameters,
            remappings=rtabmap_remappings
        ),
        
        # 3. RTABMAP SLAM (map frame 생성)
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            output='screen',
            parameters=rtabmap_parameters,
            remappings=rtabmap_remappings,
            arguments=['-d']  # Delete database on start
        ),
        
        # 4. NavOCR Python 스크립트 (OCR detection)
        ExecuteProcess(
            cmd=['/home/sehyeon/miniconda3/envs/navocr/bin/python3.10', navocr_script],
            output='screen',
            shell=False
        ),
        
        # 5. NavOCR-SLAM C++ 노드 (Landmark clustering)
        # NOTE: Parameters optimized for 0.5x playback speed
        # For 1.0x speed, use: sensor_noise=0.3, min_obs=3, merge_radius=10.0
        Node(
            package='navocr_projection',
            executable='navocr_slam_cpp',
            name='navocr_slam_cpp',
            output='screen',
            parameters=[{
                'output_dir': output_dir,
                'confidence_threshold': 0.3,
                'camera_frame': 'camera_infra1_optical_frame',
                'world_frame': 'map',  # rtabmap이 생성하는 map frame 사용
                'sensor_noise_std': 0.35,       # 0.3 → 0.35 (slower playback = more uncertainty)
                'min_observations': 4,          # 3 → 4 (require more consistent observations)
                'merge_search_radius': 11.0,    # 10.0 → 11.0 (wider search for duplicates)
                'use_sim_time': True,
            }]
        ),
    ])
