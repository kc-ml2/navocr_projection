#!/usr/bin/env python3
"""
NavOCR-SLAM 통합 Launch 파일
rtabmap + NavOCR detection + 3D projection을 모두 실행
"""

from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    
    # rtabmap 파라미터
    rtabmap_parameters = [{
        'frame_id': 'camera_link',
        'subscribe_depth': True,
        'subscribe_odom_info': True,
        'approx_sync': False,
        'wait_imu_to_init': True,
        'use_sim_time': True,  # rosbag 시간 사용
    }]
    
    rtabmap_remappings = [
        ('imu', '/imu/data'),
        ('rgb/image', '/camera/infra1/image_rect_raw'),
        ('rgb/camera_info', '/camera/infra1/camera_info'),
        ('depth/image', '/camera/depth/image_rect_raw')
    ]
    
    return LaunchDescription([
        # Global parameter: use simulation time
        SetParameter(name='use_sim_time', value=True),
        
        # 1. IMU 필터 (quaternion 계산)
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_madgwick_node',
            output='screen',
            parameters=[{
                'use_mag': False,
                'world_frame': 'enu',
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
        
        # 3. RTABMAP SLAM
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            output='screen',
            parameters=rtabmap_parameters,
            remappings=rtabmap_remappings,
            arguments=['-d']  # Delete database on start
        ),
        
        # 4. NavOCR-SLAM C++ 노드 (3D projection)
        Node(
            package='navocr_projection',
            executable='navocr_slam_cpp',
            name='navocr_slam_cpp',
            output='screen',
            parameters=[{
                'output_dir': '/home/sehyeon/ros2_ws/src/navocr_projection/results_cpp',
                'confidence_threshold': 0.3,
                'save_images': True,
                'camera_frame': 'camera_infra1_optical_frame',
                'world_frame': 'map',  # rtabmap이 생성하는 map 프레임 사용
                'use_sim_time': True,
            }]
        ),
    ])
