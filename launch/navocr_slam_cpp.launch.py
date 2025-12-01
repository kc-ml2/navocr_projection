#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Launch file for C++ NavOCR-SLAM node"""
    
    # Launch arguments
    world_frame_arg = DeclareLaunchArgument(
        'world_frame',
        default_value='odom',  # Changed from 'map' to 'odom' for rosbag testing
        description='Target frame for 3D coordinates (odom or map)'
    )
    
    navocr_slam_cpp = Node(
        package='navocr_projection',
        executable='navocr_slam_cpp',
        name='navocr_slam_cpp',
        output='screen',
        parameters=[{
            'output_dir': '/home/sehyeon/ros2_ws/src/navocr_projection/results_cpp',
            'confidence_threshold': 0.3,
            'save_images': True,
            'camera_frame': 'camera_infra1_optical_frame',
            'world_frame': LaunchConfiguration('world_frame'),
        }]
    )
    
    return LaunchDescription([
        world_frame_arg,
        navocr_slam_cpp
    ])
