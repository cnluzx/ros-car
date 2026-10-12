#!/usr/bin/env python3
"""
激光雷达循迹ROS2包启动文件
功能：启动激光雷达循迹节点，并从参数文件加载参数
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('lidar_tracking')
    
    # 优先加载 config/param.yaml，若不存在则回退至 config/params.yaml
    params_file = os.path.join(pkg_share, 'config', 'param.yaml')
    if not os.path.exists(params_file):
        params_file = os.path.join(pkg_share, 'config', 'params.yaml')

    return LaunchDescription([
        Node(
            package='lidar_tracking',
            executable='lidar_tracking',
            name='lidar_tracking',
            output='screen',
            parameters=[params_file]
        )
    ])
