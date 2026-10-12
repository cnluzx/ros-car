#!/usr/bin/env python3
"""激光雷达锥桶循迹 ROS2 启动文件 (v7)。

参数加载优先级 (与节点内逻辑一致, 唯一入口是 config/param.yaml):
    1) 环境变量 LIDAR_TRACKING_PARAM 指定的文件
    2) 工作空间源码  ~/racecar/src/lidar_tracking/config/param.yaml   ← 首选
    3) 安装目录副本  <install>/lidar_tracking/share/lidar_tracking/config/param.yaml
    4) 旧命名        .../config/params.yaml

这样调参只改源码里的 config/param.yaml, 重启节点即可生效, 不用重装/重编译。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def _find_params_file():
    pkg_share = get_package_share_directory('lidar_tracking')
    home = os.path.expanduser('~')
    candidates = [
        os.environ.get('LIDAR_TRACKING_PARAM', ''),
        os.path.join(home, 'racecar', 'src', 'lidar_tracking', 'config', 'param.yaml'),
        os.path.join(pkg_share, 'config', 'param.yaml'),
        os.path.join(pkg_share, 'config', 'params.yaml'),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def generate_launch_description():
    params_file = _find_params_file()

    node_kwargs = dict(
        package='lidar_tracking',
        executable='lidar_tracking',
        name='lidar_tracking',
        output='screen',
    )
    if params_file:
        node_kwargs['parameters'] = [params_file]
        print('[lidar_tracking.launch] 使用参数文件: %s' % params_file)
    else:
        print('[lidar_tracking.launch] 警告: 未找到 param.yaml, 将使用代码内置默认值')

    return LaunchDescription([Node(**node_kwargs)])
