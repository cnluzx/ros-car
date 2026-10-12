import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # usb_cam 不能正确处理 /dev/v4l/by-id 符号链接，启动时解析成当前真实 /dev/videoN。
    stable_camera = '/dev/v4l/by-id/usb-LRCP_HD60fps_USB_2.0_Camera-video-index0'
    default_camera = os.path.realpath(stable_camera)
    # 声明摄像头设备参数
    video_device_arg = DeclareLaunchArgument(
        # /dev/videoN 会在重启、拔插后变化；此值由稳定 by-id 链接实时解析得到。
        'video_device', default_value=default_camera,
        description='Video device path for the USB camera'
    )

    # 启动 usb_cam 节点
    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        parameters=[{
            # 使用启动参数，默认值为当前实车检测到的视频采集节点 /dev/video2。
            'video_device': LaunchConfiguration('video_device'),
            'image_width': 320,            # 图像宽度
            'image_height': 240,           # 图像高度
            'pixel_format': 'yuyv',        # 修改为 yuyv 格式
            'camera_frame_id': 'camera'    # 摄像头帧 ID
        }],
        output='screen'
    )

    # 创建 LaunchDescription 并添加动作
    ld = LaunchDescription()
    ld.add_action(video_device_arg)
    ld.add_action(usb_cam_node)

    return ld

#查看摄像头路径：ls /dev/video*
# 安装usb_cam：
#     sudo apt update
#     sudo apt install ros-humble-usb-cam
# 启动摄像头：ros2 launch turn_on_wheeltec_robot wheeltec_camera.launch.py
# ros2 topic list
#显示画面：ros2 run rqt_image_view rqt_image_view
