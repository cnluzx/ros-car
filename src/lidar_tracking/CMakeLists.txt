# =============================================================================
# 激光雷达循迹ROS2包构建配置文件
# Python版 lidar_tracking：保留 ament_cmake 包结构，仅替换节点实现
# =============================================================================

cmake_minimum_required(VERSION 3.8)
project(lidar_tracking)

find_package(ament_cmake REQUIRED)

# 将 Python 节点仍安装为 lidar_tracking。
# 原 launch 中 package='lidar_tracking', executable='lidar_tracking' 无需修改。
install(
  PROGRAMS
    scripts/lidar_tracking.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME lidar_tracking
)

# 看门狗: 控制静默自动急停 (独立终端/nohup 运行; 跟踪节点挂掉时它负责刹车)
install(
  PROGRAMS
    scripts/teleop_watchdog.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME teleop_watchdog
)

# 安装 launch 文件目录
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}
    USE_SOURCE_PERMISSIONS
  )
endif()

# 安装 config 参数目录
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(DIRECTORY
    config
    DESTINATION share/${PROJECT_NAME}
    USE_SOURCE_PERMISSIONS
  )
endif()

# =============================================================================
# C++ 版（圆弧拟合中線）
# 独立可执行文件 lidar_tracking_arcfit —— 与上面的 Python 节点并存, 互不影响:
#   ros2 run lidar_tracking lidar_tracking_arcfit    (launch 里仍是 Python 版)
# =============================================================================
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(lidar_tracking_arcfit src/lidar_tracking_arcfit.cpp)
target_include_directories(lidar_tracking_arcfit PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
ament_target_dependencies(lidar_tracking_arcfit
  rclcpp
  sensor_msgs
  geometry_msgs
)
install(TARGETS
  lidar_tracking_arcfit
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
