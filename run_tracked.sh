#!/bin/bash
# =============================================================================
#  带运行记录的启动脚本 (run_tracked.sh)
#  · 启动循迹节点 (自动在 ~/racecar/logs/ 录制 CSV, 头部含本次参数快照)
#  · 你 Ctrl-C 结束之后, 自动分析本次记录并打印调参报告
#  用法:  bash ~/racecar/run_tracked.sh
#  (不想录时直接 ros2 launch ... 即可, 或用 param.yaml 里 debug_csv: 0 关闭)
# =============================================================================
source /opt/ros/humble/setup.bash
source "$HOME/racecar/install/setup.bash"

echo '[run] 开始循迹。运行记录自动进行中 (保存在 ~/racecar/logs/)。'
echo '[run] 结束时按 Ctrl-C, 会自动生成本次运行的分析报告。'
echo
ros2 launch lidar_tracking lidar_tracking.launch.py
echo
echo '[run] 已退出, 分析本次运行记录:'
echo
python3 "$HOME/racecar/src/lidar_tracking/scripts/track_report.py"
