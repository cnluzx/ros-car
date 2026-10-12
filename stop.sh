#!/bin/bash
# =============================================================================
#  一键急停 (stop.sh) —— 任何终端里敲:  ~/racecar/stop.sh
#  1) 先停掉循迹节点(否则它下一帧又把速度覆盖回去)
#  2) 再向 /teleop_cmd_vel 连发 15 次急停 (电机 PWM 1500 / 舵机中位)
#  快速调用建议:  alias stop='bash ~/racecar/stop.sh'   (写进 ~/.bashrc)
# =============================================================================
pkill -f 'ros2 run lidar_tracking'              2>/dev/null
pkill -f 'lib/lidar_tracking/lidar_tracking'    2>/dev/null

source /opt/ros/humble/setup.bash
source "$HOME/racecar/install/setup.bash" 2>/dev/null

CENTER=$(grep -E '^[[:space:]]*servo_center:' \
    "$HOME/racecar/src/lidar_tracking/config/param.yaml" 2>/dev/null \
    | head -1 | awk '{print $2}')
[ -z "$CENTER" ] && CENTER=80.0

echo "[stop] 急停中: 循迹节点已停, 连发急停 (电机 1500 / 舵机 ${CENTER})"
for _ in $(seq 1 15); do
  ros2 topic pub --once /teleop_cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 1500.0}, angular: {z: ${CENTER}}}" >/dev/null 2>&1
  sleep 0.04
done
echo "[stop] 完成。(若看门狗 watchdog 在运行, 它会持续保持停车; 重新发车前启动控制器即可恢复)"
