#!/bin/bash
# =============================================================================
#  run_all.sh — 一键启动 ROS2 小车 (硬件 + 雷达循迹 + 看门狗 + 可视化面板)
# =============================================================================
#  启动顺序 (刻意让看门狗先于循迹就绪, 这样车一动就有急停保护):
#      1) 硬件      ros2 launch racecar Run_car.launch.py
#      2) 看门狗    teleop_watchdog.py         (话题静默 -> 连发急停)
#      3) 循迹      ros2 launch lidar_tracking lidar_tracking.launch.py
#      4) 可视化    cone.py                    (Web 面板)
#
#  用法:
#      bash ~/racecar/run_all.sh                 # 全部启动
#      bash ~/racecar/run_all.sh --no-hw         # 硬件已单独启动, 其余照常
#      bash ~/racecar/run_all.sh --no-watchdog   # 不启动看门狗 (不推荐)
#      bash ~/racecar/run_all.sh --no-cone       # 不启动可视化面板
#      bash ~/racecar/run_all.sh --port 8081     # 指定面板端口 (默认 8080)
#      bash ~/racecar/run_all.sh --delay 5       # 循迹启动前额外等待秒数 (默认 3)
#      bash ~/racecar/run_all.sh --check         # 只检查环境(source ROS), 不启动任何进程
#
#  Ctrl-C 会停掉本脚本启动的所有子进程, 并补发一次底盘急停。
#  实时日志:  tail -f ~/racecar/logs/run_all_*/lidar_tracking.log
# =============================================================================

# 注意: 不能开 set -u (nounset)。ROS 的 /opt/ros/humble/setup.bash 会引用
# 未定义变量 (如 AMENT_TRACE_SETUP_FILES)，开了 set -u 直接报错退出。

WS="$HOME/racecar"
TOPIC="/teleop_cmd_vel"
SERVO_CENTER=82.5
PORT=8080
DELAY=3
START_HW=1
START_WD=1
START_CONE=1
CHECK_ONLY=0

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --no-hw)        START_HW=0 ;;
    --no-watchdog)  START_WD=0 ;;
    --no-cone)      START_CONE=0 ;;
    --check)        CHECK_ONLY=1 ;;
    --port)         shift; PORT="${1:-8080}" ;;
    --delay)        shift; DELAY="${1:-3}" ;;
    -h|--help)      usage ;;
    *) echo "[run_all] 未知参数: $1 (用 --help 查看)"; exit 1 ;;
  esac
  shift
done

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

if [ "${CHECK_ONLY:-0}" = 1 ]; then
  echo "[run_all] 环境检查通过: ROS + $WS/install 已 source, 未启动任何进程"
  echo "[run_all]   ros2: $(command -v ros2)"
  echo "[run_all]   lidar_tracking 可执行: $(ls "$WS/install/lidar_tracking/lib/lidar_tracking/lidar_tracking" 2>/dev/null || echo '缺失')"
  exit 0
fi

LOGDIR="$WS/logs/run_all_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

PIDS=()
NAMES=()

add_proc() {   # add_proc <名字> <pid>
  NAMES+=("$1")
  PIDS+=("$2")
}

cleanup() {
  echo
  echo "[run_all] 收到退出信号, 正在停止 (先停循迹 -> 回中, 看门狗兜底急停) ..."
  # 1) 先给循迹 SIGINT, 让它主动回中并退出
  for i in "${!PIDS[@]}"; do
    case "${NAMES[$i]}" in
      lidar_tracking) kill -INT "${PIDS[$i]}" 2>/dev/null ;;
    esac
  done
  sleep 1
  # 2) 其余进程 SIGINT (看门狗会连发急停)
  for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null; done
  sleep 1
  # 3) 兜底 TERM / KILL
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
  sleep 1
  for p in "${PIDS[@]}"; do kill -KILL "$p" 2>/dev/null; done
  # 4) 向底盘补发一次急停 (中位/停车)
  timeout 3 ros2 topic pub --once "$TOPIC" geometry_msgs/msg/Twist \
      "{linear: {x: 1500.0}, angular: {z: $SERVO_CENTER}}" >/dev/null 2>&1 || true
  echo "[run_all] 已全部停止。日志目录: $LOGDIR"
  exit 0
}
trap cleanup INT TERM

echo "============================================================"
echo "[run_all] 日志目录: $LOGDIR"
echo "============================================================"

# ---- 1) 硬件 ----
if [ "$START_HW" = 1 ]; then
  if pgrep -f "[r]acecar_driver_node" >/dev/null 2>&1; then
    echo "[run_all] 检测到 racecar_driver 已在运行 -> 跳过硬件启动 (用 --no-hw 可显式跳过)"
  else
    echo "[run_all] 启动硬件: Run_car.launch.py"
    ros2 launch racecar Run_car.launch.py > "$LOGDIR/hardware.log" 2>&1 &
    add_proc hardware $!
    echo "[run_all] 等待雷达/驱动就绪 (8s) ..."
    sleep 8
  fi
fi

# ---- 2) 看门狗 (先于循迹, 保证车一动就有急停) ----
if [ "$START_WD" = 1 ]; then
  if pgrep -f "[t]eleop_watchdog" >/dev/null 2>&1; then
    echo "[run_all] 看门狗已在运行 -> 跳过"
  else
    echo "[run_all] 启动看门狗: teleop_watchdog.py"
    python3 "$WS/src/lidar_tracking/scripts/teleop_watchdog.py" \
        > "$LOGDIR/watchdog.log" 2>&1 &
    add_proc watchdog $!
    sleep 1
  fi
fi

# ---- 3) 循迹 (延迟几秒, 给你时间确认车已就位/架空) ----
if [ "${DELAY:-0}" -gt 0 ] 2>/dev/null; then
  echo "[run_all] ${DELAY}s 后启动雷达循迹 (车将开始行驶, Ctrl-C 可取消) ..."
  for i in $(seq "$DELAY" -1 1); do printf '\r[run_all] %s ... ' "$i"; sleep 1; done
  printf '\r'
fi

echo "[run_all] 启动雷达循迹: lidar_tracking.launch.py"
ros2 launch lidar_tracking lidar_tracking.launch.py \
    > "$LOGDIR/lidar_tracking.log" 2>&1 &
add_proc lidar_tracking $!
sleep 1

# ---- 4) 可视化面板 ----
if [ "$START_CONE" = 1 ]; then
  echo "[run_all] 启动可视化面板: cone.py (http://<车IP>:$PORT)"
  python3 "$WS/src/lidar_tracking/cone.py" --port "$PORT" \
      > "$LOGDIR/cone.log" 2>&1 &
  add_proc cone $!
fi

IPADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$IPADDR" ] && IPADDR="<车IP>"

cat <<EOF

============================================================
[run_all] 全部已启动:
    硬件    : $([ "$START_HW" = 1 ] && echo "Run_car.launch.py (或已在运行)" || echo "(跳过)")
    看门狗  : $([ "$START_WD" = 1 ] && echo "teleop_watchdog.py (25Hz 急停)" || echo "(跳过)")
    循迹    : lidar_tracking.launch.py
    面板    : $([ "$START_CONE" = 1 ] && echo "http://$IPADDR:$PORT" || echo "(跳过)")
    日志    : $LOGDIR
------------------------------------------------------------
  实时看循迹日志:  tail -f $LOGDIR/lidar_tracking.log
  急停 (另开终端): bash $WS/stop.sh
  Ctrl-C          : 一键停止并回中
============================================================
EOF

wait
