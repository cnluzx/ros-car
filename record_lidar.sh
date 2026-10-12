#!/bin/bash
# =============================================================================
# record_lidar.sh —— 把激光雷达数据录制为 ros2 bag
#
# 用法：
#   ./record_lidar.sh                 一直录，按 Ctrl+C 停止
#   ./record_lidar.sh 30              录 30 秒
#   ./record_lidar.sh 30 /scan        指定时长和话题
#   ./record_lidar.sh 0 /scan         效果同第一条
#
# 输出：racecar/data/bags/lidar_YYYYmmdd_HHMMSS/
#       （保存位置固定在"脚本所在目录"下，与你在哪运行无关；
#         想换到别处用环境变量： LIDAR_OUT=/media/usb/lidar ./record_lidar.sh 30）
#
# 回放（另开终端）：
#   source /opt/ros/humble/setup.bash
#   source install/setup.bash
#   ros2 bag play data/bags/lidar_20261005_143000
#
# 提示：录制前先让雷达跑起来
#   ./car.sh                                   # 整车（含雷达）
#   或 ros2 launch lslidar_driver lslidar_launch.py
# =============================================================================

set -u

DURATION="${1:-0}"
TOPIC="${2:-/scan}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${LIDAR_OUT:-$SCRIPT_DIR/data/bags}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_ROOT/lidar_$STAMP"

mkdir -p "$OUT_ROOT"

# ---------- 环境检查 ----------
if ! command -v ros2 >/dev/null 2>&1; then
  echo "[错误] 找不到 ros2 命令。请先执行："
  echo "         source /opt/ros/humble/setup.bash"
  exit 1
fi

if ! ros2 bag --help >/dev/null 2>&1; then
  echo "[错误] ros2 bag 不可用，请安装："
  echo "         sudo apt install -y ros-humble-ros2bag ros-humble-rosbag2-storage-default-plugins"
  exit 1
fi

# ---------- 话题检查 ----------
echo "检查话题 $TOPIC ..."
if ros2 topic list 2>/dev/null | grep -qx "$TOPIC"; then
  echo "  [OK] 话题存在，等待第一帧数据 ..."
  if ! timeout 5 ros2 topic hz "$TOPIC" 2>/dev/null | grep -q "average rate"; then
    echo "  [警告] 5 秒内没收到数据。可能雷达没启动，或驱动停在未激活状态。"
    echo "         检查： ros2 topic hz $TOPIC"
    echo "         雷达未激活时先发一次： ros2 topic pub -1 /lslidar_order std_msgs/msg/Int8 data:\ 1"
  fi
else
  echo "  [警告] 话题 $TOPIC 当前不在列表中。"
  echo "         若雷达稍后才启动，录制仍会继续并自动捕获。"
fi

# ---------- 录制 ----------
echo
echo "开始录制 -> $OUT_DIR"
echo "话题：$TOPIC"
if [ "$DURATION" -gt 0 ] 2>/dev/null; then
  echo "时长：${DURATION} 秒"
else
  echo "时长：不限（Ctrl+C 停止）"
fi
echo

if [ "$DURATION" -gt 0 ] 2>/dev/null; then
  # ros2 bag record 在录制期间完全不输出任何内容（只在停止时才打印汇总），
  # 直接前台跑看起来就像卡死了。这里改成后台跑 + 自己打倒计时。
  ros2 bag record -o "$OUT_DIR" "$TOPIC" &
  BAG_PID=$!

  # Ctrl+C：把 SIGINT 转发给 ros2 bag，让它正常收尾（SIGINT 就是它设计用来停止的信号，不会损坏 bag）
  on_int() {
    echo
    echo "收到 Ctrl+C，正在停止录制并收尾 ..."
    kill -INT "$BAG_PID" 2>/dev/null
    wait "$BAG_PID" 2>/dev/null
    echo "已停止。"
    exit 130
  }
  trap on_int INT TERM

  ELAPSED=0
  while [ "$ELAPSED" -lt "$DURATION" ]; do
    if ! kill -0 "$BAG_PID" 2>/dev/null; then
      echo
      echo "[提示] 录制进程提前退出了（可能目录已存在或参数有误）。"
      break
    fi
    printf "\r  录制中  %3d / %d 秒    （Ctrl+C 可提前结束）" "$ELAPSED" "$DURATION"
    sleep 1
    ELAPSED=$((ELAPSED + 1))
  done
  printf "\r  录制中  %3d / %d 秒    时间到，正在收尾 ...\n" "$DURATION" "$DURATION"

  if kill -0 "$BAG_PID" 2>/dev/null; then
    kill -INT "$BAG_PID" 2>/dev/null
    wait "$BAG_PID" 2>/dev/null
  fi
else
  echo "按 Ctrl+C 停止"
  ros2 bag record -o "$OUT_DIR" "$TOPIC"
fi

# ---------- 结果汇总 ----------
echo
echo "======================= 录制结果 ======================="
if [ -d "$OUT_DIR" ]; then
  SIZE="$(du -sh "$OUT_DIR" 2>/dev/null | cut -f1)"
  echo "目录：$OUT_DIR"
  echo "大小：$SIZE"
  echo
  ros2 bag info "$OUT_DIR" 2>/dev/null

  echo
  echo "回放："
  echo "  ros2 bag play $OUT_DIR"
  echo "只想看数据流："
  echo "  ros2 bag play $OUT_DIR &  ros2 topic hz $TOPIC"
else
  echo "[错误] 没有生成 bag 目录，录制失败。"
  exit 1
fi
echo "======================================================="
