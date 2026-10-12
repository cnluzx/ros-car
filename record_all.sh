#!/usr/bin/env bash
# =============================================================================
# record_all.sh —— 一键录制小车所有调试频道数据（ROS 2 Bag）并可自动导出 CSV
#
# 用法：
#   ./record_all.sh                 # 录制所有频道（Ctrl+C 停止）
#   ./record_all.sh 30              # 录制 30 秒自动停止
#   ./record_all.sh --core          # 仅录制核心调试频道（雷达+控制+IMU+里程计+TF）
#   ./record_all.sh --all           # 强制录制系统所有正在发布的话题 (-a)
#   ./record_all.sh --csv           # 录制停止后自动调用工具导出同步 CSV/JSON
#
# 存储位置：
#   ~/racecar/data/bags/run_YYYYmmdd_HHMMSS/
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RACECAR_ROOT="$HOME/racecar"
OUT_ROOT="$RACECAR_ROOT/data/bags"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_ROOT/run_$STAMP"
DURATION=0
RECORD_MODE="smart" # smart / all / core
AUTO_CSV=0

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      RECORD_MODE="all"
      shift
      ;;
    --core)
      RECORD_MODE="core"
      shift
      ;;
    --csv)
      AUTO_CSV=1
      shift
      ;;
    [0-9]*)
      DURATION="$1"
      shift
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: ./record_all.sh [时长秒数] [--core|--all] [--csv]"
      exit 1
      ;;
  esac
done

mkdir -p "$OUT_ROOT"

# 环境检查
if ! command -v ros2 >/dev/null 2>&1; then
  if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
  else
    echo "[错误] 找不到 ROS 2 环境"
    exit 1
  fi
fi

if [ -f "$RACECAR_ROOT/install/setup.bash" ]; then
  source "$RACECAR_ROOT/install/setup.bash"
fi

# 核心话题清单（用于 --core 或过滤掉高带宽大图像话题）
CORE_TOPICS=(
  /scan
  /teleop_cmd_vel
  /car_cmd_vel
  /IMU_data
  /odom_combined
  /encoder
  /tf
  /tf_static
)

RECORD_ARGS=()

if [ "$RECORD_MODE" == "all" ]; then
  echo ">>> [模式: 全量录制] 录制系统当前及随后发布的所有频道 (-a)"
  RECORD_ARGS=(-a)
elif [ "$RECORD_MODE" == "core" ]; then
  echo ">>> [模式: 核心调试] 仅录制核心控制与传感器频道"
  # 检查存在的话题
  ACTIVE_TOPICS=()
  CURRENT_LIST="$(ros2 topic list 2>/dev/null)"
  for t in "${CORE_TOPICS[@]}"; do
    if echo "$CURRENT_LIST" | grep -qx "$t"; then
      ACTIVE_TOPICS+=("$t")
    fi
  done
  if [ ${#ACTIVE_TOPICS[@]} -eq 0 ]; then
    ACTIVE_TOPICS=("${CORE_TOPICS[@]}")
  fi
  RECORD_ARGS=("${ACTIVE_TOPICS[@]}")
else
  # smart 模式：自动搜寻系统当前所有活跃话题，但默认剔除超大图像数据避免卡死或塞满存储
  echo ">>> [模式: 智能全频录制] 录制当前所有活跃调试频道（排除未压缩的高清图像）"
  CURRENT_LIST="$(ros2 topic list 2>/dev/null)"
  DETECTED_TOPICS=()
  for t in $CURRENT_LIST; do
    # 过滤掉高带宽原始图像数据（若需录图像可加 --all）
    if [[ "$t" =~ "image_raw" || "$t" =~ "compressed" || "$t" =~ "theora" ]]; then
      continue
    fi
    DETECTED_TOPICS+=("$t")
  done
  if [ ${#DETECTED_TOPICS[@]} -eq 0 ]; then
    DETECTED_TOPICS=(-a)
  fi
  RECORD_ARGS=("${DETECTED_TOPICS[@]}")
fi

echo "=========================================================="
echo " 目标存储目录 : $OUT_DIR"
echo " 录制频道列表 : ${RECORD_ARGS[*]}"
echo " 录制持续时间 : $( [ "$DURATION" -gt 0 ] && echo "${DURATION} 秒" || echo "手动停止 (按 Ctrl+C)" )"
echo "=========================================================="

# 启动录制
if [ "$DURATION" -gt 0 ]; then
  ros2 bag record -o "$OUT_DIR" "${RECORD_ARGS[@]}" &
  BAG_PID=$!

  cleanup() {
    echo -e "\n[提示] 收到终止信号，正在安全停止并写入元数据..."
    kill -INT "$BAG_PID" 2>/dev/null || true
    wait "$BAG_PID" 2>/dev/null || true
    echo "[提示] 录制已完整保存。"
  }
  trap cleanup INT TERM

  ELAPSED=0
  while [ "$ELAPSED" -lt "$DURATION" ]; do
    if ! kill -0 "$BAG_PID" 2>/dev/null; then
      break
    fi
    printf "\r  [录制中] 已录制 %3d / %d 秒 (Ctrl+C 可提前停止)..." "$ELAPSED" "$DURATION"
    sleep 1
    ELAPSED=$((ELAPSED + 1))
  done
  printf "\r  [录制完成] 时间到，正在写入数据库...\n"
  if kill -0 "$BAG_PID" 2>/dev/null; then
    kill -INT "$BAG_PID" 2>/dev/null || true
    wait "$BAG_PID" 2>/dev/null || true
  fi
else
  echo "[提示] 开始录制，按 Ctrl+C 安全停止..."
  ros2 bag record -o "$OUT_DIR" "${RECORD_ARGS[@]}"
fi

echo
echo "======================= 录制结果汇总 ======================="
if [ -d "$OUT_DIR" ]; then
  SIZE="$(du -sh "$OUT_DIR" 2>/dev/null | cut -f1)"
  echo "存储位置 : $OUT_DIR"
  echo "数据大小 : $SIZE"
  echo "--- 包内频道详情 ---"
  ros2 bag info "$OUT_DIR" 2>/dev/null || true
  echo "=========================================================="
  echo "回放指令 : ros2 bag play $OUT_DIR"

  # 自动导出 CSV
  if [ "$AUTO_CSV" -eq 1 ] && [ -f "$RACECAR_ROOT/tools/bag_export_telemetry.py" ]; then
    echo
    echo ">>> 正在自动导出同步遥测 CSV 与雷达数据表..."
    python3 "$RACECAR_ROOT/tools/bag_export_telemetry.py" "$OUT_DIR" || true
  fi
else
  echo "[错误] 录制未能生成目标文件夹。"
  exit 1
fi
