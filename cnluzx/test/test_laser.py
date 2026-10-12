#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_laser.py —— 激光雷达实时数据质量与障碍物距离检测工具

功能：
  1. 实时计算雷达帧率 (Hz)、总光束数、有效回波点数百分比
  2. 实时分方位监测障碍物距离：[左侧 90°] [左前 45°] [正前方 0°] [右前 45°] [右侧 90°]
  3. 实时追踪全周最近障碍物的位置与角度
  4. 终端实时动态刷新仪表盘（按 Ctrl+C 退出）
"""

import math
import sys
import time

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import LaserScan
except ImportError:
    print("ERROR: 未找到 ROS 2 环境。请执行: source /opt/ros/humble/setup.bash")
    sys.exit(1)


class LidarTester(Node):
    def __init__(self):
        super().__init__('lidar_test_node')
        self.sub = self.create_subscription(LaserScan, '/scan', self.cb_scan, QoSProfile(depth=5))
        self.last_scan = None
        self.scan_count = 0
        self.t_start = time.time()
        self.last_t = time.time()
        self.fps = 0.0

    def cb_scan(self, msg: LaserScan):
        now = time.time()
        dt = now - self.last_t
        self.last_t = now
        self.scan_count += 1
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * instant_fps if self.fps > 0 else instant_fps

        r = np.array(msg.ranges, dtype=np.float32)
        valid_mask = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        
        # 寻找最近点
        if np.any(valid_mask):
            valid_idx = np.flatnonzero(valid_mask)
            min_i = valid_idx[np.argmin(r[valid_mask])]
            min_dist = float(r[min_i])
            min_ang_rad = msg.angle_min + min_i * msg.angle_increment
            min_ang_deg = math.degrees(min_ang_rad)
        else:
            min_dist = 99.0
            min_ang_deg = 0.0

        # 分区统计扇区距离函数
        def get_sector_dist(center_deg, span_deg=15.0):
            th = msg.angle_min + np.arange(len(r)) * msg.angle_increment
            deg = np.degrees(th)
            # 处理 0~360 与 -180~180 的周期
            diff = (deg - center_deg + 180) % 360 - 180
            sector = (np.abs(diff) <= span_deg) & valid_mask
            if np.any(sector):
                return float(np.min(r[sector]))
            return 99.0

        fwd_dist = get_sector_dist(0.0)      # 正前
        fl_dist  = get_sector_dist(45.0)     # 左前
        fr_dist  = get_sector_dist(-45.0)    # 右前
        left_dist = get_sector_dist(90.0)    # 左侧
        right_dist = get_sector_dist(-90.0)  # 右侧

        self.last_scan = {
            'beams': len(r),
            'valid_count': int(np.sum(valid_mask)),
            'valid_ratio': float(np.mean(valid_mask) * 100.0),
            'min_dist': min_dist,
            'min_ang_deg': min_ang_deg,
            'fwd': fwd_dist,
            'fl': fl_dist,
            'fr': fr_dist,
            'left': left_dist,
            'right': right_dist,
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
        }


def format_dist(d: float) -> str:
    if d > 20.0 or d >= 99.0:
        return " > 20m  "
    return f"{d:5.2f} m "


def main():
    rclpy.init()
    node = LidarTester()
    import threading
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("\n" + "=" * 56)
    print("           激光雷达 (/scan) 实时诊断仪表盘")
    print("=" * 56)
    print("正在等待 /scan 数据（若5秒无数据，请检查雷达驱动或是否发送激活指令）...")

    try:
        wait_t = 0
        while node.last_scan is None:
            time.sleep(0.5)
            wait_t += 0.5
            if wait_t > 4.0 and wait_t % 3.0 == 0:
                print("[提示] 仍未收到数据。如雷达未激活，可另开终端执行: ros2 topic pub -1 /lslidar_order std_msgs/msg/Int8 data:\\ 1")

        while True:
            s = node.last_scan
            fps_str = f"{node.fps:4.1f} Hz" if node.fps > 0 else "--- Hz"
            
            # 清屏与实时刷新显示
            sys.stdout.write("\033[H\033[J")
            print("=" * 56)
            print(f"       激光雷达实时数据监控 (按 Ctrl+C 退出)")
            print("=" * 56)
            print(f"扫描频率 : {fps_str}  |  总光束数 : {s['beams']} 束")
            print(f"有效回波 : {s['valid_count']} 点 ({s['valid_ratio']:.1f}%)")
            print("-" * 56)
            print("【全周各扇区最近距离 (±15° 窗口)】")
            print(f"                  [ 正前方 0° ]")
            print(f"                  {format_dist(s['fwd'])}")
            print(f"     [ 左前 45° ]             [ 右前 -45° ]")
            print(f"      {format_dist(s['fl'])}              {format_dist(s['fr'])}")
            print(f"  [ 左侧 90° ]                   [ 右侧 -90° ]")
            print(f"   {format_dist(s['left'])}                    {format_dist(s['right'])}")
            print("-" * 56)
            print(f"【最近障碍物追踪】")
            print(f"  距离: {s['min_dist']:.3f} m  |  方位角: {s['min_ang_deg']:+6.1f}°")
            print("=" * 56)
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n\n[*] 退出雷达检测。")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
