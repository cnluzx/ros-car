#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bag_export_telemetry.py —— 将 ROS 2 bag 中的所有调试频道导出为多维同步 CSV 与 JSON

导出的文件：
1. telemetry_sync.csv : 控制(Twist) + 姿态(IMU) + 里程计(Odom) + 雷达最小距离的同步时序总表
2. scan_wide.csv      : 宽表雷达点云（每束距离值一列，支持 Excel/Numpy 直接分析）
3. summary.json       : 录制全周期的统计摘要（持续时间、平均频率、极值等）
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import LaserScan, Imu
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
except ImportError:
    print("[错误] 未找到 ROS 2 Python 库。请先执行：source /opt/ros/humble/setup.bash")
    sys.exit(1)


def euler_from_quaternion(x, y, z, w):
    """四元数转航向角 (yaw in degrees)。"""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def parse_bag(bag_path, storage_id='sqlite3'):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
        rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    )

    topics_meta = {t.name: t.type for t in reader.get_all_topics_and_types()}
    print(f"[读取] Bag 中包含的话题: {list(topics_meta.keys())}")

    scans = []       # (t, ranges, min_r, front_r)
    scan_meta = None
    cmds = []        # (t, speed, angle)
    imus = []        # (t, yaw, gyro_z, acc_x)
    odoms = []       # (t, x, y, speed, yaw)

    t0 = None

    while reader.has_next():
        topic, data, ts_raw = reader.read_next()
        t_sec = ts_raw * 1e-9
        if t0 is None:
            t0 = t_sec
        rel_t = round(t_sec - t0, 4)

        if topic == '/scan':
            msg = deserialize_message(data, LaserScan)
            r = np.array(msg.ranges, dtype=np.float32)
            valid = r[(r >= msg.range_min) & (r <= msg.range_max) & np.isfinite(r)]
            min_r = float(valid.min()) if len(valid) > 0 else 99.0
            
            # 正前方 ±15 度内的最近距离
            n_beams = len(r)
            if n_beams > 0:
                mid = n_beams // 2
                span = int(math.radians(15.0) / max(msg.angle_increment, 1e-4))
                fwd_slice = r[max(0, mid - span): min(n_beams, mid + span)]
                valid_fwd = fwd_slice[(fwd_slice >= msg.range_min) & (fwd_slice <= msg.range_max) & np.isfinite(fwd_slice)]
                front_r = float(valid_fwd.min()) if len(valid_fwd) > 0 else 99.0
            else:
                front_r = 99.0

            scans.append((rel_t, r, min_r, front_r))
            if scan_meta is None:
                scan_meta = {
                    'angle_min': msg.angle_min,
                    'angle_max': msg.angle_max,
                    'angle_increment': msg.angle_increment,
                    'range_min': msg.range_min,
                    'range_max': msg.range_max,
                    'beams': n_beams
                }

        elif topic in ('/teleop_cmd_vel', '/car_cmd_vel', '/cmd_vel'):
            msg = deserialize_message(data, Twist)
            cmds.append((rel_t, msg.linear.x, msg.angular.z))

        elif topic == '/IMU_data':
            msg = deserialize_message(data, Imu)
            q = msg.orientation
            yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
            imus.append((rel_t, yaw, msg.angular_velocity.z, msg.linear_acceleration.x))

        elif topic in ('/odom_combined', '/odom'):
            msg = deserialize_message(data, Odometry)
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
            v = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
            odoms.append((rel_t, p.x, p.y, v, yaw))

    return scans, scan_meta, cmds, imus, odoms


def export_files(out_dir, scans, scan_meta, cmds, imus, odoms):
    os.makedirs(out_dir, exist_ok=True)

    # 1. 导出 telemetry_sync.csv
    # 建立统一时间轴（以最密集的雷达或控制指令为基准）
    time_series = sorted(list(set([s[0] for s in scans] + [c[0] for c in cmds])))
    
    sync_csv_path = os.path.join(out_dir, 'telemetry_sync.csv')
    with open(sync_csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'rel_time_s', 'cmd_speed_pwm', 'cmd_servo_deg', 
            'imu_yaw_deg', 'imu_gyro_z_rad_s', 'imu_acc_x_m_s2',
            'odom_x_m', 'odom_y_m', 'odom_speed_m_s', 'odom_yaw_deg',
            'lidar_min_range_m', 'lidar_front_range_m'
        ])

        # 辅助快速查找
        c_idx = i_idx = o_idx = s_idx = 0
        last_c = (0, 0.0, 82.5)
        last_i = (0, 0.0, 0.0, 0.0)
        last_o = (0, 0.0, 0.0, 0.0, 0.0)
        last_s = (0, [], 99.0, 99.0)

        for t in time_series:
            while c_idx < len(cmds) and cmds[c_idx][0] <= t:
                last_c = cmds[c_idx]; c_idx += 1
            while i_idx < len(imus) and imus[i_idx][0] <= t:
                last_i = imus[i_idx]; i_idx += 1
            while o_idx < len(odoms) and odoms[o_idx][0] <= t:
                last_o = odoms[o_idx]; o_idx += 1
            while s_idx < len(scans) and scans[s_idx][0] <= t:
                last_s = scans[s_idx]; s_idx += 1

            w.writerow([
                f"{t:.4f}",
                f"{last_c[1]:.1f}", f"{last_c[2]:.2f}",
                f"{last_i[1]:.2f}", f"{last_i[2]:.4f}", f"{last_i[3]:.3f}",
                f"{last_o[1]:.3f}", f"{last_o[2]:.3f}", f"{last_o[3]:.2f}", f"{last_o[4]:.2f}",
                f"{last_s[2]:.3f}", f"{last_s[3]:.3f}"
            ])
    print(f"  [OK] 同步遥测表已保存: {sync_csv_path}")

    # 2. 导出 scan_wide.csv
    if scans and scan_meta:
        scan_csv_path = os.path.join(out_dir, 'scan_wide.csv')
        beams = scan_meta['beams']
        with open(scan_csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            header = ['rel_time_s'] + [f'r{i}' for i in range(beams)]
            w.writerow(header)
            for s in scans:
                row = [f"{s[0]:.4f}"] + [f"{val:.3f}" if np.isfinite(val) else "" for val in s[1]]
                w.writerow(row)
        print(f"  [OK] 雷达宽表已保存: {scan_csv_path}")

    # 3. 导出 summary.json
    summary_path = os.path.join(out_dir, 'summary.json')
    duration = time_series[-1] if time_series else 0.0
    summary_data = {
        'total_duration_s': duration,
        'frame_counts': {
            'scan': len(scans),
            'cmd_vel': len(cmds),
            'imu': len(imus),
            'odom': len(odoms),
        },
        'scan_meta': scan_meta,
        'summary_notes': '由 bag_export_telemetry.py 自动生成'
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"  [OK] 概要信息已保存: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='ROS 2 Bag 全频道数据导出工具')
    parser.add_argument('bag_path', help='Bag 目录路径')
    parser.add_argument('-o', '--out', default=None, help='导出目标文件夹（默认保存在 bag 目录下的 csv/ 子目录中）')
    args = parser.parse_args()

    bag = os.path.abspath(args.bag_path)
    if not os.path.isdir(bag):
        print(f"[错误] 指定路径不是目录: {bag}")
        sys.exit(1)

    out = args.out or os.path.join(bag, 'csv')
    print(f">>> 开始解析 Bag: {bag}")
    scans, scan_meta, cmds, imus, odoms = parse_bag(bag)
    print(f">>> 解析完成: Scan={len(scans)}帧, Cmd={len(cmds)}帧, IMU={len(imus)}帧, Odom={len(odoms)}帧")
    export_files(out, scans, scan_meta, cmds, imus, odoms)
    print(f">>> 全部完成！数据保存在: {out}")


if __name__ == '__main__':
    main()
