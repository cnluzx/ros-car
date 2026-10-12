#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_encoder.py —— 轮速编码器与里程计 (/encoder & /odom_combined) 标定测试工具

功能：
  1. 实时读取轮速与里程计话题 (/encoder 和 /odom_combined)
  2. 实时监测线速度 (m/s)、角速度 (rad/s)、(X, Y) 坐标轨迹
  3. 【推车测距标定助手】：按回车起点清零，手动推车前进 1.00 米，自动比对编码器脉冲距离与真实距离，计算修正系数！
"""

import math
import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from nav_msgs.msg import Odometry
except ImportError:
    print("ERROR: 未找到 ROS 2 环境。请执行: source /opt/ros/humble/setup.bash")
    sys.exit(1)


class EncoderTester(Node):
    def __init__(self):
        super().__init__('encoder_test_node')
        self.sub_enc = self.create_subscription(Odometry, '/encoder', self.cb_enc, QoSProfile(depth=10))
        self.sub_odom = self.create_subscription(Odometry, '/odom_combined', self.cb_odom, QoSProfile(depth=10))
        self.last_enc = None
        self.last_odom = None
        
        # 测距计程
        self.start_x = None
        self.start_y = None
        self.acc_dist = 0.0
        self.prev_x = None
        self.prev_y = None

    def cb_enc(self, msg: Odometry):
        v = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.last_enc = {
            'speed': v,
            'vx': msg.twist.twist.linear.x,
            'wz': msg.twist.twist.angular.z,
        }

    def cb_odom(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        if self.prev_x is not None:
            step = math.hypot(px - self.prev_x, py - self.prev_y)
            self.acc_dist += step
        self.prev_x = px
        self.prev_y = py

        v = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.last_odom = {
            'x': px,
            'y': py,
            'speed': v,
            'dist': self.acc_dist,
        }

    def reset_distance(self):
        self.acc_dist = 0.0
        if self.last_odom:
            self.start_x = self.last_odom['x']
            self.start_y = self.last_odom['y']
            self.prev_x = self.last_odom['x']
            self.prev_y = self.last_odom['y']


def distance_calibration_test(node: EncoderTester):
    print("\n" + "=" * 52)
    print("【推车测距标定助手】")
    print("=" * 52)
    print("说明：")
    print("  1. 请在地面标出起点，并用卷尺量出一条 1.00 米（或 2.00 米）直线")
    print("  2. 将车轮置于起点，按回车键开始计程")
    print("  3. 缓慢向前直线推车至终点，按回车键结束")
    print("-" * 52)
    input("对齐起点后，按回车键开始...")
    node.reset_distance()

    print("\n>>> 正在计程中... 请向前推车（按回车键结束）...")
    import select
    while True:
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            sys.stdin.readline()
            break
        d = node.acc_dist
        v = node.last_odom['speed'] if node.last_odom else 0.0
        sys.stdout.write(f"\r  当前推移距离: {d:6.3f} m  |  当前线速度: {v:5.2f} m/s  (推到终点按回车) ")
        sys.stdout.flush()

    measured = node.acc_dist
    print("\n\n" + "=" * 52)
    print(f"测量完成！编码器计程值 = {measured:.3f} 米")
    print("=" * 52)
    actual_str = input("请输入您实际测量的真实距离 (例如 1.0 或 2.0，直接回车默认为 1.0): ").strip()
    actual = float(actual_str) if actual_str else 1.0
    err = measured - actual
    err_pct = (err / actual) * 100.0
    scale = actual / max(measured, 1e-4)

    print("-" * 52)
    print(f"真实距离 : {actual:.3f} 米")
    print(f"测量距离 : {measured:.3f} 米")
    print(f"误差     : {err:+.3f} 米 ({err_pct:+.1f}%)")
    print(f"建议轮径/脉冲比例调整系数: {scale:.4f}")
    if abs(err_pct) < 3.0:
        print("评价结论 : [优良] 轮速计测距精度很高！")
    else:
        print("评价结论 : [需校准] 建议根据上述系数在驱动层调整轮径或减速比参数。")
    print("-" * 52)
    input("按回车键返回主菜单...")


def live_monitor(node: EncoderTester):
    print("\n>>> 进入里程计实时监视器（按 Ctrl+C 退出，按空格键里程清零）...")
    import termios
    import tty
    import select
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                ch = sys.stdin.read(1)
                if ch == ' ':
                    node.reset_distance()
                elif ch in ('q', 'Q', '\x03'):
                    break

            sys.stdout.write("\033[H\033[J")
            print("=" * 52)
            print("       轮速编码器与里程计实时监控 (按 Q 退出)")
            print("=" * 52)
            if node.last_enc:
                print(f"【底层轮速 (/encoder)】")
                print(f"  前向线速度 : {node.last_enc['vx']:+6.3f} m/s  (标量: {node.last_enc['speed']:5.3f} m/s)")
                print(f"  转向角速度 : {node.last_enc['wz']:+6.3f} rad/s")
            else:
                print("【底层轮速 (/encoder)】等待数据...")

            print("-" * 52)
            if node.last_odom:
                od = node.last_odom
                print(f"【位姿融合里程计 (/odom_combined)】")
                print(f"  坐标 X     : {od['x']:+7.3f} m")
                print(f"  坐标 Y     : {od['y']:+7.3f} m")
                print(f"  累计行驶距 : {od['dist']:+7.3f} m  (按空格键归零)")
                print(f"  融合线速度 : {od['speed']:+7.3f} m/s")
            else:
                print("【位姿融合里程计 (/odom_combined)】等待数据...")
            print("=" * 52)
            time.sleep(0.1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    rclpy.init()
    node = EncoderTester()
    import threading
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("\n等待里程计话题数据...")
    t0 = time.time()
    while node.last_enc is None and node.last_odom is None:
        time.sleep(0.2)
        if time.time() - t0 > 3.0:
            print("[提示] 暂未收到里程计数据，如果车辆静止请尝试推车转动车轮...")
            break

    try:
        while True:
            print("\n" + "=" * 52)
            print("        编码器与里程计测试标定工具")
            print("=" * 52)
            v = node.last_enc['speed'] if node.last_enc else 0.0
            d = node.acc_dist
            print(f"当前速度: {v:.2f} m/s  |  累计位移: {d:.3f} m")
            print("-" * 52)
            print(" [1] 实时里程计动态监视器 (速度、坐标、位移)")
            print(" [2] 推车测距标定助手 (测定1米直线行驶距离与误差)")
            print(" [3] 累计里程计清零")
            print(" [0] 退出")
            print("=" * 52)
            choice = input("请选择 [0-3]: ").strip()
            if choice == '1':
                live_monitor(node)
            elif choice == '2':
                distance_calibration_test(node)
            elif choice == '3':
                node.reset_distance()
                print("[OK] 已清零。")
            elif choice == '0':
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
