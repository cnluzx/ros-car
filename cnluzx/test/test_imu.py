#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_imu.py —— IMU (惯导陀螺仪与加速度计) 实时监测与零偏漂移测试工具

功能：
  1. 实时读取 /IMU_data 话题
  2. 实时四元数解算欧拉角：航向角 (Yaw), 俯仰角 (Pitch), 横滚角 (Roll)
  3. 实时显示三轴角速度 (Gyro) 与三轴线性加速度 (Acc)
  4. 支持【静止零偏漂移标定测试】：静止5秒，计算Yaw漂移率 (deg/min)
  5. 支持【当前航向一键清零】：方便手动旋转小车 90°/180° 检验角度精度
"""

import math
import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import Imu
except ImportError:
    print("ERROR: 未找到 ROS 2 环境。请执行: source /opt/ros/humble/setup.bash")
    sys.exit(1)


def euler_from_quaternion(x, y, z, w):
    """四元数转欧拉角 (返回 roll, pitch, yaw 单位：度)"""
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    return roll, pitch, yaw


class ImuTester(Node):
    def __init__(self):
        super().__init__('imu_test_node')
        self.sub = self.create_subscription(Imu, '/IMU_data', self.cb_imu, QoSProfile(depth=10))
        self.last_imu = None
        self.msg_count = 0
        self.last_t = time.time()
        self.fps = 0.0
        self.yaw_offset = 0.0

    def cb_imu(self, msg: Imu):
        now = time.time()
        dt = now - self.last_t
        self.last_t = now
        self.msg_count += 1
        if dt > 0:
            instant = 1.0 / dt
            self.fps = 0.95 * self.fps + 0.05 * instant if self.fps > 0 else instant

        q = msg.orientation
        roll, pitch, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)

        # 减去用户设定的相对零点
        rel_yaw = (yaw - self.yaw_offset + 180.0) % 360.0 - 180.0

        self.last_imu = {
            'yaw': rel_yaw,
            'raw_yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'gyro_x': math.degrees(msg.angular_velocity.x),
            'gyro_y': math.degrees(msg.angular_velocity.y),
            'gyro_z': math.degrees(msg.angular_velocity.z),
            'acc_x': msg.linear_acceleration.x,
            'acc_y': msg.linear_acceleration.y,
            'acc_z': msg.linear_acceleration.z,
        }

    def zero_yaw(self):
        if self.last_imu:
            self.yaw_offset = self.last_imu['raw_yaw']


def static_drift_test(node: ImuTester, duration_s: float = 6.0):
    print("\n" + "=" * 52)
    print(f"【静止零偏测试】请保持小车绝对静止，测试将持续 {duration_s:.0f} 秒...")
    print("=" * 52)
    time.sleep(1.0)
    yaws = []
    gyros = []
    t0 = time.time()
    while time.time() - t0 < duration_s:
        if node.last_imu:
            yaws.append((time.time(), node.last_imu['raw_yaw']))
            gyros.append(node.last_imu['gyro_z'])
        rem = duration_s - (time.time() - t0)
        sys.stdout.write(f"\r  采样中... 剩余 {rem:4.1f} 秒 ")
        sys.stdout.flush()
        time.sleep(0.05)

    if len(yaws) < 10:
        print("\n[!] 采样点不足，请检查 IMU 话题是否正常。")
        return

    dt = yaws[-1][0] - yaws[0][0]
    dyaw = (yaws[-1][1] - yaws[0][1] + 180.0) % 360.0 - 180.0
    drift_rate_sec = dyaw / max(dt, 1e-4)
    drift_rate_min = drift_rate_sec * 60.0
    mean_gyro = sum(gyros) / len(gyros)

    print("\n" + "-" * 52)
    print(f"测试时长       : {dt:.2f} 秒")
    print(f"航向角度总漂移 : {dyaw:+6.2f}°")
    print(f"折算漂移速率   : {drift_rate_min:+6.2f}° / 分钟")
    print(f"Z轴陀螺仪均值  : {mean_gyro:+6.3f}°/s")
    if abs(drift_rate_min) < 3.0:
        print("评价结论       : [优良] 零偏极小，适合高精度建图与循迹。")
    elif abs(drift_rate_min) < 10.0:
        print("评价结论       : [正常] 轻微温漂，短距离循迹无影响。")
    else:
        print("评价结论       : [偏大] 漂移较明显，建议校准 IMU 或检查附近电机磁场干扰。")
    print("-" * 52)
    input("按回车键返回主菜单...")


def live_monitor(node: ImuTester):
    print("\n>>> 进入实时监视器（按 Ctrl+C 退出返回菜单，按空格键可重设当前航向为 0°）...")
    import termios
    import tty
    import select
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            # 检查是否有按键
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                ch = sys.stdin.read(1)
                if ch == ' ':
                    node.zero_yaw()
                elif ch in ('q', 'Q', '\x03'):
                    break

            if node.last_imu:
                im = node.last_imu
                fps_str = f"{node.fps:4.1f} Hz" if node.fps > 0 else "--- Hz"
                sys.stdout.write("\033[H\033[J")
                print("=" * 52)
                print(f"         IMU 实时姿态数据监视 (按 Q 退出)")
                print("=" * 52)
                print(f"采样刷新率 : {fps_str}")
                print(f"【欧拉角姿态 (度)】")
                print(f"  航向角 (Yaw)   : {im['yaw']:+7.2f}°   (按空格键将当前重设为 0°)")
                print(f"  俯仰角 (Pitch) : {im['pitch']:+7.2f}°")
                print(f"  横滚角 (Roll)  : {im['roll']:+7.2f}°")
                print("-" * 52)
                print(f"【角速度 Gyro (°/s)】")
                print(f"  X: {im['gyro_x']:+6.2f}  |  Y: {im['gyro_y']:+6.2f}  |  Z(自转): {im['gyro_z']:+6.2f}")
                print(f"【线加速度 Acc (m/s²)】")
                print(f"  X: {im['acc_x']:+6.2f}  |  Y: {im['acc_y']:+6.2f}  |  Z: {im['acc_z']:+6.2f}")
                print("=" * 52)
            time.sleep(0.1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    rclpy.init()
    node = ImuTester()
    import threading
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("\n等待 /IMU_data 话题数据...")
    while node.last_imu is None:
        time.sleep(0.2)

    try:
        while True:
            print("\n" + "=" * 52)
            print("          IMU (/IMU_data) 诊断与标定工具")
            print("=" * 52)
            im = node.last_imu
            print(f"当前航向角 (Yaw): {im['yaw']:+6.1f}° | 自转角速度: {im['gyro_z']:+5.2f}°/s")
            print("-" * 52)
            print(" [1] 实时姿态动态监视器 (显示三轴姿态/角速度/加速度)")
            print(" [2] 静止零偏漂移测试 (静止6秒计算飘移率)")
            print(" [3] 将当前航向清零 (设为 0.0°，便于测试手动旋转转角)")
            print(" [0] 退出")
            print("=" * 52)
            choice = input("请选择 [0-3]: ").strip()
            if choice == '1':
                live_monitor(node)
            elif choice == '2':
                static_drift_test(node)
            elif choice == '3':
                node.zero_yaw()
                print("[OK] 当前航向已重置为 0.0°。")
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
