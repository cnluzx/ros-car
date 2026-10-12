#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_motor.py —— 驱动电机 (电调 ESC) 安全测试工具

控制模型 (teleop_cmd_vel)：
  小车底盘驱动 racecar_driver_node 在 /teleop_cmd_vel 话题下，
  将 twist.linear.x 作为无损原始 PWM (μs) 直接发送给电调：
    - PWM = 1500.0 : 停止 (中立驻车)
    - PWM = 1525.0 : 低速蠕行 (适合架空/离地测试)
    - PWM = 1540.0 : 常规巡航速度 (循迹推荐基准)
    - PWM = 1560.0 : 较快车速
    - PWM = 1475.0 : 倒车 (部分双向电调需中位双推触发)

安全提示：
  测试电机前强烈建议【将小车车身垫起，使驱动轮离地悬空】！
"""

import sys
import time

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
except ImportError:
    print("ERROR: 未找到 ROS 2 环境。请执行: source /opt/ros/humble/setup.bash")
    sys.exit(1)

CONFIG = {
    'topic': '/teleop_cmd_vel',
    'servo_center': 82.5,     # 电机测试时保持舵机正中
    'pwm_stop': 1500.0,       # 停止 PWM
    'pwm_slow': 1525.0,       # 低速 PWM
    'pwm_mid': 1540.0,        # 中速 PWM
    'pwm_fast': 1560.0,       # 高速 PWM
    'pwm_safe_max': 1580.0,   # 测试模式下的硬保护上限
    'pwm_safe_min': 1450.0,   # 测试模式下的硬保护下限
    'pub_rate_hz': 20.0,
}


class MotorTester(Node):
    def __init__(self):
        super().__init__('motor_test_node')
        self.pub = self.create_publisher(Twist, CONFIG['topic'], QoSProfile(depth=5))
        self.speed_pwm = CONFIG['pwm_stop']
        self.twist = Twist()
        timer_period = 1.0 / CONFIG['pub_rate_hz']
        self.timer = self.create_timer(timer_period, self._publish_cmd)

    def set_speed(self, pwm: float):
        clamped = max(CONFIG['pwm_safe_min'], min(CONFIG['pwm_safe_max'], pwm))
        self.speed_pwm = clamped
        self._publish_cmd()

    def stop(self):
        self.speed_pwm = CONFIG['pwm_stop']
        self._publish_cmd()

    def _publish_cmd(self):
        self.twist.linear.x = float(self.speed_pwm)
        self.twist.angular.z = float(CONFIG['servo_center'])
        self.pub.publish(self.twist)


def timed_spin_test(node: MotorTester, target_pwm: float, duration_s: float):
    print(f"\n>>> 启动电机 PWM = {target_pwm:.1f}，持续 {duration_s:.1f} 秒...")
    node.set_speed(target_pwm)
    t0 = time.time()
    try:
        while time.time() - t0 < duration_s:
            rem = duration_s - (time.time() - t0)
            sys.stdout.write(f"\r  运行中... 剩余 {rem:4.1f} 秒 (按 Ctrl+C 可立即紧急刹车) ")
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[!] 收到人工急停信号！")
    finally:
        node.stop()
        print("\n>>> 电机已断电停转。")


def ramp_test(node: MotorTester, target_pwm: float, step: float = 2.0, delay_s: float = 0.15):
    print(f"\n>>> 平滑加速测试: 1500 -> {target_pwm} -> 1500...")
    s = CONFIG['pwm_stop']
    try:
        # 加速
        while s < target_pwm:
            s += step
            node.set_speed(s)
            sys.stdout.write(f"\r  加速阶段: PWM = {s:6.1f}  ")
            sys.stdout.flush()
            time.sleep(delay_s)
        time.sleep(0.5)
        # 减速
        while s > CONFIG['pwm_stop']:
            s -= step
            node.set_speed(s)
            sys.stdout.write(f"\r  减速阶段: PWM = {s:6.1f}  ")
            sys.stdout.flush()
            time.sleep(delay_s)
    except KeyboardInterrupt:
        print("\n[!] 人工急停！")
    finally:
        node.stop()
        print("\n>>> 加减速测试完成，已恢复停止。")


def main():
    rclpy.init()
    node = MotorTester()
    import threading
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    time.sleep(0.2)
    node.stop()

    print("\n" + "!" * 52)
    print("【安全确认】请确保小车车轮已悬空离地，防止意外窜动！")
    print("!" * 52)

    try:
        while True:
            print("\n" + "=" * 52)
            print("          驱动电机 (ESC) 测试工具")
            print("=" * 52)
            print(f"当前状态: PWM = {node.speed_pwm:.1f} ({'停止' if node.speed_pwm == 1500 else '运转中'})")
            print("-" * 52)
            print(" [1] 立即停止 (PWM 1500)")
            print(" [2] 低速定长转动 (PWM 1525, 运行 2 秒)")
            print(" [3] 中速定长转动 (PWM 1540, 运行 2 秒)")
            print(" [4] 平滑加速-减速曲线测试 (Ramp)")
            print(" [5] 手动输入 PWM 值 (1450 ~ 1580)")
            print(" [0] 退出")
            print("=" * 52)
            choice = input("请选择 [0-5]: ").strip()

            if choice == '1':
                node.stop()
                print("[OK] 已停止。")
            elif choice == '2':
                timed_spin_test(node, CONFIG['pwm_slow'], 2.0)
            elif choice == '3':
                timed_spin_test(node, CONFIG['pwm_mid'], 2.0)
            elif choice == '4':
                ramp_test(node, CONFIG['pwm_mid'])
            elif choice == '5':
                val = input("请输入目标 PWM 值 (1500=停止, 1520~1560=前进): ").strip()
                try:
                    f = float(val)
                    if 1450.0 <= f <= 1580.0:
                        timed_spin_test(node, f, 2.0)
                    else:
                        print("[!] 超出安全范围 (1450 ~ 1580)")
                except ValueError:
                    print("[!] 输入无效")
            elif choice == '0':
                break
            else:
                print("[!] 无效选择")
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[*] 正在紧急停机并退出...")
        node.stop()
        time.sleep(0.3)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("[*] 完成。")


if __name__ == '__main__':
    main()
