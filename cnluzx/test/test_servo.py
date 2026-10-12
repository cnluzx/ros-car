#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_servo.py —— 舵机全方位标定与角度测试工具

物理与算法对应关系：
  小车前轮物理转角约在：右转 -45° ~ 正中 0° ~ 左转 +45°
  驱动层话题: /teleop_cmd_vel (twist.angular.z)
  驱动层PWM映射公式: pwm_us = 2500 - angular_z * (2000 / 180)

基准标定关系：
  物理 0° (正中直行) <==> 指令角 82.5° <==> 舵机 PWM 1583 μs
  物理 +45° (左转极限) <==> 指令角 180.0° <==> 舵机 PWM 500 μs
  物理 -45° (右转极限) <==> 指令角 0.0°   <==> 舵机 PWM 2500 μs
  每 1° 物理转角 ≈ 2.17° 算法指令角 ≈ 22.2 μs PWM
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
    'servo_center': 82.5,     # 算法中位指令角度 (deg)
    'cmd_min': 0.0,           # 算法指令下限 (右转极限 2500us)
    'cmd_max': 180.0,         # 算法指令上限 (左转极限 500us)
    'phys_max_deg': 45.0,     # 物理单侧最大摆角 (deg)
    'pub_rate_hz': 20.0,
}


def phys_to_cmd(phys_deg: float) -> float:
    """物理角度 (-45°右 ~ 0°中 ~ +45°左) -> 算法指令角度 (0° ~ 180°)"""
    c = CONFIG['servo_center']
    # 物理左转(+)对应指令角增加；物理右转(-)对应指令角减小
    cmd = c + (phys_deg / CONFIG['phys_max_deg']) * (CONFIG['cmd_max'] - c) if phys_deg >= 0 else \
          c + (phys_deg / CONFIG['phys_max_deg']) * (c - CONFIG['cmd_min'])
    return max(CONFIG['cmd_min'], min(CONFIG['cmd_max'], cmd))


def cmd_to_phys(cmd: float) -> float:
    """算法指令角度 (0° ~ 180°) -> 物理角度 (-45° ~ +45°)"""
    c = CONFIG['servo_center']
    if cmd >= c:
        return (cmd - c) / (CONFIG['cmd_max'] - c) * CONFIG['phys_max_deg']
    else:
        return (cmd - c) / (c - CONFIG['cmd_min']) * CONFIG['phys_max_deg']


def cmd_to_pwm(cmd: float) -> float:
    """算法指令角度 -> 下位机底层舵机脉宽 PWM (μs)"""
    return 2500.0 - cmd * (2000.0 / 180.0)


class ServoTester(Node):
    def __init__(self):
        super().__init__('servo_test_node')
        self.pub = self.create_publisher(Twist, CONFIG['topic'], QoSProfile(depth=5))
        self.cmd_angle = CONFIG['servo_center']
        self.twist = Twist()
        timer_period = 1.0 / CONFIG['pub_rate_hz']
        self.timer = self.create_timer(timer_period, self._publish_cmd)

    def set_cmd_angle(self, cmd_angle: float):
        self.cmd_angle = max(CONFIG['cmd_min'], min(CONFIG['cmd_max'], cmd_angle))
        self._publish_cmd()

    def set_phys_angle(self, phys_deg: float):
        cmd = phys_to_cmd(phys_deg)
        self.set_cmd_angle(cmd)

    def _publish_cmd(self):
        self.twist.linear.x = 0.0  # 电机锁定静止
        self.twist.angular.z = float(self.cmd_angle)
        self.pub.publish(self.twist)


def print_status(node: ServoTester):
    cmd = node.cmd_angle
    phys = cmd_to_phys(cmd)
    pwm = cmd_to_pwm(cmd)
    direction = "正中 (直行)" if abs(phys) < 0.5 else ("左转" if phys > 0 else "右转")
    print(f"\n[当前状态] 物理转角: {phys:+5.1f}° ({direction}) | 算法指令: {cmd:5.1f}° | 底层PWM: {pwm:5.0f} μs")


def test_5point(node: ServoTester):
    """5点标准定标测试：右45° -> 右22.5° -> 中位0° -> 左22.5° -> 左45° -> 回中"""
    points = [0.0, -22.5, -45.0, -22.5, 0.0, 22.5, 45.0, 22.5, 0.0]
    print("\n" + "=" * 50)
    print(">>> 开始 5 点阶梯定位测试（每个角度停留 2.0 秒供肉眼检查车轮）")
    print("=" * 50)
    for p in points:
        node.set_phys_angle(p)
        print_status(node)
        time.sleep(2.0)
    print(">>> 5 点测试完成，车轮已回正。")


def sweep_test(node: ServoTester, cycles: int = 2):
    """平滑连续摆动测试"""
    print(f"\n>>> 连续平滑扫描测试 ({cycles} 次循环)...")
    for c in range(cycles):
        print(f"  -> 第 {c+1}/{cycles} 轮...")
        for p in range(0, 46, 2):
            node.set_phys_angle(float(p))
            time.sleep(0.04)
        time.sleep(0.2)
        for p in range(45, -46, -2):
            node.set_phys_angle(float(p))
            time.sleep(0.04)
        time.sleep(0.2)
        for p in range(-45, 1, 2):
            node.set_phys_angle(float(p))
            time.sleep(0.04)
        time.sleep(0.3)
    node.set_phys_angle(0.0)
    print(">>> 扫描完成，车轮已回正。")


def fine_tune_mode(node: ServoTester):
    """微调模式（可用来精确校准正前方向中位）"""
    print("\n--- 舵机实时微调标定模式 ---")
    print("  [A] 物理向左微调 +1°  |  [D] 物理向右微调 -1°")
    print("  [W] 物理向左大调 +5°  |  [S] 物理向右大调 -5°")
    print("  [空格] 回到理论中位    |  [Q] 退出并保存/输出当前推荐中位")
    print("-" * 40)
    import termios
    import tty
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            ch = sys.stdin.read(1)
            cmd = node.cmd_angle
            phys = cmd_to_phys(cmd)
            if ch in ('a', 'A'):
                node.set_phys_angle(phys + 1.0)
            elif ch in ('d', 'D'):
                node.set_phys_angle(phys - 1.0)
            elif ch in ('w', 'W'):
                node.set_phys_angle(phys + 5.0)
            elif ch in ('s', 'S'):
                node.set_phys_angle(phys - 5.0)
            elif ch == ' ':
                node.set_cmd_angle(CONFIG['servo_center'])
            elif ch in ('q', 'Q', '\x03'):
                break
            
            c_now = node.cmd_angle
            p_now = cmd_to_phys(c_now)
            pwm_now = cmd_to_pwm(c_now)
            sys.stdout.write(f"\r物理: {p_now:+5.1f}° | 指令: {c_now:5.1f}° | PWM: {pwm_now:4.0f}μs   ")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print(f"\n退出微调。若当前前轮刚好最直，请将 param.yaml 的 servo_center 改为: {node.cmd_angle:.1f}")


def main():
    rclpy.init()
    node = ServoTester()
    import threading
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    time.sleep(0.2)
    node.set_cmd_angle(CONFIG['servo_center'])

    try:
        while True:
            print("\n" + "=" * 52)
            print("          舵机转向测试与标定工具 (ROS 2)")
            print("=" * 52)
            print_status(node)
            print("-" * 52)
            print(" [1] 车轮立即回正 (物理 0° / 算法 82.5°)")
            print(" [2] 5点阶梯标定 (-45° -> -22.5° -> 0° -> +22.5° -> +45°)")
            print(" [3] 平滑往复摆动测试 (Sweep)")
            print(" [4] 输入物理角度 (-45° ~ +45°)")
            print(" [5] 输入算法指令角度 (0.0° ~ 180.0°)")
            print(" [6] 键盘实时微调模式 (校准绝对中位)")
            print(" [0] 退出测试")
            print("=" * 52)
            choice = input("请选择 [0-6]: ").strip()

            if choice == '1':
                node.set_cmd_angle(CONFIG['servo_center'])
                print("[OK] 已回正。")
            elif choice == '2':
                test_5point(node)
            elif choice == '3':
                sweep_test(node)
            elif choice == '4':
                val = input("请输入目标物理角度 (-45 到 +45): ").strip()
                try:
                    f = float(val)
                    node.set_phys_angle(f)
                    print(f"[OK] 已设定物理角度 {f:+.1f}°")
                except ValueError:
                    print("[!] 输入无效")
            elif choice == '5':
                val = input("请输入目标算法指令角度 (0.0 到 180.0): ").strip()
                try:
                    f = float(val)
                    node.set_cmd_angle(f)
                    print(f"[OK] 已设定指令角度 {f:.1f}°")
                except ValueError:
                    print("[!] 输入无效")
            elif choice == '6':
                fine_tune_mode(node)
            elif choice == '0':
                break
            else:
                print("[!] 无效选择")
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[*] 正在复位舵机到中位并退出...")
        node.set_cmd_angle(CONFIG['servo_center'])
        time.sleep(0.3)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("[*] 完成。")


if __name__ == '__main__':
    main()
