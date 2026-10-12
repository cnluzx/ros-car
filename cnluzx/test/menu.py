#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
menu.py —— 小车全硬件模块综合测试与标定控制台

一站式测试：
  [1] 舵机标定与角度测试 (test_servo.py)
  [2] 驱动电机安全测试   (test_motor.py)
  [3] 激光雷达实时诊断   (test_laser.py)
  [4] IMU 惯导与姿态测试 (test_imu.py)
  [5] 轮速里程计测距标定 (test_encoder.py)
  [6] 摄像头视觉与Web图传(test_vision.py)
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name):
    path = os.path.join(HERE, script_name)
    if not os.path.exists(path):
        print(f"[错误] 未找到脚本: {path}")
        return
    try:
        subprocess.run([sys.executable, path])
    except KeyboardInterrupt:
        pass


def main():
    while True:
        print("\n" + "=" * 54)
        print("      ★ 智能小车全硬件模块综合调试控制台 ★")
        print("=" * 54)
        print("  [1] 舵机转向测试与标定 (物理 -45°~+45°, 算法 0°~180°)")
        print("  [2] 驱动电机安全测试   (ESC PWM 脉宽与平滑加减速)")
        print("  [3] 激光雷达实时监视   (各扇区障碍物测距与刷新率)")
        print("  [4] IMU 惯导姿态监视   (Yaw/Pitch/Roll, 零偏漂移)")
        print("  [5] 轮速计与里程计测试 (线速度, 1米推车测距标定)")
        print("  [6] 摄像头视觉监视     (硬件检测与 Web 实时图传)")
        print("  [0] 退出控制台")
        print("=" * 54)
        c = input("请输入测试选项 [0-6]: ").strip()
        if c == '1':
            run_script('test_servo.py')
        elif c == '2':
            run_script('test_motor.py')
        elif c == '3':
            run_script('test_laser.py')
        elif c == '4':
            run_script('test_imu.py')
        elif c == '5':
            run_script('test_encoder.py')
        elif c == '6':
            run_script('test_vision.py')
        elif c == '0':
            print("\n退出测试控制台。")
            break
        else:
            print("[!] 无效选择，请重新输入。")


if __name__ == '__main__':
    main()
