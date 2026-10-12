#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
控制静默急停看门狗 (teleop watchdog)

作用: 守着控制话题(默认 /teleop_cmd_vel)。只要话题"沉默"超过 timeout 秒
(跟踪节点被 Ctrl-C / 崩溃 / 掉线 / 线缆松动), 立刻以 20Hz 连发急停指令
(电机 PWM=1500 停转, 舵机=中位), 让车自己停下 —— 不用再跑去关电机电源。

设计: "话题永不沉默"。有控制器在发指令 -> 它保持安静; 一旦没人发 -> 它接管。
      自己的急停消息会被识别并忽略, 不会自我唤醒。
      所以它必须跑在【独立终端 / nohup 常驻】, 这样 Ctrl-C 掉跟踪节点时它还活着。

v2: 同时监听第二控制话题 /car_cmd_vel (黄线巡线节点用的标准速度接口)。
      两个话题里【任意一个】有活动都算"有控制", 这样 lidar_tracking 进入
      YELLOW 状态 (停止发 /teleop_cmd_vel、把控制权交给黄线) 时,
      看门狗不会误判为失联而连发急停去抢黄线的控制权。

用法:
    source /opt/ros/humble/setup.bash
    source ~/racecar/install/setup.bash
    python3 teleop_watchdog.py          # 前台运行; 或 nohup 常驻(推荐)
    ros2 run lidar_tracking teleop_watchdog     # 编译安装后可直接 ros2 run

参数:
    --topic   主控制话题 (默认 /teleop_cmd_vel, 也是急停发布话题)
    --topic2  次级控制话题 (默认 /car_cmd_vel, 只监听不发布; 设为空串可关闭)
    --timeout 静默判定时长 [s] (默认 0.6; 控制器按 10Hz 发, 0.6s 足够宽容)
    --rate    急停发送频率 [Hz] (默认 20)
"""

import argparse
import os
import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from geometry_msgs.msg import Twist
except ImportError:
    print('需要 ROS2 环境: source /opt/ros/humble/setup.bash')
    sys.exit(1)

STOP_MOTOR = 1500.0        # 电机停转 PWM


def find_servo_center(default=80.0):
    """从唯一调参入口 param.yaml 读舵机中位 (读不到就用默认值)"""
    cands = [
        os.environ.get('LIDAR_TRACKING_PARAM', ''),
        os.path.expanduser('~/racecar/src/lidar_tracking/config/param.yaml'),
        '/home/davinci-mini/racecar/src/lidar_tracking/config/param.yaml',
    ]
    for c in cands:
        if c and os.path.exists(c):
            try:
                import yaml
                with open(c, encoding='utf-8') as f:
                    yd = yaml.safe_load(f)
                rp = (yd.get('/**', {}).get('ros__parameters', {}) or {})
                if 'servo_center' in rp:
                    return float(rp['servo_center'])
            except Exception:
                pass
    return default


class Watchdog(Node):
    def __init__(self, topic, topic2, timeout, rate):
        super().__init__('teleop_watchdog')
        self.center = find_servo_center()
        self.timeout = timeout
        self.last_rx = time.monotonic()
        self.firing = False
        self.pub = self.create_publisher(Twist, topic, QoSProfile(depth=5))
        self.create_subscription(Twist, topic, self.on_msg, QoSProfile(depth=10))
        # 次级话题: 只监听, 不发布急停 (黄线节点用的标准速度接口)
        if topic2 and topic2 != topic:
            self.create_subscription(
                Twist, topic2, self.on_other, QoSProfile(depth=10))
            self.get_logger().info('同时监听次级控制话题 %s' % topic2)
        self.stop = Twist()
        self.stop.linear.x = STOP_MOTOR
        self.stop.angular.z = self.center
        self.create_timer(1.0 / max(1.0, rate), self.tick)
        self.get_logger().info(
            '看门狗已启动: 监听 %s, 静默超过 %.2fs 即连发急停 (电机 %.0f / 舵机 %.1f)'
            % (topic, timeout, STOP_MOTOR, self.center))

    def _is_own_stop(self, msg):
        """判断是不是自己发的急停(电机1500 且 舵机≈中位)"""
        return (abs(msg.linear.x - STOP_MOTOR) < 1e-6
                and abs(msg.angular.z - self.center) < 0.5)

    def _activity(self):
        self.last_rx = time.monotonic()
        if self.firing:
            self.firing = False
            self.get_logger().info('检测到控制恢复 -> 暂停急停, 交还给控制器')

    def on_msg(self, msg):
        if self.firing and self._is_own_stop(msg):
            return                      # 自己的急停, 不算"控制恢复"
        self._activity()

    def on_other(self, msg):
        # /car_cmd_vel 上的任何指令都视为"有控制" (我们不在这个话题发急停)
        self._activity()

    def tick(self):
        silent = time.monotonic() - self.last_rx
        if silent > self.timeout:
            if not self.firing:
                self.firing = True
                self.get_logger().warn(
                    '控制话题静默 %.2fs -> 开始急停 (电机 %.0f / 舵机 %.1f), '
                    '控制器恢复前会一直保持'
                    % (silent, STOP_MOTOR, self.center))
            self.pub.publish(self.stop)


def main():
    ap = argparse.ArgumentParser(description='控制静默急停看门狗')
    ap.add_argument('--topic', default='/teleop_cmd_vel')
    ap.add_argument('--topic2', default='/car_cmd_vel',
                    help='次级控制话题(默认 /car_cmd_vel, 空串关闭)')
    ap.add_argument('--timeout', type=float, default=0.6, help='静默判定时长 [s]')
    ap.add_argument('--rate', type=float, default=20.0, help='急停发送频率 [Hz]')
    args = ap.parse_args()

    rclpy.init()
    node = Watchdog(args.topic, args.topic2, args.timeout, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 自己退出前也补发急停, 保证车是停的
        try:
            for _ in range(10):
                node.pub.publish(node.stop)
                time.sleep(0.03)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
