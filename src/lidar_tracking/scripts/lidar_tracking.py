#!/usr/bin/env python3
"""激光雷达锥桶循迹 ROS 2 节点 (v7：四态状态机 + 直道稳定化 + 黄线交接)。

v7 相对 v6 的核心变化
=====================================================================
1. 四态状态机 (取代原来的 STRAIGHT/CORNER 布尔量)：
       STRAIGHT      直道锥桶      —— 远前瞻、高速、直行稳定
       CORNER        正常弯道锥桶  —— 近前瞻、减速、贴内线 (v6 已验证良好)
       RIGHT_ANGLE   直角弯道锥桶  —— 更近前瞻、最低速、允许满舵持续
       YELLOW        无锥桶循黄线  —— 交出底盘控制权给黄线巡线节点
   状态用「曲率迟滞 + 直角几何判据」驱动，并发布成话题方便可视化。

2. 直道稳定化 (解决用户反馈的"弯道很好、直道发抖/画龙")：
       · aim_y 低通滤波 (aim_y_lpf)
       · aim_y 死区 (aim_y_deadband)，消除中轴附近的抖动
       · 单侧锥桶 aim_y 限幅 (single_side_clamp)，避免只看到一侧时猛打
       · 左右分组加中线死区 (aim_mid_deadband)，避免中缝噪点被当锥桶
       · ego-motion 用真实 dt (原代码写死 0.1s)
       · 直道曲率死区 (straight_k_deadband)

3. 直角判据 (单帧几何)：
       · 近处锥桶横向成"墙"(x 变化小、y 跨度大)
       · 或中轴目标点方位角过大
       · 与曲率门限、持续时间联合确认，避免误切

4. 黄线交接：
       · 连续 cone_lost_timeout 秒看不到锥桶 -> YELLOW
       · YELLOW 时在 yellow_enable_topic 发 True，并停止发布 /teleop_cmd_vel
         (否则会和黄线节点的 car_cmd_vel 抢底盘，最后写入者胜)
       · 重新看到锥桶 cone_found_timeout 秒 -> 发 False 并恢复控制

保留 v6 已验证良好的部分：自适应聚类、车体系短时记忆、纯追踪曲率、弧线安全
层、机械限幅 [35,125]°、PID 积分项。
"""

import csv
import math
import os
import signal
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    from geometry_msgs.msg import Twist
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Bool, String

    HAVE_ROS = True
except ImportError:
    HAVE_ROS = False
    Node = object


# 四个显式状态 (字符串常量，方便发布/可视化)
STATE_STRAIGHT = 'STRAIGHT'
STATE_CORNER = 'CORNER'
STATE_RIGHT_ANGLE = 'RIGHT_ANGLE'
STATE_YELLOW = 'YELLOW'


DEFAULTS = {
    'topic': 'teleop_cmd_vel',
    # ===== 速度控制参数 (PWM) =====
    'speed_straight': 1560,  # 直道巡航全速 (PWM)
    'speed_corner': 1552,  # 弯道低速防推头 (PWM)
    'speed_rightangle': 1545,  # 直角弯最低速 (PWM)
    'danger_slow_drop': 12,  # 逼近锥桶额外减速量 (PWM)
    # ===== 舵机物理标定 (严守 80° ± 45° 机械限幅) =====
    'servo_center': 80.0,  # 直行中位舵角 [deg]
    'servo_min': 35.0,  # 机械物理右极限 (80 - 45) [deg]
    'servo_max': 125.0,  # 机械物理左极限 (80 + 45) [deg]
    'servo_deg_per_curv': 68.0,  # 转向增益: 45°对应极限曲率 0.66 [deg/(1/m)]
    'max_curvature': 0.66,  # 物理最大曲率 (最小转弯半径约 1.51m) [1/m]
    'steer_gain': 1.35,  # 纯追踪中轴导向系数
    'max_k_rate': 0.25,  # 每帧曲率变化上限 (杜绝猛打方向)
    # ===== 状态机门限 (纯曲率施密特迟滞) =====
    'curv_enter_corner': 0.18,  # 入弯曲率门限 (|k| > 0.18 坚决锁死弯道模式)
    'curv_exit_corner': 0.08,  # 出弯曲率门限 (|k| < 0.08 完全回直才切直道)
    'lookahead_straight': 1.60,  # 直道前瞻距离 [m] (走得直、不蛇形)
    'lookahead_corner': 1.05,  # 弯道前瞻距离 [m] (贴紧内线防外冲)
    'lookahead_rightangle': 0.75,  # 直角前瞻距离 [m] (贴死内角)
    # ===== 直角弯判据 =====
    'curv_enter_rightangle': 0.45,  # 弯道 -> 直角 的曲率门限
    'curv_exit_rightangle': 0.28,  # 直角 -> 弯道 的曲率门限
    'ra_hold_s': 0.30,  # 直角需持续确认时间 [s]
    'ra_min_cones': 4,  # 直角判据所需近处锥桶数
    'ra_cone_x_max': 1.80,  # 直角判据: 近处锥桶纵向范围 [m]
    'ra_wall_y_spread': 1.60,  # 直角判据: 横向墙 y 跨度门限 [m]
    'ra_wall_x_spread': 0.80,  # 直角判据: 横向墙 x 跨度上限 [m]
    'ra_bearing_deg': 55.0,  # 直角判据: 目标点方位角门限 [deg]
    # ===== 无锥桶 -> 黄线 =====
    'min_cones_for_track': 2,  # 视为"有锥桶赛道"的最少锥桶数
    'cone_lost_timeout': 1.20,  # 连续无锥桶多久切黄线 [s]
    'cone_found_timeout': 0.40,  # 重新看到锥桶多久切回 [s]
    'yellow_enable_topic': '/yellow_tracking_enable',
    # ===== 直道稳定化 (v7 新增) =====
    'aim_y_lpf': 0.40,  # aim_y 低通系数 (0~1, 越大越平滑)
    'aim_y_deadband': 0.04,  # aim_y 死区 [m] (中轴附近不再抽搐)
    'aim_mid_deadband': 0.08,  # 左右分组的中缝死区 [m]
    'single_side_clamp': 0.40,  # 只看到单侧时 aim_y 限幅 [m]
    'straight_k_deadband': 0.015,  # 直道曲率死区 [1/m]
    'ego_speed': 0.90,  # 用于记忆自车运动补偿的估计车速 [m/s]
    # ===== 自适应聚类与半径推广 =====
    'cd_range_min': 0.25,  # 雷达扫描盲区下限 [m]
    'cd_range_max': 4.20,  # 雷达扫描有效半径 [m]
    'cd_fov_deg': 105.0,  # 车头扇面半角 (共210°超大视野) [deg]
    'cd_gap_base': 0.045,  # 聚类基础间距 [m]
    'cd_gap_slope': 0.020,  # 聚类发散斜率
    'cd_min_points': 3,  # 锥桶最少激光点数
    'cd_min_extent': 0.018,  # 最小物理尺寸 [m]
    'cd_max_extent': 0.30,  # 最大物理尺寸 [m]
    'cd_cone_radius': 0.08,  # 锥桶底面半径向心补偿 [m]
    'win_x_min': 0.10,  # 车身后方硬盲区 (彻底屏蔽操作员双腿) [m]
    # ===== 车体系短时记忆 (ConeMap) =====
    'mem_ttl': 1.5,  # 锥桶记忆存活时间 [s] (消除转角盲区)
    'mem_match_gate': 0.45,  # 帧间点匹配距离门限 [m]
    'mem_min_hits': 2,  # 确认锥桶所需连续命中帧数 (过滤飞点)
    'track_width_nominal': 1.50,  # 单侧走廊补偿赛道半宽 [m]
    # ===== 弧线安全避障层 =====
    'safe_enable': True,
    'safe_lookahead': 1.30,  # 碰撞前瞻检查弧长 [m]
    'safe_radius': 0.26,  # 最小安全走廊半宽 [m]
    'safe_front': 0.20,  # 车头距雷达距离 [m]
    'safe_curv_steps': 21,  # 候选曲率采样数
    # ===== 起步保护 =====
    'start_guard_s': 1.5,  # 起步保护持续时间 [s]
    'start_k_max': 0.20,  # 起步期最大允许转向曲率
    # ===== PID 控制参数 (v6 新增) =====
    'pid_kp': 1.00,  # 比例增益 (与 steer_gain 叠加，可先设 1.0)
    'pid_ki': 0.05,  # 积分增益 (关键！从 0.01~0.1 之间调)
    'pid_kd': 0.00,  # 微分增益 (先关掉，需要时再开)
    'pid_i_limit': 0.15,  # 积分限幅 [1/m]，防止积分饱和 (anti-windup)
    'pid_i_decay': 0.98,  # 直道时积分缓慢泄放系数 (每帧)
    'pid_i_enable_corner': True,  # 弯道是否启用积分
    # ===== 运行记录与调试 =====
    'debug_every': 5,
    'debug_csv': 1,
    'debug_csv_dir': '~/racecar/logs',
}


def _find_param_yaml():
    cands = [
        os.environ.get('LIDAR_TRACKING_PARAM', ''),
        os.path.expanduser('~/racecar/src/lidar_tracking/config/param.yaml'),
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            'config',
            'param.yaml',
        ),
        os.path.join(os.getcwd(), 'config', 'param.yaml'),
    ]
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def load_param_yaml():
    path = _find_param_yaml()
    if not path:
        return {}, None
    try:
        import yaml

        with open(path, encoding='utf-8') as f:
            yd = yaml.safe_load(f)
        ros_p = (
            yd.get('/**', {}).get('ros__parameters', {})
            or yd.get('ros__parameters', {})
            or {}
        )
        out = {}
        for k, v in ros_p.items():
            if k not in DEFAULTS or k == 'topic':
                continue
            dv = DEFAULTS[k]
            try:
                if isinstance(dv, bool):
                    out[k] = bool(v)
                elif isinstance(dv, float):
                    out[k] = float(v)
                elif isinstance(dv, int):
                    out[k] = int(v)
                else:
                    out[k] = v
            except (TypeError, ValueError):
                pass
        return out, path
    except Exception as e:
        print('[lidar_tracking] 读取 %s 失败: %s' % (path, e))
        return {}, path


def detect_cones(ranges, angle_min, angle_inc, p) -> np.ndarray:
    """自适应连续欧氏聚类提取锥桶中心 (车体系 x前, y左)。"""
    r = np.asarray(ranges, dtype=np.float64)
    ok = np.isfinite(r) & (r >= p['cd_range_min']) & (r <= p['cd_range_max'])
    idx = np.flatnonzero(ok)
    if idx.size < p['cd_min_points']:
        return np.zeros((0, 2))

    th = angle_min + idx * angle_inc
    rv = r[idx]
    x = rv * np.cos(th)
    y = rv * np.sin(th)

    contiguous = np.diff(idx) == 1
    d = np.hypot(np.diff(x), np.diff(y))
    expect = p['cd_gap_base'] + p['cd_gap_slope'] * np.maximum(rv[:-1], rv[1:])
    brk = (~contiguous) | (d > expect)
    starts = np.r_[0, np.flatnonzero(brk) + 1]
    ends = np.r_[np.flatnonzero(brk), idx.size - 1]

    out = []
    for s, e in zip(starts, ends):
        if (e - s + 1) < p['cd_min_points']:
            continue
        gx, gy = x[s : e + 1], y[s : e + 1]
        ext = math.hypot(gx.max() - gx.min(), gy.max() - gy.min())
        if ext < p['cd_min_extent'] or ext > p['cd_max_extent']:
            continue
        cx, cy = float(gx.mean()), float(gy.mean())
        rr = math.hypot(cx, cy)
        if rr > 1e-6:
            s_ = (rr + p['cd_cone_radius']) / rr
            cx *= s_
            cy *= s_

        # 盲区过滤 (车身后方 x < win_x_min 硬过滤，排除人腿)
        if cx < p['win_x_min']:
            continue
        # 扇区过滤
        deg = abs(math.degrees(math.atan2(cy, cx)))
        if deg > p['cd_fov_deg']:
            continue
        out.append((cx, cy))
    return np.array(out, dtype=np.float64).reshape(-1, 2)


class LocalConeMemory:
    """车体系短时锥桶记忆 (带自身运动 Ego-motion 补偿与生命周期管理)。"""

    def __init__(self, p):
        self.p = p
        self.memory = []

    def update(
        self,
        det_cones: np.ndarray,
        v: float,
        kappa: float,
        dt: float,
        now: float,
    ) -> np.ndarray:
        w = v * kappa
        c_th = math.cos(w * dt)
        s_th = math.sin(w * dt)
        dx = v * dt

        for m in self.memory:
            x_rel = m['x'] - dx
            y_rel = m['y']
            m['x'] = c_th * x_rel + s_th * y_rel
            m['y'] = -s_th * x_rel + c_th * y_rel

        gate = self.p['mem_match_gate']
        matched = set()
        for cx, cy in det_cones:
            best_idx = -1
            best_d = gate
            for i, m in enumerate(self.memory):
                if i in matched:
                    continue
                d = math.hypot(m['x'] - cx, m['y'] - cy)
                if d < best_d:
                    best_d = d
                    best_idx = i

            if best_idx >= 0:
                m = self.memory[best_idx]
                m['x'] = 0.7 * cx + 0.3 * m['x']
                m['y'] = 0.7 * cy + 0.3 * m['y']
                m['last_seen'] = now
                m['hits'] = min(10, m['hits'] + 1)
                matched.add(best_idx)
            else:
                self.memory.append({'x': cx, 'y': cy, 'last_seen': now, 'hits': 1})

        ttl = self.p['mem_ttl']
        survived = []
        for m in self.memory:
            if (now - m['last_seen'] <= ttl) and (
                math.hypot(m['x'], m['y']) <= self.p['cd_range_max'] + 1.0
            ):
                survived.append(m)
        self.memory = survived

        confirmed = [
            [m['x'], m['y']]
            for m in self.memory
            if m['hits'] >= self.p['mem_min_hits']
        ]
        if not confirmed and len(det_cones) > 0:
            return det_cones
        return np.array(confirmed, dtype=np.float64).reshape(-1, 2)


def arc_clearance(kappa: float, cones: np.ndarray, p) -> float:
    if cones.shape[0] == 0:
        return 99.0
    s = np.linspace(0.1, p['safe_lookahead'], 15)
    if abs(kappa) < 1e-3:
        px = s
        py = np.zeros_like(s)
        hx = np.ones_like(s)
        hy = np.zeros_like(s)
    else:
        a = kappa * s
        px = np.sin(a) / kappa
        py = (1.0 - np.cos(a)) / kappa
        hx, hy = np.cos(a), np.sin(a)
    cx = px + 0.5 * p['safe_front'] * hx
    cy = py + 0.5 * p['safe_front'] * hy
    d = np.hypot(
        cones[:, 0][None, :] - cx[:, None], cones[:, 1][None, :] - cy[:, None]
    )
    return float(d.min())


def choose_safe_curvature(
    k_des: float, prev_k: float, cones: np.ndarray, p
) -> Tuple[float, float]:
    kmax = p['max_curvature']
    k_des = max(-kmax, min(kmax, k_des))
    if not p['safe_enable'] or cones.shape[0] == 0:
        return k_des, 99.0
    cand = np.unique(
        np.r_[np.linspace(-kmax, kmax, int(p['safe_curv_steps'])), k_des]
    )
    cls = [(float(k), arc_clearance(float(k), cones, p)) for k in cand]
    safe = [(k, cl) for k, cl in cls if cl >= p['safe_radius']]
    if safe:
        k, cl = min(
            safe, key=lambda kc: abs(kc[0] - k_des) + 0.15 * abs(kc[0] - prev_k)
        )
        return k, cl
    max_cl = max(cl for _, cl in cls)
    good = [(k, cl) for k, cl in cls if cl >= max_cl - 0.05]
    k, cl = min(good, key=lambda kc: abs(kc[0] - k_des))
    return k, cl


def _aim_point(cones: np.ndarray, la: float, p) -> np.ndarray:
    """从左右锥桶里取走廊中轴目标点 (车体系)。找不到时退回正前方 la。"""
    fallback = np.array([la, 0.0])
    if cones.shape[0] == 0:
        return fallback
    pts = cones[cones[:, 0] >= 0.3]
    if len(pts) < 2:
        return fallback
    dists = np.hypot(pts[:, 0], pts[:, 1])
    cand = pts[(dists >= la * 0.7) & (dists <= la * 1.5)]
    if len(cand) < 1:
        return fallback

    mid = p['aim_mid_deadband']
    cl = cand[cand[:, 1] > mid]
    cr = cand[cand[:, 1] < -mid]
    half_w = 0.5 * p['track_width_nominal']
    clamp = p['single_side_clamp']

    if len(cl) and len(cr):
        return 0.5 * (cl.mean(0) + cr.mean(0))
    if len(cl):
        y = float(cl[:, 1].mean()) - half_w
        y = max(-clamp, min(clamp, y))
        return np.array([float(cl[:, 0].mean()), y])
    if len(cr):
        y = float(cr[:, 1].mean()) + half_w
        y = max(-clamp, min(clamp, y))
        return np.array([float(cr[:, 0].mean()), y])
    return fallback


def _right_angle_hit(cones: np.ndarray, aim: np.ndarray, p) -> bool:
    """单帧几何判断是否像直角弯 (横向锥桶墙 或 中轴目标点方位角很大)。"""
    # 判据 A: 近处锥桶横向排成一堵墙 (x 变化小、y 跨度大)
    if cones.shape[0] >= p['ra_min_cones']:
        near = cones[(cones[:, 0] > 0.2) & (cones[:, 0] < p['ra_cone_x_max'])]
        if len(near) >= p['ra_min_cones']:
            x_spread = float(near[:, 0].max() - near[:, 0].min())
            y_spread = float(near[:, 1].max() - near[:, 1].min())
            if (
                y_spread > p['ra_wall_y_spread']
                and x_spread < p['ra_wall_x_spread']
            ):
                return True
    # 判据 B: 目标点方位角很大
    bearing = abs(math.atan2(float(aim[1]), max(0.2, float(aim[0]))))
    if bearing > math.radians(p['ra_bearing_deg']):
        return True
    return False


class ConeTracker:
    """四态状态机 + 感知 + 控制。

    对外接口 (cone.py 可视化依赖，保持稳定)：
        step(ranges, angle_min, angle_inc, now) -> (servo_deg, speed_pwm)
        属性: .p / .mode / .state / .dbg / .control_enabled
    """

    def __init__(self, params=None):
        self.p = dict(DEFAULTS)
        if params:
            self.p.update(params)
        self.mem = LocalConeMemory(self.p)
        self.prev_k = 0.0
        self.prev_servo = self.p['servo_center']
        self.smooth_k = 0.0
        self.state = STATE_STRAIGHT
        self.mode = STATE_STRAIGHT
        self.control_enabled = True
        self.t_start = None
        self.dbg = {}

        # PID 积分项状态
        self.k_i = 0.0
        self.prev_err_y = 0.0
        self.last_time = None

        # 直道稳定化 / 状态机计时 (v7)
        self.aim_y_f = 0.0
        self.ra_timer = 0.0
        self.cone_lost_t = 0.0
        self.cone_found_t = 0.0
        self.n_cones = 0

    # ------------------------------------------------------------------
    def step(
        self, ranges, angle_min, angle_inc, now: float
    ) -> Tuple[float, float]:
        p = self.p
        if self.t_start is None:
            self.t_start = now

        # 0. 真实 dt (v7: 原代码写死 0.1s)
        if self.last_time is None:
            dt = 0.1
        else:
            dt = max(0.01, min(0.5, now - self.last_time))
        self.last_time = now

        # 1. 提取与记忆更新 (ego-motion 用真实 dt 与估计车速)
        raw_cones = detect_cones(ranges, angle_min, angle_inc, p)
        cones = self.mem.update(
            raw_cones, p['ego_speed'], self.prev_k, dt, now
        )
        self.n_cones = int(cones.shape[0])

        # 2. 状态机动态前瞻决策 (用上一帧状态选前瞻)
        if self.state == STATE_RIGHT_ANGLE:
            la = p['lookahead_rightangle']
        elif self.state == STATE_CORNER:
            la = p['lookahead_corner']
        else:
            la = p['lookahead_straight']

        # 3. 走廊中轴目标点提取 + 直道稳定化 (低通 + 死区)
        aim_raw = _aim_point(cones, la, p)
        a_lpf = max(0.0, min(1.0, p['aim_y_lpf']))
        self.aim_y_f = (1.0 - a_lpf) * self.aim_y_f + a_lpf * float(aim_raw[1])
        db = p['aim_y_deadband']
        if abs(self.aim_y_f) <= db:
            aim_y = 0.0
        else:
            aim_y = self.aim_y_f - math.copysign(db, self.aim_y_f)
        aim = np.array([float(aim_raw[0]), aim_y])

        # 4. 纯追踪前馈期望曲率
        L_aim = max(0.5, math.hypot(aim[0], aim[1]))
        k_ff = p['steer_gain'] * (2.0 * aim[1]) / (L_aim * L_aim)
        k_ff = max(-p['max_curvature'], min(p['max_curvature'], k_ff))

        # ===== 5. PID 修正 =====
        # 5.1 横向偏差：让 aim[1] → 0 (车正对走廊中轴)
        err_y = -float(aim[1])

        # 5.2 比例项
        p_term = p['pid_kp'] * err_y

        # 5.3 积分项 (带 anti-windup + 弯道可冻结)
        i_enable = (self.state == STATE_STRAIGHT) or p['pid_i_enable_corner']
        if i_enable:
            self.k_i += p['pid_ki'] * err_y * dt
            lim = p['pid_i_limit']
            self.k_i = max(-lim, min(lim, self.k_i))
        i_term = self.k_i

        # 5.4 微分项 (可选，默认关闭)
        d_term = 0.0
        if p['pid_kd'] > 0.0:
            d_err = (err_y - self.prev_err_y) / dt
            d_term = p['pid_kd'] * d_err
        self.prev_err_y = err_y

        # 5.5 前馈 + PID 融合
        k_des = k_ff + p_term + i_term + d_term
        k_des = max(-p['max_curvature'], min(p['max_curvature'], k_des))

        # 5.6 直道时缓慢泄放积分 (防止出弯后持续偏转)
        if self.state == STATE_STRAIGHT:
            self.k_i *= p['pid_i_decay']

        # 6. 安全避障与变化率滤波
        k_safe, clear = choose_safe_curvature(k_des, self.prev_k, cones, p)
        max_step = float(p['max_k_rate'])
        k_cmd = self.prev_k + np.clip(k_safe - self.prev_k, -max_step, max_step)

        # 7. 起步保护
        if now - self.t_start < p['start_guard_s']:
            k_cmd = np.clip(k_cmd, -p['start_k_max'], p['start_k_max'])

        # 8. 直道曲率死区 (消除中轴附近的微小抽搐)
        if self.state == STATE_STRAIGHT and abs(k_cmd) < p['straight_k_deadband']:
            k_cmd = 0.0

        self.prev_k = k_cmd

        # ===== 9. 四态状态机 =====
        # 9.1 纯曲率迟滞滤波
        self.smooth_k = 0.85 * self.smooth_k + 0.15 * abs(k_cmd)

        # 9.2 无锥桶计时 (用于黄线交接)
        n_track = int(np.sum(cones[:, 0] >= 0.2)) if cones.shape[0] else 0
        if n_track < p['min_cones_for_track']:
            self.cone_lost_t += dt
            self.cone_found_t = 0.0
        else:
            self.cone_found_t += dt
            self.cone_lost_t = 0.0

        # 9.3 直角判据计时
        ra_hit = _right_angle_hit(cones, aim, p)
        if ra_hit and self.smooth_k > p['curv_enter_corner']:
            self.ra_timer += dt
        else:
            self.ra_timer = max(0.0, self.ra_timer - 1.5 * dt)

        self._update_state(p)

        self.mode = self.state
        self.control_enabled = self.state != STATE_YELLOW

        # 10. 舵机映射与 80° ± 45° 物理硬限幅
        servo = p['servo_center'] + p['servo_deg_per_curv'] * k_cmd
        servo = max(p['servo_min'], min(p['servo_max'], servo))
        self.prev_servo = servo

        # 11. 状态自适应车速
        if not self.control_enabled:
            speed = 1500.0  # YELLOW: 不控制底盘, 只给中位
        elif self.state == STATE_RIGHT_ANGLE:
            speed = p['speed_rightangle']
        elif self.state == STATE_CORNER:
            speed = p['speed_corner']
        else:
            speed = p['speed_straight']
        if self.control_enabled and p['safe_enable'] and clear < p['safe_radius']:
            speed -= p['danger_slow_drop']

        # 12. 调试信息
        self.dbg = dict(
            state=self.state,
            n_cones=int(cones.shape[0]),
            n_track=n_track,
            aim=aim,
            k_cmd=k_cmd,
            k_des=k_des,
            clear=clear,
            curv=self.smooth_k,
            k_ff=k_ff,
            k_i=self.k_i,
            err_y=err_y,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
            ra_timer=self.ra_timer,
            cone_lost_t=self.cone_lost_t,
        )
        return servo, speed

    # ------------------------------------------------------------------
    def _update_state(self, p):
        """四态转移。STRAIGHT/CORNER/RIGHT_ANGLE 之间用曲率迟滞 + 直角判据，
        任意态在长时间看不到锥桶时进入 YELLOW。"""
        # --- 黄线优先 ---
        if self.state == STATE_YELLOW:
            if self.cone_found_t >= p['cone_found_timeout']:
                self.state = STATE_STRAIGHT
                self.ra_timer = 0.0
                self.smooth_k = 0.0
                self.k_i = 0.0
            return

        if self.cone_lost_t >= p['cone_lost_timeout']:
            self.state = STATE_YELLOW
            self.k_i = 0.0
            return

        # --- 直角判定 ---
        ra_confirmed = (
            self.ra_timer >= p['ra_hold_s']
            or self.smooth_k > p['curv_enter_rightangle']
        )

        if self.state == STATE_RIGHT_ANGLE:
            if self.ra_timer <= 0.0 and self.smooth_k < p['curv_exit_rightangle']:
                self.state = (
                    STATE_CORNER
                    if self.smooth_k > p['curv_exit_corner']
                    else STATE_STRAIGHT
                )
        elif self.state == STATE_CORNER:
            if ra_confirmed:
                self.state = STATE_RIGHT_ANGLE
            elif self.smooth_k < p['curv_exit_corner']:
                self.state = STATE_STRAIGHT
        else:  # STRAIGHT
            if self.smooth_k > p['curv_enter_corner']:
                self.state = (
                    STATE_RIGHT_ANGLE if ra_confirmed else STATE_CORNER
                )


class LidarTrackingNode(Node):

    def __init__(self):
        super().__init__('lidar_tracking_node')
        # ---- 参数加载 (三级优先) ----
        #   1) 代码内置默认值 DEFAULTS
        #   2) 直接读工作空间 config/param.yaml (直接 python3 跑脚本时也能用)
        #   3) launch 传入的 parameters=[param.yaml] 覆盖 (优先级最高)
        # 这样无论有没有 launch, config/param.yaml 都是唯一调参入口。
        params = dict(DEFAULTS)
        file_params, ppath = load_param_yaml()
        params.update(file_params)

        # 全部声明成 ROS 参数, "默认值"= 上面合并后的值。
        # launch 的 parameters=[...] 会作为 override 叠在最上面。
        for k, v in params.items():
            try:
                if isinstance(v, bool):
                    self.declare_parameter(k, bool(v))
                elif isinstance(v, int):
                    self.declare_parameter(k, int(v))
                elif isinstance(v, float):
                    self.declare_parameter(k, float(v))
                else:
                    self.declare_parameter(k, str(v))
            except Exception:
                # 参数已声明 / 类型异常时跳过, 不让节点起不来
                pass
        try:
            params = {k: self.get_parameter(k).value for k in params}
        except Exception:
            pass

        self.params = params
        self.tracker = ConeTracker(params)
        self.topic = params['topic']
        self.twist = Twist()
        self.frame = 0
        self._yellow_active = False

        qos = QoSProfile(depth=5)
        self.pub_ = self.create_publisher(Twist, self.topic, qos)
        self.sub_ = self.create_subscription(LaserScan, '/scan', self.callback, qos)
        # 状态发布 (供 cone.py / rviz / 上层调度使用)
        self.state_pub_ = self.create_publisher(
            String, '/lidar_tracking/state', qos
        )
        self.active_pub_ = self.create_publisher(
            Bool, '/lidar_tracking/control_active', qos
        )
        # 黄线巡线控制权开关
        self.yellow_pub_ = self.create_publisher(
            Bool, params['yellow_enable_topic'], qos
        )
        self.get_logger().info(
            f'lidar_tracking v7 ready | 4-state FSM | center={params["servo_center"]} '
            f'limits=[{params["servo_min"]}, {params["servo_max"]}] '
            f'speed=[{params["speed_straight"]}/{params["speed_corner"]}/'
            f'{params["speed_rightangle"]}] '
            f'| curv in/out={params["curv_enter_corner"]:.2f}/'
            f'{params["curv_exit_corner"]:.2f} ra={params["curv_enter_rightangle"]:.2f}'
            f' | yellow>{params["cone_lost_timeout"]:.1f}s'
        )

        # 启动主动回中 + 默认禁用黄线
        try:
            init = Twist()
            init.linear.x = 1500.0
            init.angular.z = float(params['servo_center'])
            for _ in range(3):
                self.pub_.publish(init)
                self.yellow_pub_.publish(Bool(data=False))
                time.sleep(0.02)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _publish_twist(self, servo, speed):
        self.twist.angular.z = float(servo)
        self.twist.linear.x = float(speed)
        self.pub_.publish(self.twist)

    def _publish_state(self, state):
        try:
            self.state_pub_.publish(String(data=str(state)))
            self.active_pub_.publish(
                Bool(data=(state != STATE_YELLOW))
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    def callback(self, laser):
        now = self.get_clock().now().nanoseconds / 1e9
        servo, speed = self.tracker.step(
            laser.ranges, laser.angle_min, laser.angle_increment, now
        )
        state = self.tracker.state
        self._publish_state(state)

        if state == STATE_YELLOW:
            # 无锥桶: 交出底盘控制权给黄线节点
            if not self._yellow_active:
                self.yellow_pub_.publish(Bool(data=True))
                # 先补一帧中位急停, 避免交接瞬间残留上一条指令
                self._publish_twist(self.tracker.p['servo_center'], 1500.0)
                self._yellow_active = True
            # YELLOW 期间绝不再发 /teleop_cmd_vel, 否则会和黄线的
            # /car_cmd_vel 抢底盘 (driver 里最后写入者胜)
        else:
            if self._yellow_active:
                self.yellow_pub_.publish(Bool(data=False))
                self._yellow_active = False
            self._publish_twist(servo, speed)

        self.frame += 1
        n = int(self.tracker.p['debug_every'])
        if n > 0 and self.frame % n == 0:
            d = self.tracker.dbg
            self.get_logger().info(
                f"[{state}] servo={servo:.1f}° speed={speed:.0f} "
                f"cones={d.get('n_cones', 0)} k={d.get('k_cmd', 0):+.2f} "
                f"kff={d.get('k_ff', 0):+.2f} ki={d.get('k_i', 0):+.3f} "
                f"ey={d.get('err_y', 0):+.3f} |k|={d.get('curv', 0):.2f} "
                f"ra_t={d.get('ra_timer', 0):.2f} lost={d.get('cone_lost_t', 0):.2f} "
                f"clear={d.get('clear', 0):.2f}m"
            )


def main(args=None):
    try:
        from rclpy.signals import SignalHandlerOptions

        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init(args=args)

    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except Exception:
        pass

    node = LidarTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 退出主动驻车回中 + 关闭黄线
        try:
            stop = Twist()
            stop.linear.x = 1500.0
            stop.angular.z = float(node.tracker.p['servo_center'])
            for _ in range(8):
                node.pub_.publish(stop)
                try:
                    node.yellow_pub_.publish(Bool(data=False))
                except Exception:
                    pass
                time.sleep(0.02)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
