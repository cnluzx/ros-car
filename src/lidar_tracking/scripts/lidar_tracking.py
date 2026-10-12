#!/usr/bin/env python3
"""激光雷達錐桶循跡 ROS 2 節點 (v5.1 終極生產版：直道曲率自恢復 + 直角彎專用切線導向 + 80°±45°機械約束)。"""

import csv
import math
import os
import signal
import time
from typing import List, Optional, Tuple
import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import LaserScan
    HAVE_ROS = True
except ImportError:
    HAVE_ROS = False
    Node = object


DEFAULTS = {
    'topic': 'teleop_cmd_vel',

    # ===== 速度控制參數 (PWM: 1500=停止) =====
    'speed_straight': 1560,        # 直道巡航全速 (PWM)
    'speed_corner': 1532,          # 常規圓弧彎道速度 (PWM)
    'speed_sharp': 1522,           # 直角彎低速防推頭 (PWM)
    'danger_slow_drop': 12,        # 逼近錐桶額外減速量 (PWM)

    # ===== 舵機物理標定 (嚴守 80° ± 45° 機械物理限幅) =====
    'servo_center': 80.0,          # 直行中位舵角 [deg]
    'servo_min': 35.0,             # 機械物理右極限 (80 - 45) [deg]
    'servo_max': 125.0,            # 機械物理左極限 (80 + 45) [deg]
    'servo_deg_per_curv': 68.0,    # 轉向增益: 45°對應極限曲率 0.66 [deg/(1/m)]
    'max_curvature': 0.66,         # 物理最大曲率 (最小轉彎半徑約 1.51m) [1/m]
    'steer_gain': 1.35,            # 純追蹤中軸導向係數
    'steer_gain_sharp': 1.45,      # 直角彎快速響應轉向增益
    'max_k_rate': 0.25,            # 每幀曲率變化上限 (杜絕猛打方向)

    # ===== 直道自適應與狀態機門限 =====
    'curv_enter_corner': 0.18,     # 進入常規彎道曲率閾值 [1/m]
    'curv_exit_corner': 0.08,      # 退出常規彎道曲率閾值 [1/m]
    'lookahead_straight': 1.60,    # 直道前瞻距離 [m] (走得直、不蛇形)
    'lookahead_corner': 1.05,      # 彎道前瞻距離 [m] (貼緊內線防外衝)

    # ===== 自適應聚類與半徑推廣 =====
    'cd_range_min': 0.25,          # 雷達掃描盲區下限 [m]
    'cd_range_max': 4.20,          # 雷達掃描有效半徑 [m]
    'cd_fov_deg': 105.0,           # 車頭扇面半角 (共210°超大視野) [deg]
    'cd_gap_base': 0.045,          # 聚類基礎間距 [m]
    'cd_gap_slope': 0.020,         # 聚類發散斜率
    'cd_min_points': 3,            # 錐桶最少激光點數
    'cd_min_extent': 0.018,        # 最小物理尺寸 [m]
    'cd_max_extent': 0.30,         # 最大物理尺寸 [m]
    'cd_cone_radius': 0.08,        # 錐桶底面半徑向心補償 [m]
    'win_x_min': 0.10,             # 車身後方硬盲區 (徹底屏蔽操作員雙腿) [m]

    # ===== 車體系短時記憶 (LocalConeMemory) =====
    'mem_ttl': 1.5,                # 錐桶記憶存活時間 [s] (消除轉角盲區)
    'mem_match_gate': 0.45,        # 幀間點匹配距離門限 [m]
    'mem_min_hits': 2,             # 確認錐桶所需連續命中幀數 (過濾飛點)
    'track_width_nominal': 1.50,   # 單側走廊補償賽道半寬 [m]

    # ===== 直角彎檢測專用參數 =====
    'sharp_front_detect_dist': 2.80, # 直角彎前方外側擋牆檢測距離 [m]
    'sharp_inner_max_dist': 1.80,    # 直角彎內折頂點最大搜索距離 [m]

    # ===== 弧線安全避障層 =====
    'safe_enable': True,
    'safe_lookahead': 1.30,        # 碰撞前瞻檢查弧長 [m]
    'safe_radius': 0.26,           # 最小安全走廊半寬 [m]
    'safe_front': 0.20,            # 車頭距雷達距離 [m]
    'safe_curv_steps': 21,         # 候選曲率採樣數

    # ===== 起步保護 =====
    'start_guard_s': 1.5,          # 起步保護持續時間 [s]
    'start_k_max': 0.20,           # 起步期最大允許轉向曲率

    # ===== 運行記錄與調試 =====
    'debug_every': 5,
    'debug_csv': 1,
    'debug_csv_dir': '~/racecar/logs',
}


def _find_param_yaml():
    """唯一調參入口: config/param.yaml。

    查找優先級:
        1) 環境變量 LIDAR_TRACKING_PARAM (臨時覆蓋, 一般不設)
        2) 工作空間源碼  ~/racecar/src/lidar_tracking/config/param.yaml   ★ 預設就在用這個
        3) 安裝目錄副本  <install>/lidar_tracking/share/lidar_tracking/config/param.yaml
        4) 腳本相對路徑  <script_dir>/../config/param.yaml
        5) 當前目錄      ./config/param.yaml
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.environ.get('LIDAR_TRACKING_PARAM', ''),
        os.path.expanduser('~/racecar/src/lidar_tracking/config/param.yaml'),
        '/home/davinci-mini/racecar/src/lidar_tracking/config/param.yaml',
        os.path.join(here, '..', '..', 'share', 'lidar_tracking',
                     'config', 'param.yaml'),
        os.path.join(here, '..', 'config', 'param.yaml'),
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
        ros_p = (yd.get('/**', {}).get('ros__parameters', {})
                 or yd.get('ros__parameters', {}) or {})
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
        print('[lidar_tracking] 讀取 %s 失敗: %s' % (path, e))
        return {}, path


def detect_cones(ranges, angle_min, angle_inc, p) -> np.ndarray:
    """自適應連續歐氏聚類提取錐桶中心 (車體系 x前, y左)。"""
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
        gx, gy = x[s:e + 1], y[s:e + 1]
        ext = math.hypot(gx.max() - gx.min(), gy.max() - gy.min())
        if ext < p['cd_min_extent'] or ext > p['cd_max_extent']:
            continue
        cx, cy = float(gx.mean()), float(gy.mean())
        rr = math.hypot(cx, cy)
        if rr > 1e-6:
            s_ = (rr + p['cd_cone_radius']) / rr
            cx *= s_
            cy *= s_

        # 盲區過濾 (車身後方硬過濾排除操作員雙腿)
        if cx < p['win_x_min']:
            continue
        # 扇區過濾
        deg = abs(math.degrees(math.atan2(cy, cx)))
        if deg > p['cd_fov_deg']:
            continue
        out.append((cx, cy))
    return np.array(out, dtype=np.float64).reshape(-1, 2)


class LocalConeMemory:
    """車體系短時錐桶記憶 (帶自身運動 Ego-motion 補償與生命週期管理)。"""
    def __init__(self, p):
        self.p = p
        self.memory = []  # [{'x', 'y', 'last_seen', 'hits'}]

    def update(self, det_cones: np.ndarray, v: float, kappa: float, dt: float, now: float) -> np.ndarray:
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
            if (now - m['last_seen'] <= ttl) and (math.hypot(m['x'], m['y']) <= self.p['cd_range_max'] + 1.0):
                survived.append(m)
        self.memory = survived

        confirmed = [[m['x'], m['y']] for m in self.memory if m['hits'] >= self.p['mem_min_hits']]
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
    d = np.hypot(cones[:, 0][None, :] - cx[:, None], cones[:, 1][None, :] - cy[:, None])
    return float(d.min())


def choose_safe_curvature(k_des: float, prev_k: float, cones: np.ndarray, p) -> Tuple[float, float]:
    kmax = p['max_curvature']
    k_des = max(-kmax, min(kmax, k_des))
    if not p['safe_enable'] or cones.shape[0] == 0:
        return k_des, 99.0
    cand = np.unique(np.r_[np.linspace(-kmax, kmax, int(p['safe_curv_steps'])), k_des])
    cls = [(float(k), arc_clearance(float(k), cones, p)) for k in cand]
    safe = [(k, cl) for k, cl in cls if cl >= p['safe_radius']]
    if safe:
        k, cl = min(safe, key=lambda kc: abs(kc[0] - k_des) + 0.15 * abs(kc[0] - prev_k))
        return k, cl
    max_cl = max(cl for _, cl in cls)
    good = [(k, cl) for k, cl in cls if cl >= max_cl - 0.05]
    k, cl = min(good, key=lambda kc: abs(kc[0] - k_des))
    return k, cl


def check_straight_corridor(pts: np.ndarray) -> bool:
    """直道幾何特徵驗證器：判斷當前賽道是否為平行直線走廊。
    當滿足直道特徵時，即使計算出的控制曲率偏大，也強制切回直道狀態。
    """
    if len(pts) < 4:
        return False
    # 1. 前方正向通道無橫跨阻擋
    front_blocked = np.any((pts[:, 0] >= 0.5) & (pts[:, 0] <= 2.2) & (abs(pts[:, 1]) <= 0.45))
    if front_blocked:
        return False
    # 2. 錐桶縱向延伸深度充足 (X方向縱深 >= 2.0m)
    if pts[:, 0].max() < 2.0:
        return False
    # 3. 左右兩側均有連續走廊隊列 (縱向跨度均 >= 1.2m)
    left = pts[pts[:, 1] > 0.2]
    right = pts[pts[:, 1] < -0.2]
    if len(left) < 2 or len(right) < 2:
        return False
    l_span = left[:, 0].max() - left[:, 0].min()
    r_span = right[:, 0].max() - right[:, 0].min()
    if l_span < 1.2 or r_span < 1.2:
        return False
    return True


def detect_sharp_corner(pts: np.ndarray, p) -> Tuple[bool, int]:
    """直角彎特徵檢測器：
    返回 (is_sharp, sharp_dir)，sharp_dir: +1 為左直角，-1 為右直角，0 為非直角。
    """
    if len(pts) < 4:
        return False, 0
    # 1. 檢查前方外牆死胡同 (前方 0.6m ~ sharp_front_detect_dist 存在橫向擋牆)
    f_dist = p['sharp_front_detect_dist']
    front_cones = pts[(pts[:, 0] >= 0.6) & (pts[:, 0] <= f_dist) & (abs(pts[:, 1]) <= 0.60)]
    if len(front_cones) == 0:
        return False, 0

    # 2. 檢查單側開闊出口走廊
    l_open = np.sum((pts[:, 1] > 0.6) & (pts[:, 0] <= 2.5))
    r_open = np.sum((pts[:, 1] < -0.6) & (pts[:, 0] <= 2.5))

    if l_open >= 2 and r_open <= 1:
        return True, 1   # 左直角彎
    elif r_open >= 2 and l_open <= 1:
        return True, -1  # 右直角彎
    return False, 0


class ConeTracker:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS)
        if params:
            self.p.update(params)
        self.mem = LocalConeMemory(self.p)
        self.prev_k = 0.0
        self.prev_servo = self.p['servo_center']
        self.smooth_k = 0.0
        self.is_in_corner = False
        self.t_start = None
        self.mode = 'INIT'
        self.dbg = {}

    def step(self, ranges, angle_min, angle_inc, now: float) -> Tuple[float, float]:
        p = self.p
        if self.t_start is None:
            self.t_start = now
        dt = 0.1

        # 1. 檢測與記憶更新
        raw_cones = detect_cones(ranges, angle_min, angle_inc, p)
        cones = self.mem.update(raw_cones, 0.75, self.prev_k, dt, now)
        pts = cones[cones[:, 0] >= 0.2] if cones.shape[0] > 0 else np.zeros((0, 2))

        # 2. 幾何特徵檢測 (直道 vs 直角彎)
        is_straight_geom = check_straight_corridor(pts)
        is_sharp, sharp_dir = detect_sharp_corner(pts, p)

        aim = None
        mode = 'STRAIGHT'
        v_target = p['speed_straight']
        steer_gain = p['steer_gain']

        # -------------------------------------------------------------
        # 控制模式分支 A：直角彎專用算法 (Sharp Corner Pivot Guidance)
        # -------------------------------------------------------------
        if is_sharp:
            mode = 'SHARP_CORNER_LEFT' if sharp_dir == 1 else 'SHARP_CORNER_RIGHT'
            v_target = p['speed_sharp']
            steer_gain = p['steer_gain_sharp']

            left = pts[pts[:, 1] > 0]
            right = pts[pts[:, 1] <= 0]
            max_inner_d = p['sharp_inner_max_dist']
            half_w = 0.5 * p['track_width_nominal']

            if sharp_dir == 1:
                # 左直角彎: 鎖定左側內折頂點
                inner = left[(left[:, 0] >= -0.1) & (left[:, 0] <= max_inner_d)]
                if len(inner):
                    pivot = inner[np.argmin(np.hypot(inner[:, 0], inner[:, 1]))]
                    aim = np.array([max(0.3, pivot[0] + 0.15), pivot[1] + half_w])
                else:
                    aim = np.array([0.5, 1.2])
            else:
                # 右直角彎: 鎖定右側內折頂點
                inner = right[(right[:, 0] >= -0.1) & (right[:, 0] <= max_inner_d)]
                if len(inner):
                    pivot = inner[np.argmin(np.hypot(inner[:, 0], inner[:, 1]))]
                    aim = np.array([max(0.3, pivot[0] + 0.15), pivot[1] - half_w])
                else:
                    aim = np.array([0.5, -1.2])

            self.is_in_corner = True
            self.smooth_k = 0.40

        # -------------------------------------------------------------
        # 控制模式分支 B：直道幾何優先 (直道幾何存在時強制切回直道狀態)
        # -------------------------------------------------------------
        elif is_straight_geom:
            mode = 'STRAIGHT'
            v_target = p['speed_straight']
            la = p['lookahead_straight']
            self.is_in_corner = False
            self.smooth_k = 0.0  # 重置濾波曲率，杜絕誤觸發彎道

            dists = np.hypot(pts[:, 0], pts[:, 1])
            cand = pts[(dists >= la * 0.7) & (dists <= la * 1.5)]
            if len(cand) >= 2:
                cl = cand[cand[:, 1] > 0]
                cr = cand[cand[:, 1] <= 0]
                half_w = 0.5 * p['track_width_nominal']
                if len(cl) and len(cr):
                    aim = 0.5 * (cl.mean(0) + cr.mean(0))
                elif len(cl):
                    aim = np.array([cl[:, 0].mean(), cl[:, 1].mean() - half_w])
                else:
                    aim = np.array([cr[:, 0].mean(), cr[:, 1].mean() + half_w])

        # -------------------------------------------------------------
        # 控制模式分支 C：常規圓弧彎道循跡 (帶施密特遲滯狀態機)
        # -------------------------------------------------------------
        else:
            la = p['lookahead_corner'] if self.is_in_corner else p['lookahead_straight']
            v_target = p['speed_corner'] if self.is_in_corner else p['speed_straight']
            mode = 'CORNER' if self.is_in_corner else 'STRAIGHT'

            dists = np.hypot(pts[:, 0], pts[:, 1])
            cand = pts[(dists >= la * 0.7) & (dists <= la * 1.5)]
            if len(cand) >= 2:
                cl = cand[cand[:, 1] > 0]
                cr = cand[cand[:, 1] <= 0]
                half_w = 0.5 * p['track_width_nominal']
                if len(cl) and len(cr):
                    aim = 0.5 * (cl.mean(0) + cr.mean(0))
                elif len(cl):
                    aim = np.array([cl[:, 0].mean(), cl[:, 1].mean() - half_w])
                else:
                    aim = np.array([cr[:, 0].mean(), cr[:, 1].mean() + half_w])

        if aim is None:
            la_fallback = p['lookahead_straight'] if mode == 'STRAIGHT' else p['lookahead_corner']
            aim = np.array([la_fallback, 0.0])

        # 4. 純追蹤期望曲率計算
        L_aim = max(0.45, math.hypot(aim[0], aim[1]))
        k_des = steer_gain * (2.0 * aim[1]) / (L_aim * L_aim)
        k_des = max(-p['max_curvature'], min(p['max_curvature'], k_des))

        # 5. 安全避障與變化率濾波
        k_safe, clear = choose_safe_curvature(k_des, self.prev_k, cones, p)
        max_step = float(p['max_k_rate'])
        k_cmd = self.prev_k + np.clip(k_safe - self.prev_k, -max_step, max_step)

        # 6. 起步保護
        if now - self.t_start < p['start_guard_s']:
            k_cmd = np.clip(k_cmd, -p['start_k_max'], p['start_k_max'])

        self.prev_k = k_cmd

        # 7. 常規遲滯狀態機更新
        if not is_sharp and not is_straight_geom:
            self.smooth_k = 0.85 * self.smooth_k + 0.15 * abs(k_cmd)
            if not self.is_in_corner:
                if self.smooth_k > p['curv_enter_corner']:
                    self.is_in_corner = True
            else:
                if self.smooth_k < p['curv_exit_corner']:
                    self.is_in_corner = False
            mode = 'CORNER' if self.is_in_corner else 'STRAIGHT'

        self.mode = mode

        # 8. 舵機映射與 80° ± 45° 物理硬限幅
        servo = p['servo_center'] + p['servo_deg_per_curv'] * k_cmd
        servo = max(p['servo_min'], min(p['servo_max'], servo))
        self.prev_servo = servo

        # 9. 狀態自適應車速
        speed = v_target
        if p['safe_enable'] and clear < p['safe_radius']:
            speed -= p['danger_slow_drop']
            self.mode += '+DANGER'

        self.dbg = dict(n_cones=int(cones.shape[0]), aim=aim, k_cmd=k_cmd,
                        clear=clear, curv=self.smooth_k, is_straight_geom=is_straight_geom)
        return servo, speed


class LidarTrackingNode(Node):
    def __init__(self):
        super().__init__('lidar_tracking_node')
        params = dict(DEFAULTS)
        file_params, ppath = load_param_yaml()
        params.update(file_params)
        self.tracker = ConeTracker(params)
        self.topic = params['topic']
        self.twist = Twist()
        self.frame = 0

        qos = QoSProfile(depth=5)
        self.pub_ = self.create_publisher(Twist, self.topic, qos)
        self.sub_ = self.create_subscription(LaserScan, '/scan', self.callback, qos)

        # ★ 明確打印實際使用的參數文件, 避免"改了沒生效"的困惑
        self.get_logger().info('==============================================')
        self.get_logger().info(f'[param] 參數文件 = {ppath}')
        self.get_logger().info(
            f'[param] speed 直道/彎道/直角 = '
            f'{params["speed_straight"]}/{params["speed_corner"]}/{params["speed_sharp"]}'
        )
        self.get_logger().info(
            f'[param] servo center={params["servo_center"]} '
            f'limits=[{params["servo_min"]}, {params["servo_max"]}] '
            f'deg_per_curv={params["servo_deg_per_curv"]} '
            f'max_curv={params["max_curvature"]}'
        )
        self.get_logger().info(
            f'[param] 狀態機 curv 進/出={params["curv_enter_corner"]}/'
            f'{params["curv_exit_corner"]} '
            f'前瞻 直/彎={params["lookahead_straight"]}/{params["lookahead_corner"]}'
        )
        self.get_logger().info(
            f'lidar_tracking v5.1 ready | center={params["servo_center"]} '
            f'speed_straight={params["speed_straight"]} speed_corner={params["speed_corner"]}'
        )
        self.get_logger().info('==============================================')

        # 啟動主動回中
        try:
            init = Twist()
            init.linear.x = 1500.0
            init.angular.z = float(params['servo_center'])
            for _ in range(3):
                self.pub_.publish(init)
                time.sleep(0.02)
        except Exception:
            pass

    def callback(self, laser):
        now = self.get_clock().now().nanoseconds / 1e9
        servo, speed = self.tracker.step(laser.ranges, laser.angle_min,
                                         laser.angle_increment, now)
        self.twist.angular.z = float(servo)
        self.twist.linear.x = float(speed)
        self.pub_.publish(self.twist)

        self.frame += 1
        n = int(self.tracker.p['debug_every'])
        if n > 0 and self.frame % n == 0:
            d = self.tracker.dbg
            self.get_logger().info(
                f"[{self.tracker.mode}] servo={servo:.1f}° speed={speed:.0f} "
                f"cones={d.get('n_cones', 0)} k={d.get('k_cmd', 0):+.2f} "
                f"|k|={d.get('curv', 0):.2f} clear={d.get('clear', 0):.2f}m")


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
        # 退出主動駐車回中
        try:
            stop = Twist()
            stop.linear.x = 1500.0
            stop.angular.z = float(node.tracker.p['servo_center'])
            for _ in range(8):
                node.pub_.publish(stop)
                time.sleep(0.02)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
