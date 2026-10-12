#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cone_show.py - Live cone detector using v6 algorithm (no presets)

Subscribes to /scan (or reads a replay file), runs the same
detect_cones + LocalConeMemory + ConeTracker as the v6 node, and
draws raw scan points + detected cones + tracker status.

No synthetic scenes. No fake data.

Usage:
    # live ROS
    source /opt/ros/humble/setup.bash
    source ~/racecar/install/setup.bash
    python3 cone_show.py

    # replay
    python3 cone_show.py --replay scans.txt

    # headless (no window), save frames
    python3 cone_show.py --save-dir /tmp/shots --max-frames 50

Keys (window mode):
    a       toggle aim/arc overlay
    s       save screenshot
    q/ESC   quit
"""

import argparse
import math
import os
import sys
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:
    print('Need OpenCV'); sys.exit(1)

HAVE_ROS = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import LaserScan
    HAVE_ROS = True
except ImportError:
    pass


# ===================================================================
# v6 PARAMS (copied from DEFAULTS)
# ===================================================================
P = {
    'cd_range_min': 0.25,
    'cd_range_max': 4.20,
    'cd_fov_deg': 105.0,
    'cd_gap_base': 0.045,
    'cd_gap_slope': 0.020,
    'cd_min_points': 3,
    'cd_min_extent': 0.018,
    'cd_max_extent': 0.30,
    'cd_cone_radius': 0.08,
    'win_x_min': 0.10,
    'mem_ttl': 1.5,
    'mem_match_gate': 0.45,
    'mem_min_hits': 2,
    'track_width_nominal': 1.50,
    'safe_enable': True,
    'safe_lookahead': 1.30,
    'safe_radius': 0.26,
    'safe_front': 0.20,
    'safe_curv_steps': 21,
    'servo_center': 80.0,
    'servo_min': 35.0,
    'servo_max': 125.0,
    'servo_deg_per_curv': 68.0,
    'max_curvature': 0.66,
    'steer_gain': 1.35,
    'max_k_rate': 0.25,
    'curv_enter_corner': 0.18,
    'curv_exit_corner': 0.08,
    'lookahead_straight': 1.60,
    'lookahead_corner': 1.05,
    'speed_straight': 1560,
    'speed_corner': 1552,
    'danger_slow_drop': 12,
    'start_guard_s': 1.5,
    'start_k_max': 0.20,
    'pid_kp': 0.0,
    'pid_ki': 0.005,
    'pid_kd': 0.0,
    'pid_i_limit': 0.02,
    'pid_i_decay': 0.90,
    'pid_i_enable_corner': False,
}


# ===================================================================
# v6 detect_cones
# ===================================================================
def detect_cones(ranges, angle_min, angle_inc, p):
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
            k = (rr + p['cd_cone_radius']) / rr
            cx *= k
            cy *= k
        if cx < p['win_x_min']:
            continue
        deg = abs(math.degrees(math.atan2(cy, cx)))
        if deg > p['cd_fov_deg']:
            continue
        out.append((cx, cy))
    return np.array(out, dtype=np.float64).reshape(-1, 2)


# ===================================================================
# v6 LocalConeMemory
# ===================================================================
class LocalConeMemory:
    def __init__(self, p):
        self.p = p
        self.memory = []

    def update(self, det_cones, v, kappa, dt, now):
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
                self.memory.append({'x': cx, 'y': cy,
                                    'last_seen': now, 'hits': 1})

        ttl = self.p['mem_ttl']
        survived = []
        for m in self.memory:
            if (now - m['last_seen'] <= ttl) and (
                    math.hypot(m['x'], m['y']) <= self.p['cd_range_max'] + 1.0):
                survived.append(m)
        self.memory = survived

        confirmed = [[m['x'], m['y']] for m in self.memory
                     if m['hits'] >= self.p['mem_min_hits']]
        if not confirmed and len(det_cones) > 0:
            return det_cones
        return np.array(confirmed, dtype=np.float64).reshape(-1, 2)


# ===================================================================
# v6 arc_clearance + choose_safe_curvature
# ===================================================================
def arc_clearance(kappa, cones, p):
    if cones.shape[0] == 0:
        return 99.0
    s = np.linspace(0.1, p['safe_lookahead'], 15)
    if abs(kappa) < 1e-3:
        px = s; py = np.zeros_like(s)
        hx = np.ones_like(s); hy = np.zeros_like(s)
    else:
        a = kappa * s
        px = np.sin(a) / kappa
        py = (1.0 - np.cos(a)) / kappa
        hx, hy = np.cos(a), np.sin(a)
    cx = px + 0.5 * p['safe_front'] * hx
    cy = py + 0.5 * p['safe_front'] * hy
    d = np.hypot(cones[:, 0][None, :] - cx[:, None],
                 cones[:, 1][None, :] - cy[:, None])
    return float(d.min())


def choose_safe_curvature(k_des, prev_k, cones, p):
    kmax = p['max_curvature']
    k_des = max(-kmax, min(kmax, k_des))
    if not p['safe_enable'] or cones.shape[0] == 0:
        return k_des, 99.0
    cand = np.unique(np.r_[np.linspace(-kmax, kmax, int(p['safe_curv_steps'])),
                           k_des])
    cls = [(float(k), arc_clearance(float(k), cones, p)) for k in cand]
    safe = [(k, cl) for k, cl in cls if cl >= p['safe_radius']]
    if safe:
        k, cl = min(safe, key=lambda kc: abs(kc[0] - k_des)
                    + 0.15 * abs(kc[0] - prev_k))
        return k, cl
    max_cl = max(cl for _, cl in cls)
    good = [(k, cl) for k, cl in cls if cl >= max_cl - 0.05]
    k, cl = min(good, key=lambda kc: abs(kc[0] - k_des))
    return k, cl


# ===================================================================
# v6 ConeTracker
# ===================================================================
class ConeTracker:
    def __init__(self, params=None):
        self.p = dict(P)
        if params:
            self.p.update(params)
        self.mem = LocalConeMemory(self.p)
        self.prev_k = 0.0
        self.smooth_k = 0.0
        self.is_in_corner = False
        self.t_start = None
        self.mode = 'INIT'
        self.dbg = {}
        self.k_i = 0.0
        self.prev_err_y = 0.0
        self.last_time = None

    def step(self, ranges, angle_min, angle_inc, now):
        p = self.p
        if self.t_start is None:
            self.t_start = now
        dt = 0.1

        raw = detect_cones(ranges, angle_min, angle_inc, p)
        cones = self.mem.update(raw, 0.75, self.prev_k, dt, now)

        la = p['lookahead_corner'] if self.is_in_corner else p['lookahead_straight']

        pts = cones[cones[:, 0] >= 0.3] if cones.shape[0] > 0 else np.zeros((0, 2))
        aim = None
        if len(pts) >= 2:
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
            aim = np.array([la, 0.0])

        L_aim = max(0.5, math.hypot(aim[0], aim[1]))
        k_ff = p['steer_gain'] * (2.0 * aim[1]) / (L_aim * L_aim)
        k_ff = max(-p['max_curvature'], min(p['max_curvature'], k_ff))

        if self.last_time is None:
            dt_pid = 0.1
        else:
            dt_pid = max(0.01, min(0.5, now - self.last_time))
        self.last_time = now

        err_y = -float(aim[1])
        p_term = p['pid_kp'] * err_y
        i_enable = (not self.is_in_corner) or p['pid_i_enable_corner']
        if i_enable:
            self.k_i += p['pid_ki'] * err_y * dt_pid
            lim = p['pid_i_limit']
            self.k_i = max(-lim, min(lim, self.k_i))
        i_term = self.k_i
        d_term = 0.0
        if p['pid_kd'] > 0.0:
            d_err = (err_y - self.prev_err_y) / dt_pid
            d_term = p['pid_kd'] * d_err
        self.prev_err_y = err_y

        k_des = k_ff + p_term + i_term + d_term
        k_des = max(-p['max_curvature'], min(p['max_curvature'], k_des))
        if not self.is_in_corner:
            self.k_i *= p['pid_i_decay']

        k_safe, clear = choose_safe_curvature(k_des, self.prev_k, cones, p)
        max_step = float(p['max_k_rate'])
        k_cmd = self.prev_k + np.clip(k_safe - self.prev_k, -max_step, max_step)
        if now - self.t_start < p['start_guard_s']:
            k_cmd = np.clip(k_cmd, -p['start_k_max'], p['start_k_max'])
        self.prev_k = k_cmd

        self.smooth_k = 0.85 * self.smooth_k + 0.15 * abs(k_cmd)
        if not self.is_in_corner:
            if self.smooth_k > p['curv_enter_corner']:
                self.is_in_corner = True
        else:
            if self.smooth_k < p['curv_exit_corner']:
                self.is_in_corner = False
        self.mode = 'CORNER' if self.is_in_corner else 'STRAIGHT'

        servo = p['servo_center'] + p['servo_deg_per_curv'] * k_cmd
        servo = max(p['servo_min'], min(p['servo_max'], servo))

        speed = p['speed_corner'] if self.is_in_corner else p['speed_straight']
        if p['safe_enable'] and clear < p['safe_radius']:
            speed -= p['danger_slow_drop']
            self.mode += '+DANGER'

        self.dbg = dict(
            n_cones=int(cones.shape[0]),
            aim=aim,
            k_cmd=k_cmd,
            k_ff=k_ff,
            k_i=self.k_i,
            err_y=err_y,
            clear=clear,
            servo=servo,
            speed=speed,
        )
        return servo, speed, cones, raw


# ===================================================================
# Shared state
# ===================================================================
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.angles = np.zeros(0)
        self.ranges = np.zeros(0)
        self.raw = None
        self.mem = None
        self.dbg = {}
        self.mode = 'INIT'
        self.hz = 0.0
        self.ms = 0.0
        self.frame = 0
        self.source = 'none'


SH = Shared()


# ===================================================================
# ROS node
# ===================================================================
if HAVE_ROS:
    class ConeShowNode(Node):
        def __init__(self, tracker, topic):
            super().__init__('cone_show')
            self.tracker = tracker
            self.stamps = []
            self.times = []
            self.create_subscription(LaserScan, topic, self.on_scan, 5)
            self.get_logger().info('subscribing to %s' % topic)

        def on_scan(self, msg):
            t0 = time.perf_counter()
            r = np.asarray(msg.ranges, dtype=np.float64)
            ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment
            now = self.get_clock().now().nanoseconds / 1e9

            servo, speed, mem, raw = self.tracker.step(
                r, msg.angle_min, msg.angle_increment, now)
            ms = (time.perf_counter() - t0) * 1000.0

            self.stamps.append(now)
            while self.stamps and now - self.stamps[0] > 2.0:
                self.stamps.pop(0)
            if len(self.stamps) >= 2:
                span = self.stamps[-1] - self.stamps[0]
                hz = (len(self.stamps) - 1) / span if span > 1e-6 else 0.0
            else:
                hz = 0.0
            self.times.append(ms)
            if len(self.times) > 100:
                self.times.pop(0)
            ms_avg = sum(self.times) / len(self.times)

            with SH.lock:
                SH.angles = ang
                SH.ranges = r
                SH.raw = raw
                SH.mem = mem
                SH.dbg = dict(self.tracker.dbg)
                SH.mode = self.tracker.mode
                SH.hz = hz
                SH.ms = ms_avg
                SH.frame += 1
                SH.source = 'scan'


# ===================================================================
# Render
# ===================================================================
CW, CH = 900, 900
BG = (24, 28, 34)
COL_SCAN = (110, 122, 138)
COL_RAW = (90, 90, 100)
COL_CONE = (60, 190, 250)
COL_AIM = (255, 200, 60)
COL_CAR = (120, 230, 160)
COL_ARC = (255, 100, 100)
SCALE_REF = [90.0]
SHOW_OV = [True]


def w2p(x, y):
    s = SCALE_REF[0]
    ox = CW // 2
    oy = CH - 120
    return int(ox - y * s), int(oy - x * s)


def render():
    with SH.lock:
        ang = SH.angles.copy()
        rng = SH.ranges.copy()
        raw = SH.raw
        mem = SH.mem
        dbg = dict(SH.dbg)
        mode = SH.mode
        hz = SH.hz
        ms = SH.ms
        frame = SH.frame
        source = SH.source
        show_ov = SHOW_OV[0]

    s = SCALE_REF[0]
    img = np.full((CH, CW, 3), BG, dtype=np.uint8)
    ox, oy = CW // 2, CH - 120

    max_m = max(CW, CH) / s
    gstep = 0.5
    while gstep * s < 45 and gstep < 8.0:
        gstep *= 2.0
    n = int(max_m / gstep) + 2
    for i in range(-n, n + 1):
        col = (44, 50, 58) if i % 2 else (34, 39, 46)
        px, _ = w2p(0, i * gstep)
        if 0 <= px < CW:
            cv2.line(img, (px, 0), (px, CH), col, 1)
        _, py = w2p(i * gstep, 0)
        if 0 <= py < CH:
            cv2.line(img, (0, py), (CW, py), col, 1)

    for ring in (1.0, 2.0, 3.0):
        cv2.circle(img, (ox, oy), int(ring * s), (58, 66, 76), 1, cv2.LINE_AA)
        cv2.putText(img, '%dm' % ring, (ox + int(ring * s) - 26, oy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 120, 134), 1,
                    cv2.LINE_AA)

    # raw scan points
    if len(rng) > 0:
        ok = np.isfinite(rng) & (rng > 0.05) & (rng < 12.0)
        if ok.any():
            rv = rng[ok]
            av = ang[ok]
            px = (ox - (rv * np.sin(av)) * s).astype(np.int32)
            py = (oy - (rv * np.cos(av)) * s).astype(np.int32)
            inb = (px >= 0) & (px < CW) & (py >= 0) & (py < CH)
            if inb.any():
                img[py[inb], px[inb]] = COL_SCAN

    # raw detect_cones output (gray dot)
    if raw is not None and len(raw):
        for (cx, cy) in raw:
            X, Y = w2p(cx, cy)
            cv2.circle(img, (X, Y), 6, COL_RAW, 1, cv2.LINE_AA)

    # confirmed cones after memory (blue circle)
    if mem is not None and len(mem):
        for (cx, cy) in mem:
            X, Y = w2p(cx, cy)
            cv2.circle(img, (X, Y), 9, COL_CONE, 2, cv2.LINE_AA)
            cv2.circle(img, (X, Y), 2, COL_CONE, -1, cv2.LINE_AA)
            cv2.line(img, (ox, oy), (X, Y),
                     tuple(int(v * 0.35) for v in COL_CONE), 1, cv2.LINE_AA)

    if show_ov:
        aim = dbg.get('aim')
        if aim is not None:
            X, Y = w2p(float(aim[0]), float(aim[1]))
            cv2.circle(img, (X, Y), 11, COL_AIM, 2, cv2.LINE_AA)
            cv2.drawMarker(img, (X, Y), COL_AIM, cv2.MARKER_CROSS, 16, 2)
            cv2.line(img, (ox, oy), (X, Y), COL_AIM, 1, cv2.LINE_AA)

        k_cmd = float(dbg.get('k_cmd', 0.0))
        L = 1.3
        ss = np.arange(0.0, L + 1e-9, 0.06)
        if abs(k_cmd) < 1e-3:
            pix = np.array([w2p(float(v), 0.0) for v in ss], dtype=np.int32)
        else:
            pix = np.array([w2p(float(np.sin(k_cmd * v) / k_cmd),
                                float((1.0 - np.cos(k_cmd * v)) / k_cmd))
                            for v in ss], dtype=np.int32)
        if len(pix) >= 2:
            cv2.polylines(img, [pix], False, COL_ARC, 2, cv2.LINE_AA)

    car = np.array([w2p(0.30, 0.18), w2p(0.30, -0.18),
                    w2p(-0.26, -0.18), w2p(-0.26, 0.18)], dtype=np.int32)
    cv2.fillPoly(img, [car], COL_CAR)
    nose = np.array([w2p(0.42, 0.10), w2p(0.30, 0.0),
                     w2p(0.42, -0.10)], dtype=np.int32)
    cv2.fillPoly(img, [nose], COL_CAR)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (CW, 180), (18, 21, 26), -1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)

    def txt(s_, x, y, col=(220, 228, 238), sz=0.52, th=1):
        cv2.putText(img, s_, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sz, col, th,
                    cv2.LINE_AA)

    if 'STRAIGHT' in mode:
        mcol = (120, 240, 160)
    elif 'CORNER' in mode:
        mcol = (80, 200, 255)
    else:
        mcol = (200, 200, 200)

    n_raw = len(raw) if raw is not None else 0
    n_mem = len(mem) if mem is not None else 0

    txt('CONE SHOW   [%s]' % mode, 14, 32, mcol, 0.82, 2)
    txt('servo %.1f deg   speed %.0f PWM'
        % (dbg.get('servo', 0.0), dbg.get('speed', 0.0)),
        14, 62, (210, 225, 240), 0.58, 1)
    txt('k_cmd %+.3f   k_ff %+.3f   k_i %+.3f   err_y %+.3f'
        % (dbg.get('k_cmd', 0.0), dbg.get('k_ff', 0.0),
           dbg.get('k_i', 0.0), dbg.get('err_y', 0.0)),
        14, 88, (200, 220, 240), 0.55, 1)
    txt('clear %.2f m   cones raw/mem %d/%d'
        % (dbg.get('clear', 99.0), n_raw, n_mem),
        14, 114, (200, 220, 240), 0.55, 1)
    txt('%s   %.1f Hz   %.1f ms   frame %d'
        % (source, hz, ms, frame), 14, 140, (150, 200, 240), 0.5, 1)
    txt('blue = confirmed cones (memory)   gray dot = raw detect',
        14, 164, (150, 160, 180), 0.46, 1)
    txt('[a]overlay [s]ave [q]uit', 14, CH - 12,
        (130, 140, 155), 0.46, 1)

    return img


# ===================================================================
# Replay
# ===================================================================
def load_scans_file(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return []
    out = []
    first = lines[0].split()
    header_style = len(first) == 2
    if header_style:
        amin_g = float(first[0]); inc_g = float(first[1])
        body = lines[1:]
    else:
        amin_g = 0.0; inc_g = None
        body = lines
    for ln in body:
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            if header_style:
                n = int(parts[1]); vals = parts[2:2 + n]
                inc = inc_g; amin = amin_g
            else:
                inc = float(parts[1]); n = int(parts[2])
                vals = parts[3:3 + n]; amin = amin_g
            arr = np.array([np.inf if v in ('inf', 'Inf', 'INF')
                            else (np.nan if v in ('nan', 'NaN') else float(v))
                            for v in vals], dtype=np.float64)
            out.append((inc, amin, arr))
        except (ValueError, IndexError):
            continue
    if out and inc_g is None:
        c = out[0][0]
        out = [(c,) + f[1:] for f in out]
    return out


class ReplayRunner(threading.Thread):
    def __init__(self, tracker, frames, loop_hz=10.0):
        super().__init__(daemon=True)
        self.tracker = tracker
        self.frames = frames
        self.loop_hz = loop_hz
        self.stop_flag = False

    def run(self):
        idx = 0
        while not self.stop_flag:
            inc, amin, arr = self.frames[idx]
            now = time.time()
            t0 = time.perf_counter()
            servo, speed, mem, raw = self.tracker.step(arr, amin, inc, now)
            ms = (time.perf_counter() - t0) * 1000.0
            with SH.lock:
                SH.angles = amin + np.arange(len(arr)) * inc
                SH.ranges = arr
                SH.raw = raw
                SH.mem = mem
                SH.dbg = dict(self.tracker.dbg)
                SH.mode = self.tracker.mode
                SH.ms = ms
                SH.frame += 1
                SH.source = 'replay'
            idx = (idx + 1) % len(self.frames)
            time.sleep(1.0 / self.loop_hz)


# ===================================================================
# Main
# ===================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', type=str, default='/scan')
    ap.add_argument('--replay', type=str, default=None)
    ap.add_argument('--save-dir', type=str, default=None)
    ap.add_argument('--max-frames', type=int, default=50)
    ap.add_argument('--scale', type=float, default=90.0)
    ap.add_argument('--save', type=str, default=None,
                    help='Save one frame and exit (replay mode)')
    args = ap.parse_args()

    SCALE_REF[0] = args.scale
    tracker = ConeTracker({})

    runner = None
    ros_node = None

    if args.replay:
        frames = load_scans_file(args.replay)
        if not frames:
            print('no frames in replay'); sys.exit(1)
        print('replay: %d frames' % len(frames))
        runner = ReplayRunner(tracker, frames, loop_hz=10.0)
        runner.start()
    else:
        if not HAVE_ROS:
            print('ERROR: rclpy not available. Source ROS, or use --replay.')
            sys.exit(1)
        rclpy.init()
        ros_node = ConeShowNode(tracker, args.topic)
        threading.Thread(target=rclpy.spin, args=(ros_node,),
                         daemon=True).start()
        print('source: live /scan')

    # One-shot save
    if args.save:
        time.sleep(0.5)
        img = render()
        cv2.imwrite(args.save, img)
        print('saved:', args.save)
        if runner is not None:
            runner.stop_flag = True
        if ros_node is not None:
            try: ros_node.destroy_node()
            except Exception: pass
            try: rclpy.shutdown()
            except Exception: pass
        return

    # Headless save-dir
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for i in range(args.max_frames):
            time.sleep(0.1)
            img = render()
            path = os.path.join(args.save_dir, 'frame_%04d.png' % i)
            cv2.imwrite(path, img)
            print('saved:', path)
        if runner is not None:
            runner.stop_flag = True
        if ros_node is not None:
            try: ros_node.destroy_node()
            except Exception: pass
            try: rclpy.shutdown()
            except Exception: pass
        return

    # Window mode
    cv2.namedWindow('cone_show', cv2.WINDOW_AUTOSIZE)
    print('window open - press q to quit')

    while True:
        img = render()
        cv2.imshow('cone_show', img)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord('a'):
            SHOW_OV[0] = not SHOW_OV[0]
        elif k == ord('s'):
            cv2.imwrite('cone_show_shot.png', img)
            print('saved: cone_show_shot.png')

    cv2.destroyAllWindows()
    if runner is not None:
        runner.stop_flag = True
    if ros_node is not None:
        try: ros_node.destroy_node()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass


if __name__ == '__main__':
    main()