#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cone.py —— 雷达锥桶检测 / 计数 / 可视化调参工具

订阅 /scan，用「自适应聚类」找出锥桶，并提供一个内置的 Web 可视化面板：
    http://<车IP>:8080

面板功能：
    · 俯视图实时显示：原始扫描点、聚类、检出的锥桶（左/右分类）、车辆朝向
    · 左侧滑块实时调参，无需重启节点
    · 实时统计：锥桶总数、左/右数量、聚类数、单帧耗时
    · 点击「保存参数」把当前参数写入 JSON 文件

为什么用 Web 面板而不是 cv2.imshow：
    车端 DISPLAY 未设置，OpenCV 的 GTK 后端无法连接 X server
    （实测报 "Client is not authorized to connect to Server"）。
    用浏览器访问就不需要 X11，也不依赖 rviz。

与 lidar_tracking.cpp 的关系：
    本脚本是**独立诊断工具**，不发布任何控制指令，不碰 /teleop_cmd_vel。
    可以和安全地跟旧节点同时运行。

用法：
    source /opt/ros/humble/setup.bash
    source ~/racecar/install/setup.bash
    python3 cone.py                          # 用 /scan 实时数据
    python3 cone.py --port 8080              # 指定 Web 端口
    python3 cone.py --replay scans.txt       # 离线回放（循环播放，便于调参）
    python3 cone.py --no-web                 # 只打印统计，不开 Web
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

# ------------------------------------------------------------------ ROS
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    HAVE_ROS = True
except ImportError:
    HAVE_ROS = False

# ------------------------------------------------------------------ OpenCV
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    cv2 = None            # 显式绑定为模块级名字，供 render() 等函数使用
    HAVE_CV2 = False


# ===================================================================
# 检测算法
# ===================================================================
DEFAULT_PARAMS = {
    # --- 有效距离 ---
    'range_min': 0.15,      # 最近有效距离 [m]（雷达自身 range_min 之上再收紧）
    'range_max': 5.50,      # 最远有效距离 [m]

    # --- 自适应聚类 ---
    # 相邻两点的"期望间距"随距离线性增长：gap = base + slope * r
    # 雷达角分辨率约 0.0064 rad，在 r 米处相邻束间距约 0.0064*r
    # base 覆盖锥桶表面连续回波，slope 保证远处不会把锥桶切断
    'gap_base': 0.045,      # 基础间距 [m]
    'gap_slope': 0.020,     # 距离系数 [m/m]

    # --- 尺寸过滤（锥桶底座 0.28m，高 0.63m；回波只覆盖可见面）---
    # 实测：extent 撞到上限（0.40）的那些簇基本都是墙面/挡板，不是锥桶。
    #       真实锥桶可见面的回波 extent 通常在 0.02~0.20 之间。
    'min_points': 3,        # 一个簇至少这么多点（2 点时噪声太多）
    'min_extent': 0.018,    # 簇最小尺寸 [m]
    'max_extent': 0.25,     # 簇最大尺寸 [m]（超过认为是墙/挡板）

    # --- 锥桶半径补偿（回波在可见面，沿视线外推到表面）---
    'cone_radius': 0.10,    # 锥桶半径 [m]
    'min_dist': 0.40,       # 离车体最近距离 [m]

    # --- 左右分类窗口（车体坐标系：x 前、y 左）---
    # 【颜色约定·已由用户确认】
    #   赛道为环形，逆时针运行 → 左侧 = 红锥（内圈）、右侧 = 蓝锥（外圈）
    #   所以 y > 0 是【红锥】，y < 0 是【蓝锥】。
    #   注意：雷达本身无法分辨颜色，这是按规则的空间不变性推断的，
    #         只在"环形赛道 + 逆时针"时成立；S 弯/折线区/跑反方向会失效。
    'angle_offset': 0.0,    # 雷达安装角修正 [deg]（把点云旋到车体系）
    # 快速自检：勾选后自动扫描 0~359°，找出让"左右锥桶数量最接近 1:1"的角度。
    # 用途：判断雷达 0° 到底指向哪。注意这只在【对称赛道】上有意义。
    'auto_offset': 0,       # 1=自动寻找最佳 angle_offset

    'left_max_y': 2.50,     # 左侧（红锥）窗口 [m]
    'right_max_y': 2.50,    # 右侧（蓝锥）窗口 [m]
    'fwd_max_x': 4.00,      # 前方最远纵向 [m]
    'fwd_min_x': -0.30,     # 后方容许 [m]
    'min_sep_axis': 0.10,   # 中线死区

    # --- 中心线累积 + 切向判定（解决"急转弯时左右判定翻转"）---
    # 原理：颜色（内圈红 / 外圈蓝）应相对【赛道中心线切向】判定，而不是车头朝向。
    #       单帧常常只配出 1 个门中点，无法定义切向；用里程计把历史中点累积到
    #       世界坐标系，就能拟合出真正的赛道走向。
    'use_tangent': 1,        # 1=切向法判内外圈（推荐），0=退回车头法(y 符号)
    'path_memory_s': 6.0,    # 中心线累积时间窗 [s]
    'tangent_radius': 1.8,   # 用车身多少米内的历史中点估计切向 [m]
    'tangent_min_pts': 3,    # 估计切向所需最少历史点数
    'max_heading_err': 55.0, # 车头与切向夹角超过此值 → 判定不可信 [deg]
    'freeze_on_unreliable': 1,  # 不可信时冻结为上一次可靠分类（1=是）
    'use_flip_guard': 1,     # 1=启用全局翻转保护（红蓝比倒挂时整体翻转）
    'flip_ratio': 2.0,       # 倒挂判据：当前比值 < 1/此值 且 历史比值 > 此值
    'flip_min_cones': 5,     # 至少这么多个锥桶才做翻转判断
}

PARAM_META = {
    'range_min':    ('最近有效距离', 0.05, 1.00, 0.01, 'm'),
    'range_max':    ('最远有效距离', 1.00, 20.0, 0.10, 'm'),
    'gap_base':     ('聚类基础间距', 0.005, 0.300, 0.005, 'm'),
    'gap_slope':    ('聚类距离系数', 0.000, 0.100, 0.001, 'm/m'),
    'min_points':   ('簇最少点数', 1, 20, 1, ''),
    'min_extent':   ('簇最小尺寸', 0.005, 0.200, 0.001, 'm'),
    'max_extent':   ('簇最大尺寸', 0.050, 1.000, 0.005, 'm'),
    'cone_radius':  ('锥桶半径', 0.00, 0.30, 0.01, 'm'),
    'min_dist':     ('最近保留距离', 0.00, 2.00, 0.05, 'm'),
    'angle_offset': ('雷达安装角修正', -180.0, 180.0, 1.0, '°'),
    'left_max_y':   ('左/红锥窗口', 0.10, 6.00, 0.05, 'm'),
    'right_max_y':  ('右/蓝锥窗口', 0.10, 6.00, 0.05, 'm'),
    'fwd_max_x':    ('前方窗口', 0.50, 10.0, 0.10, 'm'),
    'fwd_min_x':    ('后方容许', -2.00, 0.50, 0.05, 'm'),
    'min_sep_axis': ('中线死区', 0.00, 0.50, 0.01, 'm'),
}


def detect_cones(angles, ranges, p):
    """
    自适应聚类找锥桶（numpy 向量化，车端 ARM 上也要能跑 10Hz）。

    参数
    ----
    angles : (N,) 每个束的角度 [rad]
    ranges : (N,) 每个束的距离 [m]，无效值为 nan/inf
    p      : 参数字典

    返回
    ----
    cones  : list of dict(x, y, r, a, n, extent, side)
             车体坐标 x 前、y 左；side 为 'red'(左/内圈) / 'blue'(右/外圈) / 'center'
    clusters : list of (n_points, cx, cy) —— 所有簇（调试用，未做尺寸过滤）
    """
    r = np.asarray(ranges, dtype=np.float64)
    a = np.asarray(angles, dtype=np.float64)
    N = r.size

    # 1) 有效点筛选（先算掩码，后续所有运算只在有效点上做，
    #    避免 inf * sin() 之类的 numpy 警告）
    valid = np.isfinite(r) & (r >= p['range_min']) & (r <= p['range_max'])
    if not valid.any():
        return [], []

    # 2) 转车体坐标：x 前 = r*cos(theta)，y 左 = r*sin(theta)
    #    注意 ROS LaserScan 的角度以 +x 为 0、逆时针为正
    rv = r[valid]
    av = a[valid]
    # 安装角修正：把点云旋到车体坐标系（angle_offset=0 时不做任何事）
    off = math.radians(p.get('angle_offset', 0.0))
    if off:
        av = av - off
    xs = np.empty_like(r)
    ys = np.empty_like(r)
    xs[valid] = rv * np.cos(av)
    ys[valid] = rv * np.sin(av)
    xs[~valid] = 0.0
    ys[~valid] = 0.0

    # 3) 向量化找聚类断点
    #    相邻有效点的欧氏距离 > 期望间距(base + slope*r) 处断开。
    #    gap_base 只有在连续两点间才有意义，所以 gap_slope=0 时用固定间距。
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return [], []

    # 连续段：valid 中相邻索引差为 1 的才比较距离
    vi = valid[idx]                     # 全 True（按构造）
    # 相邻点在原数组里是否连续
    contiguous = np.diff(idx) == 1

    dx = np.diff(xs[idx])
    dy = np.diff(ys[idx])
    dist = np.hypot(dx, dy)
    r_mid = np.maximum(r[idx[:-1]], r[idx[1:]])
    expect = p['gap_base'] + p['gap_slope'] * r_mid

    # 断点：非连续 或 距离超限
    brk = (~contiguous) | (dist > expect)
    # 段边界（在原 idx 数组中的位置）
    seg_starts = np.concatenate(([0], np.flatnonzero(brk) + 1))
    seg_ends = np.concatenate((np.flatnonzero(brk), [idx.size - 1]))

    groups = []
    for s, e in zip(seg_starts, seg_ends):
        if e - s + 1 >= 2:
            groups.append(idx[s:e + 1])

    # 环形扫描首尾相接（0 度和 360 度附近可能是同一个物体）
    if len(groups) > 1:
        f, l = groups[0], groups[-1]
        if math.hypot(xs[f[0]] - xs[l[-1]], ys[f[0]] - ys[l[-1]]) < 0.08:
            groups[0] = np.concatenate((l, f))
            groups.pop()

    # 4) 尺寸 / 点数过滤 + 锥桶半径补偿
    cones = []
    clusters = []
    for g in groups:
        gx = xs[g]
        gy = ys[g]
        n = int(g.size)
        dxe = float(gx.max() - gx.min())
        dye = float(gy.max() - gy.min())
        extent = math.hypot(dxe, dye)
        cx = float(gx.mean())
        cy = float(gy.mean())
        clusters.append((n, cx, cy))

        if n < p['min_points']:
            continue
        if extent < p['min_extent'] or extent > p['max_extent']:
            continue

        # 沿视线把簇心外推到锥桶表面
        rr = math.hypot(cx, cy)
        if rr < p['min_dist']:
            continue
        if rr > 1e-6:
            scale = (rr + p['cone_radius']) / rr
            cx *= scale
            cy *= scale

        # 5) 位置窗口 + 左右分类
        #    颜色按规则的空间不变性推断：环形赛道逆时针跑时
        #    y > 0（左）= 红锥（内圈），y < 0（右）= 蓝锥（外圈）
        if not (p['fwd_min_x'] <= cx <= p['fwd_max_x']):
            continue
        if cy > p['left_max_y'] or cy < -p['right_max_y']:
            continue
        if cy > p['min_sep_axis']:
            side = 'red'          # 左 → 红锥（内圈）
        elif cy < -p['min_sep_axis']:
            side = 'blue'         # 右 → 蓝锥（外圈）
        else:
            side = 'center'

        cones.append({
            'x': cx, 'y': cy,
            'r': math.hypot(cx, cy),
            'a': math.degrees(math.atan2(cy, cx)),
            'n': n, 'extent': extent, 'side': side,
        })

    # 按距离排序，近的在前
    cones.sort(key=lambda c: c['r'])
    return cones, clusters


# ===================================================================
# 中心线累积器 + 切向判定
# ===================================================================
class SingleFrameGeometry:
    """
    纯单帧几何：不从任何位姿推算赛道方向。

    为什么这样做：
      车上的 /odom_combined 在 ekf_params.yaml 里配了 imu0_relative=true、
      odom0/1_differential=true —— 它的 yaw 是相对参考、位置是微分累积，
      转弯打滑时会漂。而 IMU 也不被采用。所以完全不依赖位姿。

    单帧能做什么（有实测依据，269 帧真实数据）：
      两排锥桶分居赛道两侧，它们的【质心连线】横跨赛道，
      即垂直于赛道走向 —— 这个方向单帧就能算，且很稳定
      （门连线方位角 std 仅 8.7°）。
      切向 = 法向旋转 90°，再按切向判定内外圈。

    单帧做不到什么：
      · 无法跨帧累积中心线（那需要世界坐标）
      · 弯道里法向是局部平均，会有偏差
      · 看不到赛道转弯趋势

    实测效果（对比车头法 y 符号）：
      数量一致率  4.8% -> 21.1%
      差<=1 比例 18.6% -> 45.5%
    """

    def __init__(self):
        self.last_normal = None
        self.last_sep = 0.0
        self.last_agree = 1.0      # 与实际 y 符号的一致率（0~1）
        self.last_flipped = False
        self.flip_count = 0

    # ---------------------------------------------------------- 分组
    @staticmethod
    def split_two_groups(cones, min_sep=0.10):
        """
        把锥桶分成两组，不需要预先知道左右。
        优先用 y 符号；若一侧不足 2 个，就在 y 的最大间隙处切分。
        """
        if len(cones) < 2:
            return [], []
        ys = np.array([c['y'] for c in cones])

        a = [c for c in cones if c['y'] > min_sep]
        b = [c for c in cones if c['y'] < -min_sep]
        if len(a) >= 2 and len(b) >= 2:
            return a, b

        # 一侧太少：找 y 的最大间隙处切分
        order = np.argsort(ys)
        sy = ys[order]
        gaps = np.diff(sy)
        if len(gaps) > 0 and float(gaps.max()) > 0.15:
            k = int(np.argmax(gaps))
            thr = float(sy[k] + sy[k + 1]) * 0.5
        else:
            thr = float(np.median(ys))
        a = [cones[i] for i in order if ys[i] > thr]
        b = [cones[i] for i in order if ys[i] <= thr]
        return a, b

    # ---------------------------------------------------------- 单帧法向
    def frame_normal(self, cones):
        """
        单帧求赛道法向。返回 (nx, ny, sep)；sep = 两组质心间距。
        两组任一侧不足 2 个点、或间距过小 -> 返回 None。
        """
        a, b = self.split_two_groups(cones)
        if len(a) < 2 or len(b) < 2:
            return None
        ca = np.array([np.mean([c['x'] for c in a]), np.mean([c['y'] for c in a])])
        cb = np.array([np.mean([c['x'] for c in b]), np.mean([c['y'] for c in b])])
        d = cb - ca
        sep = float(np.hypot(d[0], d[1]))
        if sep < 0.30:                     # 两组太近，方向不可信
            return None
        return float(d[0]) / sep, float(d[1]) / sep, sep

    # ---------------------------------------------------------- 分类
    def classify(self, cones, params):
        """
        用单帧法向重新分类内外圈。
        法向 = 组A质心 -> 组B质心。组A 侧 = 内圈（红），组B 侧 = 外圈（蓝）。
        返回 (n_red, n_blue, reliable, sep)
        """
        r = self.frame_normal(cones)
        if r is None:
            # 定不出方向：保留 detect_cones 给的车头法结果，并标记不可信
            nr = sum(1 for c in cones if c.get('side') == 'red')
            nb = sum(1 for c in cones if c.get('side') == 'blue')
            self.last_normal = None
            return nr, nb, False, 0.0

        nx, ny, sep = r
        n_red = n_blue = n_center = 0
        agree = 0
        total = 0
        for c in cones:
            # 相对"过车体的切线"的法向投影：<0 = 组A侧（内圈/红）
            proj = c['x'] * nx + c['y'] * ny
            old = c.get('side')
            if proj < -params['min_sep_axis']:
                c['side'] = 'red'
            elif proj > params['min_sep_axis']:
                c['side'] = 'blue'
            else:
                c['side'] = 'center'
            if c['side'] == 'red':
                n_red += 1
            elif c['side'] == 'blue':
                n_blue += 1
            else:
                n_center += 1
            # 统计与实际 y 符号的一致率（作为"这一帧可不可信"的旁证）
            if old in ('red', 'blue') and c['side'] in ('red', 'blue'):
                total += 1
                if old == c['side']:
                    agree += 1

        self.last_normal = (nx, ny)
        self.last_sep = sep
        self.last_agree = (agree / total) if total else 0.0

        # ---- 翻转保护：一致率极低说明整体反了 ----
        flipped = False
        if int(params.get('use_flip_guard', 1)) and total >= 4 and self.last_agree < 0.25:
            for c in cones:
                if c.get('side') == 'red':
                    c['side'] = 'blue'
                elif c.get('side') == 'blue':
                    c['side'] = 'red'
                c['flipped'] = True
            n_red, n_blue = n_blue, n_red
            self.last_agree = 1.0 - self.last_agree
            self.flip_count += 1
            flipped = True
        self.last_flipped = flipped
        return n_red, n_blue, True, sep


# 全局实例
CL = SingleFrameGeometry()
# ===================================================================
# 可视化渲染
# ===================================================================
CANVAS_W, CANVAS_H = 820, 820
SCALE_DEFAULT = 90.0    # px / m
BG = (24, 28, 34)

# 出图格式：JPEG 编码在车端 ARM 上比 PNG 快数倍（实测 PNG zlib 是主要开销之一）
# 面板是连续刷新的诊断图，JPEG 的轻微压缩损失完全可接受。
JPEG_QUALITY = 72


def render(angles, ranges, cones, clusters, params, stats, view_scale):
    """渲染俯视图 PNG。返回 BGR ndarray。"""
    img = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
    ox, oy = CANVAS_W // 2, CANVAS_H - 120        # 车体位置（下方居中）
    s = view_scale

    def to_px(x, y):
        # 车体 x 前 → 屏幕向上；y 左 → 屏幕向左
        return int(ox - y * s), int(oy - x * s)

    # ---- 网格（自适应步长：缩放越小格子越大，减少绘制调用）----
    # 车端 ARM 实测：每条 cv2.line 约 0.5ms，网格是主要的渲染开销。
    # 因此按缩放选择步长，保证屏幕上的格子不小于 ~45px。
    max_m = max(CANVAS_W, CANVAS_H) / s
    gstep = 0.5
    while gstep * s < 45 and gstep < 8.0:
        gstep *= 2.0
    n = int(max_m / gstep) + 2
    for i in range(-n, n + 1):
        col = (44, 50, 58) if i % 2 else (34, 39, 46)
        px, _ = to_px(0, i * gstep)
        if 0 <= px < CANVAS_W:
            cv2.line(img, (px, 0), (px, CANVAS_H), col, 1)
        _, py = to_px(i * gstep, 0)
        if 0 <= py < CANVAS_H:
            cv2.line(img, (0, py), (CANVAS_W, py), col, 1)

    # ---- 距离圈（1m / 2m / 3m）----
    for ring in (1.0, 2.0, 3.0):
        cv2.circle(img, (ox, oy), int(ring * s), (58, 66, 76), 1, cv2.LINE_AA)
        cv2.putText(img, '%dm' % ring, (ox + int(ring * s) - 26, oy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 120, 134), 1, cv2.LINE_AA)

    # ---- 原始扫描点（向量化写入，避免逐点 Python 循环）----
    r = np.asarray(ranges, dtype=np.float64)
    a = np.asarray(angles, dtype=np.float64)
    ok = np.isfinite(r) & (r >= params['range_min']) & (r <= params['range_max'])
    if ok.any():
        rv = r[ok]
        av = a[ok] - math.radians(params.get('angle_offset', 0.0))
        px = (ox - (rv * np.sin(av)) * s).astype(np.int32)
        py = (oy - (rv * np.cos(av)) * s).astype(np.int32)
        inb = (px >= 0) & (px < CANVAS_W) & (py >= 0) & (py < CANVAS_H)
        if inb.any():
            img[py[inb], px[inb]] = (120, 132, 148)

    # ---- 位置窗口边框 ----
    p = params
    corners = [(p['fwd_max_x'], p['left_max_y']), (p['fwd_max_x'], -p['right_max_y']),
               (p['fwd_min_x'], -p['right_max_y']), (p['fwd_min_x'], p['left_max_y'])]
    pts = np.array([to_px(x, y) for x, y in corners], dtype=np.int32)
    cv2.polylines(img, [pts], True, (70, 110, 150), 1, cv2.LINE_AA)

    # ---- 所有簇（未过滤，淡色）----
    for (n, cx, cy) in clusters:
        X, Y = to_px(cx, cy)
        if 0 <= X < CANVAS_W and 0 <= Y < CANVAS_H:
            cv2.circle(img, (X, Y), 3, (90, 90, 100), -1, cv2.LINE_AA)

    # ---- 检出的锥桶 ----
    # 配色按真实颜色：红锥（左侧/内圈）用红，蓝锥（右侧/外圈）用蓝
    cols = {'red': (80, 80, 245), 'blue': (245, 170, 60), 'center': (200, 200, 200)}
    for i, c in enumerate(cones):
        X, Y = to_px(c['x'], c['y'])
        col = cols[c['side']]
        cv2.circle(img, (X, Y), int(max(4, c['extent'] * s * 0.5)),
                   col, 2, cv2.LINE_AA)
        cv2.circle(img, (X, Y), 2, col, -1, cv2.LINE_AA)
        # 从车体到锥桶的连线
        cv2.line(img, (ox, oy), (X, Y), tuple(int(v * 0.35) for v in col), 1, cv2.LINE_AA)

        # 标签：沿"车→锥桶"方向再往外推一段，避免压到相邻点
        dx, dy = X - ox, Y - oy
        L = math.hypot(dx, dy) or 1.0
        off = 16
        lx = int(X + dx / L * off)
        ly = int(Y + dy / L * off)
        if not (0 <= lx < CANVAS_W - 40):
            lx = X - 30
        if not (0 <= ly < CANVAS_H):
            ly = Y + 4
        cv2.putText(img, '#%d' % (i + 1), (lx - 10, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    # ---- 车道中心线：把左右（红/蓝）锥桶按纵向配对，连中点 ----
    # 这是判断"车是否在赛道中央"最直观的显示：
    #   中心线在车正前方 → 车居中
    #   中心线偏左/偏右   → 车偏右/偏左
    reds = [c for c in cones if c['side'] == 'red']
    blues = [c for c in cones if c['side'] == 'blue']
    if reds and blues:
        mids = []
        for rb in reds:
            # 找纵向最接近的蓝锥
            best, bd = None, 1e9
            for bb in blues:
                d = abs(bb['x'] - rb['x'])
                if d < bd:
                    bd, best = d, bb
            if best is not None and bd <= 1.0:
                mids.append(((rb['x'] + best['x']) * 0.5,
                             (rb['y'] + best['y']) * 0.5))
        if mids:
            mids.sort(key=lambda m: m[0])
            pts = np.array([to_px(mx, my) for mx, my in mids], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(img, [pts], False, (120, 240, 160), 2, cv2.LINE_AA)
            for (mx, my) in mids:
                X, Y = to_px(mx, my)
                cv2.circle(img, (X, Y), 3, (120, 240, 160), -1, cv2.LINE_AA)
            # 最近一段中心线的横向偏移（车体坐标 y）
            y0 = mids[0][1]
            cv2.putText(img, 'center off %+.2fm' % y0, (14, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (120, 240, 160) if abs(y0) < 0.25 else (80, 160, 245),
                        1, cv2.LINE_AA)

    # ---- 赛道切向箭头（单帧几何算出，与任何位姿无关）----
    # tan_ 存的是"两组质心连线"= 赛道【法向】（横跨赛道方向）。
    # 箭头要画的是【切向】= 法向旋转 90°，并让它指向前方（x>0 侧）。
    # 绿色 = 本帧定出了赛道方向；橙红 = 定不出（两组点数不足或间距太小）
    tan_ = stats.get('tangent')
    rel_ = stats.get('reliable', True)
    sep_ = stats.get('sep', 0.0)
    if tan_ is not None:
        nx, ny = tan_                       # 法向（组A → 组B）
        tx, ty = -ny, nx                    # 切向 = 法向逆时针 90°
        if tx < 0:                          # 让箭头指向前方
            tx, ty = -tx, -ty
        L = 1.3
        A = to_px(0.0, 0.0)
        B = to_px(tx * L, ty * L)
        col = (120, 240, 160) if rel_ else (80, 120, 240)
        cv2.arrowedLine(img, A, B, col, 2, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(img, 'track dir', (B[0] + 6, B[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        # 两组质心间距：越大说明两排锥桶分得越开、方向越可信
        cv2.putText(img, 'sep %.2fm' % sep_, (14, 152),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)
    else:
        # 定不出方向：显式画在图上，避免误以为箭头"丢了"
        cv2.putText(img, 'NO TRACK DIR (cones on one side only)',
                    (14, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (80, 120, 240), 1, cv2.LINE_AA)

    # ---- 车辆 ----
    car = np.array([to_px(0.30, 0.18), to_px(0.30, -0.18),
                    to_px(-0.26, -0.18), to_px(-0.26, 0.18)], dtype=np.int32)
    cv2.fillPoly(img, [car], (120, 230, 160))
    cv2.polylines(img, [car], True, (60, 150, 100), 1, cv2.LINE_AA)
    # 车头三角
    nose = np.array([to_px(0.42, 0.10), to_px(0.30, 0.0), to_px(0.42, -0.10)],
                    dtype=np.int32)
    cv2.fillPoly(img, [nose], (120, 230, 160))

    # ---- 中心虚线（参考）----
    cv2.line(img, (ox, oy), to_px(3.0, 0), (70, 90, 110), 1, cv2.LINE_AA)

    # ---- 上半部统计面板 ----
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (CANVAS_W, 118), (18, 21, 26), -1)
    cv2.addWeighted(overlay, 0.86, img, 0.14, 0, img)

    def txt(s_, x, y, col=(220, 228, 238), sz=0.52, th=1):
        cv2.putText(img, s_, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sz, col, th, cv2.LINE_AA)

    n_red = sum(1 for c in cones if c['side'] == 'red')
    n_blue = sum(1 for c in cones if c['side'] == 'blue')
    n_center = sum(1 for c in cones if c['side'] == 'center')
    # 正前方 ±20° 内（已应用 angle_offset）的锥桶数。
    # 这个数大 = angle_offset 接近真实安装角；配合左右比值接近 1 一起看。
    n_fwd20 = sum(1 for c in cones if abs(c['a']) <= 20.0)

    txt('CONE  %d' % len(cones), 14, 30, (120, 240, 180), 0.78, 2)
    txt('red %d' % n_red, 14, 56, (80, 80, 245), 0.55, 1)
    txt('blue %d' % n_blue, 110, 56, (245, 170, 60), 0.55, 1)
    txt('mid %d' % n_center, 215, 56, (200, 200, 200), 0.55, 1)
    txt('fwd20 %d' % n_fwd20, 320, 56, (150, 220, 160), 0.55, 1)

    off = params.get('angle_offset', 0.0)
    txt('ang_off %.0f deg' % off, 445, 56, (230, 190, 120), 0.55, 1)

    txt('clusters %d' % len(clusters), 14, 82, (170, 180, 195), 0.48, 1)
    txt('beams %d' % int(ok.sum()), 190, 82, (170, 180, 195), 0.48, 1)
    txt('%s' % stats.get('hz_str', ''), 330, 82, (170, 180, 195), 0.48, 1)
    txt('%s' % stats.get('ms_str', ''), 470, 82, (170, 180, 195), 0.48, 1)

    txt('scale %.0f px/m' % s, CANVAS_W - 210, 30, (140, 150, 165), 0.46, 1)
    txt('source: %s' % stats.get('source', '-'), CANVAS_W - 210, 56,
        (140, 150, 165), 0.46, 1)

    # 单帧几何可信度：定不出赛道方向、或与车头法一致率过低时显示告警
    _rel = bool(stats.get('reliable', True))
    _ag = float(stats.get('agree', 1.0))
    txt('single-frame  sep=%.2fm agree=%.0f%%  %s'
        % (float(stats.get('sep', 0.0)), _ag * 100.0,
           'OK' if _rel else 'NO DIRECTION'),
        CANVAS_W - 470, 106,
        (120, 240, 160) if _rel else (80, 120, 240), 0.46, 1)

    # 帧号画进图里：这样不用看网页也能确认图像在刷新
    txt('frame %s' % stats.get('frame', '-'), CANVAS_W - 210, 82,
        (110, 170, 220), 0.46, 1)

    txt('x=forward  y=left', 14, 106, (130, 140, 155), 0.44, 1)

    # ---- 节点算法镜像：四态状态机横幅 + aim 点 + 期望曲率圆弧 ----
    node = stats.get('node')
    if node and node.get('state'):
        stt = str(node.get('state'))
        ncol = {'STRAIGHT': (120, 240, 160), 'CORNER': (80, 200, 250),
                'RIGHT_ANGLE': (60, 140, 255), 'YELLOW': (80, 220, 230)}.get(
            stt, (200, 200, 200))
        tail = '' if node.get('control', True) else '   >> 黄线接管(停止发 teleop)'
        txt('NODE [%s]%s' % (stt, tail), 14, 176, ncol, 0.62, 2)

        def _f(v, dv=0.0):
            try:
                return float(v)
            except (TypeError, ValueError):
                return dv

        txt('servo %.1f  speed %.0f  k_cmd %+.2f (des %+.2f)  kff %+.2f  ki %+.3f'
            % (_f(node.get('servo')), _f(node.get('speed')),
               _f(node.get('k_cmd')), _f(node.get('k_des')),
               _f(node.get('k_ff')), _f(node.get('k_i'))),
            14, 198, (200, 210, 225), 0.50, 1)
        txt('ey %+.3f  clear %.2f  cones %d  ra_t %.2f  lost %.2f'
            % (_f(node.get('err_y')), _f(node.get('clear'), 99.0),
               int(node.get('n_cones') or 0), _f(node.get('ra_timer')),
               _f(node.get('cone_lost_t'))),
            14, 218, (170, 180, 195), 0.48, 1)

        # 目标点
        aim = node.get('aim')
        if aim is not None and len(aim) == 2:
            AX, AY = to_px(float(aim[0]), float(aim[1]))
            cv2.circle(img, (AX, AY), 6, (60, 140, 255), 2, cv2.LINE_AA)
        # 节点当前曲率对应的圆弧
        k_node = _f(node.get('k_cmd'))
        ss = np.arange(0.0, 1.30 + 1e-9, 0.08)
        if abs(k_node) < 1e-3:
            pix = np.array([to_px(float(s), 0.0) for s in ss], dtype=np.int32)
        else:
            pix = np.array([to_px(float(np.sin(k_node * s) / k_node),
                                  float((1.0 - np.cos(k_node * s)) / k_node))
                            for s in ss], dtype=np.int32)
        if len(pix) >= 2:
            cv2.polylines(img, [pix], False, (60, 140, 255), 2, cv2.LINE_AA)
    else:
        txt('NODE: (未加载 scripts/lidar_tracking.py)', 14, 176,
            (150, 150, 160), 0.48, 1)

    return img


# ===================================================================
# 共享状态（ROS 线程写，HTTP 线程读）
# ===================================================================
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.params = dict(DEFAULT_PARAMS)
        self.view_scale = SCALE_DEFAULT
        self.angles = np.zeros(0)
        self.ranges = np.zeros(0)
        self.cones = []
        self.clusters = []
        self.stats = {'hz_str': '-', 'ms_str': '-', 'source': 'none'}
        self.frame_id = 0
        self.enable = True
        self.last_error = ''
        # 单帧几何判定结果（供面板显示）—— 不使用任何位姿
        self.tangent = None       # 赛道切向（车体系单位向量）
        self.sep = 0.0            # 两排锥桶质心间距 [m]
        self.agree = 1.0          # 与车头法(y符号)的一致率，旁证可信度
        self.reliable = True      # 本帧是否定出了赛道方向
        self.flipped = False      # 本帧是否触发了整体翻转
        # 节点算法镜像 (scripts/lidar_tracking.py 的四态状态机) 输出
        self.node_out = None


SH = Shared()


# ===================================================================
# ROS 节点
# ===================================================================
if HAVE_ROS:
    class ConeNode(Node):
        def __init__(self, topic, replay):
            super().__init__('cone_detector')
            self.replay = replay
            self.last_t = None
            self.hz = 0.0
            self.times = []

            if replay:
                self.get_logger().info('回放模式：%s' % replay)
                self._replay_loop()
            else:
                # 只订阅 /scan —— 不使用里程计，也不使用 IMU
                self.create_subscription(LaserScan, topic, self.on_scan, 5)
                self.get_logger().info(
                    '订阅 %s（单帧几何判定，不依赖里程计/IMU）' % topic)


        # ---------------------------------------------------------- 回放
        def _replay_loop(self):
            frames = load_scans_file(self.replay)
            if not frames:
                self.get_logger().error('回放文件为空')
                return
            self.get_logger().info('载入 %d 帧，循环播放' % len(frames))
            inc = frames[0][0]
            idx = 0
            while rclpy.ok():
                amin, n, rng = frames[idx][1], frames[idx][2], frames[idx][3]
                angles = amin + np.arange(n) * inc
                self.process(angles, rng, 'replay')
                idx = (idx + 1) % len(frames)
                time.sleep(0.10)

        # ---------------------------------------------------------- 实时
        def on_scan(self, msg):
            r = np.asarray(msg.ranges, dtype=np.float64)
            a = msg.angle_min + np.arange(len(r)) * msg.angle_increment
            self.process(a, r, 'scan')

        # ---------------------------------------------------------- 公共
        def process(self, angles, ranges, source):
            if not SH.enable:
                return
            t0 = time.perf_counter()
            now = time.time()

            with SH.lock:
                p = dict(SH.params)

            cones, clusters = detect_cones(angles, ranges, p)

            # ============================================================
            # 节点算法镜像：直接调用 scripts/lidar_tracking.py 的 ConeTracker，
            # 用同一份 param.yaml 跑同一套四态状态机，实时显示状态/舵角/曲率。
            # ============================================================
            node_out = None
            if NODE_TRACKER is not None:
                try:
                    amin = float(angles[0]) if len(angles) else 0.0
                    ainc = (float(angles[1] - angles[0])
                            if len(angles) > 1 else 0.0)
                    servo_n, speed_n = NODE_TRACKER.step(ranges, amin, ainc, now)
                    d = NODE_TRACKER.dbg
                    node_out = {
                        'state': NODE_TRACKER.state,
                        'control': bool(NODE_TRACKER.control_enabled),
                        'servo': float(servo_n),
                        'speed': float(speed_n),
                        'k_cmd': float(d.get('k_cmd') or 0.0),
                        'k_des': float(d.get('k_des') or 0.0),
                        'k_ff': float(d.get('k_ff') or 0.0),
                        'k_i': float(d.get('k_i') or 0.0),
                        'err_y': float(d.get('err_y') or 0.0),
                        'clear': float(d.get('clear', 99.0)),
                        'aim': d.get('aim'),
                        'n_cones': int(d.get('n_cones', 0)),
                        'ra_timer': float(d.get('ra_timer', 0.0)),
                        'cone_lost_t': float(d.get('cone_lost_t', 0.0)),
                    }
                except Exception as e:
                    node_out = {'state': 'ERR', 'err': str(e)}

            # ============================================================
            # 单帧几何：不用里程计、不用 IMU
            #   两排锥桶的质心连线 = 赛道法向 → 旋转 90° = 切向
            #   按切向重新分类内外圈，避免"急转弯时车头法翻转"
            # ============================================================
            n_red = n_blue = 0
            reliable = True
            sep = 0.0
            agree = 1.0
            flipped = False

            if int(p.get('use_tangent', 1)) and cones:
                n_red, n_blue, reliable, sep = CL.classify(cones, p)
                agree = CL.last_agree
                flipped = getattr(CL, 'last_flipped', False)

            ms = (time.perf_counter() - t0) * 1000.0

            if self.last_t is not None:
                dt = now - self.last_t
                if dt > 0:
                    inst = 1.0 / dt
                    self.hz = inst if self.hz == 0 else (self.hz * 0.9 + inst * 0.1)
            self.last_t = now

            self.times.append(ms)
            if len(self.times) > 100:
                self.times.pop(0)
            ms_avg = sum(self.times) / len(self.times)
            ms_max = max(self.times)

            with SH.lock:
                SH.angles = angles
                SH.ranges = ranges
                SH.cones = cones
                SH.clusters = clusters
                SH.frame_id += 1
                SH.tangent = CL.last_normal
                SH.sep = CL.last_sep
                SH.agree = CL.last_agree
                SH.reliable = reliable
                SH.flipped = flipped
                SH.node_out = node_out
                SH.stats = {
                    'hz_str': '%.1f Hz' % self.hz,
                    'ms_str': '%.2f/%.1f ms' % (ms_avg, ms_max),
                    'source': source,
                    'frame': SH.frame_id,
                    'n_path': 0,
                    'sep': round(float(CL.last_sep), 2),
                    'agree': round(float(CL.last_agree), 2),
                }
                SH.last_error = ''

# ===================================================================
# 回放文件加载
# ===================================================================
def load_scans_file(path):
    """
    支持两种格式：
      A) 第一行 "angle_min angle_increment"，后续每行 "t n r0 r1 ..."
         （与 lidar_turn_study 的 scans.txt 一致）
      B) 每行 "t angle_increment n r0 r1 ..."
    返回 [(inc, amin, n, np.array(ranges)), ...]
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return []

    out = []
    first = lines[0].split()
    header_style = len(first) == 2

    if header_style:
        amin_global = float(first[0])
        inc_global = float(first[1])
        body = lines[1:]
    else:
        amin_global = 0.0
        inc_global = None
        body = lines

    for ln in body:
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            if header_style:
                t = float(parts[0]); n = int(parts[1])
                vals = parts[2:2 + n]
                inc = inc_global
                amin = amin_global
            else:
                # 格式 B: t inc n r...
                t = float(parts[0]); inc = float(parts[1]); n = int(parts[2])
                vals = parts[3:3 + n]
                amin = amin_global
            arr = np.array([np.inf if v in ('inf', 'Inf', 'INF')
                            else (np.nan if v in ('nan', 'NaN') else float(v))
                            for v in vals], dtype=np.float64)
            out.append((inc, amin, len(arr), arr))
        except (ValueError, IndexError):
            continue

    # 统一 inc
    if out and inc_global is None:
        inc_common = out[0][0]
        out = [(inc_common,) + f[1:] for f in out]
    return out


# ===================================================================
# Web 面板
# ===================================================================
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>锥桶检测 · 调参面板</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:"Microsoft YaHei",system-ui,sans-serif;
     display:flex;height:100vh;overflow:hidden;font-size:13px}
#side{width:340px;min-width:340px;height:100vh;overflow-y:auto;background:#161b22;
      border-right:1px solid #30363d;padding:12px}
#main{flex:1;display:flex;flex-direction:column;align-items:center;
      justify-content:center;background:#0d1117;padding:8px}
h1{font-size:14px;color:#58a6ff;margin-bottom:10px}
.sec{font-size:11px;color:#8b949e;margin:12px 0 6px;border-bottom:1px solid #21262d;
     padding-bottom:4px;letter-spacing:.5px}
.row{display:flex;align-items:center;gap:6px;margin:5px 0}
.row label{flex:0 0 116px;font-size:11px;color:#8b949e}
.row input[type=range]{flex:1;accent-color:#58a6ff;height:18px}
.row .v{flex:0 0 52px;text-align:right;font-family:ui-monospace,monospace;
        font-size:11px;color:#7ee787}
button{background:#238636;border:none;color:#fff;padding:6px 10px;border-radius:5px;
       cursor:pointer;font-size:11px;font-family:inherit}
button:hover{background:#2ea043}
button.sec{background:#21262d;border:1px solid #30363d}
button.sec:hover{background:#30363d}
#stats{background:#0d1117;border-radius:6px;padding:8px;font-family:ui-monospace,monospace;
       font-size:11px;line-height:1.7}
#stats b{color:#7ee787}
img{max-width:100%;max-height:calc(100vh - 24px);border-radius:6px;
    border:1px solid #30363d;image-rendering:auto}
.lg{display:flex;gap:10px;flex-wrap:wrap;font-size:10.5px;color:#8b949e;margin-top:6px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}
</style></head><body>
<div id="side">
  <h1>锥桶检测调参面板</h1>
  <div id="stats">加载中…</div>
  <div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap">
    <button onclick="save()">保存参数</button>
    <button class="sec" onclick="reset()">恢复默认</button>
    <button class="sec" id="tg" onclick="toggleRun()">暂停</button>
    <button class="sec" onclick="sweep()">扫描最佳安装角</button>
  </div>
  <div id="sweepOut" style="font-size:10.5px;color:#8b949e;margin-top:6px"></div>
  <div class="sec">视角</div>
  <div class="row"><label>缩放 px/m</label>
    <input type="range" id="view_scale" min="25" max="260" step="1" value="90">
    <span class="v" id="view_scale_v">90</span></div>
  <div id="params"></div>
  <div class="lg">
    <span><i class="dot" style="background:#f55a5a"></i>红锥(左/内圈)</span>
    <span><i class="dot" style="background:#3caaf5"></i>蓝锥(右/外圈)</span>
    <span><i class="dot" style="background:#c8c8c8"></i>中线</span>
    <span><i class="dot" style="background:#787880"></i>未过滤簇</span>
  </div>
</div>
<div id="main">
  <img id="img" src="/frame.jpg">
</div>
<script>
const META = __META__;
let params = {}, timer = null, paused = false;

function buildSliders(){
  const box = document.getElementById('params');
  const groups = {
    '距离范围': ['range_min','range_max','min_dist'],
    '自适应聚类': ['gap_base','gap_slope'],
    '尺寸过滤': ['min_points','min_extent','max_extent','cone_radius'],
    '位置窗口与左右分类': ['left_max_y','right_max_y','fwd_max_x','fwd_min_x','min_sep_axis'],
  };
  for (const [gname, keys] of Object.entries(groups)){
    const h = document.createElement('div');
    h.className = 'sec'; h.textContent = gname; box.appendChild(h);
    for (const k of keys){
      const m = META[k]; if(!m) continue;
      const [label, lo, hi, step, unit] = m;
      const row = document.createElement('div'); row.className='row';
      row.innerHTML = `<label title="${k}">${label}</label>
        <input type="range" id="p_${k}" min="${lo}" max="${hi}" step="${step}">
        <span class="v" id="v_${k}"></span>`;
      box.appendChild(row);
    }
  }
}

function syncFromServer(){
  fetch('/params').then(r=>r.json()).then(d=>{
    params = d.params;
    for (const k in params){
      const el = document.getElementById('p_'+k); if(!el) continue;
      el.value = params[k];
      const m = META[k];
      document.getElementById('v_'+k).textContent =
        m && m[4] ? params[k].toFixed(2)+m[4] : params[k];
      el.oninput = ()=>{
        const val = parseFloat(el.value);
        document.getElementById('v_'+k).textContent =
          m && m[4] ? val.toFixed(2)+m[4] : val;
        setParam(k, val);
      };
    }
    const vs = document.getElementById('view_scale');
    vs.value = d.view_scale;
    document.getElementById('view_scale_v').textContent = d.view_scale;
  });
}

const pending = {};
let pushTimer = null;
function setParam(k, v){
  pending[k] = v;
  if (pushTimer) return;
  pushTimer = setTimeout(()=>{
    pushTimer = null;
    fetch('/set', {method:'POST', headers:{'Content-Type':'application/json'},
                   body: JSON.stringify(pending)})
      .then(()=>{ for(const kk in pending) delete pending[kk]; });
  }, 40);
}

function save(){
  fetch('/save', {method:'POST'}).then(r=>r.text()).then(t=>alert(t));
}
function reset(){
  if(!confirm('恢复默认参数？')) return;
  fetch('/reset', {method:'POST'}).then(()=>syncFromServer());
}
function toggleRun(){
  paused = !paused;
  looping = !paused;
  document.getElementById('tg').textContent = paused ? '继续' : '暂停';
  fetch('/enable', {method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({on: !paused})});
  if (!paused) tick();                     // 恢复时重启出图循环
}

function sweep(){
  const out = document.getElementById('sweepOut');
  out.textContent = '扫描中…';
  fetch('/sweep').then(r=>r.json()).then(d=>{
    if (!d.ok){ out.textContent = '扫描失败: ' + (d.err||'?'); return; }
    const b = d.best;
    // 找几个"局部最好"的候选一起列出，便于判断是否真的唯一
    const sorted = d.samples.slice().sort((a,c)=>Math.abs(Math.log(Math.max(1e-6,a[3])))-Math.abs(Math.log(Math.max(1e-6,c[3]))));
    let html = `最佳安装角 <b style="color:#7ee787">${b.offset}°</b> ` +
               `(红 ${b.n_red} : 蓝 ${b.n_blue} = ${b.ratio})<br>`;
    html += '前 5 个候选：<br>';
    for (const s of sorted.slice(0,5)){
      html += `&nbsp;&nbsp;${s[0]}° → 红${s[1]} 蓝${s[2]} (${s[3]})<br>`;
    }
    html += `<span style="color:#f0883e">${d.note}</span>`;
    out.innerHTML = html;
    // 提一下：是否要应用
    if (confirm('把 angle_offset 设为 ' + b.offset + '° 吗？')) {
      setParam('angle_offset', b.offset);
      const el = document.getElementById('p_angle_offset');
      if (el){ el.value = b.offset; document.getElementById('v_angle_offset').textContent = b.offset+'°'; }
    }
  }).catch(e=>{ out.textContent = '扫描出错: ' + e; });
}

function refresh(){
  const im = document.getElementById('img');
  im.src = '/frame.jpg?t=' + Date.now();
  fetch('/state').then(r=>r.json()).then(d=>{
    const s = d.stats;
    const lag = (Date.now() - lastFrameAt) / 1000;
    document.getElementById('stats').innerHTML =
      `锥桶总数 <b>${s.n_cones}</b><br>` +
      `红锥 <b>${s.n_red}</b> &nbsp; 蓝锥 <b>${s.n_blue}</b> &nbsp; 中线 ${s.n_center}<br>` +
      `聚类数 ${s.n_clusters} &nbsp; 有效束 ${s.n_beams}<br>` +
      `${s.hz} &nbsp; ${s.ms}<br>` +
      `帧 ${s.frame} &nbsp; 来源 ${s.source}<br>` +
      `<span style="color:${lag>3?'#f85149':'#7ee787'}">距上次出图 ${lag.toFixed(1)}s</span>`;
  }).catch(()=>{});
}

// ---- 自调度刷新：等上一帧图片加载完再发下一帧 ----
// 不能用 setInterval(fn, 200)：服务端每帧要 100~115ms 渲染，固定间隔轮询
// 会让请求堆积，浏览器取消未完成的加载，表现就是"页面不更新"。
let looping = true;
let lastFrameAt = Date.now();

function tick(){
  if (!looping) return;
  const im = document.getElementById('img');
  im.onload = () => { lastFrameAt = Date.now(); if (looping) setTimeout(tick, 60); };
  im.onerror = () => { if (looping) setTimeout(tick, 600); };
  im.src = '/frame.jpg?t=' + Date.now();
}

function fnum(x, n){
  return (x === null || x === undefined) ? '-' : Number(x).toFixed(n);
}

function refreshStats(){
  fetch('/state').then(r=>r.json()).then(d=>{
    const s = d.stats;
    const lag = (Date.now() - lastFrameAt) / 1000;
    const okCol = s.reliable ? '#7ee787' : '#f0883e';
    const okTxt = s.reliable ? '切向可信' : '车头偏离赛道，判定冻结';
    document.getElementById('stats').innerHTML =
      `锥桶总数 <b>${s.n_cones}</b><br>` +
      `红锥 <b>${s.n_red}</b> &nbsp; 蓝锥 <b>${s.n_blue}</b> &nbsp; 中线 ${s.n_center}<br>` +
      `聚类数 ${s.n_clusters} &nbsp; 有效束 ${s.n_beams}<br>` +
      `${s.hz} &nbsp; ${s.ms}<br>` +
      `帧 ${s.frame} &nbsp; 来源 ${s.source}<br>` +
      `<span style="color:${okCol}">${okTxt}</span><br>` +
      `两组质心间距 <b>${s.sep}m</b> &nbsp; 与车头法一致率 ${(s.agree*100).toFixed(0)}%` +
      `${s.flipped?' &nbsp; <b style="color:#f0883e">已翻转</b>':''}<br>` +
      (s.node_state
        ? `<span style="color:#ff8c3c">NODE [${s.node_state}]</span>` +
          `${s.node_control===false?' <b style="color:#50e6e6">黄线接管</b>':''}<br>` +
          `servo ${fnum(s.node_servo,1)}&deg; speed ${fnum(s.node_speed,0)} ` +
          `k ${fnum(s.node_kcmd,2)} (des ${fnum(s.node_kdes,2)}) ` +
          `ey ${fnum(s.node_ey,2)} clear ${fnum(s.node_clear,2)}<br>`
        : '<span style="color:#8b949e">NODE --</span><br>') +
      `<span style="color:${lag>3?'#f85149':'#7ee787'}">距上次出图 ${lag.toFixed(1)}s</span>`;
  }).catch(()=>{});
}

document.getElementById('view_scale').oninput = function(){
  document.getElementById('view_scale_v').textContent = this.value;
  fetch('/set', {method:'POST', headers:{'Content-Type':'application/json'},
                 body: JSON.stringify({_view_scale: parseFloat(this.value)})});
};

buildSliders();
syncFromServer();
tick();                                    // 启动自调度出图循环
setInterval(refreshStats, 400);            // 统计数字单独轮询（很轻）
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'      # 复用连接，避免每个请求重建 TCP

    def log_message(self, *a):
        pass

    def handle(self):
        # 客户端提前断开是常态（连续图片流），不要让 socketserver 打 traceback
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    # ---------------------------------------------------------- GET
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/' or u.path == '/index.html':
            page = PAGE.replace('__META__', json.dumps(PARAM_META))
            self._send(200, 'text/html; charset=utf-8', page.encode('utf-8'))
            return

        if u.path == '/params':
            with SH.lock:
                body = json.dumps({'params': SH.params,
                                   'view_scale': SH.view_scale}).encode()
            self._send(200, 'application/json', body)
            return

        if u.path == '/state':
            with SH.lock:
                cones = list(SH.cones)
                clusters = list(SH.clusters)
                st = dict(SH.stats)
                ang = SH.angles
                rng = SH.ranges
                frame = SH.frame_id
                reliable = SH.reliable
                sep = SH.sep
                agree = SH.agree
                flipped = SH.flipped
                node_out = SH.node_out
            n_beams = 0
            if len(rng):
                ok = np.isfinite(rng) & (rng >= 0.15)
                n_beams = int(ok.sum())
            body = json.dumps({
                'stats': {
                    'n_cones': len(cones),
                    'n_red': sum(1 for c in cones if c['side'] == 'red'),
                    'n_blue': sum(1 for c in cones if c['side'] == 'blue'),
                    'n_center': sum(1 for c in cones if c['side'] == 'center'),
                    'n_clusters': len(clusters),
                    'n_beams': n_beams,
                    'hz': st.get('hz_str', '-'),
                    'ms': st.get('ms_str', '-'),
                    'frame': frame,
                    'source': st.get('source', '-'),
                    'reliable': bool(reliable),
                    'sep': round(float(sep), 2),
                    'agree': round(float(agree), 2),
                    'flipped': bool(flipped),
                    'node_state': (node_out.get('state') if node_out else None),
                    'node_control': (bool(node_out.get('control', True))
                                     if node_out else None),
                    'node_servo': jnum(node_out.get('servo')) if node_out else None,
                    'node_speed': jnum(node_out.get('speed')) if node_out else None,
                    'node_kcmd': jnum(node_out.get('k_cmd')) if node_out else None,
                    'node_kdes': jnum(node_out.get('k_des')) if node_out else None,
                    'node_kff': jnum(node_out.get('k_ff')) if node_out else None,
                    'node_ki': jnum(node_out.get('k_i')) if node_out else None,
                    'node_ey': jnum(node_out.get('err_y')) if node_out else None,
                    'node_clear': jnum(node_out.get('clear')) if node_out else None,
                },
                'cones': cones,
            }).encode()
            self._send(200, 'application/json', body)
            return

        if u.path == '/sweep':
            # 一次性自检：扫描 angle_offset 0~359°，找出让"两侧锥桶数量最接近 1:1"
            # 的角度。仅在对称赛道上结果才有意义；房间/走廊数据会给出误导性答案。
            with SH.lock:
                ang = SH.angles.copy()
                rng = SH.ranges.copy()
                p = dict(SH.params)
            if len(ang) < 3:
                self._send(200, 'application/json',
                           b'{"ok":false,"err":"no scan data"}')
                return
            best = None
            rows = []
            for off in range(0, 360, 3):
                pp = dict(p)
                pp['angle_offset'] = float(off)
                cs, _ = detect_cones(ang, rng, pp)
                nr = sum(1 for c in cs if c['side'] == 'red')
                nb = sum(1 for c in cs if c['side'] == 'blue')
                if nr + nb < 4:
                    continue
                ratio = nr / max(1, nb)
                score = abs(math.log(max(1e-6, ratio)))
                rows.append((off, nr, nb, round(ratio, 2), round(score, 3)))
                if best is None or score < best[4]:
                    best = (off, nr, nb, ratio, score)
            body = json.dumps({
                'ok': True,
                'samples': rows,
                'best': None if best is None else {
                    'offset': best[0], 'n_red': best[1],
                    'n_blue': best[2], 'ratio': round(best[3], 2)},
                'note': '仅在对称赛道上结果才有意义',
            }).encode()
            self._send(200, 'application/json', body)
            return

        if u.path == '/frame.png' or u.path == '/frame.jpg':
            with SH.lock:
                ang = SH.angles.copy()
                rng = SH.ranges.copy()
                cones = list(SH.cones)
                clusters = list(SH.clusters)
                p = dict(SH.params)
                st = dict(SH.stats)
                s = SH.view_scale
                # 单帧几何判定的可视化数据
                st['tangent'] = SH.tangent
                st['sep'] = SH.sep
                st['agree'] = SH.agree
                st['reliable'] = SH.reliable
                st['node'] = SH.node_out
            if len(ang) < 3:
                img = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
                cv2.putText(img, 'waiting for /scan ...', (40, CANVAS_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 160, 175), 2,
                            cv2.LINE_AA)
            else:
                img = render(ang, rng, cones, clusters, p, st, s)

            if u.path == '/frame.jpg':
                ok, buf = cv2.imencode('.jpg', img,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                ctype = 'image/jpeg'
            else:
                ok, buf = cv2.imencode('.png', img,
                                       [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                ctype = 'image/png'
            if not ok:
                self._send(500, 'text/plain', b'encode failed')
                return
            self._send(200, ctype, buf.tobytes())
            return

        self._send(404, 'text/plain', b'not found')

    # ---------------------------------------------------------- POST
    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get('Content-Length', 0) or 0)
        raw = self.rfile.read(n) if n else b'{}'
        try:
            data = json.loads(raw.decode('utf-8')) if raw else {}
        except Exception:
            data = {}

        if u.path == '/set':
            with SH.lock:
                for k, v in data.items():
                    if k == '_view_scale':
                        SH.view_scale = float(v)
                    elif k in SH.params:
                        try:
                            SH.params[k] = float(v)
                        except (TypeError, ValueError):
                            pass
            self._send(200, 'application/json', b'{"ok":true}')
            return

        if u.path == '/reset':
            with SH.lock:
                SH.params = dict(DEFAULT_PARAMS)
                SH.view_scale = SCALE_DEFAULT
            self._send(200, 'application/json', b'{"ok":true}')
            return

        if u.path == '/enable':
            with SH.lock:
                SH.enable = bool(data.get('on', True))
            self._send(200, 'application/json', b'{"ok":true}')
            return

        if u.path == '/save':
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'cone_params.json')
            try:
                with SH.lock:
                    payload = {'params': SH.params, 'view_scale': SH.view_scale}
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                msg = '已保存到 %s' % path
            except Exception as e:
                msg = '保存失败: %s' % e
            self._send(200, 'text/plain; charset=utf-8', msg.encode('utf-8'))
            return

        self._send(404, 'text/plain', b'not found')

    # ---------------------------------------------------------- 工具
    def _send(self, code, ctype, body):
        try:
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # 浏览器在图片还没传完就发了新请求（旧版固定间隔轮询会这样）。
            # 这是正常的客户端行为，不是错误，直接忽略避免日志被 traceback 刷屏。
            pass


class QuietServer(ThreadingHTTPServer):
    """客户端断连时不打印 traceback（连续图片流里这是常态）。"""
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        pass


# ===================================================================
# 主程序
# ===================================================================
def load_saved_params():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_paths = [
        os.path.join(base_dir, 'config', 'param.yaml'),
        os.path.join(base_dir, 'config', 'params.yaml'),
    ]
    loaded_from_yaml = False
    for yp in yaml_paths:
        if os.path.exists(yp):
            try:
                import yaml
                with open(yp, encoding='utf-8') as f:
                    yd = yaml.safe_load(f)
                ros_p = yd.get('/**', {}).get('ros__parameters', {}) or yd.get('ros__parameters', {})
                mapping = {
                    'cd_range_min': 'range_min',
                    'cd_range_max': 'range_max',
                    'cd_gap_base': 'gap_base',
                    'cd_gap_slope': 'gap_slope',
                    'cd_min_points': 'min_points',
                    'cd_min_extent': 'min_extent',
                    'cd_max_extent': 'max_extent',
                    'cd_cone_radius': 'cone_radius',
                    'lidar_angle_offset': 'angle_offset',
                    'win_y_max': 'left_max_y',
                    'win_x_max': 'fwd_max_x',
                    'win_x_min': 'fwd_min_x',
                }
                for yk, ck in mapping.items():
                    if yk in ros_p and ck in DEFAULT_PARAMS:
                        DEFAULT_PARAMS[ck] = ros_p[yk]
                if 'win_y_max' in ros_p and 'right_max_y' in DEFAULT_PARAMS:
                    DEFAULT_PARAMS['right_max_y'] = ros_p['win_y_max']
                print('[cone] 已从 YAML 加载参数: %s' % yp)
                loaded_from_yaml = True
                break
            except Exception as e:
                print('[cone] 从 YAML 加载失败: %s' % e)

    path = os.path.join(base_dir, 'cone_params.json')
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                d = json.load(f)
            for k, v in d.get('params', {}).items():
                if k in DEFAULT_PARAMS and not loaded_from_yaml:
                    DEFAULT_PARAMS[k] = v
            SH.view_scale = float(d.get('view_scale', SCALE_DEFAULT))
            print('[cone] 已加载 %s' % path)
        except Exception as e:
            print('[cone] 加载参数失败: %s' % e)

    SH.params = dict(DEFAULT_PARAMS)


# ===================================================================
# 节点算法镜像加载 (实时显示 lidar_tracking 的四态状态机)
# ===================================================================
NODE_MODULE = None
NODE_TRACKER = None


def jnum(x):
    """float -> JSON 安全值 (NaN/inf 转 None)。"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _find_node_param_yaml(base_dir):
    cands = [
        os.path.join(base_dir, 'config', 'param.yaml'),
        os.path.join(base_dir, '..', 'config', 'param.yaml'),
        os.path.expanduser('~/racecar/src/lidar_tracking/config/param.yaml'),
    ]
    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def _load_node_module():
    """加载 scripts/lidar_tracking.py，用同一份 param.yaml 构造 ConeTracker。

    这样面板显示的舵角/曲率/状态 = 车真正在跑的那套算法，避免"两套实现"。
    """
    global NODE_MODULE, NODE_TRACKER
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.path.join(base_dir, 'scripts', 'lidar_tracking.py'),
        os.path.join(base_dir, '..', 'scripts', 'lidar_tracking.py'),
        os.path.expanduser('~/racecar/src/lidar_tracking/scripts/lidar_tracking.py'),
        '/home/davinci-mini/racecar/src/lidar_tracking/scripts/lidar_tracking.py',
    ]
    for c in cands:
        if not os.path.exists(c):
            continue
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                'lt_node', os.path.abspath(c))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not hasattr(mod, 'ConeTracker'):
                continue
            params = {}
            yp = _find_node_param_yaml(base_dir)
            if yp:
                try:
                    import yaml
                    with open(yp, encoding='utf-8') as f:
                        yd = yaml.safe_load(f)
                    ros_p = (yd.get('/**', {}).get('ros__parameters', {})
                             or yd.get('ros__parameters', {}) or {})
                    params = {k: v for k, v in ros_p.items()
                              if k in getattr(mod, 'DEFAULTS', {})}
                except Exception:
                    pass
            NODE_MODULE = mod
            NODE_TRACKER = mod.ConeTracker(params)
            print('[cone] 已加载节点算法镜像: %s' % os.path.abspath(c))
            return True
        except Exception as e:
            print('[cone] 加载节点算法失败 %s: %s' % (c, e))
    print('[cone] 未找到 scripts/lidar_tracking.py, 面板只显示独立检测结果')
    return False


def main():
    ap = argparse.ArgumentParser(description='雷达锥桶检测 / 计数 / 可视化调参')
    ap.add_argument('--topic', default='/scan', help='扫描话题（默认 /scan）')
    ap.add_argument('--port', type=int, default=8080, help='Web 端口（默认 8080）')
    ap.add_argument('--replay', default=None, help='离线回放文件（循环播放）')
    ap.add_argument('--no-web', action='store_true', help='不开 Web，只打印统计')
    args = ap.parse_args()

    if not HAVE_CV2:
        print('需要 OpenCV（cv2）'); sys.exit(1)
    if not args.no_web and not HAVE_ROS and not args.replay:
        print('需要 rclpy 或 --replay'); sys.exit(1)

    load_saved_params()
    _load_node_module()

    # ---- Web 服务器 ----
    if not args.no_web:
        try:
            srv = QuietServer(('0.0.0.0', args.port), Handler)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            print('=' * 66)
            print('可视化面板:  http://<车IP>:%d' % args.port)
            print('  例如:     http://192.168.5.4:%d' % args.port)
            print('=' * 66)
        except OSError as e:
            print('Web 端口 %d 启动失败: %s' % (args.port, e))
            print('换端口: python3 cone.py --port 8081')

    if args.no_web and not HAVE_ROS:
        print('--no-web 需要 rclpy'); sys.exit(1)

    # ---- ROS ----
    if not HAVE_ROS:
        print('未找到 rclpy。请先 source ROS 环境：')
        print('  source /opt/ros/humble/setup.bash')
        print('  source ~/racecar/install/setup.bash')
        # 无 ROS 时至少让 Web 面板可访问（用于看默认参数）
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return

    rclpy.init()
    node = ConeNode(args.topic, args.replay)
    try:
        if args.no_web:
            # 打印模式
            last = 0
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
                now = time.time()
                if now - last > 1.0:
                    last = now
                    with SH.lock:
                        cones = list(SH.cones)
                        cl = len(SH.clusters)
                        st = dict(SH.stats)
                    nl = sum(1 for c in cones if c['side'] == 'red')
                    nr = sum(1 for c in cones if c['side'] == 'blue')
                    print('[cone] total=%2d left=%2d right=%2d clusters=%3d %s %s'
                          % (len(cones), nl, nr, cl,
                             st.get('hz_str', ''), st.get('ms_str', '')))
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
