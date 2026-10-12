#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cone.py —— 雷??桶?? / 中?提取 / 可?化??工具

?? /scan，用「自适?聚?」找出?桶，并提供一??置的 Web 可?化面板：
    http://<?IP>:8080

【2026-10 改版】去掉?/?（?外圈）分?，改用"??色"中心?方案：
    ?桶按几何分成?排（y 最大?隙）→ 每排各自?合直?（排的方向=?道方向）
    → ???取平均 = ?道中心? → 直接?出?? PID ?差：
        e_lat  中?在?前方的?向位置 [m]（中?在?左?正）
        e_head 中?走向与???角 [度]（?道向左?正）
    ??舵角 = trim + k_head*e_head + k_lat*e_lat
    另?"合理性?查"：几何不像?道（?排?距/偏移/?角超限）??出"?效"，
    提示控制器保持上一舵角 —— ?面退化（撞桶/?出走廊）?防止舵机被打死。
    （背景：???注在急?/掉??必然翻?；中???排的"命名"免疫。）

面板功能：
    · 俯?????示：原始?描?、聚?、?出的?桶（A/B ?排）、?合中?、??
    · 同??示?种中?做?比：
        ?色 = 直??合中?（分?排+每排直?, 直道好、?道差）
        青色 = ?弧?合中?（arcfit, 与 lidar_tracking_arcfit.cpp 同??,
               ?道里准得多; ??在左?"?弧?合 arcfit"一?里）
    · 左?滑?????，?需重???
    · ????：?桶??、e_lat / e_head / ??舵角 / 中?有效性、??耗?
    · ??「保存??」把?前???入 JSON 文件

?什么用 Web 面板而不是 cv2.imshow：
    ?端 DISPLAY 未?置，OpenCV 的 GTK 后端?法?接 X server
    （??? "Client is not authorized to connect to Server"）。
    用??器??就不需要 X11，也不依? rviz。

与 lidar_tracking.cpp 的?系：
    本?本是**?立??工具**，不?布任何控制指令，不碰 /teleop_cmd_vel。
    可以和安全地跟???同??行。

用法：
    source /opt/ros/humble/setup.bash
    source ~/racecar/install/setup.bash
    python3 cone.py                          # 用 /scan ???据
    python3 cone.py --port 8080              # 指定 Web 端口
    python3 cone.py --replay scans.txt       # 离?回放（循?播放，便于??）
    python3 cone.py --no-web                 # 只打印??，不? Web
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
    cv2 = None            # ?式?定?模??名字，供 render() 等函?使用
    HAVE_CV2 = False


# ===================================================================
# ????
# ===================================================================
DEFAULT_PARAMS = {
    # --- 有效距离 ---
    'range_min': 0.15,      # 最近有效距离 [m]（雷?自身 range_min 之上再收?）
    'range_max': 5.50,      # 最?有效距离 [m]

    # --- 自适?聚? ---
    # 相???的"期望?距"?距离?性增?：gap = base + slope * r
    # 雷?角分辨率? 0.0064 rad，在 r 米?相?束?距? 0.0064*r
    # base 覆??桶表面??回波，slope 保???不?把?桶切?
    'gap_base': 0.045,      # 基??距 [m]
    'gap_slope': 0.020,     # 距离系? [m/m]

    # --- 尺寸??（?桶底座 0.28m，高 0.63m；回波只覆?可?面）---
    # ??：extent 撞到上限（0.40）的那些簇基本都是?面/?板，不是?桶。
    #       真??桶可?面的回波 extent 通常在 0.02~0.20 之?。
    'min_points': 3,        # 一?簇至少?么多?（2 ??噪?太多）
    'min_extent': 0.018,    # 簇最小尺寸 [m]
    'max_extent': 0.25,     # 簇最大尺寸 [m]（超???是?/?板）

    # --- ?桶半???（回波在可?面，沿??外推到表面）---
    'cone_radius': 0.10,    # ?桶半? [m]
    'min_dist': 0.40,       # 离?体最近距离 [m]

    # --- 坐?与安?角（?体坐?系：x 前、y 左）---
    'angle_offset': 0.0,    # 雷?安?角修正 [deg]（把?云旋到?体系）

    'left_max_y': 2.50,     # ?出窗口: 左?最大 [m]
    'right_max_y': 2.50,    # ?出窗口: 右?最大 [m]
    'fwd_max_x': 4.00,      # ?出窗口: 前方最??向 [m]
    'fwd_min_x': -0.30,     # ?出窗口: 后方容? [m]

    # --- 中心?提取（???方案）---
    # 分?排(几何) → 每排?合直? → 取平均 = ?道中心?
    'x_look': 1.20,          # ?差取??（?前方）[m]
    'cl_lat_max': 1.20,      # 中?只用 |y| ? 此值的近?走廊?桶 [m]（不?自?放?一?）
    'max_cones_used': 8,     # 最多用最近的?么多?桶
    'split_min_gap': 0.55,   # 分?排的最大?隙?值 [m]（?道?1.5m ? 排??距~1.1m）
    'min_per_side': 2,       # 每排最少??
    'track_half_width': 0.75,  # ?道半? [m]（?排兜底用; ???定?道? 1.5m）
    'allow_single_side': 1,  # 1=只看到一排?用"半?先?"兜底

    # --- 合理性?查（??: ?住"垃圾几何 → 舵角打?"）---
    'sep_min': 0.80,         # ?排?距下限 [m]
    'sep_max': 2.20,         # ?排?距上限 [m]
    'e_lat_max': 1.20,       # |e_lat| 超?判?效 [m]
    'e_head_max': 60.0,      # |e_head| 超?判?效 [deg]

    # --- ??舵角 = trim + k_head*e_head(度) + k_lat*e_lat(米) ---
    'k_head': 0.70,          # 航向增益 [度/度]（?道方向偏 10° → 舵? 7°）
    'k_lat': 30.0,           # ?向增益 [度/米]（偏 0.3m → 舵? 9°）
    'trim': 82.5,            # 直行基准角 [deg]（用??打表??确定）

    # --- ?弧?合中?（arcfit，与 lidar_tracking_arcfit.cpp 同一套??）---
    # 不?分左右/??：找一??弧，使每??桶到它的距离都?半?道?。
    # ?道里比"?排直??合"准得多（直??不出?曲率）。
    'arc_enable': 1,         # 1=?算并?示?弧?合
    'arc_every': 2,          # 每 N ?算一次（?端 Python ?慢, 降低占用）
    'arc_track_width': 1.5,  # ?道名??度 [m]
    'arc_tau': 0.25,         # ??容差 [m]
    'arc_len': 2.4,          # ?合弧? [m]
    'arc_max_curv': 1.2,     # 最大曲率 [1/m]
    'arc_block_cap': 3.0,    # 走廊中??路?桶的?分上限
    'arc_reg_psi': 0.10,     # 冷??正?: 偏向??方向
    'arc_reg_kappa': 0.05,   # 冷??正?: 偏向直?
    'arc_kx': 1.0,           # ?向?差->曲率增益 [1/(m·m)] (与??同式: 面板?示????的 k)
    'arc_kpsi': 1.2,         # 航向?差->曲率增益 [1/(m·rad)]
    'arc_win_x_max': 2.40,   # ?与?合的最??桶 [m]
    'arc_la1': 0.6,          # 3 ?前??弧? [m]
    'arc_la2': 1.2,
    'arc_la3': 1.8,
    'arc_rate1': 0.8,        # ?差加?（沿用 C++ rate1/2/3）
    'arc_rate2': 2.2,
    'arc_rate3': 0.8,

    # --- ??算法??（直接加? scripts/lidar_tracking.py 里的同一份代?,
    #     每?跑一遍, 面板展示??????的 模式/舵角/速度）---
    'node_vis': 1,           # 1=????算法可?化; 0=?(省 CPU)
}

PARAM_META = {
    'range_min':    ('最近有效距离', 0.05, 1.00, 0.01, 'm'),
    'range_max':    ('最?有效距离', 1.00, 20.0, 0.10, 'm'),
    'gap_base':     ('聚?基??距', 0.005, 0.300, 0.005, 'm'),
    'gap_slope':    ('聚?距离系?', 0.000, 0.100, 0.001, 'm/m'),
    'min_points':   ('簇最少??', 1, 20, 1, ''),
    'min_extent':   ('簇最小尺寸', 0.005, 0.200, 0.001, 'm'),
    'max_extent':   ('簇最大尺寸', 0.050, 1.000, 0.005, 'm'),
    'cone_radius':  ('?桶半?', 0.00, 0.30, 0.01, 'm'),
    'min_dist':     ('最近保留距离', 0.00, 2.00, 0.05, 'm'),
    'angle_offset': ('雷?安?角修正', -180.0, 180.0, 1.0, '°'),
    'left_max_y':   ('?出窗口左', 0.10, 6.00, 0.05, 'm'),
    'right_max_y':  ('?出窗口右', 0.10, 6.00, 0.05, 'm'),
    'fwd_max_x':    ('前方窗口', 0.50, 10.0, 0.10, 'm'),
    'fwd_min_x':    ('后方容?', -2.00, 0.50, 0.05, 'm'),
    'x_look':       ('?差取??', 0.40, 3.00, 0.05, 'm'),
    'cl_lat_max':   ('中??向窗口', 0.50, 3.00, 0.05, 'm'),
    'max_cones_used': ('最多取??', 4, 16, 1, ''),
    'split_min_gap': ('分排?隙?值', 0.20, 1.50, 0.05, 'm'),
    'min_per_side': ('每排最少??', 2, 6, 1, ''),
    'track_half_width': ('?道半?', 0.40, 1.50, 0.05, 'm'),
    'allow_single_side': ('?排兜底??', 0, 1, 1, ''),
    'sep_min':      ('?排?距下限', 0.30, 2.00, 0.05, 'm'),
    'sep_max':      ('?排?距上限', 1.00, 4.00, 0.05, 'm'),
    'e_lat_max':    ('e_lat 有效性上限', 0.30, 3.00, 0.05, 'm'),
    'e_head_max':   ('e_head 有效性上限', 10.0, 90.0, 1.0, '°'),
    'k_head':       ('航向增益 K_head', 0.00, 3.00, 0.05, '°/°'),
    'k_lat':        ('?向增益 K_lat', 0.0, 90.0, 1.0, '°/m'),
    'trim':         ('直行基准角 trim', 60.0, 105.0, 0.5, '°'),
    'arc_enable':   ('?弧?合??', 0, 1, 1, ''),
    'arc_every':    ('每 N ?算一次', 1, 6, 1, ''),
    'arc_track_width': ('?道?(?弧用)', 0.8, 3.0, 0.05, 'm'),
    'arc_tau':      ('??容差 tau', 0.05, 0.50, 0.01, 'm'),
    'arc_len':      ('?合弧?', 1.0, 3.0, 0.1, 'm'),
    'arc_max_curv': ('最大曲率', 0.2, 2.0, 0.1, '1/m'),
    'arc_block_cap': ('?路?分上限', 1.0, 12.0, 0.5, 'x'),
    'arc_reg_psi':  ('冷??正? psi', 0.0, 1.0, 0.05, ''),
    'arc_reg_kappa': ('冷??正? kappa', 0.0, 1.0, 0.05, ''),
    'arc_kx':       ('?向增益 Kx', 0.0, 3.0, 0.1, ''),
    'arc_kpsi':     ('航向增益 Kpsi', 0.0, 3.0, 0.1, ''),
    'arc_win_x_max': ('?合最??', 1.0, 4.0, 0.1, 'm'),
    'arc_la1':      ('前??1', 0.1, 2.4, 0.1, 'm'),
    'arc_la2':      ('前??2', 0.1, 2.4, 0.1, 'm'),
    'arc_la3':      ('前??3', 0.1, 2.4, 0.1, 'm'),
    'arc_rate1':    ('?重 rate1', 0.0, 4.0, 0.1, ''),
    'arc_rate2':    ('?重 rate2', 0.0, 4.0, 0.1, ''),
    'arc_rate3':    ('?重 rate3', 0.0, 4.0, 0.1, ''),
    'node_vis':     ('??算法可?化', 0, 1, 1, ''),
}


def detect_cones(angles, ranges, p):
    """
    自适?聚?找?桶（numpy 向量化，?端 ARM 上也要能跑 10Hz）。

    ??
    ----
    angles : (N,) 每?束的角度 [rad]
    ranges : (N,) 每?束的距离 [m]，?效值? nan/inf
    p      : ??字典

    返回
    ----
    cones  : list of dict(x, y, r, a, n, extent, grp)
             ?体坐? x 前、y 左。grp 先?一置 'un'，
             由 extract_centerline 之后在 process() 里?成 'A'/'B'（?用于?示）。
    clusters : list of (n_points, cx, cy) —— 所有簇（??用，未做尺寸??）
    """
    r = np.asarray(ranges, dtype=np.float64)
    a = np.asarray(angles, dtype=np.float64)
    N = r.size

    # 1) 有效???（先算掩?，后?所有?算只在有效?上做，
    #    避免 inf * sin() 之?的 numpy 警告）
    valid = np.isfinite(r) & (r >= p['range_min']) & (r <= p['range_max'])
    if not valid.any():
        return [], []

    # 2) ??体坐?：x 前 = r*cos(theta)，y 左 = r*sin(theta)
    #    注意 ROS LaserScan 的角度以 +x ? 0、逆???正
    rv = r[valid]
    av = a[valid]
    # 安?角修正：把?云旋到?体坐?系（angle_offset=0 ?不做任何事）
    off = math.radians(p.get('angle_offset', 0.0))
    if off:
        av = av - off
    xs = np.empty_like(r)
    ys = np.empty_like(r)
    xs[valid] = rv * np.cos(av)
    ys[valid] = rv * np.sin(av)
    xs[~valid] = 0.0
    ys[~valid] = 0.0

    # 3) 向量化找聚???
    #    相?有效?的?氏距离 > 期望?距(base + slope*r) ???。
    #    gap_base 只有在?????才有意?，所以 gap_slope=0 ?用固定?距。
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return [], []

    # ??段：valid 中相?索引差? 1 的才比?距离
    vi = valid[idx]                     # 全 True（按构造）
    # 相??在原??里是否??
    contiguous = np.diff(idx) == 1

    dx = np.diff(xs[idx])
    dy = np.diff(ys[idx])
    dist = np.hypot(dx, dy)
    r_mid = np.maximum(r[idx[:-1]], r[idx[1:]])
    expect = p['gap_base'] + p['gap_slope'] * r_mid

    # ??：非?? 或 距离超限
    brk = (~contiguous) | (dist > expect)
    # 段?界（在原 idx ??中的位置）
    seg_starts = np.concatenate(([0], np.flatnonzero(brk) + 1))
    seg_ends = np.concatenate((np.flatnonzero(brk), [idx.size - 1]))

    groups = []
    for s, e in zip(seg_starts, seg_ends):
        if e - s + 1 >= 2:
            groups.append(idx[s:e + 1])

    # ?形?描首尾相接（0 度和 360 度附近可能是同一?物体）
    if len(groups) > 1:
        f, l = groups[0], groups[-1]
        if math.hypot(xs[f[0]] - xs[l[-1]], ys[f[0]] - ys[l[-1]]) < 0.08:
            groups[0] = np.concatenate((l, f))
            groups.pop()

    # 4) 尺寸 / ???? + ?桶半???
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

        # 沿??把簇心外推到?桶表面
        rr = math.hypot(cx, cy)
        if rr < p['min_dist']:
            continue
        if rr > 1e-6:
            scale = (rr + p['cone_radius']) / rr
            cx *= scale
            cy *= scale

        # 5) 位置窗口
        if not (p['fwd_min_x'] <= cx <= p['fwd_max_x']):
            continue
        if cy > p['left_max_y'] or cy < -p['right_max_y']:
            continue

        cones.append({
            'x': cx, 'y': cy,
            'r': math.hypot(cx, cy),
            'a': math.degrees(math.atan2(cy, cx)),
            'n': n, 'extent': extent, 'grp': 'un',
        })

    # 按距离排序，近的在前
    cones.sort(key=lambda c: c['r'])
    return cones, clusters


# ===================================================================
# 中心?提取（???方案）
# ===================================================================
# 原理:
#   ?排?桶分居?道??, 把?按 y 排序后在【最大?隙】?切?即得?排
#   （?道? 1.5m ? 排??桶?距 ~1.1m, 最大?隙必然在?排之?）。
#   每排各自?合直? —— 一排?桶的方向就是?道方向 —— ???取平均
#   就是?道中心?。中??"哪排是?、哪排是?"完全免疫, 不需要任何?注。
#
#   ?出??可直接? PID 的?差:
#     e_lat  [m]  中?在?前方 x_look ?的?向位置(中?在?左?正)
#     e_head [度] 中?走向与???角(?道向左?正)
#   ??舵角 = trim + k_head*e_head + k_lat*e_lat   (??增益都?正)
#
# 合理性?查: ?排?距/?向偏移/?角超限 → 判?效, 控制器保持上一舵角。
# (?拍?据??: 走廊?中?提取率 100%, e_lat ??跳? 0 次;
#  ?面退化段全部被合理性?查?下, 避免"垃圾几何 → 舵角打?"。)

def split_two_groups(xy, p):
    """
    按 y 最大?隙把?分成?排。xy: (n,2) ?体系。
    返回 (索引A, 索引B) 或 None（?隙不?大 / 任一排??不足）。
    """
    order = np.argsort(xy[:, 1])
    sy = xy[order, 1]
    gaps = np.diff(sy)
    if gaps.size == 0 or gaps.max() < p['split_min_gap']:
        return None
    k = int(np.argmax(gaps))
    thr = 0.5 * (sy[k] + sy[k + 1])
    mask = xy[:, 1] > thr
    ia = np.flatnonzero(mask)
    ib = np.flatnonzero(~mask)
    if len(ia) < p['min_per_side'] or len(ib) < p['min_per_side']:
        return None
    return ia, ib


def fit_line_xy(xy):
    """
    最小二乘?合 y = a*x + b。>=4 ??先?合一次,
    剔除?差 > 0.35m 的野?(比如??的?板??)后重?合。
    返回 (a, b) 或 None。
    """
    if len(xy) < 2:
        return None
    x = xy[:, 0]
    y = xy[:, 1]
    if len(xy) == 2:
        d = x[1] - x[0]
        if abs(d) < 1e-6:
            return 0.0, float(y.mean())
        a = (y[1] - y[0]) / d
        return float(a), float(y[0] - a * x[0])
    a, b = np.polyfit(x, y, 1)
    if len(xy) >= 4:
        resid = np.abs(y - (a * x + b))
        keep = resid <= 0.35
        if 2 <= int(keep.sum()) < len(xy):
            a, b = np.polyfit(x[keep], y[keep], 1)
    return float(a), float(b)


def extract_centerline(cones, p):
    """
    ???中心?提取。

    流程: 取 |y| ? cl_lat_max 的最近 max_cones_used ??桶 →
          按 y 最大?隙分?排 → 每排?合直? → 合理性?查 →
          ???加?平均 = 中? → ?出 e_lat / e_head。

    返回 dict 或 None(本?判?效):
        e_lat / e_head_deg / line=(a,b) / fitA=(a,b) / fitB=(a,b)(或None)
        / A_ids / B_ids / sep / method('fit'|'single')
    """
    n = len(cones)
    if n < 2:
        return None
    xs = np.array([c['x'] for c in cones])
    ys = np.array([c['y'] for c in cones])
    ids = np.arange(n)
    x_look = p['x_look']

    # 逐?放??向窗口: 优先只用近?走廊?的?桶
    for lat in (p['cl_lat_max'], p['cl_lat_max'] + 0.6, 2.5):
        sel = ids[np.abs(ys) <= lat]
        if sel.size < 2 * p['min_per_side']:
            continue
        use = sel[:int(p['max_cones_used'])]
        if use.size < 2 * p['min_per_side']:
            continue
        xy = np.column_stack([xs[use], ys[use]])

        sp = split_two_groups(xy, p)
        if sp is None:
            continue
        ia, ib = sp
        fit_a = fit_line_xy(xy[ia])
        fit_b = fit_line_xy(xy[ib])
        if fit_a is None or fit_b is None:
            continue
        a_a, b_a = fit_a
        a_b, b_b = fit_b

        # ---- 合理性?查（??）----
        # 崩?/?出走廊?, 2+2 ?散?也能"?合出直?", 但几何毫?意?;
        # 放行??舵角瞬?打到极限, 必?在?里?掉。
        sep_look = abs((a_a * x_look + b_a) - (a_b * x_look + b_b))
        if not (p['sep_min'] <= sep_look <= p['sep_max']):
            continue

        # 按 x 跨度加?: ?少而近的一??斜率不可靠
        w_a = max(0.5, float(xy[ia][:, 0].max() - xy[ia][:, 0].min()))
        w_b = max(0.5, float(xy[ib][:, 0].max() - xy[ib][:, 0].min()))
        a = (a_a * w_a + a_b * w_b) / (w_a + w_b)
        b = (b_a * w_a + b_b * w_b) / (w_a + w_b)

        e_lat = a * x_look + b
        e_head_deg = math.degrees(math.atan(a))
        if abs(e_lat) > p['e_lat_max'] or abs(e_head_deg) > p['e_head_max']:
            continue

        return {'e_lat': float(e_lat), 'e_head_deg': float(e_head_deg),
                'line': (float(a), float(b)),
                'fitA': (float(a_a), float(b_a)), 'fitB': (float(a_b), float(b_b)),
                'A_ids': use[ia], 'B_ids': use[ib],
                'sep': float(sep_look), 'method': 'fit'}

    # ---- 兜底: 只看到一排 → 用?道半? 0.75m 先? ----
    # （急?瞬?可能只看到一?; ???定?道? 1.5m, 所以?排也能推出中?）
    if int(p.get('allow_single_side', 1)):
        sel = ids[np.abs(ys) <= p['cl_lat_max']]
        if sel.size >= 3:
            use = sel[:int(p['max_cones_used'])]
            xy = np.column_stack([xs[use], ys[use]])
            fit = fit_line_xy(xy)
            y_mean = float(xy[:, 1].mean())
            if fit is not None and abs(y_mean) > 0.15:
                a, b = fit
                sign = 1.0 if y_mean > 0 else -1.0
                y_look = a * x_look + b - sign * p['track_half_width']
                e_head_deg = math.degrees(math.atan(a))
                if abs(y_look) <= 1.5 and abs(e_head_deg) <= p['e_head_max']:
                    ids_a = use if sign > 0 else np.zeros(0, dtype=int)
                    ids_b = use if sign < 0 else np.zeros(0, dtype=int)
                    return {'e_lat': float(y_look),
                            'e_head_deg': float(e_head_deg),
                            'line': (float(a), float(y_look - a * x_look)),
                            'fitA': (float(a), float(b)), 'fitB': None,
                            'A_ids': ids_a, 'B_ids': ids_b,
                            'sep': float(p['track_half_width'] * 2.0),
                            'method': 'single'}
    return None


# ===================================================================
# ?弧?合中?（arcfit）
# ===================================================================
# 与 lidar_tracking_arcfit.cpp 同一套??：找一??弧 (x0, psi, kappa)，
# 使每??桶到弧的距离都 ? 半?道?（与左右/?色??）；落在走廊中?的
# ?桶重?，????不?。粗网格 + 局部?化，?上一??果做??先?。
# ???定: x = 左?正, y = 前?正（和本文件 detect_cones ?出一致）。
#
# ?什么?道里比"分?排+直??合"好:
#   直??不出曲率 —— 左右?排各?一?直?, ?道里???合的是"弦",
#   中?的近端?被弦的弓高系?性?偏 (R=5 ?前 1m ?就偏 ~0.3m)。

class ArcFitter:
    def __init__(self):
        self.prev = None          # (x0, psi, kappa)
        self.have_prev = False
        self.last = None          # 最近一次成功?果 dict
        self._cnt = 0             # ?算次?（每 10 次做一次全局粗搜）

    @staticmethod
    def arc_point(x0, psi, kappa, s):
        """中?上弧? s ?的? (left, fwd)"""
        if abs(kappa) < 1e-3:
            return x0 + s * math.sin(psi), s * math.cos(psi)
        th = psi + kappa * s
        return (x0 + (math.cos(psi) - math.cos(th)) / kappa,
                (math.sin(th) - math.sin(psi)) / kappa)

    def _clamp(self, f, p):
        x_max = 0.6 * p['arc_track_width']
        return (min(x_max, max(-x_max, f[0])),
                min(1.4, max(-1.4, f[1])),
                min(p['arc_max_curv'], max(-p['arc_max_curv'], f[2])))

    def _score(self, f, qx, qy, qr, p):
        x0, psi, kappa = f
        tau = p['arc_tau']
        half = 0.5 * p['arc_track_width']
        L = p['arc_len']
        sp, cp = math.sin(psi), math.cos(psi)
        straight = abs(kappa) < 1e-3
        if straight:
            t = np.clip((qx - x0) * sp + qy * cp, -0.3, L)
            d = np.hypot(qx - (x0 + t * sp), qy - t * cp)
        else:
            R = 1.0 / kappa
            cx = x0 + R * cp
            cy = -R * sp
            vx = qx - cx
            vy = qy - cy
            thq = np.arctan2(vy, -vx) if kappa > 0 else np.arctan2(-vy, vx)
            dth = thq - psi
            dth = (dth + math.pi) % (2 * math.pi) - math.pi
            s = np.clip(dth / kappa, -0.3, L)
            th = psi + kappa * s
            px = x0 + (cp - np.cos(th)) / kappa
            py = (np.sin(th) - sp) / kappa
            d = np.hypot(qx - px, qy - py)
        e = d - half
        w = 1.0 / (1.0 + 0.25 * np.maximum(0.0, qr - 1.0))
        ci = np.where(e > tau, tau * tau,
                      np.where(e >= -tau, e * e,
                               np.minimum(e * e, p['arc_block_cap'] * tau * tau)))
        inliers = int(np.sum((e >= -tau) & (e <= tau)))
        cost = float(np.sum(w * ci)) + 0.3 * x0 * x0
        if not self.have_prev:
            scale = min(1.0, max(0.1, (7.0 - len(qx)) / 3.0))
            cost += scale * (p['arc_reg_psi'] * psi * psi
                             + p['arc_reg_kappa'] * kappa * kappa)
        else:
            px0, ppsi, pk = self.prev
            cost += (0.5 * (x0 - px0) ** 2 + 0.7 * (psi - ppsi) ** 2
                     + 0.05 * (kappa - pk) ** 2)
        return cost, inliers

    def fit(self, cones, p):
        """cones: [(left, fwd, r)]; 返回 dict 或 None"""
        if len(cones) < 2:
            return None
        arr = np.asarray(cones, dtype=np.float64)
        qx, qy, qr = arr[:, 0], arr[:, 1], arr[:, 2]

        top = []

        def consider(f):
            f = self._clamp(f, p)
            cost, inl = self._score(f, qx, qy, qr, p)
            top.append((cost, f, inl))
            top.sort(key=lambda t: t[0])
            del top[3:]

        x_max = 0.6 * p['arc_track_width']
        k_max = p['arc_max_curv']
        self._cnt += 1
        if (not self.have_prev) or (self._cnt % 10 == 0):
            # 全局粗搜（冷?? / 定期兜底）
            for x0 in np.arange(-x_max, x_max + 1e-9, 0.3):
                for pd in range(-60, 61, 15):
                    for k in np.arange(-k_max, k_max + 1e-9, 0.4):
                        consider((float(x0), pd * math.pi / 180.0, float(k)))
        else:
            # 暖??：只在上一次?果附近撒种子（?端 ARM 上省??）
            px0, ppsi, pk = self.prev
            for dx in (-0.3, 0.0, 0.3):
                for dpd in (-15, 0, 15):
                    for dk in (-0.4, 0.0, 0.4):
                        consider((px0 + dx, ppsi + dpd * math.pi / 180.0, pk + dk))
        if self.have_prev:
            consider(self.prev)

        best = None
        for _, f0, _ in top:
            f = f0
            cost, inl = self._score(f, qx, qy, qr, p)
            sx, spd, sk = 0.12, 6.0 * math.pi / 180.0, 0.18
            for _round in range(3):
                improved, it = True, 0
                while improved and it < 10:
                    improved = False
                    it += 1
                    for dim in range(3):
                        for dr in (-1.0, 1.0):
                            g = list(f)
                            g[dim] += dr * (sx if dim == 0 else spd if dim == 1 else sk)
                            g = self._clamp(tuple(g), p)
                            c2, i2 = self._score(g, qx, qy, qr, p)
                            if c2 < cost - 1e-9:
                                f, cost, inl = g, c2, i2
                                improved = True
                sx *= 0.5
                spd *= 0.5
                sk *= 0.5
            if best is None or cost < best[0]:
                best = (cost, f, inl)

        cost, f, inl = best
        if inl < 2:
            return None
        self.prev = f
        self.have_prev = True

        # 前?? + ?差（与 C++ 的 error 同式: 加? 2*x ×53.33）
        las = (p['arc_la1'], p['arc_la2'], p['arc_la3'])
        rates = (p['arc_rate1'], p['arc_rate2'], p['arc_rate3'])
        pts = [self.arc_point(f[0], f[1], f[2], s) for s in las]
        wsum = sum(r for r in rates if r > 0)
        err = 0.0
        if wsum > 0:
            err = sum(2.0 * pt[0] * r for pt, r in zip(pts, rates) if r > 0) / wsum
        # ??????的期望曲率 (同一公式), 面板直接?示便于?照??
        k_node = (f[2] + p.get('arc_kx', 1.0) * f[0] + p.get('arc_kpsi', 1.2) * f[1])
        res = {'x0': f[0], 'psi': f[1], 'kappa': f[2], 'inliers': inl,
               'err_deg': err * 53.33, 'k_node': k_node, 'la': pts, 'fit': f}
        self.last = res
        return res


ARC = ArcFitter()


# ===================================================================
# 可?化渲染
# ===================================================================
CANVAS_W, CANVAS_H = 820, 820
SCALE_DEFAULT = 90.0    # px / m
BG = (24, 28, 34)

# 出?格式：JPEG ??在?端 ARM 上比 PNG 快?倍（?? PNG zlib 是主要??之一）
# 面板是??刷新的???，JPEG 的?微???失完全可接受。
JPEG_QUALITY = 72

# ?排的?示配色（BGR）。只是"分?"的?色, 不代表?/??桶！
GRP_COLS = {'A': (60, 190, 250), 'B': (230, 120, 230), 'un': (170, 170, 180)}


def render(angles, ranges, cones, clusters, params, stats, view_scale):
    """渲染俯?? PNG。返回 BGR ndarray。"""
    img = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
    ox, oy = CANVAS_W // 2, CANVAS_H - 120        # ?体位置（下方居中）
    s = view_scale

    def to_px(x, y):
        # ?体 x 前 → 屏幕向上；y 左 → 屏幕向左
        return int(ox - y * s), int(oy - x * s)

    # ---- 网格（自适?步?：?放越小格子越大，?少?制?用）----
    # ?端 ARM ??：每? cv2.line ? 0.5ms，网格是主要的渲染??。
    # 因此按?放??步?，保?屏幕上的格子不小于 ~45px。
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

    # ---- 原始?描?（向量化?入，避免逐? Python 循?）----
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

    # ---- 位置窗口?框 ----
    p = params
    corners = [(p['fwd_max_x'], p['left_max_y']), (p['fwd_max_x'], -p['right_max_y']),
               (p['fwd_min_x'], -p['right_max_y']), (p['fwd_min_x'], p['left_max_y'])]
    pts = np.array([to_px(x, y) for x, y in corners], dtype=np.int32)
    cv2.polylines(img, [pts], True, (70, 110, 150), 1, cv2.LINE_AA)

    # ---- 所有簇（未??，淡色）----
    for (n, cx, cy) in clusters:
        X, Y = to_px(cx, cy)
        if 0 <= X < CANVAS_W and 0 <= Y < CANVAS_H:
            cv2.circle(img, (X, Y), 3, (90, 90, 100), -1, cv2.LINE_AA)

    # ---- ?出的?桶（按?排分??色; 只表示"哪一排", ??色含?）----
    for i, c in enumerate(cones):
        X, Y = to_px(c['x'], c['y'])
        col = GRP_COLS.get(c.get('grp', 'un'), GRP_COLS['un'])
        cv2.circle(img, (X, Y), int(max(4, c['extent'] * s * 0.5)),
                   col, 2, cv2.LINE_AA)
        cv2.circle(img, (X, Y), 2, col, -1, cv2.LINE_AA)
        # ??体到?桶的??
        cv2.line(img, (ox, oy), (X, Y), tuple(int(v * 0.35) for v in col), 1, cv2.LINE_AA)

        # ??：沿"?→?桶"方向再往外推一段，避免?到相??
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

    # ---- ?合中?（???方案的核心?示）----
    #   ?色粗? = ?道中心?(?排?合?的平均)
    #   ??(琥珀/品?) = ?排各自的?合?
    #   白十字 = ?差取??(x_look ?), e_lat 就是它的?向坐?
    cl = stats.get('cl')
    if cl is not None:
        for key, gid in (('fitA', 'A'), ('fitB', 'B')):
            fit = cl.get(key)
            if not fit:
                continue
            a_g, b_g = fit
            xs_g = [c['x'] for c in cones if c.get('grp') == gid]
            if len(xs_g) >= 2:
                x0, x1 = min(xs_g), max(xs_g)
                if x1 - x0 < 0.6:
                    x0, x1 = x0 - 0.3, x1 + 0.3
                cv2.line(img, to_px(x0, a_g * x0 + b_g),
                         to_px(x1, a_g * x1 + b_g), GRP_COLS[gid], 1, cv2.LINE_AA)

        a_c, b_c = cl['line']
        x_end = max(3.0, float(p.get('x_look', 1.2)) + 1.5)
        cv2.line(img, to_px(0.0, b_c), to_px(x_end, a_c * x_end + b_c),
                 (120, 240, 160), 2, cv2.LINE_AA)
        # ?道方向箭?(沿中?走向, 指向前方)
        d = np.array([1.0, a_c])
        d = d / (np.hypot(d[0], d[1]) + 1e-9)
        cv2.arrowedLine(img, to_px(0.0, 0.0), to_px(d[0] * 1.3, d[1] * 1.3),
                        (120, 240, 160), 2, cv2.LINE_AA, tipLength=0.18)
        # ?差取??
        xl = float(p.get('x_look', 1.2))
        LX, LY = to_px(xl, a_c * xl + b_c)
        cv2.drawMarker(img, (LX, LY), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)

    # ---- ?弧?合中?（arcfit, 青色）: 与上面?色直?中?的?照 ----
    arc = stats.get('arc')
    if arc is not None:
        f = arc['fit']
        ss = np.arange(0.0, float(p.get('arc_len', 2.4)) + 1e-9, 0.08)
        pts = [ArcFitter.arc_point(f[0], f[1], f[2], s) for s in ss]
        pix = np.array([to_px(pt[1], pt[0]) for pt in pts], dtype=np.int32)
        if len(pix) >= 2:
            cv2.polylines(img, [pix], False, (230, 230, 80), 2, cv2.LINE_AA)
        for (lx, ly) in arc['la']:
            cv2.circle(img, to_px(ly, lx), 5, (230, 230, 80), 1, cv2.LINE_AA)
        cv2.circle(img, to_px(0.0, f[0]), 3, (230, 230, 80), -1, cv2.LINE_AA)

    # ---- ??算法??（橙色）: ??此刻?走的弧?(k_cmd, 安全?之后) + 瞄准? ----
    node_out = stats.get('node_out')
    if node_out and node_out.get('mode') != 'ERR':
        k_node = float(node_out.get('k_cmd') or 0.0)
        L_arc = float(p.get('safe_lookahead', 1.5)) or 1.5
        ss = np.arange(0.0, L_arc + 1e-9, 0.08)
        if abs(k_node) < 1e-3:
            pix = np.array([to_px(float(s), 0.0) for s in ss], dtype=np.int32)
        else:
            pix = np.array([to_px(float(np.sin(k_node * s) / k_node),
                                  float((1.0 - np.cos(k_node * s)) / k_node))
                            for s in ss], dtype=np.int32)
        if len(pix) >= 2:
            cv2.polylines(img, [pix], False, (60, 140, 255), 2, cv2.LINE_AA)
        aim = node_out.get('aim')
        if aim:
            cv2.circle(img, to_px(float(aim[0]), float(aim[1])), 6,
                       (60, 140, 255), 2, cv2.LINE_AA)

    # ---- ?? ----
    car = np.array([to_px(0.30, 0.18), to_px(0.30, -0.18),
                    to_px(-0.26, -0.18), to_px(-0.26, 0.18)], dtype=np.int32)
    cv2.fillPoly(img, [car], (120, 230, 160))
    cv2.polylines(img, [car], True, (60, 150, 100), 1, cv2.LINE_AA)
    # ??三角
    nose = np.array([to_px(0.42, 0.10), to_px(0.30, 0.0), to_px(0.42, -0.10)],
                    dtype=np.int32)
    cv2.fillPoly(img, [nose], (120, 230, 160))

    # ---- 中心??（?考）----
    cv2.line(img, (ox, oy), to_px(3.0, 0), (70, 90, 110), 1, cv2.LINE_AA)

    # ---- 上半部??面板 ----
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (CANVAS_W, 118), (18, 21, 26), -1)
    cv2.addWeighted(overlay, 0.86, img, 0.14, 0, img)

    def txt(s_, x, y, col=(220, 228, 238), sz=0.52, th=1):
        cv2.putText(img, s_, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sz, col, th, cv2.LINE_AA)

    n_a = sum(1 for c in cones if c.get('grp') == 'A')
    n_b = sum(1 for c in cones if c.get('grp') == 'B')
    n_un = len(cones) - n_a - n_b
    valid = bool(stats.get('valid'))

    txt('CONE  %d' % len(cones), 14, 30, (120, 240, 180), 0.78, 2)
    txt('A %d' % n_a, 14, 56, GRP_COLS['A'], 0.55, 1)
    txt('B %d' % n_b, 80, 56, GRP_COLS['B'], 0.55, 1)
    txt('un %d' % n_un, 146, 56, GRP_COLS['un'], 0.55, 1)
    txt('valid %s' % ('YES' if valid else 'NO'), 235, 56,
        (120, 240, 160) if valid else (80, 120, 240), 0.55, 1)

    off = params.get('angle_offset', 0.0)
    txt('ang_off %.0f deg' % off, 445, 56, (230, 190, 120), 0.55, 1)

    txt('clusters %d' % len(clusters), 14, 82, (170, 180, 195), 0.48, 1)
    txt('beams %d' % int(ok.sum()), 190, 82, (170, 180, 195), 0.48, 1)
    txt('%s' % stats.get('hz_str', ''), 330, 82, (170, 180, 195), 0.48, 1)
    txt('%s' % stats.get('ms_str', ''), 470, 82, (170, 180, 195), 0.48, 1)

    txt('scale %.0f px/m' % s, CANVAS_W - 210, 30, (140, 150, 165), 0.46, 1)
    txt('source: %s' % stats.get('source', '-'), CANVAS_W - 210, 56,
        (140, 150, 165), 0.46, 1)
    txt('frame %s' % stats.get('frame', '-'), CANVAS_W - 210, 82,
        (110, 170, 220), 0.46, 1)

    # 中??差与??舵角（面板最?眼的一行）
    if valid:
        line3 = ('e_lat %+.2f m   e_head %+.1f deg   servo %.1f deg'
                 % (float(stats.get('e_lat') or 0.0),
                    float(stats.get('e_head') or 0.0),
                    float(stats.get('servo') or 0.0)))
        col3 = (120, 240, 160)
    else:
        line3 = 'NO CENTERLINE  ->  hold last steering'
        col3 = (80, 120, 240)
    txt(line3, 14, 130, col3, 0.55, 1)
    txt('sep %.2fm  method %s' % (float(stats.get('sep') or 0.0),
                                  stats.get('method', '-')),
        14, 152, (170, 180, 195), 0.46, 1)

    # ?弧?合??行（青色）
    if arc is not None:
        txt('arcfit x0 %+.2f  psi %+.1f  kappa %+.2f  k_cmd %+.2f  inl %d  err %+.1f'
            % (arc['x0'], math.degrees(arc['psi']), arc['kappa'],
               arc.get('k_node', 0.0), arc['inliers'], arc['err_deg']),
            14, 174, (230, 230, 80), 0.48, 1)
    else:
        txt('arcfit: --', 14, 174, (150, 150, 160), 0.48, 1)

    # ??算法??行（橙色）: ??此刻????的 舵角/速度/曲率/避障?距
    if node_out and node_out.get('mode') != 'ERR':
        txt('NODE [%s] servo %.1f  speed %.0f  k_cmd %+.2f (k_des %+.2f)  clear %.2f'
            % (node_out.get('mode', '?'), float(node_out.get('servo') or 0.0),
               float(node_out.get('speed') or 0.0),
               float(node_out.get('k_cmd') or 0.0),
               float(node_out.get('k_des') or 0.0),
               float(node_out.get('clear') or 0.0)),
            14, 196, (60, 140, 255), 0.52, 1)
    elif node_out:
        txt('NODE ERR: %s' % node_out.get('err', '?'),
            14, 196, (80, 120, 240), 0.48, 1)

    txt('x=forward  y=left', 14, 106, (130, 140, 155), 0.44, 1)

    return img


# ===================================================================
# 共享??（ROS ?程?，HTTP ?程?）
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
        # 中心?提取?果（供面板?示）
        self.cl = None            # extract_centerline 的返回(dict) 或 None
        self.e_lat = float('nan')
        self.e_head_deg = float('nan')
        self.servo = float('nan')
        self.valid = False        # 本?中?是否有效
        self.sep = 0.0            # x_look ??排?距 [m]
        self.method = '-'         # fit / single / -
        self.arc = None           # ?弧?合?果(ArcFitter.fit 的返回) 或 None
        self.arc_skip = 0         # arc_every 的降???
        self.node_out = None      # ??算法??(同一份代?)每??出: mode/servo/speed/k...


SH = Shared()


def jnum(x):
    """float ? JSON 安全值（NaN/inf → None, 否? json ??生非法 JSON）"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


# ===================================================================
# ROS ??
# ===================================================================
if HAVE_ROS:
    class ConeNode(Node):
        def __init__(self, topic, replay):
            super().__init__('cone_detector')
            self.replay = replay
            self.hz = 0.0
            self.stamps = []          # 最近 2 秒的??戳(算真??率)
            self.times = []

            if replay:
                self.get_logger().info('回放模式：%s' % replay)
                self._replay_loop()
            else:
                # 只?? /scan —— 不使用里程?，也不使用 IMU
                self.create_subscription(LaserScan, topic, self.on_scan, 5)
                self.get_logger().info(
                    '?? %s（???中心?提取, 不依?里程?/IMU）' % topic)

        # ---------------------------------------------------------- 回放
        def _replay_loop(self):
            frames = load_scans_file(self.replay)
            if not frames:
                self.get_logger().error('回放文件?空')
                return
            self.get_logger().info('?入 %d ?，循?播放' % len(frames))
            inc = frames[0][0]
            idx = 0
            while rclpy.ok():
                amin, n, rng = frames[idx][1], frames[idx][2], frames[idx][3]
                angles = amin + np.arange(n) * inc
                self.process(angles, rng, 'replay')
                idx = (idx + 1) % len(frames)
                time.sleep(0.10)

        # ---------------------------------------------------------- ??
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
            # ?弧?合（arcfit）: 与 lidar_tracking_arcfit.cpp 同??。
            # ?端 Python ?慢, 按 arc_every 降??算, 其余?沿用上次?果。
            # ============================================================
            arc = None
            if int(p.get('arc_enable', 1)):
                if SH.arc_skip <= 0:
                    SH.arc_skip = max(1, int(p.get('arc_every', 2)))
                    win = [(c['y'], c['x'], c['r']) for c in cones
                           if -0.3 < c['x'] < 2.4 and abs(c['y']) <= 2.5]
                    try:
                        arc = ARC.fit(win[:30], p)
                    except Exception:
                        arc = None
                else:
                    SH.arc_skip -= 1
                    arc = ARC.last

            # ============================================================
            # ??算法??: 直接? scripts/lidar_tracking.py 的 ConeTracker
            # （同一份代? + 同一份 param.yaml），展示??此刻??的
            # 模式 / 舵角 / 速度 / 曲率(安全?前后) / 避障?距
            # ============================================================
            node_out = None
            if NODE_TRACKER is not None:
                try:
                    a_min = float(angles[0]) if len(angles) else 0.0
                    a_inc = (float(angles[1] - angles[0])
                             if len(angles) > 1 else 0.0)
                    servo_n, speed_n = NODE_TRACKER.step(
                        ranges, a_min, a_inc, now)
                    d = NODE_TRACKER.dbg
                    node_out = {
                        'mode': NODE_TRACKER.mode,
                        'servo': float(servo_n),
                        'speed': float(speed_n),
                        'k_cmd': float(d.get('k_cmd') or 0.0),
                        'k_des': float(d.get('k_des') or 0.0),
                        'clear': float(d.get('clear', 99.0)),
                        'aim': d.get('aim'),
                        'n_cones': int(d.get('n_cones', 0)),
                        'n_gates': int(d.get('n_gates', 0)),
                    }
                except Exception as e:
                    node_out = {'mode': 'ERR', 'err': str(e)}

            # ============================================================
            # 中心?提取（???）: 分?排 → 每排?合 → 加?平均
            #   ?出 e_lat / e_head / ??舵角; 几何不合理 → 判?效
            # ============================================================
            res = extract_centerline(cones, p)
            if res is not None:
                e_lat = res['e_lat']
                e_head_deg = res['e_head_deg']
                servo = (p['trim'] + p['k_head'] * e_head_deg
                         + p['k_lat'] * e_lat)
                valid = True
                sep = res['sep']
                method = res['method']
                for i in res['A_ids']:
                    cones[int(i)]['grp'] = 'A'
                for i in res['B_ids']:
                    cones[int(i)]['grp'] = 'B'
            else:
                e_lat = e_head_deg = servo = float('nan')
                valid = False
                sep = 0.0
                method = '-'

            ms = (time.perf_counter() - t0) * 1000.0

            # 真??率：2 秒滑?窗口（?理耗?不均? EMA(1/dt) ?失真）
            self.stamps.append(now)
            while self.stamps and now - self.stamps[0] > 2.0:
                self.stamps.pop(0)
            if len(self.stamps) >= 2:
                span = self.stamps[-1] - self.stamps[0]
                self.hz = (len(self.stamps) - 1) / span if span > 1e-6 else 0.0

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
                SH.cl = res
                SH.e_lat = e_lat
                SH.e_head_deg = e_head_deg
                SH.servo = servo
                SH.valid = valid
                SH.sep = sep
                SH.method = method
                SH.arc = arc
                SH.node_out = node_out
                SH.stats = {
                    'hz_str': '%.1f Hz' % self.hz,
                    'ms_str': '%.2f/%.1f ms' % (ms_avg, ms_max),
                    'source': source,
                    'frame': SH.frame_id,
                    'valid': valid,
                    'e_lat': e_lat,
                    'e_head': e_head_deg,
                    'servo': servo,
                    'sep': round(float(sep), 2),
                    'method': method,
                }
                SH.last_error = ''


# ===================================================================
# 回放文件加?
# ===================================================================
def load_scans_file(path):
    """
    支持?种格式：
      A) 第一行 "angle_min angle_increment"，后?每行 "t n r0 r1 ..."
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

    # ?一 inc
    if out and inc_global is None:
        inc_common = out[0][0]
        out = [(inc_common,) + f[1:] for f in out]
    return out


# ===================================================================
# Web 面板
# ===================================================================
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>?桶中? · ??面板</title>
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
  <h1>?桶中???面板</h1>
  <div id="stats">加?中…</div>
  <div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap">
    <button onclick="save()">保存??</button>
    <button class="sec" onclick="reset()">恢复默?</button>
    <button class="sec" id="tg" onclick="toggleRun()">?停</button>
    <button class="sec" onclick="sweep()">?描最佳安?角</button>
  </div>
  <div id="sweepOut" style="font-size:10.5px;color:#8b949e;margin-top:6px"></div>
  <div class="sec">?角</div>
  <div class="row"><label>?放 px/m</label>
    <input type="range" id="view_scale" min="25" max="260" step="1" value="90">
    <span class="v" id="view_scale_v">90</span></div>
  <div id="params"></div>
  <div class="lg">
    <span><i class="dot" style="background:#fabe3c"></i>A 排</span>
    <span><i class="dot" style="background:#e678e6"></i>B 排</span>
    <span><i class="dot" style="background:#78f0a0"></i>直?中?(linefit)</span>
    <span><i class="dot" style="background:#50e6e6"></i>?弧中?(arcfit)</span>
    <span><i class="dot" style="background:#ff8c3c"></i>??算法 k_cmd</span>
    <span><i class="dot" style="background:#aaaab4"></i>未分?簇</span>
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
    '距离范?': ['range_min','range_max','min_dist'],
    '自适?聚?': ['gap_base','gap_slope'],
    '尺寸??': ['min_points','min_extent','max_extent','cone_radius'],
    '位置窗口': ['left_max_y','right_max_y','fwd_max_x','fwd_min_x'],
    '中心?提取': ['x_look','cl_lat_max','max_cones_used','split_min_gap','min_per_side','track_half_width','allow_single_side'],
    '合理性?查': ['sep_min','sep_max','e_lat_max','e_head_max'],
    '??舵角': ['k_head','k_lat','trim'],
    '?弧?合 arcfit': ['arc_enable','arc_every','arc_track_width','arc_tau','arc_len','arc_max_curv','arc_block_cap','arc_reg_psi','arc_reg_kappa','arc_la1','arc_la2','arc_la3','arc_rate1','arc_rate2','arc_rate3','node_vis'],
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
  if(!confirm('恢复默???？')) return;
  fetch('/reset', {method:'POST'}).then(()=>syncFromServer());
}
function toggleRun(){
  paused = !paused;
  looping = !paused;
  document.getElementById('tg').textContent = paused ? '??' : '?停';
  fetch('/enable', {method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({on: !paused})});
  if (!paused) tick();                     // 恢复?重?出?循?
}

function sweep(){
  const out = document.getElementById('sweepOut');
  out.textContent = '?描中…';
  fetch('/sweep').then(r=>r.json()).then(d=>{
    if (!d.ok || !d.best){
      out.textContent = '未找到有效角度 —— ?先把??止放在?桶走廊中?再?描';
      return;
    }
    const b = d.best;
    let html = `最佳安?角 <b style="color:#7ee787">${b.offset}°</b> ` +
               `(?排?距 ${b.sep}m, 目? 1.5m)<br>`;
    html += '候?(?距最接近 1.5m)：<br>';
    for (const s of d.samples.slice(0,5)){
      html += `&nbsp;&nbsp;${s[0]}° → sep ${s[1]==null?'-':s[1]}m ${s[2]?'?':''}<br>`;
    }
    html += `<span style="color:#f0883e">${d.note}</span>`;
    out.innerHTML = html;
    if (confirm('把 angle_offset ?? ' + b.offset + '° ?？')) {
      setParam('angle_offset', b.offset);
      const el = document.getElementById('p_angle_offset');
      if (el){ el.value = b.offset; document.getElementById('v_angle_offset').textContent = b.offset+'°'; }
    }
  }).catch(e=>{ out.textContent = '?描出?: ' + e; });
}

// ---- 自?度刷新：等上一??片加?完再?下一? ----
// 不能用 setInterval(fn, 200)：服?端每?要 100~115ms 渲染，固定?隔??
// ???求堆?，??器取消未完成的加?，表?就是"?面不更新"。
let looping = true;
let lastFrameAt = Date.now();

function tick(){
  if (!looping) return;
  const im = document.getElementById('img');
  im.onload = () => { lastFrameAt = Date.now(); if (looping) setTimeout(tick, 60); };
  im.onerror = () => { if (looping) setTimeout(tick, 600); };
  im.src = '/frame.jpg?t=' + Date.now();
}

function fmt(x, digits){
  return (x === null || x === undefined) ? '-' : x.toFixed(digits);
}

function refreshStats(){
  fetch('/state').then(r=>r.json()).then(d=>{
    const s = d.stats;
    const lag = (Date.now() - lastFrameAt) / 1000;
    const okCol = s.valid ? '#7ee787' : '#f0883e';
    const okTxt = s.valid ? '中?有效' : '中??效 → 保持上一舵角';
    document.getElementById('stats').innerHTML =
      `?桶?? <b>${s.n_cones}</b>  (A ${s.n_a} / B ${s.n_b} / 未分? ${s.n_un})<br>` +
      `e_lat <b>${fmt(s.e_lat,2)} m</b> &nbsp; e_head <b>${fmt(s.e_head,1)}°</b><br>` +
      `??舵角 <b>${fmt(s.servo,1)}°</b> &nbsp; ?排?距 ${s.sep}m<br>` +
      `<span style="color:${okCol}">${okTxt}</span> (${s.method})<br>` +
      (s.arc_ok
        ? `<span style="color:#50e6e6">arcfit</span> x0 ${fmt(s.arc_x0,2)} ψ ${fmt(s.arc_psi,1)}° κ ${fmt(s.arc_kappa,2)} k_cmd ${fmt(s.arc_kcmd,2)} ?? ${s.arc_inl} err ${fmt(s.arc_err,1)}°<br>`
        : `<span style="color:#8b949e">arcfit --</span><br>`) +
      (s.node_ok
        ? `<span style="color:#ff8c3c">NODE [${s.node_mode}]</span> servo ${fmt(s.node_servo,1)}° speed ${fmt(s.node_speed,0)} k ${fmt(s.node_kcmd,2)} (des ${fmt(s.node_kdes,2)}) clear ${fmt(s.node_clear,2)}<br>`
        : '') +
      `聚?? ${s.n_clusters} &nbsp; 有效束 ${s.n_beams} &nbsp; ${s.hz} &nbsp; ${s.ms}<br>` +
      `? ${s.frame} &nbsp; ?源 ${s.source}<br>` +
      `<span style="color:${lag>3?'#f85149':'#7ee787'}">距上次出? ${lag.toFixed(1)}s</span>`;
  }).catch(()=>{});
}

document.getElementById('view_scale').oninput = function(){
  document.getElementById('view_scale_v').textContent = this.value;
  fetch('/set', {method:'POST', headers:{'Content-Type':'application/json'},
                 body: JSON.stringify({_view_scale: parseFloat(this.value)})});
};

buildSliders();
syncFromServer();
tick();                                    // ??自?度出?循?
setInterval(refreshStats, 400);            // ???字????（很?）
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'      # 复用?接，避免每??求重建 TCP

    def log_message(self, *a):
        pass

    def handle(self):
        # 客?端提前??是常?（???片流），不要? socketserver 打 traceback
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
                rng = SH.ranges
                frame = SH.frame_id
                e_lat = SH.e_lat
                e_head = SH.e_head_deg
                servo = SH.servo
                valid = SH.valid
                sep = SH.sep
                method = SH.method
                arc = SH.arc
                node_out = SH.node_out
            n_beams = 0
            if len(rng):
                ok = np.isfinite(rng) & (rng >= 0.15)
                n_beams = int(ok.sum())
            body = json.dumps({
                'stats': {
                    'n_cones': len(cones),
                    'n_a': sum(1 for c in cones if c.get('grp') == 'A'),
                    'n_b': sum(1 for c in cones if c.get('grp') == 'B'),
                    'n_un': sum(1 for c in cones if c.get('grp') == 'un'),
                    'n_clusters': len(clusters),
                    'n_beams': n_beams,
                    'hz': st.get('hz_str', '-'),
                    'ms': st.get('ms_str', '-'),
                    'frame': frame,
                    'source': st.get('source', '-'),
                    'valid': bool(valid),
                    'e_lat': jnum(e_lat),
                    'e_head': jnum(e_head),
                    'servo': jnum(servo),
                    'sep': round(float(sep), 2),
                    'method': method,
                    'arc_ok': arc is not None,
                    'arc_x0': jnum(arc['x0']) if arc else None,
                    'arc_psi': jnum(math.degrees(arc['psi'])) if arc else None,
                    'arc_kappa': jnum(arc['kappa']) if arc else None,
                    'arc_inl': (arc['inliers'] if arc else None),
                    'arc_err': jnum(arc['err_deg']) if arc else None,
                    'arc_kcmd': jnum(arc.get('k_node')) if arc else None,
                    'node_ok': bool(node_out is not None and node_out.get('mode') != 'ERR'),
                    'node_mode': (node_out.get('mode') if node_out else None),
                    'node_servo': jnum(node_out.get('servo')) if node_out else None,
                    'node_speed': jnum(node_out.get('speed')) if node_out else None,
                    'node_kcmd': jnum(node_out.get('k_cmd')) if node_out else None,
                    'node_kdes': jnum(node_out.get('k_des')) if node_out else None,
                    'node_clear': jnum(node_out.get('clear')) if node_out else None,
                },
                'cones': cones,
            }).encode()
            self._send(200, 'application/json', body)
            return

        if u.path == '/sweep':
            # 一次性自?: ?描 angle_offset 0~359°, 找?"?排?距"最接近 1.5m
            # 的角度（等效于校准雷?安?正方向）。
            # ??止在?桶走廊中??才有意?；不在走廊里?据?意?。
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
                res = extract_centerline(cs, pp)
                # 只??排都看到(fit)的?果; ?排兜底的 sep 是先?值, ?有校准意?
                if res is None or res['method'] != 'fit':
                    rows.append((off, None, 0))
                    continue
                sep = round(res['sep'], 2)
                rows.append((off, sep, 1))
                score = abs(sep - 1.5)
                if best is None or score < best[2]:
                    best = (off, sep, score)
            # 有效优先, 然后?距越接近 1.5m 越靠前
            rows.sort(key=lambda r: (0 if r[2] else 1,
                                     abs((r[1] if r[1] is not None else 99.0) - 1.5)))
            body = json.dumps({
                'ok': True,
                'samples': rows,
                'best': None if best is None else {
                    'offset': best[0], 'sep': best[1], 'valid': True},
                'note': '??止在?桶走廊中??使用; 以?排?距最接近 1.5m ?准',
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
                st['cl'] = SH.cl          # 中心??果(供 render ?中?/取??)
                st['arc'] = SH.arc        # ?弧?合?果(供 render ?青色弧)
                st['node_out'] = SH.node_out   # ??算法??(供 render ?期望行?弧/文字)
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
                msg = '保存失?: %s' % e
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
            # ??器在?片???完就?了新?求（?版固定?隔?????）。
            # ?是正常的客?端行?，不是??，直接忽略避免日志被 traceback 刷屏。
            pass


class QuietServer(ThreadingHTTPServer):
    """客?端???不打印 traceback（???片流里?是常?）。"""
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        pass


# ===================================================================
# 主程序
# ===================================================================
def _find_param_yaml():
    """
    找到共享??文件 param.yaml (唯一??入口)。
    注意: cone.py 在 src/lidar_tracking/src/ 下, 而??文件在上一? config/,
    所以第一?候?是 ../config/param.yaml —— ?版本的路?差了一?, ????到?。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.path.join(here, '..', 'config', 'param.yaml'),
        os.path.join(here, 'config', 'param.yaml'),
        os.path.join(os.getcwd(), 'config', 'param.yaml'),
        '/home/davinci-mini/racecar/src/lidar_tracking/config/param.yaml',
    ]
    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


# ??算法??: 直接加? scripts/lidar_tracking.py 里的同一份代?（同一套
# ??/几何?/arcfit/安全?/舵机/速度??），面板每?跑一遍它，展示??
# ????什么。??面板看到的 = ?上??在跑的，不存在"?套近似??"。
NODE_MODULE = None
NODE_TRACKER = None


def _load_node_module():
    """加?同包的循???模? (scripts/lidar_tracking.py)。"""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.path.join(here, '..', 'scripts', 'lidar_tracking.py'),
        os.path.join(here, 'lidar_tracking.py'),
        os.path.expanduser('~/racecar/src/lidar_tracking/scripts/lidar_tracking.py'),
        '/home/davinci-mini/racecar/src/lidar_tracking/scripts/lidar_tracking.py',
    ]
    for c in cands:
        if os.path.exists(c):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    'lt_node', os.path.abspath(c))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if not hasattr(mod, 'ConeTracker'):
                    continue
                return mod, os.path.abspath(c)
            except Exception as e:
                print('[cone] 加???模? %s 失?: %s' % (c, e))
    return None, None


def load_saved_params():
    """加??序: param.yaml (共享真源, 唯一??入口) → cone_params.json (面板本地??值)"""
    global NODE_MODULE, NODE_TRACKER
    # ---- 1) 共享?? param.yaml ----
    yaml_keys = set()
    raw_ros_p = None
    yp = _find_param_yaml()
    if yp:
        try:
            import yaml
            with open(yp, encoding='utf-8') as f:
                yd = yaml.safe_load(f)
            ros_p = (yd.get('/**', {}).get('ros__parameters', {})
                     or yd.get('ros__parameters', {}) or {})
            raw_ros_p = ros_p
            # 名?不同的?: ??名 -> 面板?部名 (可以一?多)
            mapping = {
                'cd_range_min': ('range_min',),
                'cd_range_max': ('range_max',),
                'cd_gap_base': ('gap_base',),
                'cd_gap_slope': ('gap_slope',),
                'cd_min_points': ('min_points',),
                'cd_min_extent': ('min_extent',),
                'cd_max_extent': ('max_extent',),
                'cd_cone_radius': ('cone_radius',),
                'lidar_angle_offset': ('angle_offset',),
                'win_y_max': ('left_max_y', 'right_max_y'),
                'win_x_max': ('fwd_max_x',),
                'win_x_min': ('fwd_min_x',),
                'servo_center': ('trim',),   # 面板??舵角与??用同一?中位
            }
            for yk, targets in mapping.items():
                if yk in ros_p:
                    for t in targets:
                        if t in DEFAULT_PARAMS:
                            DEFAULT_PARAMS[t] = ros_p[yk]
                            yaml_keys.add(t)
            # 同名?直接套用 (arc_* 等)
            for yk, v in ros_p.items():
                if yk in DEFAULT_PARAMS:
                    DEFAULT_PARAMS[yk] = v
                    yaml_keys.add(yk)
            print('[cone] 共享??已? %s ?入 %d ? (以后??只改??文件)'
                  % (yp, len(yaml_keys)))
        except Exception as e:
            print('[cone] ?取 param.yaml 失?: %s' % e)
    else:
        print('[cone] 未找到 config/param.yaml, 使用?置默?值')

    # ---- 2) cone_params.json 只? param.yaml ?管的? (面板自己的????) ----
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cone_params.json')
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                d = json.load(f)
            n = 0
            for k, v in d.get('params', {}).items():
                if k in DEFAULT_PARAMS and k not in yaml_keys:
                    DEFAULT_PARAMS[k] = v
                    n += 1
            SH.view_scale = float(d.get('view_scale', SCALE_DEFAULT))
            print('[cone] 面板本地???入 %d ? (不?覆? param.yaml)' % n)
        except Exception as e:
            print('[cone] 加?面板??失?: %s' % e)

    SH.params = dict(DEFAULT_PARAMS)

    # ---- 3) ??算法??: 用同一份 param.yaml ?例化??的 ConeTracker ----
    if int(DEFAULT_PARAMS.get('node_vis', 1)) and raw_ros_p is not None:
        mod, mpath = _load_node_module()
        if mod is not None:
            node_params = {k: v for k, v in raw_ros_p.items()
                           if k in getattr(mod, 'DEFAULTS', {})}
            NODE_MODULE = mod
            NODE_TRACKER = mod.ConeTracker(node_params)
            print('[cone] ??算法??已?? (同一份代?: %s), 面板?展示?????出' % mpath)
        else:
            print('[cone] 未找到 scripts/lidar_tracking.py, 面板不展示??算法?出')


def main():
    ap = argparse.ArgumentParser(description='雷??桶?? / 中?提取 / 可?化??')
    ap.add_argument('--topic', default='/scan', help='?描??（默? /scan）')
    ap.add_argument('--port', type=int, default=8080, help='Web 端口（默? 8080）')
    ap.add_argument('--replay', default=None, help='离?回放文件（循?播放）')
    ap.add_argument('--no-web', action='store_true', help='不? Web，只打印??')
    args = ap.parse_args()

    if not HAVE_CV2:
        print('需要 OpenCV（cv2）'); sys.exit(1)
    if not args.no_web and not HAVE_ROS and not args.replay:
        print('需要 rclpy 或 --replay'); sys.exit(1)

    load_saved_params()

    # ---- Web 服?器 ----
    if not args.no_web:
        try:
            srv = QuietServer(('0.0.0.0', args.port), Handler)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            print('=' * 66)
            print('可?化面板:  http://<?IP>:%d' % args.port)
            print('  例如:     http://192.168.5.4:%d' % args.port)
            print('=' * 66)
        except OSError as e:
            print('Web 端口 %d ??失?: %s' % (args.port, e))
            print('?端口: python3 cone.py --port 8081')

    if args.no_web and not HAVE_ROS:
        print('--no-web 需要 rclpy'); sys.exit(1)

    # ---- ROS ----
    if not HAVE_ROS:
        print('未找到 rclpy。?先 source ROS ?境：')
        print('  source /opt/ros/humble/setup.bash')
        print('  source ~/racecar/install/setup.bash')
        # ? ROS ?至少? Web 面板可??（用于看默???）
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
                        e_lat = SH.e_lat
                        e_head = SH.e_head_deg
                        servo = SH.servo
                        valid = SH.valid
                    el = ('%+.2f' % e_lat) if valid else ' -- '
                    eh = ('%+.1f' % e_head) if valid else ' -- '
                    sv = ('%.1f' % servo) if valid else ' -- '
                    print('[cone] total=%2d clusters=%3d  e_lat=%sm e_head=%sdeg '
                          'servo=%sdeg  %s %s'
                          % (len(cones), cl, el, eh, sv,
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
