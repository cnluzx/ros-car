#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行记录分析 (track_report.py) —— 配合 lidar_tracking 节点自动录制的 CSV

用法:
    python3 track_report.py                        # 自动分析 logs 里最新一份
    python3 track_report.py ~/racecar/logs/track_20261006_210000.csv
    python3 track_report.py ~/racecar/logs/track_*.csv   # 多份(逐份输出)

输出内容:
    · 本次生效的参数快照 (从 CSV 头部读取)
    · 模式分布 / 实际速度分布 (直道弯道各用什么速度)
    · 安全层的干预比例 / 曲率顶上限比例 / DANGER 次数 / clear 统计
    · arcfit 拟合健康度 (内点比例)
    · 转向命令最大跳变 Top5 (定位入弯"猛打舵"发生在第几秒)
    · 针对性的调参提示
"""

import glob
import math
import os
import sys
import csv
from collections import Counter

TRK_PREFIX = ('ARCFIT', 'GATE', 'SINGLE')       # 循迹类模式前缀


def load(path):
    params, rows = {}, []
    with open(path, encoding='utf-8', errors='replace') as f:
        rd = csv.reader(f)
        for r in rd:
            if not r:
                continue
            if r[0] == '# param':
                if len(r) >= 3:
                    params[r[1]] = r[2]
                continue
            if r[0].startswith('#'):
                continue
            if r[0] == 't':
                continue
            try:
                rows.append({
                    't': float(r[0]), 'frame': int(r[1]), 'mode': r[2],
                    'n_cones': int(r[3]), 'n_gates': int(r[4]),
                    'k_des': float(r[5]), 'k_cmd': float(r[6]),
                    'clear': float(r[7]), 'servo': float(r[8]),
                    'speed': float(r[9]), 'x0': float(r[10]),
                    'psi': float(r[11]), 'kappa': float(r[12]),
                    'inl': int(r[13]), 'err': float(r[14]),
                })
            except (ValueError, IndexError):
                continue
    return params, rows


def fget(params, key, default):
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return default


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def report(path):
    params, rows = load(path)
    print('=' * 76)
    print('  运行记录分析: %s' % os.path.basename(path))
    print('=' * 76)
    if not rows:
        print('  !! 没有数据行')
        return
    dur = rows[-1]['t'] - rows[0]['t']
    print('  时长 %.1fs, %d 帧 (%.1f Hz)' % (dur, len(rows), len(rows) / max(dur, 1e-9)))

    keys = ['speed', 'turn_slow_drop', 'turn_slow_threshold', 'danger_slow_drop',
            'servo_center', 'controller', 'arc_kx', 'arc_kpsi', 'arc_win_x_max',
            'arc_tau', 'safe_enable', 'safe_radius', 'max_curvature']
    print('  参数: ' + ' '.join('%s=%s' % (k, params.get(k, '?')) for k in keys))

    # ---------- 模式 ----------
    modes = Counter(r['mode'] for r in rows)
    print('\n  [模式分布]')
    for m, c in modes.most_common():
        print('    %-16s %5d 帧 (%.0f%%)' % (m, c, pct(c, len(rows))))

    trk = [r for r in rows if r['mode'].startswith(TRK_PREFIX)]
    if not trk:
        print('\n  !! 没有任何循迹帧(全程丢失/搜索), 先检查检测与启动位置')
        return

    # ---------- 速度 ----------
    sp = Counter(int(r['speed']) for r in trk)
    print('\n  [实际速度分布] (仅循迹帧; 括号内为约m/s, 按1545≈0.8估算)')
    for v, c in sorted(sp.items()):
        print('    PWM %d (%.2f m/s)  %5d 帧  %.0f%%'
              % (v, (v - 1500) * 0.8 / 45.0, c, pct(c, len(trk))))

    # ---------- 安全层 / 转向裕量 ----------
    kcap = fget(params, 'max_curvature', 0.65)
    center = fget(params, 'servo_center', 82.5)
    interv = [r for r in trk if abs(r['k_des'] - r['k_cmd']) > 0.02]
    sat = [r for r in trk if abs(r['k_cmd']) >= 0.98 * kcap]
    danger = sum(c for m, c in modes.items() if '+DANGER' in m)
    clears = [r['clear'] for r in trk if r['clear'] < 90]
    serv_dev = [abs(r['servo'] - center) for r in trk]
    print('\n  [安全层 / 转向裕量]')
    print('    安全层修正过转向: %d 帧 (%.1f%%)   [k_cmd 明显小于 k_des]'
          % (len(interv), pct(len(interv), len(trk))))
    print('    DANGER 触发:      %d 帧 (%.1f%%)' % (danger, pct(danger, len(trk))))
    print('    k_cmd 顶上限(±%.2f): %d 帧 (%.1f%%)' % (kcap, len(sat), pct(len(sat), len(trk))))
    if clears:
        print('    clear 均值 %.2fm, 最小 %.2fm' % (sum(clears) / len(clears), min(clears)))
    print('    舵机偏离中位: 均值 %.1f°, 最大 %.1f°' % (
        sum(serv_dev) / len(serv_dev), max(serv_dev)))

    # ---------- arcfit 健康 ----------
    inl_pts = [r['inl'] for r in trk if r['inl'] >= 0]
    if inl_pts:
        ok = sum(1 for v in inl_pts if v >= 3)
        print('\n  [arcfit 拟合健康]')
        print('    内点≥3 的比例: %.0f%% (低→查 arc_track_width/arc_tau/检测参数)'
              % pct(ok, len(inl_pts)))

    # ---------- 转向跳变 Top5 ----------
    jumps = []
    for a, b in zip(trk, trk[1:]):
        dk = abs(b['k_des'] - a['k_des'])
        jumps.append((dk, a['t'], a['k_des'], b['k_des'], a['mode'], b['mode']))
    jumps.sort(key=lambda x: -x[0])
    print('\n  [转向命令最大跳变 Top5] (1/m; 用来定位"入弯猛打舵"发生在第几秒)')
    for dk, t, k1, k2, m1, m2 in jumps[:5]:
        print('    t=%7.1fs  k_des %+.2f → %+.2f  (Δ%.2f)  %s→%s' % (t, k1, k2, dk, m1, m2))

    # ---------- 调参提示 ----------
    print('\n  [调参提示]')
    hints = []
    if pct(len(interv), len(trk)) > 15:
        hints.append('安全层频繁修正转向(>15%): 调小 safe_radius(0.28→0.25), 或降低 arc 增益')
    if pct(len(sat), len(trk)) > 5:
        hints.append('k_cmd 经常顶上限(>5%): 弯道需要真正减速(turn_slow_drop↑ 或 threshold↓), '
                     '否则只能靠提高 max_curvature(小心舵机行程)')
    if pct(danger, len(trk)) > 10:
        hints.append('DANGER 频繁(>10%): 直道上也出现就调小 safe_radius; 只在弯道出现就先看入弯速度')
    if jumps and jumps[0][0] > 0.25:
        hints.append('单帧 k_des 跳变 >0.25 = 入弯猛打舵: 降 arc_kpsi/arc_kx, '
                     '或加大 servo_smooth(0.85→0.80)')
    lost = sum(c for m, c in modes.items() if m in ('HOLD', 'SEARCH'))
    if pct(lost, len(rows)) > 10:
        hints.append('HOLD/SEARCH 占比 >10%: 检测不稳定或 arcfit 频繁无效, 先查检测参数/内点比例')
    if not hints:
        hints.append('各项平稳。想更快: speed 每次 +5~10 并重测; 想更稳: 保持现参数多跑几圈')
    for h in hints:
        print('    · ' + h)
    print()


def main():
    args = sys.argv[1:]
    if not args:
        files = glob.glob(os.path.expanduser('~/racecar/logs/track_*.csv'))
        if not files:
            print('logs 目录里没有 track_*.csv; 直接指定文件: track_report.py <csv>')
            return 1
        files = [max(files, key=os.path.getmtime)]
        print('(自动选择最新一份: %s)\n' % os.path.basename(files[0]))
    else:
        files = []
        for a in args:
            files.extend(sorted(glob.glob(os.path.expanduser(a))) or [a])
    for f in files:
        if os.path.exists(f):
            report(f)
        else:
            print('!! 文件不存在: %s' % f)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
