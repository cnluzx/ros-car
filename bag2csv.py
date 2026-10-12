#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bag2csv.py —— 把 ros2 bag 里的激光雷达数据转成 CSV / JSON / HTML

用途：用 record_lidar.sh 录完 bag 之后，想直接拿数据做分析或画图时用。

用法：
  source /opt/ros/humble/setup.bash
  source install/setup.bash

  ./bag2csv.py data/bags/lidar_20261005_143000
  ./bag2csv.py data/bags/lidar_20261005_143000 --html
  ./bag2csv.py data/bags/lidar_20261005_143000 -t /scan -o ./out

输出与 lidar_capture.py 完全一致，方便两套流程混用。
"""

import argparse
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(
        description='把 ros2 bag 中的 /scan 转成 CSV / JSON / HTML',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag', help='bag 目录路径（含 metadata.yaml 的那个）')
    ap.add_argument('-t', '--topic', default='/scan', help='话题名（默认 /scan）')
    ap.add_argument('-o', '--outdir', default=None, help='输出目录（默认与 bag 同级）')
    ap.add_argument('--html', action='store_true', help='额外生成 HTML 点云回放')
    ap.add_argument('--html-max-frames', type=int, default=120)
    ap.add_argument('--max-frames', type=int, default=0, help='最多转换多少帧，0 为不限')
    ap.add_argument('--storage-id', default='sqlite3', help='bag 存储后端（默认 sqlite3）')
    args = ap.parse_args()

    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import LaserScan
    except ImportError:
        print('[错误] 找不到 ROS2 Python 库。请先执行：')
        print('         source /opt/ros/humble/setup.bash')
        print('         source install/setup.bash      # 若 bag 在工作空间里')
        return 2

    try:
        import lidar_capture as lc
    except ImportError:
        print('[错误] 同目录下没找到 lidar_capture.py，请把两个脚本放在一起。')
        return 2

    if not os.path.isdir(args.bag):
        print('[错误] 不是目录：%s' % args.bag)
        return 2
    if not (os.path.exists(os.path.join(args.bag, 'metadata.yaml')) or
            os.path.exists(os.path.join(args.bag, 'metadata.yaml.bak'))):
        print('[警告] 该目录里没有 metadata.yaml，可能不是有效的 bag 目录。继续尝试 ...')

    # ---------- 打开 bag ----------
    try:
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=args.bag, storage_id=args.storage_id),
            rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr'))
    except Exception as e:
        print('[错误] 打开 bag 失败：%s' % e)
        print('       尝试指定存储后端： --storage-id mcap')
        return 1

    topics = [t.name for t in reader.get_all_topics_and_types()]
    print('bag 中的话题：%s' % ', '.join(topics))
    if args.topic not in topics:
        print('[错误] 话题 %s 不在 bag 里。' % args.topic)
        return 1

    # ---------- 逐帧读取 ----------
    rec = lc.Recorder()
    n_seen = 0
    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic != args.topic:
            continue
        msg = deserialize_message(data, LaserScan)
        rec.add(ts * 1e-9, msg.ranges, {
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment,
            'time_increment': msg.time_increment,
            'scan_time': msg.scan_time,
            'range_min': msg.range_min,
            'range_max': msg.range_max,
            'frame_id': msg.header.frame_id,
        })
        n_seen += 1
        if args.max_frames and n_seen >= args.max_frames:
            break
        if n_seen % 100 == 0:
            sys.stdout.write('\r已读取 %d 帧 ...' % n_seen)
            sys.stdout.flush()

    sys.stdout.write('\r已读取 %d 帧\n' % n_seen)
    if rec.count == 0:
        print('[错误] 没有读到任何 %s 消息。' % args.topic)
        return 1

    # ---------- 导出 ----------
    outdir = args.outdir or os.path.join(os.path.dirname(args.bag.rstrip('/')) or '.', 'csv')
    os.makedirs(outdir, exist_ok=True)

    base_name = os.path.basename(args.bag.rstrip('/'))
    base = os.path.join(outdir, base_name)

    files = []
    rec.save_csv(base + '.csv')
    files.append(('CSV', base + '.csv'))
    rec.save_json(base + '.json', args.topic, {
        'source_bag': args.bag,
        'note': 'CSV 为宽表：首列相对时间，其后每列一束，inf/nan/0 记为空白',
        'angle_formula': 'theta_i = angle_min + i * angle_increment',
    })
    files.append(('JSON', base + '.json'))
    if args.html:
        rec.save_html(base + '.html', args.topic, max_frames=args.html_max_frames)
        files.append(('HTML', base + '.html'))

    lc.print_report(rec, files, args.topic)
    return 0


if __name__ == '__main__':
    sys.exit(main())
