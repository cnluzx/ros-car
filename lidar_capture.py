#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lidar_capture.py —— 采集激光雷达 /scan 数据流

干什么：
  1. 订阅 /scan，按"时长"或"帧数"采集（哪个先到就停）
  2. 输出 CSV（宽表：每行一帧，列是各束距离）+ JSON（元数据与统计）
  3. 可选生成自包含 HTML，在浏览器里俯视回放点云（不需要 RViz）

用法：
  python3 lidar_capture.py                        # 默认采 20 秒
  python3 lidar_capture.py -d 60                  # 采 60 秒
  python3 lidar_capture.py -n 300                 # 采 300 帧
  python3 lidar_capture.py -o ~/lidar_data        # 指定输出目录
  python3 lidar_capture.py --html -n 120          # 顺便生成 HTML 回放
  python3 lidar_capture.py -t /scan --reliable    # 指定话题 / 用 reliable QoS

采集前先确认雷达在发数据：
  ros2 topic hz /scan
  ros2 topic echo /scan --once

ROS2 环境准备好后运行：
  source /opt/ros/humble/setup.bash
  source ~/racecar/install/setup.bash      # 按你的实际路径
  python3 lidar_capture.py -d 30 --html
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime


# ============================================================================
# 一、数据处理层（不依赖 ROS2，可单独测试）
# ============================================================================

class Recorder:
    """收集扫描帧、维护统计、导出文件。"""

    def __init__(self):
        self.meta = None          # 首帧的元数据（角度/量程等）
        self.frames = []          # [{'t': 相对秒, 'ranges': [...]}]
        self.t0 = None            # 首帧到达时刻（墙钟）
        self.beam_counts = {}     # 每帧束数 → 出现次数，用于发现束数异常

    # ---------------------------------------------------------------- 收集
    def add(self, wall_time, ranges, meta):
        if self.t0 is None:
            self.t0 = wall_time
            self.meta = dict(meta)
        self.frames.append({
            't': round(wall_time - self.t0, 4),
            'ranges': list(ranges),
        })
        n = len(ranges)
        self.beam_counts[n] = self.beam_counts.get(n, 0) + 1

    @property
    def count(self):
        return len(self.frames)

    @property
    def duration(self):
        return self.frames[-1]['t'] if self.frames else 0.0

    # ---------------------------------------------------------------- 统计
    def stats(self):
        if not self.frames:
            return {}

        n_frames = len(self.frames)
        dur = self.duration
        hz = (n_frames - 1) / dur if dur > 1e-6 and n_frames > 1 else 0.0

        inf_cnt = 0
        nan_cnt = 0
        zero_cnt = 0
        valid = []                 # 有效距离样本（抽样上限，避免占内存）
        per_frame_valid = []

        for fr in self.frames:
            v = 0
            for r in fr['ranges']:
                if math.isnan(r):
                    nan_cnt += 1
                    continue
                if math.isinf(r):
                    inf_cnt += 1
                    continue
                if r <= 0.0:
                    zero_cnt += 1
                    continue
                v += 1
                if len(valid) < 400000:
                    valid.append(r)
            per_frame_valid.append(v)

        total_slots = sum(len(f['ranges']) for f in self.frames)

        def pct(xs, p):
            if not xs:
                return 0.0
            s = sorted(xs)
            return s[min(len(s) - 1, int(len(s) * p))]

        beams = max(self.beam_counts, key=self.beam_counts.get) if self.beam_counts else 0

        return {
            'frames': n_frames,
            'duration_s': round(dur, 3),
            'measured_hz': round(hz, 3),
            'beams_per_frame': beams,
            'beam_count_variants': len(self.beam_counts),
            'total_slots': total_slots,
            'valid_points': len(valid),
            'valid_ratio': round(len(valid) / total_slots, 4) if total_slots else 0.0,
            'nan_count': nan_cnt,
            'inf_count': inf_cnt,
            'zero_count': zero_cnt,
            'range_min_seen': round(min(valid), 3) if valid else 0.0,
            'range_p10': round(pct(valid, 0.10), 3) if valid else 0.0,
            'range_p50': round(pct(valid, 0.50), 3) if valid else 0.0,
            'range_p90': round(pct(valid, 0.90), 3) if valid else 0.0,
            'range_max_seen': round(max(valid), 3) if valid else 0.0,
            'valid_per_frame_min': min(per_frame_valid) if per_frame_valid else 0,
            'valid_per_frame_max': max(per_frame_valid) if per_frame_valid else 0,
        }

    # ---------------------------------------------------------------- 导出
    def save_csv(self, path):
        """宽表：第一列相对时间，其后每列一束。"""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            head = ['t'] + ['r%d' % i for i in range(len(self.frames[0]['ranges']))]
            w.writerow(head)
            for fr in self.frames:
                row = [fr['t']]
                for r in fr['ranges']:
                    if math.isnan(r) or math.isinf(r):
                        row.append('')
                    else:
                        row.append('%.3f' % r)
                w.writerow(row)

    def save_json(self, path, topic, extra=None):
        doc = {
            'topic': topic,
            'captured_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'meta': self.meta,
            'stats': self.stats(),
            'timestamps': [f['t'] for f in self.frames],
        }
        if extra:
            doc.update(extra)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

    def save_html(self, path, topic, max_frames=120, range_clip=None):
        """生成自包含 HTML：浏览器里俯视回放点云。"""
        fr = self.frames[:max_frames]
        beams = len(fr[0]['ranges']) if fr else 0

        # 压缩：只保留有效值，inf/nan → 0（渲染时跳过）
        packed = []
        for f in fr:
            row = []
            for r in f['ranges']:
                if math.isnan(r) or math.isinf(r) or r <= 0.0:
                    row.append(0)
                elif range_clip and r > range_clip:
                    row.append(0)
                else:
                    row.append(round(r, 2))
            packed.append(row)

        payload = {
            'topic': topic,
            'angleMin': self.meta.get('angle_min', 0.0),
            'angleInc': self.meta.get('angle_increment', 0.0),
            'rangeMin': self.meta.get('range_min', 0.0),
            'rangeMax': self.meta.get('range_max', 0.0),
            'frameId': self.meta.get('frame_id', ''),
            'beams': beams,
            'times': [f['t'] for f in fr],
            'totalFrames': self.count,
            'frames': packed,
        }
        html = HTML_TEMPLATE.replace('__PAYLOAD__', json.dumps(payload, separators=(',', ':')))
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        return len(html)


# ============================================================================
# 二、ROS2 订阅层
# ============================================================================

def run_capture(args):
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                               HistoryPolicy, DurabilityPolicy)
        from sensor_msgs.msg import LaserScan
    except ImportError:
        print('找不到 rclpy。请先 source ROS2 环境：')
        print('  source /opt/ros/humble/setup.bash')
        return 2

    rec = Recorder()
    stop = {'flag': False}

    class Capture(Node):
        def __init__(self):
            super().__init__('lidar_capture')
            if args.reliable:
                qos = QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    history=HistoryPolicy.KEEP_LAST,
                    durability=DurabilityPolicy.VOLATILE,
                    depth=10)
            else:
                # 雷达驱动多数用 best_effort；best_effort 订阅也能接住 reliable 发布
                qos = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST,
                    durability=DurabilityPolicy.VOLATILE,
                    depth=10)
            self.create_subscription(LaserScan, args.topic, self.cb, qos)

        def cb(self, msg):
            if stop['flag']:
                return
            now = self.get_clock().now().nanoseconds * 1e-9
            rec.add(now, msg.ranges, {
                'angle_min': msg.angle_min,
                'angle_max': msg.angle_max,
                'angle_increment': msg.angle_increment,
                'time_increment': msg.time_increment,
                'scan_time': msg.scan_time,
                'range_min': msg.range_min,
                'range_max': msg.range_max,
                'frame_id': msg.header.frame_id,
            })

            # 实时状态行
            if not args.quiet:
                valid = sum(1 for r in msg.ranges
                            if not (math.isnan(r) or math.isinf(r)) and r > 0)
                near = min((r for r in msg.ranges
                            if not (math.isnan(r) or math.isinf(r)) and r > 0),
                           default=0.0)
                sys.stdout.write(
                    '\r[%6.1fs] %4d 帧 | %.1f Hz | 有效 %4d/%d | 最近 %.2fm   ' %
                    (rec.duration, rec.count,
                     (rec.count - 1) / rec.duration if rec.duration > 1e-6 else 0.0,
                     valid, len(msg.ranges), near))
                sys.stdout.flush()

            # 停止条件
            if args.frames and rec.count >= args.frames:
                stop['flag'] = True
            elif rec.duration >= args.duration:
                stop['flag'] = True

    rclpy.init()
    node = Capture()
    print('开始采集 %s ...（Ctrl+C 可提前结束）' % args.topic)
    try:
        while rclpy.ok() and not stop['flag']:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print('\n收到 Ctrl+C，提前结束。')
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not args.quiet:
        sys.stdout.write('\n')
    return rec


# ============================================================================
# 三、报表
# ============================================================================

def print_report(rec, files, topic):
    s = rec.stats()
    if not s:
        print('没有采集到任何帧。请检查：')
        print('  1) 雷达是否在发布    ros2 topic hz %s' % topic)
        print('  2) 话题名是否正确    ros2 topic list')
        print('  3) QoS 是否匹配      加 --reliable 再试一次')
        return

    print()
    print('=' * 62)
    print('采集完成')
    print('=' * 62)
    print('  帧数            %d' % s['frames'])
    print('  时长            %.2f s' % s['duration_s'])
    print('  实测频率        %.2f Hz' % s['measured_hz'])
    print('  每帧束数        %d' % s['beams_per_frame'])
    if s['beam_count_variants'] > 1:
        print('  !! 束数有 %d 种取值，说明数据不稳定' % s['beam_count_variants'])
    print('-' * 62)
    print('  有效点          %d / %d  (%.1f%%)' %
          (s['valid_points'], s['total_slots'], s['valid_ratio'] * 100))
    print('  nan / inf / 0   %d / %d / %d' %
          (s['nan_count'], s['inf_count'], s['zero_count']))
    print('  每帧有效点范围  %d ~ %d' %
          (s['valid_per_frame_min'], s['valid_per_frame_max']))
    print('-' * 62)
    print('  距离分布  min %.2f  p10 %.2f  p50 %.2f  p90 %.2f  max %.2f  (m)' %
          (s['range_min_seen'], s['range_p10'], s['range_p50'],
           s['range_p90'], s['range_max_seen']))
    print('-' * 62)

    # 数据健康度提示
    if s['inf_count'] > s['total_slots'] * 0.5:
        print('  提示：inf 占比超过一半，多半是朝天空/开阔方向没有回波，正常。')
    if s['beam_count_variants'] > 1:
        print('  提示：束数不固定会让固定下标的算法失效，务必用 ranges.size()。')
    if s['measured_hz'] < 5.0 and s['frames'] > 5:
        print('  提示：实测频率偏低，检查串口带宽或是否与其他节点抢 CPU。')
    if s['range_min_seen'] < 0.10:
        print('  提示：出现很近距离的回波，可能是车体自身或线缆被扫到了。')
    print()
    print('已保存：')
    for label, p in files:
        size = os.path.getsize(p) if os.path.exists(p) else 0
        print('  %-6s %s  (%.1f KB)' % (label, p, size / 1024.0))
    print()


# ============================================================================
# 四、HTML 回放模板
# ============================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>雷达点云回放</title>
<style>
  body{margin:0;padding:18px;background:#f8fafc;color:#0f172a;
       font-family:system-ui,-apple-system,"PingFang SC",sans-serif;font-size:13px;}
  h1{font-size:15px;font-weight:500;margin:0 0 3px;}
  .sub{color:#64748b;font-size:12px;margin:0 0 14px;}
  .wrap{display:flex;gap:16px;align-items:flex-start;}
  canvas{background:#fff;border:1px solid #e2e8f0;border-radius:12px;display:block;}
  .card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px;margin-bottom:12px;}
  .tit{font-size:12px;font-weight:500;margin:0 0 9px;}
  .stat{display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px;}
  .stat span{color:#475569;} .stat b{font-weight:500;}
  input[type=range]{width:100%;accent-color:#2563eb;}
  button{font:inherit;font-size:12px;padding:5px 12px;border-radius:8px;
         border:1px solid #cbd5e1;background:#fff;cursor:pointer;}
  button:hover{background:#f1f5f9;}
  .hint{font-size:11.5px;color:#94a3b8;line-height:1.6;}
</style>
</head>
<body>
<h1>雷达点云回放</h1>
<p class="sub" id="sub"></p>
<div class="wrap">
  <canvas id="cv" width="620" height="620"></canvas>
  <div style="min-width:280px;">
    <div class="card">
      <p class="tit">当前帧</p>
      <div class="stat"><span>帧序号</span><b id="s-idx">0</b></div>
      <div class="stat"><span>相对时间</span><b id="s-time">0.00 s</b></div>
      <div class="stat"><span>有效点</span><b id="s-valid">0</b></div>
      <div class="stat"><span>最近距离</span><b id="s-near">- m</b></div>
    </div>
    <div class="card">
      <p class="tit">播放</p>
      <input type="range" id="slider" min="0" max="0" value="0">
      <div style="display:flex;gap:8px;margin-top:10px;">
        <button id="btn-play">播放</button>
        <button id="btn-start">回到首帧</button>
      </div>
    </div>
    <div class="card">
      <p class="tit">显示</p>
      <label style="display:flex;gap:7px;align-items:center;font-size:12px;color:#475569;margin-bottom:7px;">
        <input type="checkbox" id="ck-ray"> 雷达射线
      </label>
      <label style="display:flex;gap:7px;align-items:center;font-size:12px;color:#475569;margin-bottom:10px;">
        <input type="checkbox" id="ck-trail" checked> 残留上一帧
      </label>
      <div class="stat"><span>缩放 (m)</span><b id="s-zoom">5.0</b></div>
      <input type="range" id="zoom" min="1.5" max="12" step="0.5" value="5">
    </div>
    <div class="card">
      <p class="hint" id="hint"></p>
    </div>
  </div>
</div>
<script>
const DATA = __PAYLOAD__;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const CX = cv.width/2, CY = cv.height/2;
let zoom = 5.0, idx = 0, playing = false, timer = null;

document.getElementById('sub').textContent =
  '话题 ' + DATA.topic + '  ·  ' + DATA.beams + ' 束/帧  ·  共 ' + DATA.totalFrames +
  ' 帧（本页载入前 ' + DATA.frames.length + ' 帧）  ·  frame_id=' + (DATA.frameId || '-');

function scale(){ return (cv.width/2 - 24) / zoom; }
function W2C(x,y){ const s=scale(); return [CX + x*s, CY - y*s]; }

function draw(){
  const s = scale();
  ctx.fillStyle = '#fff'; ctx.fillRect(0,0,cv.width,cv.height);

  // 距离环
  ctx.strokeStyle = '#f1f5f9'; ctx.lineWidth = 1;
  for(let r=1; r<=zoom; r++){
    ctx.beginPath(); ctx.arc(CX, CY, r*s, 0, Math.PI*2); ctx.stroke();
    if(r % 1 === 0){
      ctx.fillStyle = '#cbd5e1'; ctx.font = '10px sans-serif';
      ctx.fillText(r+'m', CX+4, CY - r*s + 11);
    }
  }
  // 坐标轴
  ctx.strokeStyle = '#e2e8f0';
  ctx.beginPath(); ctx.moveTo(CX,10); ctx.lineTo(CX,cv.height-10); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(10,CY); ctx.lineTo(cv.width-10,CY); ctx.stroke();

  const showTrail = document.getElementById('ck-trail').checked;
  const showRay   = document.getElementById('ck-ray').checked;

  if(showTrail && idx > 0){
    ctx.fillStyle = 'rgba(148,163,184,0.30)';
    const prev = DATA.frames[idx-1];
    for(let i=0;i<prev.length;i++){
      const r = prev[i]; if(r<=0) continue;
      const th = DATA.angleMin + i*DATA.angleInc;
      const x = r*Math.sin(th), y = r*Math.cos(th);
      const [px,py] = W2C(x,y);
      ctx.beginPath(); ctx.arc(px,py,1.6,0,Math.PI*2); ctx.fill();
    }
  }

  const row = DATA.frames[idx] || [];
  if(showRay) ctx.strokeStyle = 'rgba(191,219,254,0.55)';
  ctx.fillStyle = '#2563eb';
  let valid = 0, near = Infinity;
  for(let i=0;i<row.length;i++){
    const r = row[i]; if(r<=0) continue;
    valid++;
    if(r < near) near = r;
    const th = DATA.angleMin + i*DATA.angleInc;
    const x = r*Math.sin(th), y = r*Math.cos(th);
    const [px,py] = W2C(x,y);
    if(showRay){ ctx.beginPath(); ctx.moveTo(CX,CY); ctx.lineTo(px,py); ctx.stroke(); }
    ctx.beginPath(); ctx.arc(px,py,2.4,0,Math.PI*2); ctx.fill();
  }

  // 车体
  ctx.fillStyle = '#334155';
  ctx.beginPath();
  ctx.moveTo(CX, CY-16); ctx.lineTo(CX+11, CY+13);
  ctx.lineTo(CX, CY+8); ctx.lineTo(CX-11, CY+13);
  ctx.closePath(); ctx.fill();

  document.getElementById('s-idx').textContent = idx + ' / ' + (DATA.frames.length-1);
  document.getElementById('s-time').textContent = (DATA.times[idx]||0).toFixed(2) + ' s';
  document.getElementById('s-valid').textContent = valid;
  document.getElementById('s-near').textContent = isFinite(near) ? near.toFixed(2)+' m' : '- m';
}

document.getElementById('slider').max = Math.max(0, DATA.frames.length-1);
document.getElementById('slider').addEventListener('input', e => {
  idx = +e.target.value; draw();
});

document.getElementById('btn-play').addEventListener('click', e => {
  playing = !playing;
  e.target.textContent = playing ? '暂停' : '播放';
  if(playing){
    timer = setInterval(() => {
      idx = (idx + 1) % DATA.frames.length;
      document.getElementById('slider').value = idx;
      draw();
    }, 100);
  } else { clearInterval(timer); }
});

document.getElementById('btn-start').addEventListener('click', () => {
  idx = 0; document.getElementById('slider').value = 0; draw();
});

document.getElementById('zoom').addEventListener('input', e => {
  zoom = +e.target.value;
  document.getElementById('s-zoom').textContent = zoom.toFixed(1);
  draw();
});

document.getElementById('hint').innerHTML =
  '点云是原始回波，没有做任何过滤。近距离密集的点通常是锥桶或障碍物；' +
  '沿同一半径铺开的一串点往往是墙壁或地面边缘；' +
  '空白方向表示该角度没有回波（超出量程或吸收太强）。';

document.getElementById('ck-trail').addEventListener('change', draw);
document.getElementById('ck-ray').addEventListener('change', draw);

draw();
</script>
</body>
</html>
"""


# ============================================================================
# 五、入口
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='采集激光雷达 /scan 数据流并导出 CSV / JSON / HTML',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-t', '--topic', default='/scan', help='扫描话题名（默认 /scan）')
    ap.add_argument('-d', '--duration', type=float, default=20.0,
                    help='采集时长秒数（默认 20）')
    ap.add_argument('-n', '--frames', type=int, default=0,
                    help='采集帧数上限，0 表示不限制（默认 0）')
    ap.add_argument('-o', '--outdir', default='lidar_data',
                    help='输出目录（默认 ./lidar_data）')
    ap.add_argument('--html', action='store_true', help='额外生成 HTML 点云回放')
    ap.add_argument('--html-max-frames', type=int, default=120,
                    help='HTML 中最多包含多少帧（默认 120，太多会卡）')
    ap.add_argument('--reliable', action='store_true',
                    help='用 reliable QoS 订阅（默认 best_effort）')
    ap.add_argument('-q', '--quiet', action='store_true', help='不打印实时状态行')
    args = ap.parse_args()

    rec = run_capture(args)
    if isinstance(rec, int):        # 出错返回码
        return rec

    if rec.count == 0:
        print('没有采集到任何帧。请检查：')
        print('  1) 雷达是否在发布    ros2 topic hz %s' % args.topic)
        print('  2) 话题名是否正确    ros2 topic list')
        print('  3) QoS 是否匹配      加 --reliable 再试一次')
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.join(args.outdir, 'scan_' + stamp)

    files = []
    csv_path = base + '.csv'
    rec.save_csv(csv_path)
    files.append(('CSV', csv_path))

    json_path = base + '.json'
    rec.save_json(json_path, args.topic, {
        'note': 'CSV 为宽表：首列相对时间，其后每列一束，inf/nan/0 记为空白',
        'angle_formula': 'theta_i = angle_min + i * angle_increment',
    })
    files.append(('JSON', json_path))

    if args.html:
        html_path = base + '.html'
        rec.save_html(html_path, args.topic, max_frames=args.html_max_frames)
        files.append(('HTML', html_path))

    print_report(rec, files, args.topic)
    return 0


if __name__ == '__main__':
    sys.exit(main())
