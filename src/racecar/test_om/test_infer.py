#!/usr/bin/env python3
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ais_bench.infer.interface import InferSession


CFG = {
    # ---------- 模型与 ROS 话题 ----------
    "model_path": "/home/davinci-mini/racecar/src/racecar/test_om/line_seg_v2_310B4_int8.om",
    "image_topic": "/image_raw",
    "cmd_topic": "/car_cmd_vel",


    # ---------- 黄线掩码扫描与边界提取 ----------
    # mask 过少时的后备扫描位置。正常情况下使用 adaptive_scan_rows 自动选行。
    "scan_ratios": (0.82, 0.72, 0.62, 0.52, 0.42),
    # 根据本帧 mask 中有白线候选段的连续区域，自动选择五条扫描线。
    "adaptive_scan_rows": True,
    # 自适应扫描只在图像该比例以下搜索，排除天花板、锥桶等远处误分割。
    "scan_search_top_ratio": 0.35,
    # 每条扫描线向上、向下扩展的像素数。5 表示实际统计 11 行图像，用列投票判断黄线。增大：更不容易漏掉细线或轻微抖动的黄线；过大：容易把相邻区域、噪点或交叉线混入同一条边界。
    "scan_half_height": 5,
    # 一段黄色像素至少连续多少列，才认为它是一条黄线候选段。增大可过滤小噪点；过大可能漏掉远处很细的黄线。
    "min_segment_px": 3,
    # 左右黄线之间允许的最小像素间距。小于此值的两段黄色不会被当成赛道边界。增大可减少把噪声误配成赛道；过大可能在远处丢失赛道。
    "min_road_width": 35,
    # 左右黄线之间允许的最大像素间距。防止把画面两侧无关的黄色区域拼成一条赛道。
    "max_road_width": 290,
    # 单边黄线补全时，使用上一扫描层赛道宽度的比例；小于 1 可避免把边界补得过宽。
    "width_decay": 0.84,
    # 所有双边观测共同维护的赛道宽度。自适应扫描行变化时，它可为单边观测提供稳定宽度。
    "lane_width_smooth": 0.20,
    # 扫描线每向上移动 10 像素时，赛道宽度保留的比例。用于单边黄线的透视宽度推算。
    "perspective_width_decay": 0.94,
    # 双边宽度与当前透视预测相差过大时，认为是斜线/噪声误配，拒绝更新赛道宽度。
    "max_width_ratio_error": 0.32,
    # 相邻扫描层或相邻帧中，赛道中心允许变化的最大像素量。
    # 减小：更不容易跳到错误黄线；
    # 增大：急弯、车身晃动时更不容易 LOST。
    # 60 是安全起点，实车急弯丢线可逐步调到 65、70。
    "max_track_jump": 60,
    # 至少需要几条扫描线识别有效，才能允许车辆输出非零速度。
    "min_valid_rows": 2,

    # ---------- 前视目标与弯道判断 ----------
    # 直道时目标点在“近处中心”和“远处中心”之间的比例。0 表示完全看近处，1 表示完全看远处。
    # 增大：直道更平滑，但面对近处偏差的反应更慢；减小：反应更快，但更容易左右摆。
    "lookahead_straight": 0.55,
    # 弯道时的前视比例。弯道建议低于直道，避免目标点过远导致错过近处折角。太低会频繁修正，太高会转向滞后。
    "lookahead_corner": 0.30,
    # 远近中心点横向差达到多少像素时，判定为弯道。减小：更早进入弯道控制；增大：只有明显转弯才降速和改变前视距离。
    "corner_delta_px": 18.0,
    # 用近、中、远三点判断折线程度的阈值。数值越小，越容易识别折角或 S 弯。
    "corner_bend_px": 12.0,
    # 单边黄线跨扫描行的斜率辅助转向：以 30 px 前视距离换算成附加横向误差。
    "slope_lookahead_px": 30.0,
    "min_slope_points": 2,
    # 折角期间保持最近可靠的转弯方向，避免单边误配使方向突然反转。
    "corner_commit_sec": 0.85,
    "corner_commit_error_px": 18.0,

    # ---------- 转向控制器 ----------
    # 比例控制系数。误差越大，转向命令越大。
    # 过小：车转不过去、容易在弯道外侧偏离；过大：车辆会左右摆动，甚至快速撞向边界。
    "kp": 0.0020,
    # 积分控制系数。当前建议保持 0，因为视觉分割误差会被积分累积，容易造成长时间偏转和突然反向修正。
    "ki": 0.0,
    # 微分控制系数。当前建议先保持 0。只有确认车辆持续过冲、而图像中心点稳定时，再小幅加入。不建议初始直接使用 D 项，因为分割边界抖动会放大到舵机。
    "kd": 0.0,
    # 最大转向控制量，代码会限制 steer 在 ±max_steer 内。对当前驱动节点而言，0.18 对应约 ±234 PWM 舵机变化。
    # 太小：直角或急弯转不过去；太大：高速时容易突然打满方向。
    "max_steer": 0.18,
    # 转向命令每秒允许变化的最大幅度。限制舵机瞬间跳变，影响轨迹平滑程度。
    # 太小：转向跟不上急弯；太大：会产生类似左右猛打方向的效果。
    "steer_rate": 0.70,
    # 当前帧误差在平滑误差中的权重。0：完全使用历史误差；1：完全使用当前误差。
    # 增大：反应更快，但抖动更明显；减小：更平滑，但转向会滞后。
    "error_smooth": 0.45,

    # ---------- 速度策略 ----------
    # 直道基础速度。
    # 驱动节点换算：PWM = 1500 + 100 × speed。 0.40 对应约 1540 PWM。
    "speed_straight": 0.40,
    # 识别为弯道时的基础速度。
    # 0.38 对应约 1538 PWM。与你的目标“弯道 1537～1541 PWM”一致。
    "speed_corner": 0.38,
    # 单边黄线、部分扫描层缺失、大误差时的限速值。
    # 0.36 对应约 1536 PWM。不能设置得比直道速度高，否则“不确定状态”反而跑更快。
    "speed_uncertain": 0.36,
    # 已有可信赛道宽度、但折角处仅看见一侧黄线时的速度，约为 1538 PWM。
    "speed_single_edge": 0.38,
    # 横向误差超过此值时，强制将速度限制为 speed_uncertain。
    # 降低：更早减速，安全性更高；增大：允许带更大偏差继续高速前进，但撞线/撞桶风险增加。
    "speed_max_error_px": 50.0,


    # ---------- 图像与运行保护 ----------
    # 形态学闭运算的核尺寸。
    # 用于连接黄线掩码里的小孔洞和断裂。
    # 太小：黄线断裂会变多；
    # 太大：相邻黄线、噪声可能被粘连。
    "morph_kernel": 5,

    # 是否调用 OpenCV 弹窗显示分割结果和扫描线。
    # SSH 无桌面环境必须是 False。
    # 有本机显示器调试时可设为 True。
    "show_debug": False,

    # 日志打印时间间隔，单位秒。
    # 不影响控制，只影响终端信息量。
    "log_period": 0.5,

    # 超过多少秒未收到图像时，节点发布停车命令。
    # 摄像头约 30 FPS 时，0.5 秒足够识别异常。
    # 太大：摄像头断流后，旧控制命令持续时间过长；
    # 太小：偶发推理卡顿可能误触发停车。
    "image_timeout": 0.5,
}


class Follower(Node):
    def __init__(self):
        super().__init__('follower')
        self.cfg = CFG.copy()
        # 在车载桌面终端运行时可用 -p show_debug:=true 打开模型可视化窗口。
        self.cfg['show_debug'] = bool(
            self.declare_parameter('show_debug', self.cfg['show_debug']).value)
        # 默认不向真实底盘发送运动控制，先确认 mask 和边界结果。
        # 经过桌面检查后，运行时以 -p enable_motion:=true 才会真的行驶。
        self.enable_motion = self.declare_parameter('enable_motion', False).value
        self.bridge = CvBridge()
        self.om_session = InferSession(device_id=0, model_path=self.cfg['model_path'])
        self.om_inputs = self.om_session.get_inputs()
        self.om_outputs = self.om_session.get_outputs()
        # 摄像头只保留最新图像，避免 NPU 推理时处理过期画面。
        image_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.image_sub = self.create_subscription(
            Image, self.cfg['image_topic'], self.image_callback, image_qos)
        self.cmd_vel_pub = self.create_publisher(Twist, self.cfg['cmd_topic'], 10)
        self.mask_pub = self.create_publisher(Image, '/yellow_test/mask', 1)
        self.overlay_pub = self.create_publisher(Image, '/yellow_test/overlay', 1)
        self.width_history = [None] * len(self.cfg['scan_ratios'])
        self.lane_width_px = None
        self.lane_width_ref_y = None
        self.prev_centers = None
        self.last_scan_rows = []
        self.last_mask_y_range = None
        self.prev_error = 0.0
        self.prev_steer = 0.0
        self.corner_commit_direction = 0.0
        self.corner_commit_until = 0.0
        self.last_single_slope = None
        self.last_control_time = None
        self.last_log_time = 0.0
        self.last_image_time = time.monotonic()
        self.watchdog = self.create_timer(0.2, self.check_image_timeout)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.cfg['morph_kernel'],) * 2)
        self.get_logger().info(
            'Yellow-line follower started: motion=%s, output=%s, mask=/yellow_test/mask, overlay=/yellow_test/overlay'
            % (self.enable_motion, self.cfg['cmd_topic']))
    
    def get_scan_rows(self, mask):
        """按 mask 的白线有效范围选扫描行；白线不足时回退到固定扫描行。"""
        h, _ = mask.shape
        half_h = h // 2
        fallback = [min(h - 1, half_h + int(half_h * r))
                    for r in self.cfg['scan_ratios']]
        self.last_mask_y_range = None

        white_y = np.flatnonzero(np.any(mask != 0, axis=1))
        if white_y.size:
            self.last_mask_y_range = (int(white_y.min()), int(white_y.max()))
        if not self.cfg.get('adaptive_scan_rows', True) or white_y.size == 0:
            self.last_scan_rows = fallback
            return fallback

        # 使用与边界提取相同的候选段规则寻找有效行，避免只按单个噪点选扫描线。
        search_top = max(0, int(h * self.cfg.get('scan_search_top_ratio', 0.35)))
        supported = [y for y in range(search_top, h)
                     if self.get_segments(mask, y)]
        if len(supported) < len(fallback):
            self.last_scan_rows = fallback
            return fallback

        # 只保留最长的连续有效区间，避免远处孤立噪声把扫描线拉走。
        runs, start = [], 0
        for i in range(1, len(supported) + 1):
            if i == len(supported) or supported[i] != supported[i - 1] + 1:
                runs.append(supported[start:i])
                start = i
        band = max(runs, key=len)
        if len(band) < len(fallback):
            self.last_scan_rows = fallback
            return fallback

        # 从近到远（y 大到小）均匀抽取五行；与控制器的近/远前视顺序一致。
        indices = np.linspace(len(band) - 1, 0, len(fallback)).round().astype(int)
        rows = [int(band[i]) for i in indices]
        self.last_scan_rows = rows
        return rows

    def preprocess_om(self, image):
        """输入：image:[240,320,3]，BGR图像。输出：input_tensor:[1,3,240,320]，float32。"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)

    def predict_mask(self, image):
        """输入：image:[240,320,3]。输出：mask:[240,320]，0为背景，255为黄线。"""
        output = np.asarray(self.om_session.infer([self.preprocess_om(image)])[0])
        if output.shape != (1, 2, 240, 320):
            raise ValueError(f'Unexpected OM output shape: {output.shape}')
        return (np.argmax(output[0], axis=0) == 1).astype(np.uint8) * 255

    def get_segments(self, mask, y):
        """输入：mask:[H,W]，扫描行y。输出：[(x_start,x_end,x_center),...]，按x从小到大排列。"""
        band = mask[max(0, y - self.cfg['scan_half_height']):min(mask.shape[0], y + self.cfg['scan_half_height'] + 1)]
        # 条带内以列投票抑制零散噪点；连续黄色列组成一段候选黄线。
        present = (np.count_nonzero(band, axis=0) >= max(2, band.shape[0] // 3)).astype(np.uint8)
        padded = np.pad(present, (1, 1), constant_values=0)
        starts = np.flatnonzero(np.diff(padded.astype(np.int16)) == 1)
        ends = np.flatnonzero(np.diff(padded.astype(np.int16)) == -1) - 1
        return [(int(a), int(b), 0.5 * (a + b)) for a, b in zip(starts, ends) if b - a + 1 >= self.cfg['min_segment_px']]

    def select_boundaries(self, candidates, expected_center, expected_width, previous_pair):
        """输入：候选区段、预测中心、预测宽度和上一层双边位置。输出：(left,right,source)或None。"""
        xs = [s[2] for s in candidates]
        if not xs:
            return None
        pairs = []
        for i, left in enumerate(xs):
            for right in xs[i + 1:]:
                width = right - left
                if not self.cfg['min_road_width'] <= width <= self.cfg['max_road_width']:
                    continue
                if expected_width is not None:
                    width_ratio_error = abs(width - expected_width) / max(expected_width, 1.0)
                    if width_ratio_error > self.cfg['max_width_ratio_error']:
                        continue
                center = (left + right) * 0.5
                # 首层优先选择靠近历史中心的双边；随后同时检查中心和左右边界的连续性。
                score = abs(center - expected_center) + 0.30 * abs(width - expected_width) if expected_width else abs(center - expected_center)
                if previous_pair is not None:
                    score += 0.35 * (abs(left - previous_pair[0]) + abs(right - previous_pair[1]))
                pairs.append((score, left, right))
        if pairs:
            _, left, right = min(pairs)
            if previous_pair is None or abs((left + right) * 0.5 - expected_center) <= self.cfg['max_track_jump']:
                return left, right, 'double', None

        # 只能看见一侧时，必须有可信的宽度估计，不能把黄线本身当作赛道中心。
        if expected_width is None:
            return None
        possibilities = []
        for x in xs:
            for left, right, side in ((x, x + expected_width, 'left'),
                                      (x - expected_width, x, 'right')):
                if not (0 <= left < right < 320):
                    continue
                center = (left + right) * 0.5
                score = abs(center - expected_center)
                if previous_pair is not None:
                    score += 0.35 * (abs(left - previous_pair[0]) + abs(right - previous_pair[1]))
                possibilities.append((score, left, right, side, x))
        if not possibilities:
            return None
        score, left, right, side, visible_x = min(possibilities)
        if abs((left + right) * 0.5 - expected_center) > self.cfg['max_track_jump'] * 0.65:
            return None
        return left, right, f'single_{side}', visible_x

    def extract_centers(self, mask):
        """输入：mask:[240,320]。输出：从近到远的中心点及检测类型，缺失层为None。"""
        
        h, w = mask.shape
        rows = self.get_scan_rows(mask)
        observations = [None] * len(rows)
        previous_pair = None
        previous_y = None
        centers = []
        for i, y in enumerate(rows):
            #沿用上一帧相同扫描层的中心，否则取图像中心
            expected_center = self.prev_centers[i] if self.prev_centers and self.prev_centers[i] is not None else w * 0.5
            if centers:
                # 利用前面已跟踪的近处扫描层预测远处中心，允许左右两个方向的折线。
                expected_center = centers[-1]
                #沿用当前帧前两个扫描层的坐标变化趋势
                if len(centers) >= 2:
                    expected_center += np.clip(centers[-1] - centers[-2], -35, 35)
            # 最近处已确认/补全的边界最可信。向上扫描时按透视缩小宽度，
            # 避免把近处宽赛道错误地用于折角远处的单边黄线。
            if previous_pair is not None and previous_y is not None:
                vertical_step = max(0.0, (previous_y - y) / 10.0)
                expected_width = (previous_pair[1] - previous_pair[0]) * (
                    self.cfg['perspective_width_decay'] ** vertical_step)
            elif self.lane_width_px is not None and self.lane_width_ref_y is not None:
                vertical_step = (self.lane_width_ref_y - y) / 10.0
                expected_width = self.lane_width_px * (
                    self.cfg['perspective_width_decay'] ** vertical_step)
            else:
                expected_width = self.width_history[i]
            #get_segments找出所有候选边界，select_boundaries则是对这些边界组合，筛选出合适边界
            result = self.select_boundaries(self.get_segments(mask, y), expected_center, expected_width, previous_pair)
            if result is None:
                # 不因中间一层短暂断线丢掉近处已确认的宽度和中心预测。
                continue
            left, right, source, visible_x = result
            center = (left + right) * 0.5
            observations[i] = {
                'y': y, 'left': left, 'right': right, 'center': center,
                'source': source, 'visible_x': visible_x,
            }
            centers.append(center)
            previous_pair = (left, right)
            previous_y = y
            if source == 'double':
                width = right - left
                previous_width = self.width_history[i]
                self.width_history[i] = width if previous_width is None else 0.8 * previous_width + 0.2 * width
                alpha = self.cfg['lane_width_smooth']
                self.lane_width_px = width if self.lane_width_px is None else (
                    (1.0 - alpha) * self.lane_width_px + alpha * width)
                self.lane_width_ref_y = float(y) if self.lane_width_ref_y is None else (
                    (1.0 - alpha) * self.lane_width_ref_y + alpha * y)
        self.prev_centers = [p['center'] if p else None for p in observations]
        return observations

    def calculate_control(self, observations, dt):
        """输入：5层扫描结果和控制时间间隔。输出：(speed,steer,state,error)；steer正值左转。"""
        #找出有效扫描层
        valid = [(i, p) for i, p in enumerate(observations) if p is not None]
        # 只有一层时，仅在它是借助既往宽度完成的单边补全时继续；完全失去边界仍停车。
        single_edge_mode = (
            len(valid) == 1 and valid[0][1]['source'].startswith('single_') and
            self.lane_width_px is not None)
        if len(valid) < self.cfg['min_valid_rows'] and not single_edge_mode:
            self.prev_error = 0.0
            self.prev_steer = 0.0
            return 0.0, 0.0, 'LOST', 0.0
        #根据valid划分三层
        near = valid[0][1]['center']
        far = valid[-1][1]['center']
        mid = valid[len(valid) // 2][1]['center']
        #根据远处和近处计算道路趋势
        trend = far - near
        #判断是否出现了折角
        bend = abs(far - 2 * mid + near)
        # 对同一侧可见黄线做 x(y) 线性拟合。左转时通常 slope>0，右转时 slope<0。
        side_points = {}
        for _, point in valid:
            if point['source'].startswith('single_') and point['visible_x'] is not None:
                side = point['source']
                side_points.setdefault(side, []).append((point['y'], point['visible_x']))
        slope = None
        if side_points:
            points = max(side_points.values(), key=len)
            if len(points) >= self.cfg['min_slope_points']:
                ys, xs = zip(*points)
                if max(ys) != min(ys):
                    slope = float(np.polyfit(ys, xs, 1)[0])
        self.last_single_slope = slope
        #根据bend或者trend去判断是否应该进入corner模式，即拐弯
        slope_turn = slope is not None and abs(slope * self.cfg['slope_lookahead_px']) >= self.cfg['corner_delta_px']
        corner = (single_edge_mode or abs(trend) >= self.cfg['corner_delta_px'] or
                  bend >= self.cfg['corner_bend_px'] or slope_turn)
        alpha = self.cfg['lookahead_corner'] if corner else self.cfg['lookahead_straight']
        target = near * (1 - alpha) + far * alpha
        error = target - 160.0
        # 单边边界的斜率给出即将到来的横向位移，用作提前转向的前馈项。
        if slope is not None:
            error -= slope * self.cfg['slope_lookahead_px']

        uncertain = any(p['source'].startswith('single_') for _, p in valid)
        now = time.monotonic()
        # 在可靠的折角迹象出现时锁定转弯方向一小段时间；单边误配不能把左转翻成右转。
        if corner and abs(error) >= self.cfg['corner_commit_error_px'] and (not uncertain or slope is not None):
            self.corner_commit_direction = float(np.sign(error))
            self.corner_commit_until = now + self.cfg['corner_commit_sec']
        elif not uncertain and not corner:
            self.corner_commit_direction = 0.0
        if uncertain and now < self.corner_commit_until and self.corner_commit_direction != 0.0:
            error = self.corner_commit_direction * max(
                abs(error), self.cfg['corner_commit_error_px'])
        #用历史误差和求出的误差加权，即对误差做时间平滑
        error = self.cfg['error_smooth'] * error + (1 - self.cfg['error_smooth']) * self.prev_error
        # 计算误差变化速度，就是计算PD中的D项
        derivative = (error - self.prev_error) / max(dt, 0.03)
        self.prev_error = error

        # 实车标定结果：正 angular.z 会让车辆向右。
        # 因而目标中心在左侧（error<0）时应发送负转向，车辆才能向左回正。
        raw_steer = self.cfg['kp'] * error + self.cfg['kd'] * derivative
        raw_steer = float(np.clip(raw_steer, -self.cfg['max_steer'], self.cfg['max_steer']))
        # 限制steer 最多变化量，防止车辆猛转弯
        max_change = self.cfg['steer_rate'] * min(dt, 0.2)
        steer = float(np.clip(raw_steer, self.prev_steer - max_change, self.prev_steer + max_change))
        self.prev_steer = steer

        # 根据是否拐弯设置基础速度
        speed = self.cfg['speed_corner'] if corner else self.cfg['speed_straight']
        # 有全局宽度支撑的单边观测属于折角的正常情况，保持连续过弯速度；
        # 尚未建立宽度历史的单边观测才按低置信度限速。
        if uncertain:
            speed = (min(speed, self.cfg['speed_single_edge'])
                     if self.lane_width_px is not None else min(speed, self.cfg['speed_uncertain']))
        if abs(error) > self.cfg['speed_max_error_px']:
            speed = min(speed, self.cfg['speed_uncertain'])
        state = 'SINGLE_EDGE' if uncertain else 'CORNER' if corner else 'STRAIGHT'
        return speed, steer, state, error

    def publish_cmd(self, speed, steer):
        """输入：speed和steer控制量。输出：发布geometry_msgs/Twist到黄线循迹话题。"""
        msg = Twist()
        msg.linear.x = float(speed) if self.enable_motion else 0.0
        msg.angular.z = float(steer) if self.enable_motion else 0.0
        self.cmd_vel_pub.publish(msg)

    def make_overlay(self, image, mask, observations, state, error, speed, steer):
        """将模型 mask、扫描线、边界和控制状态叠加到原图，供桌面 rqt_image_view 查看。"""
        canvas = image.copy()
        yellow = np.zeros_like(canvas)
        yellow[mask > 0] = (0, 255, 255)
        canvas = cv2.addWeighted(canvas, 1.0, yellow, 0.55, 0)
        cv2.line(canvas, (160, 0), (160, 239), (110, 110, 110), 1)
        # extract_centers 已按当前 mask 选好行；叠加图复用同一组行，避免二次选行不一致。
        rows = self.last_scan_rows
        for i, p in enumerate(observations):
            y = rows[i]
            cv2.line(canvas, (0, y), (319, y), (80, 80, 80), 1)
            if p is None:
                continue
            color = (0, 220, 0) if p['source'] == 'double' else (0, 150, 255)
            cv2.circle(canvas, (int(p['center']), y), 4, color, -1)
            cv2.circle(canvas, (int(p['left']), y), 2, (255, 0, 0), -1)
            cv2.circle(canvas, (int(p['right']), y), 2, (255, 0, 0), -1)
        cv2.putText(canvas, f'{state} e={error:.1f}', (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(canvas, f'v={speed:.2f} s={steer:.3f} motion={self.enable_motion}',
                    (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        return canvas

    def check_image_timeout(self):
        """输入：最近接收图像的时间。输出：图像超时后发布停止指令。"""
        if time.monotonic() - self.last_image_time > self.cfg['image_timeout']:
            self.prev_steer = 0.0
            self.prev_error = 0.0
            self.publish_cmd(0.0, 0.0)

    def image_callback(self, msg):
        """输入：ROS2摄像头Image。输出：黄线控制指令，必要时显示调试窗口。"""
        now = time.monotonic()
        self.last_image_time = now
        dt = now - self.last_control_time if self.last_control_time is not None else 0.1
        self.last_control_time = now
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if image.shape != (240, 320, 3):
                raise ValueError(f'Expected image shape (240,320,3), got {image.shape}')
            mask = self.predict_mask(image)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            observations = self.extract_centers(mask)
            speed, steer, state, error = self.calculate_control(observations, dt)
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)
            overlay = self.make_overlay(image, mask, observations, state, error, speed, steer)
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            overlay_msg.header = msg.header
            self.overlay_pub.publish(overlay_msg)
        except Exception as exc:
            self.get_logger().error(f'Yellow-line processing failed: {exc}')
            self.prev_steer = 0.0
            self.prev_error = 0.0
            self.publish_cmd(0.0, 0.0)
            return
        self.publish_cmd(speed, steer)
        if now - self.last_log_time >= self.cfg['log_period']:
            valid_rows = sum(p is not None for p in observations)
            yellow_pixels = int(np.count_nonzero(mask))
            self.get_logger().info(
                f'{state} | valid={valid_rows}/5 mask={yellow_pixels} px '
                f'mask_y={self.last_mask_y_range} scan_y={self.last_scan_rows} '
                f'lane_w={self.lane_width_px if self.lane_width_px is not None else 0.0:.1f}@'
                f'{self.lane_width_ref_y if self.lane_width_ref_y is not None else 0.0:.0f} '
                f'slope={self.last_single_slope if self.last_single_slope is not None else 0.0:.2f} '
                f'commit={self.corner_commit_direction:.0f} '
                f'error={error:.1f} steer={steer:.3f} speed={speed:.2f} motion={self.enable_motion}')
            self.last_log_time = now
        if self.cfg['show_debug']:
            cv2.imshow('yellow_line_tracking', overlay)
            cv2.imshow('yellow_model_mask', mask)
            cv2.waitKey(1)

    def stop(self):
        """输出：发送停止和回正指令。"""
        # Ctrl+C 时 rclpy 可能已先关闭上下文；此时不再发布，避免退出报错。
        if rclpy.ok():
            self.publish_cmd(0.0, 0.0)
        if self.cfg['show_debug']:
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    follower = None
    try:
        follower = Follower()
        rclpy.spin(follower)
    except KeyboardInterrupt:
        pass
    finally:
        if follower is not None:
            follower.stop()
            follower.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
