#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS2 dual-yellow-tape zigzag follower + single-window paged tuning and road-only ROI.

SAFETY: tune_mode is TRUE by default. In tune_mode this node NEVER publishes
car_cmd_vel, even if /yellow_tracking_enable is True. Tune with wheels stationary.
Click or drag a slider in the single tuning window (four tabs).
Press 1/2/3/4 to switch tabs, S to save ROS2 YAML, R to reset,
Q to quit; click the original-image window on yellow tape to sample HSV.

Original basis: ROS2 dual-yellow-tape zigzag follower for the 2026 outdoor ROS race.

Track: two 10-cm yellow tapes forming a ~1.5-m-wide lane, with a 135-degree
interior angle in the zigzag. This program controls ONLY the yellow-tape
section; cones, red/green lights, SLAM and parking are outside its scope.

ROS2 interfaces:
    Sub: /image_raw               sensor_msgs/Image
    Sub: /yellow_tracking_enable  std_msgs/Bool (True=take control)
    Pub: car_cmd_vel              geometry_msgs/Twist
    Pub: /yellow_track_detected   std_msgs/Bool (diagnostic, NOT a turn switch)

Safety: DISABLED on startup by default. While disabled, it never publishes
car_cmd_vel. Only the active motion controller may drive the chassis. A
supervisor should switch between cone navigation and yellow tracking using
an exclusive cmd_vel arbiter (or stop the old publisher before enabling this).

Tuning without motion commands:
    python3 line_follow2_tuner.py --ros-args -p tune_mode:=true

After pressing S, reload parameters and inspect without moving:
    python3 line_follow2_tuner.py --ros-args \
        --params-file yellow_tuned.yaml -p tune_mode:=true

To enable control later, use --params-file yellow_tuned.yaml together
with tune_mode:=false, and exclusively grant ownership of car_cmd_vel.

Integration:
    ros2 topic pub --once /yellow_tracking_enable std_msgs/msg/Bool '{data: true}'
    ros2 topic pub --once /yellow_tracking_enable std_msgs/msg/Bool '{data: false}'

Notes: image coordinates, normalized control gains and steering sign MUST be
calibrated on the actual camera/chassis. A 1.5-m physical lane does not imply
any fixed pixel width without camera calibration.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import copy

import cv2
import numpy as np

# ROS imports are optional here so the vision algorithm can be unit-tested on
# a non-ROS PC; the node cannot run until ROS2 packages are installed.
try:
    import rclpy
    from cv_bridge import CvBridge, CvBridgeError
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool
except ImportError:
    rclpy = None
    Node = object


@dataclass
class ScanPoint:
    row_index: int
    x: float
    y: int
    source: str                # 'pair', 'left_only', 'right_only'
    left: Optional[float] = None
    right: Optional[float] = None


@dataclass
class LaneObservation:
    valid: bool
    reason: str
    points: List[ScanPoint] = field(default_factory=list)
    mask: Optional[np.ndarray] = None
    roi_top: int = 0
    target_x: Optional[float] = None
    target_y: Optional[int] = None
    heading: float = 0.0       # normalized lateral trend toward distant rows
    corner: float = 0.0        # normalized change of piecewise centerline slope
    paired_rows: int = 0
    single_rows: int = 0
    width: int = 0
    height: int = 0
    mask_pixels: int = 0
    mask_fraction: float = 0.0
    run_count: int = 0


class YellowLaneVision:
    """Vision-only geometry. No ROS dependencies or motion commands."""

    def __init__(
        self, hsv_lower=(20, 75, 70), hsv_upper=(38, 255, 255),
        roi_start=0.70, roi_end=0.97, scan_rows=11,
        roi_top_half_width_ratio=0.42, roi_bottom_half_width_ratio=0.50,
        roi_center_ratio=0.50,
        scan_half_band=3, min_run_px=3,
        min_gap_ratio=0.12, max_gap_ratio=0.95,
        lookahead_ratio=0.82, near_ratio=0.92, far_ratio=0.74,
        memory_seconds=0.80,
        blur_kernel=5, open_kernel=3, close_kernel=5,
        band_coverage=0.30, max_run_ratio=0.18,
        min_center_points=4, min_pair_rows=2, min_span_ratio=0.10,
    ):
        self.lower = np.array(hsv_lower, dtype=np.uint8)
        self.upper = np.array(hsv_upper, dtype=np.uint8)
        self.roi_start = float(roi_start)
        self.roi_end = float(roi_end)
        self.roi_top_half_width_ratio = float(roi_top_half_width_ratio)
        self.roi_bottom_half_width_ratio = float(roi_bottom_half_width_ratio)
        self.roi_center_ratio = float(roi_center_ratio)
        self.scan_rows = int(scan_rows)
        self.band = int(scan_half_band)
        self.min_run_px = int(min_run_px)
        self.min_gap_ratio = float(min_gap_ratio)
        self.max_gap_ratio = float(max_gap_ratio)
        self.lookahead_ratio = float(lookahead_ratio)
        self.near_ratio = float(near_ratio)
        self.far_ratio = float(far_ratio)
        self.memory_seconds = float(memory_seconds)
        self.blur_kernel = int(blur_kernel)
        self.open_kernel = int(open_kernel)
        self.close_kernel = int(close_kernel)
        self.band_coverage = float(band_coverage)
        self.max_run_ratio = float(max_run_ratio)
        self.min_center_points = int(min_center_points)
        self.min_pair_rows = int(min_pair_rows)
        self.min_span_ratio = float(min_span_ratio)
        if not (0.0 <= self.roi_start < self.far_ratio < self.lookahead_ratio <
                self.near_ratio <= self.roi_end <= 1.0):
            raise ValueError('Require roi_start < far < lookahead < near <= roi_end')
        if self.scan_rows < 5:
            raise ValueError('scan_rows must be at least 5')
        self.reset()

    def reset(self):
        self.pair_model = {}         # index -> (left, right, gap), EMA per scan height
        self.last_pair_time = None

    @staticmethod
    def _runs(binary_row, min_run_px, max_run_px):
        """Return horizontal yellow segments (center x, width) in image coords."""
        active = np.asarray(binary_row, dtype=np.bool_)
        edges = np.diff(np.concatenate(([False], active, [False])).astype(np.int8))
        starts = np.where(edges == 1)[0]
        stops = np.where(edges == -1)[0]
        return [((int(a) + int(b) - 1) / 2.0, int(b - a))
                for a, b in zip(starts, stops)
                if min_run_px <= b - a <= max_run_px]

    @staticmethod
    def _interp(points: List[ScanPoint], y: float):
        """Interpolate within measured row range; never extrapolate."""
        ordered = sorted(points, key=lambda p: p.y)
        if not ordered or y < ordered[0].y or y > ordered[-1].y:
            return None
        return float(np.interp(y, [p.y for p in ordered], [p.x for p in ordered]))

    def _pair(self, runs, row_idx, w):
        """Choose two separated horizontal tape intersections.

        Crucially, both borders may lie in the SAME image half on sharp turns.
        Therefore candidate pairs are selected by x-order, not half-image.
        """
        if len(runs) < 2:
            return None
        model = self.pair_model.get(row_idx)
        candidates = []
        ordered = sorted(runs)
        for i, (left, _) in enumerate(ordered[:-1]):
            for right, _ in ordered[i + 1:]:
                gap = right - left
                if not self.min_gap_ratio * w <= gap <= self.max_gap_ratio * w:
                    continue
                center = (left + right) / 2
                if model is None:
                    score = abs(center - 0.5 * w) / w - 0.12 * gap / w
                else:
                    prev_left, prev_right, prev_gap = model
                    score = (0.75 * abs(center - (prev_left + prev_right) / 2) / w
                             + 1.1 * abs(gap - prev_gap) / w)
                candidates.append((score, left, right))
        if not candidates:
            return None
        _, left, right = min(candidates)
        return float(left), float(right)

    def roi_polygon(self, h: int, w: int):
        """Camera-space trapezoid: narrow at distant edge, wide at near edge.

        This is a pixel ROI, not ground-plane geometry or camera calibration.
        It removes background from processing, not just from display.
        """
        y0 = int(round(h * self.roi_start))
        y1 = max(y0 + 1, min(h-1, int(round(h * self.roi_end))))
        center = w * self.roi_center_ratio
        top_half = self.roi_top_half_width_ratio * w
        bottom_half = self.roi_bottom_half_width_ratio * w
        pts = np.array([
            [center-top_half, y0], [center+top_half, y0],
            [center+bottom_half, y1], [center-bottom_half, y1],
        ], dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w-1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h-1)
        return pts.astype(np.int32)

    def analyze(self, image: np.ndarray, now: Optional[float] = None):
        if now is None:
            now = time.monotonic()
        h, w = image.shape[:2]
        roi_top = int(round(h * self.roi_start))
        roi_bottom = min(h, int(round(h * self.roi_end)) + 1)
        roi = image[roi_top:roi_bottom]
        blur_size = max(1, int(self.blur_kernel) | 1)  # odd; 1 = disabled
        blurred = cv2.GaussianBlur(roi, (blur_size, blur_size), 0) if blur_size > 1 else roi
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # On inverted sliders, return an empty mask rather than accidentally
        # classifying the entire frame as road.
        if np.any(self.lower > self.upper):
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        else:
            mask = cv2.inRange(hsv, self.lower, self.upper)
        if self.open_kernel > 1:
            k = self.open_kernel | 1
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                       cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        if self.close_kernel > 1:
            k = self.close_kernel | 1
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                       cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

        # Apply real trapezoid exclusion *before* finding tape runs.
        # All pixels outside the polygon are zero, so walls and side objects
        # cannot participate in paired-yellow-line detection.
        polygon = self.roi_polygon(h, w).copy()
        polygon[:, 1] -= roi_top
        valid_area = np.zeros_like(mask)
        cv2.fillPoly(valid_area, [polygon], 255)
        mask = cv2.bitwise_and(mask, valid_area)

        result = LaneObservation(False, 'no_valid_centerline', mask=mask,
                                 roi_top=roi_top, width=w, height=h,
                                 mask_pixels=int(cv2.countNonZero(mask)),
                                 mask_fraction=float(cv2.countNonZero(mask) / max(1, mask.size)))
        ys = np.linspace(int(h * self.roi_end) - 1,
                         int(h * self.roi_start) + self.band + 1,
                         self.scan_rows).astype(int)
        row_runs = {}
        for idx, y in enumerate(ys):
            relative_y = y - roi_top
            lo = max(0, relative_y - self.band)
            hi = min(mask.shape[0], relative_y + self.band + 1)
            band_view = mask[lo:hi]
            # Require a yellow pixel in >= 30% of the small vertical band.
            visible = np.count_nonzero(band_view, axis=0) >= max(
                1, int(np.ceil(self.band_coverage * band_view.shape[0])))
            row_runs[idx] = self._runs(visible, self.min_run_px,
                                       max(self.min_run_px, int(self.max_run_ratio * w)))
        result.run_count = sum(len(runs) for runs in row_runs.values())

        paired = {}
        for idx, runs in row_runs.items():
            pair = self._pair(runs, idx, w)
            if pair is not None:
                left, right = pair
                paired[idx] = (left, right)
                result.points.append(ScanPoint(idx, (left + right) / 2,
                                               int(ys[idx]), 'pair', left, right))
                if idx in self.pair_model:
                    pleft, pright, _ = self.pair_model[idx]
                    left = 0.65 * pleft + 0.35 * left
                    right = 0.65 * pright + 0.35 * right
                self.pair_model[idx] = (left, right, right - left)
        result.paired_rows = len(paired)
        if len(paired) >= 2:
            self.last_pair_time = now

        # If one tape is temporarily hidden, estimate the center using a
        # recently measured width. Never guess left/right from x < w/2.
        memory_fresh = (self.last_pair_time is not None and
                        now - self.last_pair_time <= self.memory_seconds)
        if memory_fresh:
            for idx, runs in row_runs.items():
                if idx in paired or len(runs) != 1:
                    continue
                x = runs[0][0]
                model = self.pair_model.get(idx)
                if model is None:
                    # A neighboring paired row (within 2 scanlines) supplies
                    # a conservative one-frame approximation of the width.
                    neighbors = [(abs(idx - k), k) for k in paired]
                    if not neighbors:
                        continue
                    distance, nearest_idx = min(neighbors)
                    if distance > 2:
                        continue
                    ml, mr = paired[nearest_idx]
                    ratio = np.clip((ys[idx] - roi_top) /
                                    max(1., ys[nearest_idx] - roi_top), 0.70, 1.30)
                    center = (ml + mr) / 2
                    gap = (mr - ml) * ratio
                    model = (center - gap / 2, center + gap / 2, gap)
                left_pred, right_pred, gap = model
                left_distance = abs(x - left_pred)
                right_distance = abs(x - right_pred)
                # Reject ambiguous observations; a wrong side assignment can
                # steer directly into the painted boundary.
                if abs(left_distance - right_distance) < 0.18 * gap:
                    continue
                if min(left_distance, right_distance) > 0.35 * w:
                    continue
                is_left = left_distance < right_distance
                center_x = x + gap / 2 if is_left else x - gap / 2
                if not -0.10 * w <= center_x <= 1.10 * w:
                    continue
                result.points.append(ScanPoint(idx, float(np.clip(center_x, 0, w-1)),
                                               int(ys[idx]),
                                               'left_only' if is_left else 'right_only',
                                               left=x if is_left else None,
                                               right=x if not is_left else None))
                result.single_rows += 1

        result.points.sort(key=lambda p: p.y)
        enough_rows = len(result.points) >= self.min_center_points
        enough_range = bool(result.points) and (
            result.points[-1].y - result.points[0].y >= self.min_span_ratio * h)
        # At least two paired rows now, OR recent pairing + enough single rows.
        credible = result.paired_rows >= self.min_pair_rows or (
            memory_fresh and result.single_rows >= max(4, self.min_center_points))
        if not (enough_rows and enough_range and credible):
            result.reason = ('insufficient_yellow_tape' if result.paired_rows < self.min_pair_rows
                             else 'insufficient_centerline_span')
            return result

        target_y = int(round(h * self.lookahead_ratio))
        target_x = self._interp(result.points, target_y)
        if target_x is None:
            result.reason = 'lookahead_not_observed'
            return result

        result.target_x = target_x
        result.target_y = target_y
        result.valid = True
        result.reason = 'both_boundaries' if result.paired_rows >= self.min_pair_rows else 'one_boundary'

        far_y = h * self.far_ratio
        near_y = h * self.near_ratio
        mid_y = 0.5 * (far_y + near_y)
        far_x = self._interp(result.points, far_y)
        mid_x = self._interp(result.points, mid_y)
        near_x = self._interp(result.points, near_y)
        if far_x is not None and near_x is not None:
            # Positive: further road goes to the right in camera coordinates.
            result.heading = float(np.clip((far_x - near_x) / (w / 2), -1., 1.))
        if far_x is not None and mid_x is not None and near_x is not None:
            # Discontinuity in two centerline slopes captures an impending V
            # corner; it is a heuristic, not a calibrated steering angle.
            far_slope = ((mid_x - far_x) / (mid_y - far_y)) * h / (w / 2)
            near_slope = ((near_x - mid_x) / (near_y - mid_y)) * h / (w / 2)
            result.corner = float(np.clip(0.25 * abs(far_slope - near_slope), 0., 1.))
        return result


class PDTracker:
    def __init__(self, kp=1.25, kd=0.06, k_heading=0.38,
                 max_angular=0.85, angular_rate_limit=3.0,
                 max_speed=0.24, min_speed=0.09,
                 corner_threshold=0.20, corner_speed_cap=0.14,
                 one_boundary_speed_cap=0.13, acceleration_limit=0.7,
                 camera_center_ratio=0.50):
        self.kp = float(kp)
        self.kd = float(kd)
        self.k_heading = float(k_heading)
        self.max_angular = float(max_angular)
        self.angular_rate_limit = float(angular_rate_limit)
        self.max_speed = float(max_speed)
        self.min_speed = float(min_speed)
        self.corner_threshold = float(corner_threshold)
        self.corner_speed_cap = float(corner_speed_cap)
        self.one_boundary_speed_cap = float(one_boundary_speed_cap)
        self.acceleration_limit = float(acceleration_limit)
        self.camera_center_ratio = float(camera_center_ratio)
        self.reset()

    def reset(self):
        self.prev_error = None
        self.prev_time = None
        self.derivative_ema = 0.0
        self.last_angular = 0.0
        self.last_speed = 0.0

    def command(self, obs: LaneObservation, now=None) -> Tuple[float, float, float]:
        if not obs.valid or obs.target_x is None:
            self.reset()
            return 0., 0., 0.
        if now is None:
            now = time.monotonic()
        error = float(np.clip((obs.target_x - obs.width*self.camera_center_ratio) /
                              (obs.width / 2.0), -1., 1.))
        dt = (0.05 if self.prev_time is None
              else float(np.clip(now - self.prev_time, 0.02, 0.20)))
        derivative = (0. if self.prev_error is None else
                      float(np.clip((error-self.prev_error)/dt, -4., 4.)))
        self.derivative_ema = 0.75*self.derivative_ema + 0.25*derivative
        self.prev_error, self.prev_time = error, now
        raw_angular = -(self.kp*error + self.kd*self.derivative_ema
                        + self.k_heading*obs.heading)
        angular = float(np.clip(raw_angular, -self.max_angular, self.max_angular))
        # Limit steering slew, but do not delay safety STOP commands.
        angular = float(np.clip(angular,
                                self.last_angular - self.angular_rate_limit*dt,
                                self.last_angular + self.angular_rate_limit*dt))
        self.last_angular = angular

        severity = max(abs(error), abs(angular)/max(0.01, self.max_angular),
                       obs.corner)
        speed = self.max_speed - (self.max_speed-self.min_speed)*min(1., severity)
        if obs.corner >= self.corner_threshold:
            speed = min(speed, self.corner_speed_cap)
        if obs.reason == 'one_boundary':
            speed = min(speed, self.one_boundary_speed_cap)
        # First frame after enable uses low speed and ramps up smoothly.
        speed = float(np.clip(speed, 0., self.last_speed + self.acceleration_limit*dt))
        self.last_speed = speed
        return speed, angular, error



class LiveTrackbars:
    """One OpenCV window with four clickable tabs and mouse-draggable sliders.

    No stacking or scrolling of 38 native trackbars, which is especially
    painful when forwarded from an SSH X11 session. All values are editable
    in place; the algorithms receive the clamped values on the next frame.
    """
    WINDOW = 'Yellow Lane Tuner | 1 HSV  2 ROI  3 Scan  4 Control'
    WIDTH = 820
    HEIGHT = 730
    SLIDER_X0 = 405
    SLIDER_X1 = 697
    ROW_Y0 = 137
    ROW_STEP = 42
    # display label, config name, range, multiplier to represent integers
    PAGES = [
        ('HSV / FILTERS', [
            ('H min', 'h_min', 0, 179, 1),
            ('H max', 'h_max', 0, 179, 1),
            ('S min', 's_min', 0, 255, 1),
            ('S max', 's_max', 0, 255, 1),
            ('V min', 'v_min', 0, 255, 1),
            ('V max', 'v_max', 0, 255, 1),
            ('Gaussian blur (odd)', 'blur_kernel', 0, 15, 1),
            ('Morph open (odd)', 'open_kernel', 0, 19, 1),
            ('Morph close (odd)', 'close_kernel', 0, 25, 1),
        ]),
        ('ROAD ROI / LOOKAHEAD', [
            ('ROI top % (cut background)', 'roi_start_ratio', 0, 90, 100),
            ('ROI bottom %', 'roi_end_ratio', 10, 100, 100),
            ('Trapezoid TOP halfwidth %', 'roi_top_half_width_ratio', 10, 50, 100),
            ('Trapezoid BOTTOM halfwidth %', 'roi_bottom_half_width_ratio', 10, 50, 100),
            ('Trapezoid center X %', 'roi_center_x_ratio', 10, 90, 100),
            ('Far row %', 'far_y_ratio', 1, 98, 100),
            ('Target row %', 'lookahead_y_ratio', 2, 99, 100),
            ('Near row %', 'near_y_ratio', 3, 100, 100),
            ('Scan rows', 'scan_rows', 5, 31, 1),
            ('Minimum Y span %', 'min_span_ratio', 0, 50, 100),
        ]),
        ('LANE PAIRING', [
            ('Scan half band px', 'scan_half_band', 0, 15, 1),
            ('Band coverage %', 'band_coverage', 1, 100, 100),
            ('Min yellow run px', 'min_run_px', 1, 30, 1),
            ('Max run width %', 'max_run_ratio', 1, 70, 100),
            ('Min lane gap %', 'min_lane_gap_ratio', 2, 80, 100),
            ('Max lane gap %', 'max_lane_gap_ratio', 4, 100, 100),
            ('Min paired rows', 'min_pair_rows', 2, 15, 1),
            ('Min center points', 'min_center_points', 4, 15, 1),
            ('Single-line memory x100', 'single_boundary_memory_sec', 0, 300, 100),
        ]),
        ('CONTROLLER (PREVIEW)', [
            ('Kp x100', 'kp', 0, 500, 100),
            ('Kd x100', 'kd', 0, 200, 100),
            ('K heading x100', 'k_heading', 0, 300, 100),
            ('Max angular x100', 'max_angular', 1, 300, 100),
            ('Angular rate limit x100', 'angular_rate_limit', 1, 1000, 100),
            ('Max speed x100', 'max_speed', 0, 150, 100),
            ('Min speed x100', 'min_speed', 0, 100, 100),
            ('Corner threshold %', 'corner_threshold', 0, 100, 100),
            ('Corner speed x100', 'corner_speed_cap', 0, 100, 100),
            ('One-side speed x100', 'single_boundary_speed_cap', 0, 100, 100),
            ('Acceleration x100', 'acceleration_limit', 0, 500, 100),
            ('Camera center %', 'camera_center_x_ratio', 0, 100, 100),
            ('Timeout x100', 'camera_timeout_sec', 15, 300, 100),
        ]),
    ]

    def __init__(self, values):
        self.startup = copy.deepcopy(values)
        self.values = copy.deepcopy(values)
        self.page = 0
        self.drag_index = None
        self.dirty = True
        self.specs = {spec[1]: spec for _, pg in self.PAGES for spec in pg}
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW, self._mouse)
        self.draw()

    def _spec_value(self, spec):
        _, key, low, high, scale = spec
        v = float(self.values[key]) * scale
        return int(np.clip(round(v), low, high))

    def _update(self, spec, mouse_x):
        _, key, low, high, scale = spec
        frac = np.clip((mouse_x-self.SLIDER_X0)/(self.SLIDER_X1-self.SLIDER_X0), 0., 1.)
        value = (low + round(frac*(high-low))) / scale
        if self.values[key] != value:
            self.values[key] = value
            self.dirty = True

    def _mouse(self, event, x, y, flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            if 46 <= y < 93:
                tab = int(np.clip(x//(self.WIDTH//len(self.PAGES)), 0, len(self.PAGES)-1))
                self.set_page(tab)
                return
            i = int(round((y-self.ROW_Y0)/self.ROW_STEP))
            if (0 <= i < len(self.PAGES[self.page][1]) and
                    abs(y-(self.ROW_Y0+i*self.ROW_STEP)) <= 17):
                self.drag_index = i
                self._update(self.PAGES[self.page][1][i], x)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_index is not None:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self._update(self.PAGES[self.page][1][self.drag_index], x)
            else:
                self.drag_index = None
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_index = None

    def set_page(self, index):
        if 0 <= index < len(self.PAGES):
            self.page = index
            self.drag_index = None
            self.dirty = True

    def draw(self):
        bg = np.full((self.HEIGHT, self.WIDTH, 3), (247, 248, 251), np.uint8)
        cv2.rectangle(bg, (0, 0), (self.WIDTH, 46), (38, 46, 56), -1)
        cv2.putText(bg, 'YELLOW TAPE | LIVE TUNER | PREVIEW ONLY',
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .72, (250, 250, 250), 2)
        tw = self.WIDTH // len(self.PAGES)
        for i, (name, _) in enumerate(self.PAGES):
            left = i*tw
            color = (225, 234, 242) if i == self.page else (233, 235, 237)
            cv2.rectangle(bg, (left, 46), (left+tw-2, 90), color, -1)
            label = ['1  HSV', '2  ROI / FRONT', '3  PAIRING', '4  CONTROL'][i]
            cv2.putText(bg, label, (left+13, 74), cv2.FONT_HERSHEY_SIMPLEX,
                        .53, (18, 75, 138) if i == self.page else (67, 71, 75), 1)
        title, specs = self.PAGES[self.page]
        cv2.putText(bg, title, (20, 114), cv2.FONT_HERSHEY_SIMPLEX,
                    .54, (65, 70, 76), 1)
        for i, spec in enumerate(specs):
            label, key, low, high, scale = spec
            y = self.ROW_Y0 + i*self.ROW_STEP
            cv2.putText(bg, label, (20, y+5), cv2.FONT_HERSHEY_SIMPLEX,
                        .48, (45, 53, 61), 1)
            cv2.line(bg, (self.SLIDER_X0, y), (self.SLIDER_X1, y), (198, 203, 210), 5)
            value = self._spec_value(spec)
            frac = (value-low)/max(1, high-low)
            x = int(self.SLIDER_X0 + frac*(self.SLIDER_X1-self.SLIDER_X0))
            cv2.line(bg, (self.SLIDER_X0, y), (x, y), (205, 116, 28), 5)
            cv2.circle(bg, (x, y), 9, (40, 106, 185), -1)
            cv2.circle(bg, (x, y), 9, (255, 255, 255), 1)
            valstring = str(value) if scale == 1 else f'{value/scale:.2f}'
            cv2.putText(bg, valstring, (710, y+6), cv2.FONT_HERSHEY_SIMPLEX,
                        .52, (24, 74, 135), 1)
        cv2.line(bg, (18, 693), (802, 693), (210, 213, 216), 1)
        cv2.putText(bg, 'CLICK / DRAG | 1-4 TABS | S SAVE | R RESET | Q QUIT',
                    (20, 714), cv2.FONT_HERSHEY_SIMPLEX, .53, (68, 77, 86), 1)
        cv2.imshow(self.WINDOW, bg)
        self.dirty = False

    def read(self):
        v = dict(self.values)
        # Keep scan heights strictly increasing and within ROI. Their actual
        # clamped values are drawn in the panel and saved to disk, too.
        v['roi_start_ratio'] = min(v['roi_start_ratio'], .90)
        v['roi_end_ratio'] = float(np.clip(v['roi_end_ratio'],
                                           v['roi_start_ratio'] + .09, 1.0))
        v['far_y_ratio'] = float(np.clip(v['far_y_ratio'],
                                         v['roi_start_ratio'] + .02,
                                         v['roi_end_ratio'] - .06))
        v['lookahead_y_ratio'] = float(np.clip(v['lookahead_y_ratio'],
                                               v['far_y_ratio']+.02,
                                               v['roi_end_ratio']-.04))
        v['near_y_ratio'] = float(np.clip(v['near_y_ratio'],
                                            v['lookahead_y_ratio'] + .02,
                                            v['roi_end_ratio'] - .01))
        v['min_speed'] = min(v['min_speed'], v['max_speed'])
        v['max_lane_gap_ratio'] = max(v['max_lane_gap_ratio'],
                                         v['min_lane_gap_ratio'] + .02)
        v['max_lane_gap_ratio'] = min(1.0, v['max_lane_gap_ratio'])
        # 0 or 1 disables morphology, blur uses 1 to disable. Odd sizes only.
        v['blur_kernel'] = max(1, int(v['blur_kernel']) | 1)
        for key in ('open_kernel', 'close_kernel'):
            num = int(v[key]); v[key] = 0 if num <= 1 else (num | 1)
        for key in ('scan_rows', 'scan_half_band', 'min_run_px',
                    'min_pair_rows', 'min_center_points'):
            v[key] = int(v[key])
        if v != self.values:
            self.values = v
            self.dirty = True
        if self.dirty:
            self.draw()
        return v

    def restore(self):
        self.values = copy.deepcopy(self.startup)
        self.dirty = True
        self.drag_index = None
        self.draw()


class YellowFollower(Node):
    def __init__(self):
        super().__init__('yellow_zigzag_follower')
        defaults = {
            'image_topic': '/image_raw',
            'cmd_topic': 'car_cmd_vel',
            'enable_topic': '/yellow_tracking_enable',
            'detected_topic': '/yellow_track_detected',
            'enabled_on_start': False,
            'tune_mode': True,     # ALWAYS no motion commands while true
            'save_path': 'yellow_tuned.yaml',
            'show_debug': False,
            'hsv_lower': [18, 35, 40],
            'hsv_upper': [45, 255, 255],
            'roi_start_ratio': 0.70,
            'roi_end_ratio': 0.97,
            'roi_top_half_width_ratio': 0.42,
            'roi_bottom_half_width_ratio': 0.50,
            'roi_center_x_ratio': 0.50,
            'scan_rows': 11,
            'scan_half_band': 3,
            'min_run_px': 3,
            'max_run_ratio': 0.18,
            'band_coverage': 0.30,
            'blur_kernel': 5,
            'open_kernel': 3,
            'close_kernel': 5,
            'min_center_points': 4,
            'min_pair_rows': 2,
            'min_span_ratio': 0.10,
            'min_lane_gap_ratio': 0.12,
            'max_lane_gap_ratio': 0.95,
            'lookahead_y_ratio': 0.82,
            'near_y_ratio': 0.92,
            'far_y_ratio': 0.74,
            'single_boundary_memory_sec': 0.80,
            'kp': 1.25,
            'kd': 0.06,
            'k_heading': 0.38,
            'max_angular': 0.85,
            'angular_rate_limit': 3.0,
            'max_speed': 0.24,
            'min_speed': 0.09,
            'corner_threshold': 0.20,
            'corner_speed_cap': 0.14,
            'single_boundary_speed_cap': 0.13,
            'acceleration_limit': 0.7,
            'camera_center_x_ratio': 0.50,
            'camera_timeout_sec': 0.50,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value
        self.vision = YellowLaneVision(
            hsv_lower=p('hsv_lower'), hsv_upper=p('hsv_upper'),
            roi_start=p('roi_start_ratio'), roi_end=p('roi_end_ratio'),
            scan_rows=p('scan_rows'), scan_half_band=p('scan_half_band'),
            roi_top_half_width_ratio=p('roi_top_half_width_ratio'),
            roi_bottom_half_width_ratio=p('roi_bottom_half_width_ratio'),
            roi_center_ratio=p('roi_center_x_ratio'),
            min_run_px=p('min_run_px'), max_run_ratio=p('max_run_ratio'),
            min_gap_ratio=p('min_lane_gap_ratio'), max_gap_ratio=p('max_lane_gap_ratio'),
            lookahead_ratio=p('lookahead_y_ratio'), near_ratio=p('near_y_ratio'),
            far_ratio=p('far_y_ratio'), memory_seconds=p('single_boundary_memory_sec'),
            blur_kernel=p('blur_kernel'), open_kernel=p('open_kernel'),
            close_kernel=p('close_kernel'), band_coverage=p('band_coverage'),
            min_center_points=p('min_center_points'), min_pair_rows=p('min_pair_rows'),
            min_span_ratio=p('min_span_ratio'))
        self.controller = PDTracker(
            kp=p('kp'), kd=p('kd'), k_heading=p('k_heading'),
            max_angular=p('max_angular'), angular_rate_limit=p('angular_rate_limit'),
            max_speed=p('max_speed'), min_speed=p('min_speed'),
            corner_threshold=p('corner_threshold'), corner_speed_cap=p('corner_speed_cap'),
            one_boundary_speed_cap=p('single_boundary_speed_cap'),
            acceleration_limit=p('acceleration_limit'),
            camera_center_ratio=p('camera_center_x_ratio'))
        self.bridge = CvBridge()
        self.tune_mode = bool(p('tune_mode'))
        self.enabled = False if self.tune_mode else bool(p('enabled_on_start'))
        self.show_debug = self.tune_mode or bool(p('show_debug'))
        self.save_path = str(Path(str(p('save_path'))).expanduser().resolve())
        self.timeout = max(0.15, float(p('camera_timeout_sec')))
        self.exit_requested = False
        self.all_defaults = dict(defaults)
        self.startup_values = self._current_tuning_values()
        self.trackbars = None
        self._debug_image = None
        self._debug_scale = 1.0
        self._last_clicked = None
        self._debug_window = 'Yellow zigzag: original + centers'
        self._mask_window = 'Yellow zigzag: HSV mask (ROI)'
        self._debug_ready = False
        self.preview_controller = PDTracker()
        if self.tune_mode:
            self.trackbars = LiveTrackbars(self.startup_values)
            self.get_logger().warn('TUNING/PREVIEW ONLY: no car_cmd_vel will be published. '
                                   'S=save YAML, R=restore, Q=quit. Click yellow tape to sample HSV.')
        self.last_image_time = time.monotonic()
        self.last_warning = 0.0
        self.cmd_pub = self.create_publisher(Twist, str(p('cmd_topic')),
                                             QoSProfile(depth=10))
        self.detect_pub = self.create_publisher(Bool, str(p('detected_topic')),
                                                QoSProfile(depth=10))
        self.img_sub = self.create_subscription(Image, str(p('image_topic')),
                                                self.image_callback,
                                                qos_profile_sensor_data)
        self.enable_sub = self.create_subscription(Bool, str(p('enable_topic')),
                                                   self.enable_callback,
                                                   QoSProfile(depth=10))
        self.watchdog = self.create_timer(0.1, self.camera_watchdog)
        self.get_logger().info(f'Yellow zigzag follower ready; enabled={self.enabled}. '
                               f'Set {p("enable_topic")} to Bool(true) for lane section.')


    def _current_tuning_values(self):
        p = lambda name: self.get_parameter(name).value
        v = {name: p(name) for name in self.all_defaults
             if name not in ('image_topic', 'cmd_topic', 'enable_topic',
                             'detected_topic', 'show_debug', 'enabled_on_start',
                             'tune_mode', 'save_path')}
        for name, idx in [('h_min', 0), ('s_min', 1), ('v_min', 2)]:
            v[name] = int(p('hsv_lower')[idx])
        for name, idx in [('h_max', 0), ('s_max', 1), ('v_max', 2)]:
            v[name] = int(p('hsv_upper')[idx])
        del v['hsv_lower']; del v['hsv_upper']
        return v

    def apply_live_values(self, values):
        # UI handles all validation/clamping. Cache, reset history if changed.
        previous = getattr(self, 'live_values', None)
        if previous == values:
            return
        self.live_values = dict(values)
        v = self.vision
        v.lower = np.array([values['h_min'], values['s_min'], values['v_min']],
                           dtype=np.uint8)
        v.upper = np.array([values['h_max'], values['s_max'], values['v_max']],
                           dtype=np.uint8)
        for attr, key in [
            ('roi_start', 'roi_start_ratio'), ('roi_end', 'roi_end_ratio'),
            ('roi_top_half_width_ratio', 'roi_top_half_width_ratio'),
            ('roi_bottom_half_width_ratio', 'roi_bottom_half_width_ratio'),
            ('roi_center_ratio', 'roi_center_x_ratio'),
            ('scan_rows', 'scan_rows'), ('band', 'scan_half_band'),
            ('min_run_px', 'min_run_px'), ('max_run_ratio', 'max_run_ratio'),
            ('band_coverage', 'band_coverage'), ('blur_kernel', 'blur_kernel'),
            ('open_kernel', 'open_kernel'), ('close_kernel', 'close_kernel'),
            ('min_gap_ratio', 'min_lane_gap_ratio'),
            ('max_gap_ratio', 'max_lane_gap_ratio'),
            ('far_ratio', 'far_y_ratio'),
            ('lookahead_ratio', 'lookahead_y_ratio'),
            ('near_ratio', 'near_y_ratio'),
            ('memory_seconds', 'single_boundary_memory_sec'),
            ('min_center_points', 'min_center_points'),
            ('min_pair_rows', 'min_pair_rows'),
            ('min_span_ratio', 'min_span_ratio')]:
            setattr(v, attr, values[key])
        c = self.controller
        for attr, key in [
            ('kp', 'kp'), ('kd', 'kd'), ('k_heading', 'k_heading'),
            ('max_angular', 'max_angular'),
            ('angular_rate_limit', 'angular_rate_limit'),
            ('max_speed', 'max_speed'), ('min_speed', 'min_speed'),
            ('corner_threshold', 'corner_threshold'),
            ('corner_speed_cap', 'corner_speed_cap'),
            ('one_boundary_speed_cap', 'single_boundary_speed_cap'),
            ('acceleration_limit', 'acceleration_limit'),
            ('camera_center_ratio', 'camera_center_x_ratio')]:
            setattr(c, attr, values[key])
        self.timeout = values['camera_timeout_sec']
        self.preview_controller = copy.deepcopy(c)
        self.controller.reset()
        # Reusing old lane width after changing HSV/ROI can lead to unsafe
        # single-boundary estimates, so deliberately clear stored history.
        self.vision.reset()

    def save_tuning(self):
        """Save valid ROS2 YAML without a nonstandard PyYAML dependency."""
        values = getattr(self, 'live_values', None)
        if values is None:
            values = self.startup_values
        all_values = {name: self.get_parameter(name).value for name in self.all_defaults}
        for key, val in values.items():
            if key in ('h_min', 's_min', 'v_min', 'h_max', 's_max', 'v_max'):
                continue
            all_values[key] = val
        all_values['hsv_lower'] = [int(values[k]) for k in ('h_min', 's_min', 'v_min')]
        all_values['hsv_upper'] = [int(values[k]) for k in ('h_max', 's_max', 'v_max')]
        # Deliberately save disabled, no GUI, and tune_mode off for later deployment.
        # The vehicle will still NOT move until explicitly enabled.
        all_values['tune_mode'] = False
        all_values['enabled_on_start'] = False
        all_values['show_debug'] = False
        import json
        def yaml_scalar(obj):
            if isinstance(obj, str): return json.dumps(obj, ensure_ascii=False)
            if isinstance(obj, bool): return str(obj).lower()
            if isinstance(obj, list): return '['+', '.join(yaml_scalar(x) for x in obj)+']'
            if isinstance(obj, (int, float)): return str(obj)
            raise TypeError(type(obj))
        out = ['yellow_zigzag_follower:', '  ros__parameters:']
        out += [f'    {key}: {yaml_scalar(value)}'
                for key, value in sorted(all_values.items())]
        path = Path(self.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(out)+'\n', encoding='utf-8')
        self.get_logger().info(f'SAVED parameters to: {path}')
        print(f'\nSAVED: {path}\nTo preview again: -p tune_mode:=true '
              f'--params-file {path}\n')

    def sample_hsv(self, event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN or self._debug_image is None:
            return
        img = self._debug_image
        px = int(x / max(self._debug_scale, 1e-6))
        py = int(y / max(self._debug_scale, 1e-6))
        if not 0 <= px < img.shape[1] or not 0 <= py < img.shape[0]:
            return
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rad = 3
        patch = hsv[max(0,py-rad):py+rad+1, max(0,px-rad):px+rad+1]
        med = np.median(patch.reshape(-1, 3), axis=0).astype(int)
        self._last_clicked = (px, py, med.tolist())
        self.get_logger().info(f'Sampled yellow? at ({px},{py}): '
                               f'median H/S/V = {med.tolist()}; '
                               'adjust HSV sliders to include this sample.')

    def publish_command(self, speed, angular):
        if self.tune_mode:
            return  # HARD SAFETY GUARD for all publication paths
        command = Twist()
        command.linear.x = float(speed)
        command.angular.z = float(angular)
        self.cmd_pub.publish(command)

    def stop(self):
        self.publish_command(0., 0.)
        self.controller.reset()

    def warn_throttled(self, message):
        now = time.monotonic()
        if now-self.last_warning > 2.0:
            self.get_logger().warn(message)
            self.last_warning = now

    def enable_callback(self, msg):
        if self.tune_mode:
            return  # ignore enable messages during tuning
        requested = bool(msg.data)
        if requested == self.enabled:
            return
        if not requested:
            # One zero command at hand-off; no velocity publications afterwards.
            # The supervisor must ensure controllers never publish concurrently.
            self.stop()
            self.enabled = False
            self.vision.reset()
            self.get_logger().info('Yellow control disabled (no more commands).')
        else:
            self.enabled = True
            self.vision.reset()
            self.controller.reset()
            self.last_image_time = time.monotonic()
            self.get_logger().info('Yellow control enabled.')

    def camera_watchdog(self):
        if self.enabled and time.monotonic()-self.last_image_time > self.timeout:
            self.stop()
            self.warn_throttled('Camera timeout: emergency stop (yellow mode).')

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except (CvBridgeError, ValueError) as exc:
            if self.enabled:
                self.stop()
            self.warn_throttled(f'Camera conversion error: {exc}')
            return
        now = time.monotonic()
        self.last_image_time = now
        if self.trackbars is not None:
            try:
                self.apply_live_values(self.trackbars.read())
            except cv2.error as exc:
                self.warn_throttled(f'Trackbar window closed? {exc}')
        obs = self.vision.analyze(image, now=now)
        self.detect_pub.publish(Bool(data=bool(obs.valid and obs.paired_rows >= self.vision.min_pair_rows)))
        speed, angular, error = 0., 0., 0.
        if self.enabled:
            if obs.valid:
                speed, angular, error = self.controller.command(obs, now=now)
                self.publish_command(speed, angular)
            else:
                self.stop()
                self.warn_throttled('Yellow lane unavailable: ' + obs.reason)

        if self.show_debug:
            preview_speed, preview_turn, preview_error = 0.0, 0.0, 0.0
            if not self.enabled:
                preview_speed, preview_turn, preview_error = self.preview_controller.command(obs, now)
            self.draw_debug(image, obs, speed, angular, error,
                            preview_speed, preview_turn, preview_error)

    def draw_debug(self, image, obs, speed, angular, error,
                   preview_speed=0., preview_turn=0., preview_error=0.):
        h, w = image.shape[:2]
        # Strip ALL pixels outside the active trapezoid from visualisation, too.
        # What you see is the same ROI used by the image-processing algorithm.
        polygon = self.vision.roi_polygon(h, w)
        active = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(active, [polygon], 255)
        debug = cv2.bitwise_and(image, image, mask=active)
        cv2.polylines(debug, [polygon], True, (255, 190, 65), 2)
        if self.trackbars is not None:
            # Pixel samples use raw camera frames, not annotations.
            self._debug_image = image.copy()
        cv2.line(debug, (int(w*self.controller.camera_center_ratio), 0),
                 (int(w*self.controller.camera_center_ratio), h-1), (255, 255, 255), 1)
        # Show where the horizontal scans intersect the image.
        if self.tune_mode:
            rows = np.linspace(int(h*self.vision.roi_end)-1,
                               int(h*self.vision.roi_start)+self.vision.band+1,
                               self.vision.scan_rows).astype(int)
            for y in rows:
                cv2.line(debug, (0,int(y)), (w-1,int(y)), (120,120,120),1)
        for p in obs.points:
            clr = (0, 255, 0) if p.source == 'pair' else (0, 165, 255)
            if p.left is not None:
                cv2.circle(debug, (int(p.left), p.y), 3, (255, 0, 255), -1)
            if p.right is not None:
                cv2.circle(debug, (int(p.right), p.y), 3, (255, 0, 255), -1)
            cv2.circle(debug, (int(p.x), p.y), 4, clr, -1)
        for p1, p2 in zip(obs.points[:-1], obs.points[1:]):
            cv2.line(debug, (int(p1.x), p1.y), (int(p2.x), p2.y),
                     (0, 255, 0), 2)
        if obs.valid and obs.target_x is not None:
            cv2.circle(debug, (int(obs.target_x), obs.target_y), 8, (0, 0, 255), 2)
            cv2.line(debug, (0, obs.target_y), (w-1, obs.target_y), (160, 160, 160), 1)
        text1 = f'{"TUNING" if self.tune_mode else ("ACTIVE" if self.enabled else "PREVIEW")} {obs.reason} pairs={obs.paired_rows} one={obs.single_rows}'
        text2 = (f'mask={obs.mask_fraction*100:.2f}% runs={obs.run_count} '
                 f'e={preview_error if self.tune_mode else error:+.2f} '
                 f'heading={obs.heading:+.2f} corner={obs.corner:.2f}')
        text3 = (f'PREDICT ONLY v={preview_speed:.2f} w={preview_turn:+.2f}'
                 if self.tune_mode else f'ACTUAL v={speed:.2f} w={angular:+.2f}')
        cv2.putText(debug, text1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (0,255,255), 2)
        cv2.putText(debug, text2, (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                    .48, (0,255,255), 2)
        cv2.putText(debug, text3, (10, 76), cv2.FONT_HERSHEY_SIMPLEX,
                    .48, (0,255,255), 2)
        if self.tune_mode:
            cv2.putText(debug, 'S=SAVE  R=RESET  Q=QUIT  CLICK=HSV SAMPLE',
                        (10,h-12), cv2.FONT_HERSHEY_SIMPLEX,.47,(0,255,255),1)
        if self._last_clicked:
            px,py,hsv = self._last_clicked
            cv2.circle(debug, (px,py),6,(0,0,255),2)
            cv2.putText(debug, f'H/S/V={hsv}', (10,100),
                        cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),2)
        # Explicitly scale the debug window to simplify SSH mouse selection.
        display_w = min(w, 960)
        self._debug_scale = display_w / float(w)
        display_h = max(1, round(h * self._debug_scale))
        display = cv2.resize(debug,(display_w,display_h)) if display_w != w else debug
        if not self._debug_ready:
            cv2.namedWindow(self._debug_window, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self._mask_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._debug_window,display_w,display_h)
            cv2.resizeWindow(self._mask_window,display_w,display_h)
            if self.trackbars is not None:
                cv2.setMouseCallback(self._debug_window,self.sample_hsv)
            self._debug_ready = True
        cv2.imshow(self._debug_window, display)
        full_mask = np.zeros((h, w), dtype=np.uint8)
        roi_h = obs.mask.shape[0]
        full_mask[obs.roi_top:obs.roi_top+roi_h, :] = obs.mask
        if display_w != w:
            full_mask = cv2.resize(full_mask, (display_w, display_h),
                                   interpolation=cv2.INTER_NEAREST)
        cv2.imshow(self._mask_window, full_mask)
        key = cv2.waitKey(1) & 0xFF
        if self.tune_mode:
            if key in (ord('s'),ord('S')):
                self.save_tuning()
            elif key in (ord('r'),ord('R')):
                self.trackbars.restore()
                self.get_logger().info('Restored startup slider values.')
            elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
                self.trackbars.set_page(key-ord('1'))
                self.trackbars.draw()
            elif key in (ord('q'),ord('Q'),27):
                self.exit_requested = True


def main(args=None):
    if rclpy is None:
        raise RuntimeError('ROS2 rclpy / cv_bridge are required to run this node.')
    rclpy.init(args=args)
    node = None
    try:
        node = YellowFollower()
        while rclpy.ok() and not node.exit_requested:
            rclpy.spin_once(node, timeout_sec=0.10)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if node.enabled and rclpy.ok():
                try:
                    node.stop()
                except Exception as exc:
                    print(f'Warning: stop publish unavailable during shutdown: {exc}')
            if node.show_debug:
                cv2.destroyAllWindows()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
