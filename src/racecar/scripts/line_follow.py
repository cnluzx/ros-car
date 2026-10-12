#!/usr/bin/env python3
"""Run PP-LiteSeg and turn its line mask into the vision tracking command."""

from pathlib import Path
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


ACLLITE_DIR = "/usr/local/Ascend/thirdpart/aarch64/acllite"
if ACLLITE_DIR not in sys.path:
    sys.path.insert(0, ACLLITE_DIR)

from acllite_model import AclLiteModel  # noqa: E402
from acllite_resource import AclLiteResource  # noqa: E402


MODEL_WIDTH = 320
MODEL_HEIGHT = 240


class LineSegmentationNode(Node):
    def __init__(self):
        super().__init__("line_segmentation")
        default_model = "/home/davinci-mini/racecar/src/racecar/test_om/line_seg_v2_310B4_int8.om"
        self.declare_parameter("model_path", default_model)
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("mask_topic", "/line_seg/mask")
        self.declare_parameter("overlay_topic", "/line_seg/overlay")
        self.declare_parameter("cmd_topic", "/vision_cmd_raw")
        self.declare_parameter("input_scale", 1.0 / 255.0)
        self.declare_parameter("input_mean", 0.5)
        self.declare_parameter("input_std", 0.5)
        self.declare_parameter("use_rgb", False)
        self.declare_parameter("roi_top_ratio", 0.20)
        self.declare_parameter("min_component_area", 10)
        self.declare_parameter("min_component_height", 4)
        self.declare_parameter("min_bilateral_separation", 35)
        self.declare_parameter("valid_confirm_frames", 2)
        self.declare_parameter("invalid_confirm_frames", 3)
        self.declare_parameter("debug_visualization", False)

        self.model_path = self.get_parameter("model_path").value
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"OM model not found: {self.model_path}")

        self.input_scale = float(self.get_parameter("input_scale").value)
        self.input_mean = float(self.get_parameter("input_mean").value)
        self.input_std = float(self.get_parameter("input_std").value)
        if self.input_std == 0.0:
            raise ValueError("input_std must not be zero")
        self.use_rgb = bool(self.get_parameter("use_rgb").value)
        self.roi_top_ratio = float(self.get_parameter("roi_top_ratio").value)
        self.min_component_area = int(self.get_parameter("min_component_area").value)
        self.min_component_height = int(self.get_parameter("min_component_height").value)
        self.min_bilateral_separation = int(
            self.get_parameter("min_bilateral_separation").value)
        self.valid_confirm_frames = max(
            1, int(self.get_parameter("valid_confirm_frames").value))
        self.invalid_confirm_frames = max(
            1, int(self.get_parameter("invalid_confirm_frames").value))
        self.debug_visualization = bool(
            self.get_parameter("debug_visualization").value)
        self.bridge = CvBridge()
        qos = QoSProfile(depth=2)
        self.mask_pub = self.create_publisher(
            Image, self.get_parameter("mask_topic").value, qos)
        self.overlay_pub = self.create_publisher(
            Image, self.get_parameter("overlay_topic").value, qos)
        self.cmd_pub = self.create_publisher(
            Twist, self.get_parameter("cmd_topic").value, qos)
        self.yellow_valid_pub = self.create_publisher(
            Bool, "/yellow_tracking_valid", qos)
        self.yellow_bilateral_pub = self.create_publisher(
            Bool, "/yellow_tracking_bilateral", qos)
        self.yellow_mask_debug_pub = self.create_publisher(
            Image, "/yellow_mask_debug", qos)
        self.yellow_detection_debug_pub = self.create_publisher(
            Image, "/yellow_detection_debug", qos)
        self.subscription = self.create_subscription(
            Image, self.get_parameter("image_topic").value,
            self.image_callback, qos)

        self.resource = AclLiteResource(device_id=0)
        self.resource.init()
        self.model = AclLiteModel(self.model_path)
        self.frames_published = 0
        self.line_valid = {"left": False, "right": False}
        self.hit_count = {"left": 0, "right": 0}
        self.lost_count = {"left": 0, "right": 0}
        self.last_cx = {"left": None, "right": None}
        self.get_logger().info(
            "line_seg OM is ready: input=1x3x240x320 float32, output=1x2x240x320")
        self.get_logger().info(
            "publishing segmentation diagnostics and vision tracking commands")

    def preprocess(self, image):
        resized = cv2.resize(image, (MODEL_WIDTH, MODEL_HEIGHT),
                             interpolation=cv2.INTER_LINEAR)
        if self.use_rgb:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = resized.astype(np.float32) * self.input_scale
        tensor = (tensor - self.input_mean) / self.input_std
        return np.ascontiguousarray(np.transpose(tensor, (2, 0, 1))[None, ...])

    def update_line_state(self, side, raw_valid, cx):
        if raw_valid:
            self.hit_count[side] += 1
            self.lost_count[side] = 0
            if self.hit_count[side] >= self.valid_confirm_frames:
                self.line_valid[side] = True
                self.last_cx[side] = cx
        else:
            self.hit_count[side] = 0
            self.lost_count[side] += 1
            if self.lost_count[side] >= self.invalid_confirm_frames:
                self.line_valid[side] = False
        return self.last_cx[side] if self.line_valid[side] else None

    def track_mask(self, image, mask, header):
        height, width = mask.shape
        roi_top = int(height * self.roi_top_ratio)
        roi = mask[roi_top:, :]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
        label_count, _, stats, centroids = cv2.connectedComponentsWithStats(roi, 8)

        candidates = []
        for index in range(1, label_count):
            x, y, comp_width, comp_height, area = stats[index]
            if area < self.min_component_area or comp_height < self.min_component_height:
                continue
            cx = int(centroids[index][0])
            confidence = min(
                1.0,
                0.6 * area / (self.min_component_area * 4.0) +
                0.4 * comp_height / (self.min_component_height * 4.0))
            if confidence < 0.25:
                continue
            candidates.append({
                "x": x, "y": y + roi_top, "width": comp_width,
                "height": comp_height, "cx": cx,
                "score": confidence * (area + comp_height * 2)})

        left_candidates = [item for item in candidates if item["cx"] < width // 2]
        right_candidates = [item for item in candidates if item["cx"] >= width // 2]
        left = max(left_candidates, key=lambda item: item["score"]) if left_candidates else None
        right = max(right_candidates, key=lambda item: item["score"]) if right_candidates else None
        if (left is None or right is None) and len(candidates) >= 2:
            ordered = sorted(candidates, key=lambda item: item["cx"])
            if ordered[-1]["cx"] - ordered[0]["cx"] >= self.min_bilateral_separation:
                left, right = ordered[0], ordered[-1]

        cx_left = self.update_line_state("left", left is not None,
                                         left["cx"] if left else None)
        cx_right = self.update_line_state("right", right is not None,
                                          right["cx"] if right else None)
        valid = self.line_valid["left"] or self.line_valid["right"]
        bilateral = self.line_valid["left"] and self.line_valid["right"]
        valid_msg = Bool()
        valid_msg.data = valid
        self.yellow_valid_pub.publish(valid_msg)
        bilateral_msg = Bool()
        bilateral_msg.data = bilateral
        self.yellow_bilateral_pub.publish(bilateral_msg)

        if cx_left is not None and cx_right is not None:
            error = (cx_left + cx_right) // 2 - width // 2
            kp = 0.02
        elif cx_left is not None:
            error = cx_left - width // 2
            kp = 0.12
        elif cx_right is not None:
            error = cx_right - width // 2
            kp = 0.12
        else:
            error = 0
            kp = 0.12

        command = Twist()
        if valid:
            command.angular.z = -kp * error
            if abs(error) < 10:
                command.linear.x = 0.30
            elif abs(error) < 50:
                command.linear.x = 0.25
            elif abs(error) < 100:
                command.linear.x = 0.20
        self.cmd_pub.publish(command)

        debug_mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        mask_message = self.bridge.cv2_to_imgmsg(debug_mask, encoding="mono8")
        mask_message.header = header
        self.yellow_mask_debug_pub.publish(mask_message)
        if self.debug_visualization:
            debug = image.copy()
            scale_x = debug.shape[1] / float(width)
            scale_y = debug.shape[0] / float(height)
            debug_top = int(debug.shape[0] * self.roi_top_ratio)
            cv2.rectangle(debug, (0, debug_top),
                          (debug.shape[1] - 1, debug.shape[0] - 1), (255, 255, 0), 1)
            for candidate in candidates:
                selected = candidate is left or candidate is right
                color = (0, 255, 0) if selected else (0, 165, 255)
                x1 = int(candidate["x"] * scale_x)
                y1 = int(candidate["y"] * scale_y)
                x2 = int((candidate["x"] + candidate["width"]) * scale_x)
                y2 = int((candidate["y"] + candidate["height"]) * scale_y)
                cv2.rectangle(debug, (x1, y1), (x2, y2), color, 1)
            cv2.putText(debug, "L:%d R:%d V:%d B:%d" % (
                self.line_valid["left"], self.line_valid["right"], valid, bilateral),
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            debug_message = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_message.header = header
            self.yellow_detection_debug_pub.publish(debug_message)

    def image_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            outputs = self.model.execute([self.preprocess(image)])
            if not outputs or outputs[0].shape != (1, 2, MODEL_HEIGHT, MODEL_WIDTH):
                raise RuntimeError(
                    f"unexpected model output: {[x.shape for x in outputs] if outputs else outputs}")

            classes = np.argmax(outputs[0][0], axis=0).astype(np.uint8)
            mask = (classes * 255).astype(np.uint8)
            self.track_mask(image, mask, message.header)
            full_mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            overlay = image.copy()
            overlay[full_mask > 0] = (0, 255, 0)
            overlay = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)

            mask_message = self.bridge.cv2_to_imgmsg(full_mask, encoding="mono8")
            mask_message.header = message.header
            self.mask_pub.publish(mask_message)
            overlay_message = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            overlay_message.header = message.header
            self.overlay_pub.publish(overlay_message)
            self.frames_published += 1
            if self.frames_published % 30 == 0:
                self.get_logger().info(
                    f"published {self.frames_published} segmentation frames")
        except Exception as exc:
            self.get_logger().error(f"inference failed: {exc}", throttle_duration_sec=2.0)

    def destroy_node(self):
        if hasattr(self, "model"):
            self.model.destroy()
        return super().destroy_node()


def main():
    rclpy.init()
    node = LineSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
