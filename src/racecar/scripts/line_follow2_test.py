#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
import time
import signal
from pathlib import Path
from rclpy.signals import SignalHandlerOptions
import acl

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


MODEL_PATH = (
    "/home/davinci-mini/racecar/src/racecar/test_om/"
    "line_seg_v2_310B4_int8.om"
)
DEVICE_ID = 0
INPUT_SHAPE = (1, 3, 240, 320)
OUTPUT_SHAPE = (1, 2, 240, 320)
# Set True after verifying the model and steering.
ENABLE_MOTION = False


def check(ret, name):
    if ret != 0:
        raise RuntimeError(f"{name} failed, ret={ret}; details=" + str(acl.get_recent_err_msg() if hasattr(acl, "get_recent_err_msg") else "unavailable"))

class Follower(Node):
    def __init__(self):
        super().__init__('follower')
        self.bridge = CvBridge()
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.enable_motion = bool(self.declare_parameter('enable_motion', False).value)
        self.preview = bool(self.declare_parameter('preview', False).value)
        self.save_debug_images = bool(
            self.declare_parameter('save_debug_images', True).value
        )
        self.debug_dir = Path(self.declare_parameter(
            'debug_dir', '/home/davinci-mini/racecar/debug/line_follow2_test'
        ).value)
        self.debug_frame_count = 0
        if self.save_debug_images:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.max_linear = float(self.declare_parameter('max_linear', 0.25).value)
        self.max_angular = float(self.declare_parameter('max_angular', 0.30).value)
        if not (0.0 < self.max_linear <= 0.30 and 0.0 < self.max_angular <= 0.50):
            raise ValueError('max_linear must be (0,0.30]; max_angular must be (0,0.50]')
        self.last_good = time.monotonic()
        self.lane_width_px = None
        self.last_log = 0.0
        self.last_frame_wall = None
        self.last_stale_log = 0.0
        self.mat = None
        self.model_path = str(self.declare_parameter('model_path', MODEL_PATH).value)
        
        # ACL resource handles
        self.acl_ready = False
        self.device_ready = False
        self.acl_context = None
        self.model_id = None
        self.model_desc = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_ptr = None
        self.output_ptr = None
        self.input_buffer = None
        self.output_buffer = None
        self.closed = False
        # Initialize ACL and load OM once.
        try:
            self.init_om()
        except Exception:
            self.release_om()
            raise
        
        self.image_sub = self.create_subscription(
                Image,
                '/image_raw',#订阅摄像头话题
                self.image_callback,
                qos)
        self.cmd_vel_pub = self.create_publisher(Twist, 'car_cmd_vel', 10)
        self.mask_pub = self.create_publisher(Image, '/yellow_test/mask', qos)
        self.overlay_pub = self.create_publisher(Image, '/yellow_test/overlay', qos)
        self.create_timer(0.1, self.watchdog)
        self.get_logger().info('Yellow-line test: enable_motion=%s, limits=(%.2f, %.2f)' % (self.enable_motion, self.max_linear, self.max_angular))
        self.twist = Twist()
        self.tmp = 0


    # ====================================
    # 1. Initialize ACL and load OM
    # ====================================
    
    def init_om(self):
        # 1. Initialize ACL
        check(acl.init(), "acl.init")
        self.acl_ready = True

        # 2. Set device
        check(
            acl.rt.set_device(DEVICE_ID),
            "set_device"
        )
        self.device_ready = True

        # 3. Create context
        self.acl_context, ret = acl.rt.create_context(
            DEVICE_ID
        )
        check(ret, "create_context")

        # 4. Check run mode
        self.run_mode, ret = acl.rt.get_run_mode()
        check(ret, "get_run_mode")

        self.get_logger().info(
            f"ACL run mode: {self.run_mode}"
        )

        # 0 = ACL_DEVICE, 1 = ACL_HOST
        if self.run_mode not in (0, 1):
            raise RuntimeError(
                f"Unsupported ACL mode: {self.run_mode}"
            )

        # 5. Load OM
        # Use the parameterized path so a test model can be selected at launch.
        self.model_id, ret = acl.mdl.load_from_file(
            self.model_path
        )
        check(ret, "load_from_file")

        # 6. Create model description
        self.model_desc = acl.mdl.create_desc()

        if self.model_desc is None:
            raise RuntimeError("create_desc failed")

        check(
            acl.mdl.get_desc(
                self.model_desc,
                self.model_id
            ),
            "get_desc"
        )

        # 7. Check input/output counts
        input_num = acl.mdl.get_num_inputs(
            self.model_desc
        )
        output_num = acl.mdl.get_num_outputs(
            self.model_desc
        )

        if input_num != 1:
            raise ValueError(
                f"Expected 1 input, got {input_num}"
            )

        if output_num != 1:
            raise ValueError(
                f"Expected 1 output, got {output_num}"
            )

        # 8. Check model shapes
        in_dims, ret = acl.mdl.get_input_dims(
            self.model_desc, 0
        )
        check(ret, "get_input_dims")

        out_dims, ret = acl.mdl.get_output_dims(
            self.model_desc, 0
        )
        check(ret, "get_output_dims")

        if tuple(in_dims["dims"]) != INPUT_SHAPE:
            raise ValueError(
                f"Unexpected input: {in_dims['dims']}"
            )

        if tuple(out_dims["dims"]) != OUTPUT_SHAPE:
            raise ValueError(
                f"Unexpected output: {out_dims['dims']}"
            )

        # 9. Check float32 datatype
        input_dtype = acl.mdl.get_input_data_type(
            self.model_desc, 0
        )
        output_dtype = acl.mdl.get_output_data_type(
            self.model_desc, 0
        )

        if input_dtype != 0:
            raise ValueError(
                f"Expected float32 input, got {input_dtype}"
            )

        if output_dtype != 0:
            raise ValueError(
                f"Expected float32 output, got {output_dtype}"
            )

        # 10. Get memory sizes
        self.input_size = int(
            acl.mdl.get_input_size_by_index(
                self.model_desc, 0
            )
        )

        self.output_size = int(
            acl.mdl.get_output_size_by_index(
                self.model_desc, 0
            )
        )

        if self.input_size != int(
            np.prod(INPUT_SHAPE) * 4
        ):
            raise ValueError("Input size mismatch")

        if self.output_size != int(
            np.prod(OUTPUT_SHAPE) * 4
        ):
            raise ValueError("Output size mismatch")

        # 11. Allocate input memory
        self.input_ptr, ret = acl.rt.malloc(
            self.input_size, 0
        )
        check(ret, "malloc input")

        self.input_dataset = acl.mdl.create_dataset()

        if self.input_dataset is None:
            raise RuntimeError(
                "create input dataset failed"
            )

        self.input_buffer = acl.create_data_buffer(
            self.input_ptr,
            self.input_size
        )

        if self.input_buffer is None:
            raise RuntimeError(
                "create input buffer failed"
            )

        _, ret = acl.mdl.add_dataset_buffer(
            self.input_dataset,
            self.input_buffer
        )
        check(ret, "add input buffer")

        # 12. Allocate output memory
        self.output_ptr, ret = acl.rt.malloc(
            self.output_size, 0
        )
        check(ret, "malloc output")

        self.output_dataset = acl.mdl.create_dataset()

        if self.output_dataset is None:
            raise RuntimeError(
                "create output dataset failed"
            )

        self.output_buffer = acl.create_data_buffer(
            self.output_ptr,
            self.output_size
        )

        if self.output_buffer is None:
            raise RuntimeError(
                "create output buffer failed"
            )

        _, ret = acl.mdl.add_dataset_buffer(
            self.output_dataset,
            self.output_buffer
        )
        check(ret, "add output buffer")

        self.get_logger().info(
            f"OM loaded successfully: "
            f"{INPUT_SHAPE} -> {OUTPUT_SHAPE}"
        )

    
    
    def preprocess_om(self, image):

        image = cv2.resize(image, (320, 240),
                           interpolation=cv2.INTER_LINEAR)
        # The model was trained with OpenCV BGR input. Do not swap to RGB.
        # Training normalization: (pixel / 255 - 0.5) / 0.5.
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        # HWC -> CHW -> NCHW
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)

        return np.ascontiguousarray(image)
    
    
    def execute_om(self, input_tensor):
        """
        Input:
            numpy float32 [1, 3, 240, 320]

        Output:
            numpy float32 [1, 2, 240, 320]
        """

        # 1. Validate input
        if input_tensor.shape != INPUT_SHAPE:
            raise ValueError(
                f"Incorrect input shape: {input_tensor.shape}"
            )

        input_tensor = np.ascontiguousarray(
            input_tensor,
            dtype=np.float32
        )

        if input_tensor.nbytes != self.input_size:
            raise ValueError(
                "Incorrect input byte size"
            )

        # 2. Get NumPy input pointer
        input_host_ptr = int(input_tensor.ctypes.data)

        # 3. Copy input into ACL device buffer
        check(
            acl.rt.memcpy(
                self.input_ptr,
                self.input_size,
                input_host_ptr,
                input_tensor.nbytes,
                  1 if self.run_mode == 1 else 3  # HOST_TO_DEVICE or DEVICE_TO_DEVICE
            ),
            "input memcpy"
        )

        # 4. Execute OM
        check(
            acl.mdl.execute(
                self.model_id,
                self.input_dataset,
                self.output_dataset
            ),
            "model execute"
        )

        # 5. Create NumPy output
        output_tensor = np.empty(
            OUTPUT_SHAPE,
            dtype=np.float32
        )

        output_host_ptr = int(output_tensor.ctypes.data)

        # 6. Copy inference result into NumPy
        check(
            acl.rt.memcpy(
                output_host_ptr,
                output_tensor.nbytes,
                self.output_ptr,
                self.output_size,
                  2 if self.run_mode == 1 else 3  # DEVICE_TO_HOST or DEVICE_TO_DEVICE
            ),
            "output memcpy"
        )

        # 7. Return output
        return output_tensor


    # .om模型推理函数
    def predict_mask(self, image):

        original_h, original_w = image.shape[:2]
        input_tensor = self.preprocess_om(image)
        output = self.execute_om(input_tensor)
        # [1, 2, 240, 320] -> [2, 240, 320]
        output = output[0]
        # Per-pixel class prediction
        class_map = np.argmax(output, axis=0)
        # Class 0: background
        # Class 1: yellow line
        mask = ((class_map == 1).astype(np.uint8) * 255)
        mask = cv2.resize(mask,(original_w, original_h),interpolation=cv2.INTER_NEAREST)

        return mask



    def stop(self):
        if rclpy.ok():
            self.cmd_vel_pub.publish(Twist())

    def watchdog(self):
        if time.monotonic() - self.last_good > 0.5:
            self.stop()
        if self.last_frame_wall is None and time.monotonic() - self.last_log > 2.0:
            self.last_log = time.monotonic()
            self.get_logger().warning(
                'No /image_raw frames received; start the camera publisher and check ros2 topic info /image_raw'
            )

    def image_callback(self, msg):
        try:
            self.last_frame_wall = time.monotonic()
            started = time.monotonic()
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            age = self.get_clock().now().nanoseconds * 1e-9 - stamp
            if stamp > 0 and (age > 0.35 or age < -0.1):
                self.stop()
                now = time.monotonic()
                if now - self.last_stale_log >= 2.0:
                    self.last_stale_log = now
                    self.get_logger().warning(
                        'Ignoring stale /image_raw frame: age=%.3fs; check camera and system timestamps' % age
                    )
                return
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            mask = self.predict_mask(image)
            if not np.isfinite(mask).all():
                raise ValueError('Non-finite mask')
            h, w = mask.shape
            left_mask = mask.copy()
            right_mask = mask.copy()
            left_mask[:, w // 2:] = 0
            right_mask[:, :w // 2] = 0
            ml, mr = cv2.moments(left_mask), cv2.moments(right_mask)
            # Reject very small isolated blobs before centroid control.
            left_ok = cv2.countNonZero(left_mask) >= 40
            right_ok = cv2.countNonZero(right_mask) >= 40
            left = ml['m10'] / ml['m00'] if left_ok else None
            right = mr['m10'] / mr['m00'] if right_ok else None
            mid = None
            reason = 'no_line'
            if left is not None and right is not None:
                width = right - left
                if 0.20 * w <= width <= 0.95 * w:
                    self.lane_width_px = width if self.lane_width_px is None else 0.9 * self.lane_width_px + 0.1 * width
                    mid = (left + right) * 0.5
                    reason = 'both_lines'
                else:
                    reason = 'implausible_width'
            elif self.lane_width_px is not None and time.monotonic() - self.last_good < 0.5:
                mid = left + self.lane_width_px * 0.5 if left is not None else right - self.lane_width_px * 0.5 if right is not None else None
                reason = 'single_line_prediction' if mid is not None else 'no_line'
            else:
                reason = 'single_line_without_history' if left is not None or right is not None else 'no_line'
            error = float(mid - w * 0.5) if mid is not None else 0.0
            if mid is not None and (mid < 0 or mid >= w):
                mid = None
                reason = 'centre_outside_image'
            angular = float(np.clip(-0.02 * error, -self.max_angular, self.max_angular)) if mid is not None else 0.0
            speed = (0.30 if abs(error) < 10 else 0.25 if abs(error) < 50 else 0.20 if abs(error) < 100 else 0.0) if mid is not None else 0.0
            speed = min(speed, self.max_linear)
            elapsed = time.monotonic() - started
            if elapsed > 0.35 or (stamp > 0 and age + elapsed > 0.35):
                speed, angular, reason = 0.0, 0.0, 'inference_or_image_stale'
            if reason == 'both_lines':
                self.last_good = time.monotonic()
            command = Twist()
            if self.enable_motion:
                command.linear.x, command.angular.z = speed, angular
            self.cmd_vel_pub.publish(command)
            overlay = image.copy()
            overlay[mask > 0] = (0, 255, 255)
            if mid is not None:
                cv2.line(overlay, (int(mid), 0), (int(mid), h - 1), (0, 255, 0), 1)
            if self.save_debug_images:
                self.debug_frame_count += 1
                if self.debug_frame_count % 15 == 0:
                    cv2.imwrite(str(self.debug_dir / 'latest_overlay.jpg'), overlay)
                    cv2.imwrite(str(self.debug_dir / 'latest_mask.png'), mask)
            for pub, mat, encoding in [(self.mask_pub, mask, 'mono8'), (self.overlay_pub, overlay, 'bgr8')]:
                output = self.bridge.cv2_to_imgmsg(mat, encoding=encoding)
                output.header = msg.header
                pub.publish(output)
            now = time.monotonic()
            if now - self.last_log >= 0.25:
                self.last_log = now
                self.get_logger().info('reason=%s error=%.1f requested=(%.2f,%.2f) published=(%.2f,%.2f) inference_ms=%.1f' % (reason, error, speed, angular, command.linear.x, command.angular.z, elapsed * 1000))
            if self.preview:
                cv2.imshow('yellow_test_overlay', overlay)
                if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                    self.stop()
                    raise KeyboardInterrupt()
        except Exception as exc:
            self.stop()
            self.get_logger().error('Frame processing failed: ' + str(exc))

    def release_om(self):

        if self.closed:
            return
        self.closed = True
        if self.acl_context is not None:
            acl.rt.set_context(self.acl_context)
        if self.input_buffer is not None:
            acl.destroy_data_buffer(self.input_buffer)
        if self.output_buffer is not None:
            acl.destroy_data_buffer(self.output_buffer)
        if self.input_dataset is not None:
            acl.mdl.destroy_dataset(self.input_dataset)
        if self.output_dataset is not None:
            acl.mdl.destroy_dataset(self.output_dataset)
        if self.input_ptr is not None:
            acl.rt.free(self.input_ptr)
        if self.output_ptr is not None:
            acl.rt.free(self.output_ptr)
        if self.model_desc is not None:
            acl.mdl.destroy_desc(self.model_desc)
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
        if self.acl_context is not None:
            acl.rt.destroy_context(self.acl_context)
        if self.device_ready:
            acl.rt.reset_device(DEVICE_ID)
        if self.acl_ready:
            acl.finalize()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    def terminate(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    print('start patrolling')
    follower = None
    try:
        follower = Follower()
        rclpy.spin(follower)
    except KeyboardInterrupt:
        pass
    finally:
        if follower is not None:
            for _ in range(3):
                follower.stop()
                time.sleep(0.05)
            follower.release_om()
            follower.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
