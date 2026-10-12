#!/usr/bin/env python3
"""ROS2 OM entry point. image_callback is retained; inference uses a latest-frame worker.
Only the binary mask and dt enter the image-space controller.
"""
import time
import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ais_bench.infer.interface import InferSession
from pixel_line_core import ImageLineController, PIXEL_DEFAULTS

# Detailed Chinese parameter definitions are in pixel_line_lab/README.md.
CFG = dict(PIXEL_DEFAULTS,
    model_path='/home/davinci-mini/racecar/src/racecar/test_om/line_seg_v2_310B4_int8.om',
    image_topic='/image_raw', cmd_topic='/car_cmd_vel',
    show_debug=False, image_timeout=0.50, log_period=0.50)

class Follower(Node):
    def __init__(self):
        super().__init__('follower')
        self.cfg={key:self.declare_parameter(key,value).value for key,value in CFG.items()}
        self.enable_motion=bool(self.declare_parameter('enable_motion',False).value)
        self.controller=ImageLineController(self.cfg)
        self.bridge=CvBridge();self.lock=threading.Lock();self.exit=threading.Event()
        self.pending=None;self.result=None;self.capture_time=0.;self.image_time=0.
        self.last_log=0.;self.session=None
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST)
        self.image_sub=self.create_subscription(Image,self.cfg['image_topic'],self.image_callback,qos)
        self.cmd_vel_pub=self.create_publisher(Twist,self.cfg['cmd_topic'],1)
        self.mask_pub=self.create_publisher(Image,'/yellow_test/mask',1)
        self.overlay_pub=self.create_publisher(Image,'/yellow_test/overlay',1)
        # Watchdog publication remains responsive if NPU inference is delayed.
        self.timer=self.create_timer(.05,self.publish_latest)
        self.worker=threading.Thread(target=self.inference_loop,daemon=True)
        self.worker.start()
        self.get_logger().info(f'Pixel yellow follower: motion={self.enable_motion}, output={self.cfg["cmd_topic"]}; '
                              'mask=/yellow_test/mask overlay=/yellow_test/overlay')

    def image_callback(self,msg):
        """Keep only the newest ROS camera frame; no inference backlog."""
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        age=max(0.,self.get_clock().now().nanoseconds*1e-9-stamp) if stamp>0 else 0.
        if stamp>0 and age>self.cfg['image_timeout']:return
        with self.lock:
            self.image_time=time.monotonic();self.pending=(msg,self.image_time-age)

    def publish_cmd(self,speed,steer):
        msg=Twist()
        if self.enable_motion:
            msg.linear.x=float(speed);msg.angular.z=float(steer)
        self.cmd_vel_pub.publish(msg)

    def publish_latest(self):
        now=time.monotonic()
        with self.lock:
            result=self.result
            fresh=now-self.image_time<=self.cfg['image_timeout'] and now-self.capture_time<=self.cfg['image_timeout']
        self.publish_cmd(result['speed'],result['steer']) if fresh and result else self.publish_cmd(0.,0.)

    def preprocess_om(self,image):
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb.astype(np.float32).transpose(2,0,1)[None]/255.)

    def predict_mask(self,image):
        output=np.asarray(self.session.infer([self.preprocess_om(image)])[0])
        if output.shape!=(1,2,240,320):raise ValueError(f'Expected output (1,2,240,320), got {output.shape}')
        return (np.argmax(output[0],axis=0)==1).astype('uint8')*255

    def make_overlay(self,image,mask,result):
        canvas=self.controller.overlay(image,mask,result)
        cv2.putText(canvas,f'motion={self.enable_motion}',(4,48),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,255,255),1)
        return canvas

    def inference_loop(self):
        previous_time=None
        try:
            # CANN session/context and infer are created and used in the same thread.
            self.session=InferSession(device_id=0,model_path=self.cfg['model_path'])
            inputs=self.session.get_inputs();outputs=self.session.get_outputs()
            if len(inputs)!=1 or list(inputs[0].shape)!=[1,3,240,320]:
                raise ValueError(f'Unexpected OM inputs: {[v.shape for v in inputs]}')
            if len(outputs)!=1 or list(outputs[0].shape)!=[1,2,240,320]:
                raise ValueError(f'Unexpected OM outputs: {[v.shape for v in outputs]}')
            self.get_logger().info('OM ready: RGB float32 /255 -> NCHW 1x3x240x320; output 1x2x240x320')
            while not self.exit.is_set():
                with self.lock:pending=self.pending;self.pending=None
                if pending is None:self.exit.wait(.005);continue
                msg,captured=pending
                try:
                    image=self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
                    if image.shape!=(240,320,3):raise ValueError(f'Expected 320x240 BGR, got {image.shape}')
                    mask=self.predict_mask(image)
                    if self.exit.is_set():break
                    now=time.monotonic();dt=now-previous_time if previous_time is not None else 1/12
                    if dt>self.cfg['image_timeout']:self.controller=ImageLineController(self.cfg)
                    previous_time=now
                    result=self.controller.step(mask,dt)
                    if time.monotonic()-captured>self.cfg['image_timeout']:
                        result.update(speed=0.,steer=0.,state='INFERENCE_TIMEOUT')
                        self.controller=ImageLineController(self.cfg)
                    overlay=self.make_overlay(image,mask,result)
                    for pub,img,enc in [(self.mask_pub,mask,'mono8'),(self.overlay_pub,overlay,'bgr8')]:
                        packet=self.bridge.cv2_to_imgmsg(img,encoding=enc);packet.header=msg.header;pub.publish(packet)
                    with self.lock:self.result=result;self.capture_time=captured
                    if now-self.last_log>=self.cfg['log_period']:
                        self.get_logger().info(f'{result["state"]} | valid={result["valid_rows"]}/{len(result["rows"])} '
                            f'mask={result["pixels"]} mask_y={result["mask_y"]} '
                            f'armed={result["armed"]} mode={result["input_mode"]} '
                            f'near={result["near_error"]:+.3f} far={result["far_error"]:+.3f} '
                            f'width={result["width_ref_px"]:.1f}px width_slope={result["width_slope_px"]:.2f}px/row '
                            f'blind={result["blind_sec"]:.2f}s steer={result["steer"]:+.3f} '
                            f'PWM={1500+100*result["speed"]:.1f} frame_age={(time.monotonic()-captured)*1000:.1f}ms '
                            f'motion={self.enable_motion}')
                        self.last_log=now
                    if self.cfg['show_debug']:
                        cv2.imshow('yellow_model_mask',mask);cv2.imshow('yellow_line_tracking',overlay);cv2.waitKey(1)
                except Exception as exc:
                    with self.lock:self.result=None
                    self.controller=ImageLineController(self.cfg)
                    if self.exit.is_set():break
                    self.get_logger().error(f'Yellow-line processing failed: {exc}')
        except Exception as exc:self.get_logger().error(f'OM initialization failed: {exc}')
        finally:
            if self.session is not None:
                try:self.session.free_resource()
                except Exception:pass

    def stop(self):
        self.exit.set();self.timer.cancel()
        if rclpy.ok():
            for _ in range(3):self.publish_cmd(0.,0.);time.sleep(.02)
        self.worker.join(timeout=2.)
        if self.cfg['show_debug']:cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args,signal_handler_options=SignalHandlerOptions.NO)
    follower=None
    try:follower=Follower();rclpy.spin(follower)
    except KeyboardInterrupt:pass
    finally:
        if follower is not None:follower.stop();follower.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
