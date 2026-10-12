#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_vision.py —— 摄像头与视觉模组测试工具

功能：
  1. 设备测试：检查 /dev/video0 是否正常打开，读取分辨率与实测帧率
  2. 样张保存：抓取一帧真实图片保存为 camera_test.jpg
  3. Web 网页图传监视器 (端口 8081)：
     在浏览器打开 http://<小车IP>:8081 即可在手机或电脑上实时查看摄像头画面！
  4. ROS 2 话题测试：若运行了 usb_cam 节点，可订阅 /image_raw 监测帧率
"""

import argparse
import sys
import time

try:
    import cv2
except ImportError:
    print("[错误] 未安装 opencv-python (cv2)。请执行: pip install opencv-python")
    sys.exit(1)


def test_device(device_id=0):
    print(f"\n>>> 正在尝试打开摄像头 /dev/video{device_id} ...")
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"[!] 无法打开 /dev/video{device_id}。请检查摄像头连接或权限 (ls -l /dev/video*)")
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  [OK] 成功打开！硬件默认分辨率: {w}x{h}, 标称FPS: {fps}")

    # 读取 20 帧测速
    print("  正在连续捕获 20 帧测试传输速率...")
    t0 = time.time()
    frames = 0
    last_frame = None
    for _ in range(20):
        ret, frame = cap.read()
        if ret:
            frames += 1
            last_frame = frame
    dt = time.time() - t0
    real_fps = frames / max(dt, 1e-4)
    print(f"  [OK] 实测采集帧率: {real_fps:.1f} FPS")

    if last_frame is not None:
        save_path = "camera_test.jpg"
        cv2.imwrite(save_path, last_frame)
        print(f"  [OK] 已抓取测试样张并保存为: {save_path}")

    cap.release()
    return True


def start_web_stream(device_id=0, port=8081):
    """启动极轻量 Web MJPEG 视频流，浏览器直接看画面"""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"[!] 无法打开 /dev/video{device_id}")
        return

    # 稍微调低分辨率保证流畅度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    class StreamHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
                self.end_headers()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # 画面加文字时间戳
                    cv2.putText(frame, time.strftime("%H:%M:%S"), (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    _, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                    self.wfile.write(b'--jpgboundary\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg.tobytes())
                    self.wfile.write(b'\r\n')
                    time.sleep(0.04)

        def log_message(self, format, *args):
            return

    server = HTTPServer(('0.0.0.0', port), StreamHandler)
    print("\n" + "=" * 54)
    print(f" 摄像头 Web 图传服务已启动！")
    print(f" 请在电脑或手机浏览器打开: http://<小车IP>:{port}")
    print(f" 例如: http://192.168.5.4:{port}")
    print(" 按 Ctrl+C 停止图传...")
    print("=" * 54)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        cap.release()
        print("\n[*] 图传服务已停止。")


def main():
    print("\n" + "=" * 52)
    print("           摄像头与视觉模组测试工具")
    print("=" * 52)
    print(" [1] 硬件连通性与捕获测试 (抓拍测试样张 camera_test.jpg)")
    print(" [2] 启动 Web 网页实时画面监视器 (端口 8081)")
    print(" [0] 退出")
    print("=" * 52)
    choice = input("请选择 [0-2]: ").strip()
    if choice == '1':
        test_device(0)
    elif choice == '2':
        start_web_stream(0, 8081)


if __name__ == '__main__':
    main()
