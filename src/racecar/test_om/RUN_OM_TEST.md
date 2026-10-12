本次检查发现 root 的 .bashrc 配置了 /usr/local/miniconda3 和 CANN；该 Conda base 是 Python 3.9.2，已安装 ais_bench、aclruntime。davinci-mini 可以访问它，无须切换 root。已用它实际加载本项目的 OM 并进行推理，输出 1×2×240×320 float32，数值有限。

但 ROS 2 Humble 的 rclpy 是 Python 3.10 扩展，Python 3.9 导入失败。test_infer.py 同时依赖 ROS 和 InferSession，所以需要兼容 Python 3.10 的推理运行库。新增环境 /home/davinci-mini/.venvs/racecar_om 继承系统 ROS/OpenCV/NumPy，推理依赖安装在这个独立环境中。

test_infer.py 原文件为 GB18030 编码，没有编码声明，Python 在第 52 行报编码错误。已备份为 test_infer.py.bak_20261010_environment，再转换为 UTF-8。模型预处理、五条扫描线、边界配对、中心点提取、转向和速度算法保持原样。

新增 ROS 参数、enable_motion 开关（默认 false）、BEST_EFFORT 深度 1 图像订阅、/yellow_test/mask 和 /yellow_test/overlay 图像输出、--check-model 模型独立验证，以及退出时发送零命令并释放模型资源。show_debug 默认为 false，SSH 无桌面时使用图像话题查看。

以下命令均在小车上执行，无须 colcon 重编译该 Python 文件。

每个新终端先激活：

```bash
source /home/davinci-mini/racecar/src/racecar/test_om/activate_om.sh
```

独立验证模型，不发布车辆控制：

```bash
python3 /home/davinci-mini/racecar/src/racecar/test_om/test_infer.py --check-model
```

终端 1，摄像头（已有摄像头节点时不重复启动）：

```bash
source /home/davinci-mini/racecar/src/racecar/test_om/activate_om.sh
ros2 launch racecar camer.launch.py
```

终端 2，先只看识别效果，命令话题重映射到测试话题：

```bash
source /home/davinci-mini/racecar/src/racecar/test_om/activate_om.sh
python3 -u /home/davinci-mini/racecar/src/racecar/test_om/test_infer.py \
  --ros-args -p enable_motion:=false \
  -r /car_cmd_vel:=/yellow_test/dry_cmd_vel
```

有图形桌面的终端：

```bash
source /home/davinci-mini/racecar/src/racecar/test_om/activate_om.sh
ros2 run rqt_image_view rqt_image_view
```

选择 /yellow_test/overlay 查看扫描线、中心点和状态；选择 /yellow_test/mask 查看模型分割；选择 /image_raw 查看原图。SSH 无图形桌面时不要直接启用 show_debug:=true。

识别确认后，先 Ctrl+C 关闭终端 2 的测试程序。把车放在黄线中间，使用以下两个终端测试运动。不要同时启动 hybrid、雷达循迹、其他黄线节点或遥控发布者。

终端 3，底盘（已经启动时不重复）：

```bash
source /home/davinci-mini/racecar/src/racecar/test_om/activate_om.sh
ros2 run racecar_driver racecar_driver_node
```

终端 2，启用黄线运动并保存终端日志：

```bash
source /home/davinci-mini/racecar/src/racecar/test_om/activate_om.sh
python3 -u /home/davinci-mini/racecar/src/racecar/test_om/test_infer.py \
  --ros-args -p enable_motion:=true \
  2>&1 | tee "$HOME/yellow_test_$(date +%Y%m%d_%H%M%S).log"
```

Ctrl+C 结束循迹程序时会发送三次零控制。确认循迹程序结束后，如果需要再发送一次停车：

```bash
ros2 topic pub --once /car_cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.0}, angular: {z: 0.0}}'
```

实际底盘代码把 linear.x 转为 PWM=1500+100×linear.x，把 angular.z 转为舵机 PWM=1500−1300×angular.z。默认直道 0.30→1530，弯道 0.18→1518，缺边降速 0.10→1510；这些值不等于已标定的 m/s。1510/1518 如果处于你的电调死区，可能识别正常但车不走。可通过 -p speed_straight:=0.35 -p speed_corner:=0.30 -p speed_uncertain:=0.25 调参，但应先验证识别和转向方向，再逐步提高速度。不能仅凭模型推理成功证明实车能够正确循迹。

原文件恢复：

```bash
cp /home/davinci-mini/racecar/src/racecar/test_om/test_infer.py.bak_20261010_environment \
   /home/davinci-mini/racecar/src/racecar/test_om/test_infer.py
```

恢复后原编码错误也会恢复，且原脚本没有运动开关。


Verification on 2026-10-10: Python 3.10 model check passed (5 runs, mean 10.5 ms); 328 mask messages, 326 overlay messages, 334 dry commands (all zero). Model file mode changed 664 -> 644 to satisfy ais_bench input-file checks. Current camera view yielded LOST / zero valid boundary rows. Actual driving has not been tested. Test camera and follower were stopped after verification.
