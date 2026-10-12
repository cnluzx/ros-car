import csv
import math
import os
import signal
import time
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

try:
    from geometry_msgs.msg import Twist
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import LaserScan

    HAVE_ROS = True
except ImportError:
    HAVE_ROS = False
    Node = object