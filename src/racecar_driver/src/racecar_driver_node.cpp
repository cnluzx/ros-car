// racecar_driver_node.cpp
// ROS2 到底盘串口的唯一控制出口。默认安全：启动、超时和退出都发送中位 PWM。

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "../include/racecar_driver.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>

class RacecarDriver : public rclcpp::Node
{
public:
  RacecarDriver() : Node("racecar_driver")
  {
    declare_parameter<std::string>("serial_port", "/dev/car");
    declare_parameter<int>("baud_rate", 38400);
    declare_parameter<double>("throttle_neutral", 1500.0);
    declare_parameter<double>("steering_neutral", 1500.0);
    declare_parameter<double>("command_timeout_sec", 0.35);

    const auto serial_port = get_parameter("serial_port").as_string();
    const int baud_rate = get_parameter("baud_rate").as_int();
    throttle_neutral_ = get_parameter("throttle_neutral").as_double();
    steering_neutral_ = get_parameter("steering_neutral").as_double();
    command_timeout_sec_ = get_parameter("command_timeout_sec").as_double();

    auto serial_port_copy = serial_port;
    art_racecar_init(baud_rate, serial_port_copy.data());

    // 控制板可能保持上一条串口命令；驱动一启动就明确写入停车/回正。
    send_safe_stop();
    last_command_time_ = now();

    subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      "/car_cmd_vel", 1,
      std::bind(&RacecarDriver::twist_callback, this, std::placeholders::_1));
    teleop_subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      "/teleop_cmd_vel", 1,
      std::bind(&RacecarDriver::teleop_callback, this, std::placeholders::_1));

    watchdog_timer_ = create_wall_timer(
      std::chrono::milliseconds(50), std::bind(&RacecarDriver::watchdog_callback, this));

    RCLCPP_INFO(get_logger(), "Driver armed safely: neutral throttle=%.1f steering=%.1f timeout=%.2fs",
                throttle_neutral_, steering_neutral_, command_timeout_sec_);
  }

  ~RacecarDriver() override
  {
    // 正常 Ctrl+C、launch 关闭或析构时都向底盘写入一次停止命令。
    send_safe_stop();
  }

private:
  static uint16_t clamp_pwm(double value)
  {
    return static_cast<uint16_t>(std::clamp(value, 1000.0, 2000.0));
  }

  void send_safe_stop()
  {
    send_cmd(clamp_pwm(throttle_neutral_), clamp_pwm(steering_neutral_));
  }

  void twist_callback(const geometry_msgs::msg::Twist::SharedPtr twist)
  {
    // 标准速度接口：linear.x=0 映射为节流中位，而非保留上一条油门。
    const double motor_pwm = throttle_neutral_ + twist->linear.x * 100.0;
    const double steering_pwm = steering_neutral_ - twist->angular.z * 1300.0;
    send_cmd(clamp_pwm(motor_pwm), clamp_pwm(steering_pwm));
    last_command_time_ = now();
    timed_out_ = false;
    RCLCPP_INFO(get_logger(), "PWM throttle=%.1f steering=%.1f", motor_pwm, steering_pwm);
  }

  void teleop_callback(const geometry_msgs::msg::Twist::SharedPtr twist)
  {
    // 旧 teleop 接口传入的是直接 PWM，保留兼容性但同样纳入超时保护。
    const double motor_pwm = twist->linear.x;
    const double steering_pwm = 2500.0 - twist->angular.z * 2000.0 / 180.0;
    send_cmd(clamp_pwm(motor_pwm), clamp_pwm(steering_pwm));
    last_command_time_ = now();
    timed_out_ = false;
    RCLCPP_INFO(get_logger(), "Teleop PWM throttle=%.1f steering=%.1f", motor_pwm, steering_pwm);
  }

  void watchdog_callback()
  {
    if (!timed_out_ && (now() - last_command_time_).seconds() > command_timeout_sec_) {
      send_safe_stop();
      timed_out_ = true;
      RCLCPP_WARN(get_logger(), "Control command timeout; sent neutral PWM");
    }
  }

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr teleop_subscription_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;
  rclcpp::Time last_command_time_;
  double throttle_neutral_{1500.0};
  double steering_neutral_{1500.0};
  double command_timeout_sec_{0.35};
  bool timed_out_{false};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RacecarDriver>());
  rclcpp::shutdown();
  return 0;
}
