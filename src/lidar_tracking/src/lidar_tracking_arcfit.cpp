/**
 * LiDAR cone tracking node (centerline-fit version).
 *
 * 思路：不區分紅藍/左右錐桶，直接用一段圓弧去擬合「賽道中線」。
 *  - 中線參數：y=0 處的橫向位置 x0、朝向 psi、曲率 kappa（車體系：+x 右，+y 前）。
 *  - 賽道特性：每個錐桶到中線的距離都應約等於 track_width/2，與顏色、左右無關。
 *  - 代價函數：距離≈半寬的錐桶(內點)幾乎不扣分；落在走廊中間的錐桶(擋路)重罰；
 *    離得遠的錐桶(別條賽道、牆)視為雜訊，不扣分。
 *  - 搜尋：粗網格全域搜尋 + 局部細化，不依賴上一幀初值；上一幀結果只作為弱先驗與候選之一。
 *  - 誤差：取中線上前方 3 個點的橫向位置，沿用原有 rate1/2/3 加權與 PID 參數。
 * 同時保留：無錐桶模式誤判修正、丟點(inf)容忍聚類、丟失保持。
 */
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "control.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// 中線：圓弧，從 (x0, 0) 出發，朝向 psi（0=正前方，正=朝右），曲率 kappa（正=向右彎）
struct CenterFit {
  double x0 = 0.0;
  double psi = 0.0;
  double kappa = 0.0;
  double cost = 1e18;
  int inliers = 0;
};

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kServoCenter = 82.5;
constexpr std::size_t kMaxScanPoints = 4096;
constexpr std::size_t kMaxConePoints = 30;

struct Point {
  double x = 0.0;      // +x 為右側（與原代碼約定一致）
  double y = 0.0;      // +y 為正前方
  double range = 0.0;
  std::size_t index = 0;
};

struct ConeCandidate {
  Point center;
  std::size_t points = 0;
  double width = 0.0;
  double depth = 0.0;
};

double clampValue(double value, double low, double high)
{
  return std::max(low, std::min(value, high));
}

double pointDistance(const Point &a, const Point &b)
{
  return std::hypot(a.x - b.x, a.y - b.y);
}

bool validRange(float value, const sensor_msgs::msg::LaserScan &scan)
{
  if (!std::isfinite(value) || value <= 0.0f) return false;
  if (std::isfinite(scan.range_min) && scan.range_min > 0.0f && value < scan.range_min)
    return false;
  if (std::isfinite(scan.range_max) && scan.range_max > scan.range_min && value > scan.range_max)
    return false;
  return true;
}
}  // namespace

class PubAndSub : public rclcpp::Node
{
public:
  PubAndSub() : Node("lidar_tracking_node")
  {
    declare_parameter("speed", 1535);
    declare_parameter("kp", 0.88);
    declare_parameter("ki", 0.0);
    declare_parameter("kd", 1.2);
    declare_parameter("topic", std::string("/teleop_cmd_vel"));
    declare_parameter("rate1", 0.8);
    declare_parameter("rate2", 2.2);
    declare_parameter("rate3", 0.8);
    declare_parameter("no_cone_detection_range", 2.0);  // 原1.0m小於錐桶間距，實測會把86%的幀誤判為無錐桶
    declare_parameter("no_cone_sector_angle", 60.0);
    declare_parameter("no_cone_speed", 1525);
    declare_parameter("no_cone_turn_angle", 120.0);
    declare_parameter("max_right_dis", 2.5);   // +x 側檢測寬度
    declare_parameter("max_left_dis", 2.5);    // -x 側檢測寬度
    declare_parameter("lost_tracking_speed", 1515.0);
    declare_parameter("mode_enter_frames", 3);
    declare_parameter("mode_exit_frames", 2);
    declare_parameter("cone_cluster_gap", 0.20);
    declare_parameter("cone_max_width", 0.45);
    declare_parameter("cone_max_depth", 0.45);
    // 中線擬合參數
    declare_parameter("track_width", 1.5);        // 賽道名義寬度(米)，請按實際量測(錄包約 1.4)
    declare_parameter("fit_tau", 0.25);           // 內點容差(米)：到中線距離與半寬相差在此內算內點
    declare_parameter("fit_arc_len", 2.4);        // 擬合的中線長度(米)
    declare_parameter("fit_max_curvature", 1.2);  // 最大曲率(1/米)
    declare_parameter("fit_reg_psi", 0.1);       // 正則化：偏向朝向為0（證據不足時別亂轉）
    declare_parameter("fit_reg_kappa", 0.05);     // 正則化：偏向直線
    declare_parameter("fit_block_cap", 3.0);      // 走廊中間擋路錐桶的最大罰分(單位：雜訊錐桶的罰分)
    declare_parameter("lookahead1", 0.6);         // 3 個前視點沿中線的弧長(米)
    declare_parameter("lookahead2", 1.2);
    declare_parameter("lookahead3", 1.8);
    declare_parameter("lost_hold_frames", 10);    // 丟失後保持舵機角度的幀數
    declare_parameter("lost_steer_limit", 35.0);  // 丟失時舵機相對中位的最大偏轉

    speed_ = get_parameter("speed").as_int();
    no_cone_speed_ = get_parameter("no_cone_speed").as_int();
    no_cone_detection_range_ = get_parameter("no_cone_detection_range").as_double();
    no_cone_sector_angle_ = get_parameter("no_cone_sector_angle").as_double() * kPi / 180.0;
    no_cone_turn_angle_ = get_parameter("no_cone_turn_angle").as_double();
    max_right_dis_ = std::max(0.05, get_parameter("max_right_dis").as_double());
    max_left_dis_ = std::max(0.05, get_parameter("max_left_dis").as_double());
    lost_tracking_speed_ = clampValue(get_parameter("lost_tracking_speed").as_double(), 1500.0, 1600.0);
    mode_enter_frames_ = std::max(1, static_cast<int>(get_parameter("mode_enter_frames").as_int()));
    mode_exit_frames_ = std::max(1, static_cast<int>(get_parameter("mode_exit_frames").as_int()));
    cone_cluster_gap_ = clampValue(get_parameter("cone_cluster_gap").as_double(), 0.05, 0.40);
    cone_max_width_ = clampValue(get_parameter("cone_max_width").as_double(), 0.20, 0.80);
    cone_max_depth_ = clampValue(get_parameter("cone_max_depth").as_double(), 0.20, 0.80);
    track_width_ = clampValue(get_parameter("track_width").as_double(), 0.6, 3.0);
    half_w_ = 0.5 * track_width_;
    fit_tau_ = clampValue(get_parameter("fit_tau").as_double(), 0.05, 0.5);
    fit_arc_len_ = clampValue(get_parameter("fit_arc_len").as_double(), 1.0, 3.0);
    fit_max_curv_ = clampValue(get_parameter("fit_max_curvature").as_double(), 0.2, 2.0);
    reg_psi_ = clampValue(get_parameter("fit_reg_psi").as_double(), 0.0, 1.0);
    reg_kappa_ = clampValue(get_parameter("fit_reg_kappa").as_double(), 0.0, 1.0);
    block_cap_ = clampValue(get_parameter("fit_block_cap").as_double(), 1.0, 12.0);
    lookahead_[0] = clampValue(get_parameter("lookahead1").as_double(), 0.1, fit_arc_len_);
    lookahead_[1] = clampValue(get_parameter("lookahead2").as_double(), 0.1, fit_arc_len_);
    lookahead_[2] = clampValue(get_parameter("lookahead3").as_double(), 0.1, fit_arc_len_);
    lost_hold_frames_ = std::max(0, static_cast<int>(get_parameter("lost_hold_frames").as_int()));
    lost_steer_limit_ = clampValue(get_parameter("lost_steer_limit").as_double(), 5.0, 80.0);
    rates_[0] = get_parameter("rate1").as_double();
    rates_[1] = get_parameter("rate2").as_double();
    rates_[2] = get_parameter("rate3").as_double();

    pid_.Init();
    pid_.kp = get_parameter("kp").as_double();
    pid_.ki = get_parameter("ki").as_double();
    pid_.kd = get_parameter("kd").as_double();

    pub_ = create_publisher<geometry_msgs::msg::Twist>(
      get_parameter("topic").as_string(), rclcpp::QoS(5));
    sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", rclcpp::SensorDataQoS(),
      std::bind(&PubAndSub::callback, this, std::placeholders::_1));
    status_timer_ = create_wall_timer(
      std::chrono::milliseconds(500), std::bind(&PubAndSub::printStatus, this));
  }

private:
  // 前方扇區內、no_cone_detection_range 以內若沒有任何「錐桶」，才算無錐桶。
  // 注意：雷達 angle_min=0、範圍 0~2π，不能直接用 abs(theta)<=60° 判斷（只會檢查到 0~60° 一半），
  // 這裡改用錐桶座標 atan2(x,y)，天然落在 (-π, π]，左右兩側都會檢查。
  bool shouldEnterNoConeMode(const sensor_msgs::msg::LaserScan &scan,
                             const std::vector<ConeCandidate> &cones) const
  {
    if (scan.ranges.empty() || !std::isfinite(scan.angle_min) ||
        !std::isfinite(scan.angle_increment) || scan.angle_increment == 0.0) return true;
    for (const auto &c : cones) {
      const double bearing = std::atan2(c.center.x, c.center.y);
      if (std::abs(bearing) <= no_cone_sector_angle_ && c.center.range <= no_cone_detection_range_)
        return false;
    }
    return true;
  }

  void updateDrivingMode(const sensor_msgs::msg::LaserScan &scan,
                         const std::vector<ConeCandidate> &cones)
  {
    if (shouldEnterNoConeMode(scan, cones)) {
      ++no_cone_frames_;
      cone_frames_ = 0;
      if (!is_no_cone_mode_ && no_cone_frames_ >= mode_enter_frames_) {
        is_no_cone_mode_ = true;
        no_cone_start_time_ = now().seconds();
        RCLCPP_INFO(get_logger(), "进入无锥桶巡航: %.2fm, ±%.1fdeg",
          no_cone_detection_range_, no_cone_sector_angle_ * 180.0 / kPi);
      }
    } else {
      ++cone_frames_;
      no_cone_frames_ = 0;
      if (is_no_cone_mode_ && cone_frames_ >= mode_exit_frames_) {
        is_no_cone_mode_ = false;
        RCLCPP_INFO(get_logger(), "退出无锥桶巡航, 持续 %.2fs",
          now().seconds() - no_cone_start_time_);
      }
    }
  }

  // 一次性檢測所有錐桶（不再在此處分左右），結果按 y 由近到遠排序
  std::vector<ConeCandidate> detectCones(const sensor_msgs::msg::LaserScan &scan) const
  {
    std::vector<ConeCandidate> result;
    if (scan.ranges.size() < 3 || scan.ranges.size() > kMaxScanPoints ||
        !std::isfinite(scan.angle_min) || !std::isfinite(scan.angle_increment) ||
        scan.angle_increment == 0.0) return result;

    std::vector<Point> cluster;
    auto flush = [&]() {
      if (cluster.size() < 3 || cluster.size() > 80) {
        cluster.clear();
        return;
      }
      double min_x = cluster.front().x, max_x = min_x;
      double min_y = cluster.front().y, max_y = min_y;
      double mean_x = 0.0, mean_y = 0.0;
      for (const auto &p : cluster) {
        min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
        mean_x += p.x; mean_y += p.y;
      }
      const double width = max_x - min_x;
      const double depth = max_y - min_y;
      const double cx = mean_x / static_cast<double>(cluster.size());
      const double cy = mean_y / static_cast<double>(cluster.size());
      const bool size_ok = width <= cone_max_width_ && depth <= cone_max_depth_ &&
        std::hypot(cx, cy) <= 3.0 && cy > -0.3 && cy < 2.4;
      if (size_ok) {
        ConeCandidate candidate;
        candidate.center = {cx, cy, std::hypot(cx, cy), cluster.front().index};
        candidate.points = cluster.size();
        candidate.width = width;
        candidate.depth = depth;
        result.push_back(candidate);
      }
      cluster.clear();
    };

    bool have_previous = false;
    int missing = 0;   // 簇內連續無效光束數
    Point previous;
    for (std::size_t i = 0; i < scan.ranges.size(); ++i) {
      const float raw = scan.ranges[i];
      if (!validRange(raw, scan) || raw > 3.0f) {
        // 實測雷達在錐桶表面常有 1~2 束丟點(inf)，會把同一個錐桶切成兩簇，
        // 所以簇內允許最多 2 束連續無效，不立刻斷開。
        if (have_previous && ++missing <= 2) continue;
        flush(); have_previous = false; missing = 0; continue;
      }
      const int skipped = missing;
      missing = 0;
      const double theta = scan.angle_min + static_cast<double>(i) * scan.angle_increment;
      Point point;
      point.range = raw; point.index = i;
      point.x = static_cast<double>(raw) * std::sin(theta);
      point.y = static_cast<double>(raw) * std::cos(theta);
      // 窗口：+x 側 max_right_dis，-x 側 max_left_dis（原代碼此處命名與 side_ok 相反，已更正）
      const bool in_window = point.x >= -max_left_dis_ && point.x <= max_right_dis_ &&
        point.y > -0.3 && point.y < 2.4;
      const double adaptive_gap = std::max(
        cone_cluster_gap_,
        2.5 * std::max(0.1, point.range) * std::abs(scan.angle_increment) * (1 + skipped));
      if (!in_window) {
        flush();
        have_previous = false;
        continue;
      }
      if (have_previous && pointDistance(point, previous) > adaptive_gap) flush();
      cluster.push_back(point);
      previous = point; have_previous = true;
    }
    flush();

    std::sort(result.begin(), result.end(), [](const ConeCandidate &a, const ConeCandidate &b) {
      return a.center.y < b.center.y;
    });
    if (result.size() > kMaxConePoints) result.resize(kMaxConePoints);
    return result;
  }

  // ---------------- 中線擬合 ----------------

  // 中線上弧長 s 處的點
  static void arcPoint(const CenterFit &f, double s, double &x, double &y)
  {
    if (std::abs(f.kappa) < 1e-3) {
      x = f.x0 + s * std::sin(f.psi);
      y = s * std::cos(f.psi);
    } else {
      const double th = f.psi + f.kappa * s;
      x = f.x0 + (std::cos(f.psi) - std::cos(th)) / f.kappa;
      y = (std::sin(th) - std::sin(f.psi)) / f.kappa;
    }
  }

  // 計算一組中線參數的代價，同時寫入 f.cost / f.inliers
  void scoreFit(CenterFit &f, const std::vector<ConeCandidate> &cones) const
  {
    const double sp = std::sin(f.psi), cp = std::cos(f.psi);
    const bool straight = std::abs(f.kappa) < 1e-3;
    const double R = straight ? 0.0 : 1.0 / f.kappa;
    const double cx = f.x0 + R * cp;     // 圓心
    const double cy = -R * sp;
    double cost = 0.0;
    int inliers = 0;
    for (const auto &c : cones) {
      const double qx = c.center.x, qy = c.center.y;
      double d;
      if (straight) {
        const double t = clampValue((qx - f.x0) * sp + qy * cp, -0.3, fit_arc_len_);
        d = std::hypot(qx - (f.x0 + t * sp), qy - t * cp);
      } else {
        const double vx = qx - cx, vy = qy - cy;
        const double thq = f.kappa > 0.0 ? std::atan2(vy, -vx) : std::atan2(-vy, vx);
        double dth = thq - f.psi;
        while (dth > kPi) dth -= 2.0 * kPi;
        while (dth <= -kPi) dth += 2.0 * kPi;
        const double s = clampValue(dth / f.kappa, -0.3, fit_arc_len_);
        const double th = f.psi + f.kappa * s;
        const double px = f.x0 + (cp - std::cos(th)) / f.kappa;
        const double py = (std::sin(th) - sp) / f.kappa;
        d = std::hypot(qx - px, qy - py);
      }
      const double e = d - half_w_;                 // 與「應有距離」的偏差
      const double w = 1.0 / (1.0 + 0.25 * std::max(0.0, c.center.range - 1.0));  // 近處更可信
      double ci;
      if (e > fit_tau_) {
        ci = fit_tau_ * fit_tau_;                   // 太遠：視為雜訊/別條賽道，等同沒解釋到
      } else if (e >= -fit_tau_) {
        ci = e * e;                                 // 內點
        ++inliers;
      } else {
        ci = std::min(e * e, block_cap_ * fit_tau_ * fit_tau_);  // 落在走廊中間：擋路，加罰但設上限
      }
      cost += w * ci;
    }
    // 弱先驗：車應在賽道內；與上一幀結果連續
    // 正則化只在「冷啟動」（沒有上一幀結果）時啟用：此時錐桶少(如單側被遮擋)容易過擬合，偏向直線。
    // 有上一幀結果時改由下面的時間連續先驗約束，不再加正則化，避免壓制真實的彎道。
    cost += 0.3 * f.x0 * f.x0;
    if (!have_prev_fit_) {
      const double n_cones = static_cast<double>(cones.size());
      const double reg_scale = clampValue((7.0 - n_cones) / 3.0, reg_min_scale_, 1.0);
      cost += reg_scale * (reg_psi_ * f.psi * f.psi + reg_kappa_ * f.kappa * f.kappa);
    }
    if (have_prev_fit_) {
      cost += 0.5 * (f.x0 - prev_fit_.x0) * (f.x0 - prev_fit_.x0)
            + 0.3 * (f.psi - prev_fit_.psi) * (f.psi - prev_fit_.psi)
            + 0.05 * (f.kappa - prev_fit_.kappa) * (f.kappa - prev_fit_.kappa);
    }
    f.cost = cost;
    f.inliers = inliers;
  }

  void clampFit(CenterFit &f) const
  {
    f.x0 = clampValue(f.x0, -0.6 * track_width_, 0.6 * track_width_);
    f.psi = clampValue(f.psi, -1.4, 1.4);
    f.kappa = clampValue(f.kappa, -fit_max_curv_, fit_max_curv_);
  }

  // 粗網格全域搜尋 + 局部細化。返回是否擬合成功（內點數足夠）。
  bool fitCenterline(const std::vector<ConeCandidate> &cones, CenterFit &best) const
  {
    best = CenterFit();
    if (cones.size() < 2) return false;

    constexpr std::size_t kTop = 4;
    std::vector<CenterFit> top;
    auto consider = [&](CenterFit f) {
      clampFit(f);
      scoreFit(f, cones);
      top.push_back(f);
      std::sort(top.begin(), top.end(), [](const CenterFit &a, const CenterFit &b) { return a.cost < b.cost; });
      if (top.size() > kTop) top.resize(kTop);
    };

    const double x_max = 0.6 * track_width_;
    for (double x0 = -x_max; x0 <= x_max + 1e-9; x0 += 0.2) {
      for (int pd = -60; pd <= 60; pd += 10) {
        for (double k = -fit_max_curv_; k <= fit_max_curv_ + 1e-9; k += 0.3) {
          CenterFit f;
          f.x0 = x0; f.psi = pd * kPi / 180.0; f.kappa = k;
          consider(f);
        }
      }
    }
    if (have_prev_fit_) consider(prev_fit_);

    for (auto f : top) {
      double sx = 0.10, sp = 5.0 * kPi / 180.0, sk = 0.15;
      for (int round = 0; round < 4; ++round) {
        bool improved = true;
        for (int it = 0; improved && it < 15; ++it) {
          improved = false;
          for (int dim = 0; dim < 3; ++dim) {
            for (int dir = -1; dir <= 1; dir += 2) {
              CenterFit g = f;
              if (dim == 0) g.x0 += dir * sx;
              else if (dim == 1) g.psi += dir * sp;
              else g.kappa += dir * sk;
              clampFit(g);
              scoreFit(g, cones);
              if (g.cost < f.cost - 1e-9) { f = g; improved = true; }
            }
          }
        }
        sx *= 0.5; sp *= 0.5; sk *= 0.5;
      }
      if (f.cost < best.cost) best = f;
    }
    return best.inliers >= 2;
  }

  void handleNoConeMode()
  {
    twist_.linear.x = no_cone_speed_;
    twist_.angular.z = clampValue(no_cone_turn_angle_, 0.0, 180.0);
  }

  void handleLostTracking()
  {
    ++lost_tracking_frames_;
    fit_inliers_ = 0;
    if (lost_tracking_frames_ > lost_hold_frames_) {
      last_angle_ *= 0.95;          // 保持一段時間後再緩慢回正
      have_prev_fit_ = false;       // 先驗已過時，丟掉
    }
    twist_.linear.x = lost_tracking_speed_;
    const double steer = clampValue(last_angle_, -lost_steer_limit_, lost_steer_limit_);
    twist_.angular.z = clampValue(kServoCenter + steer, 0.0, 180.0);
  }

  void handleTracking(const std::vector<ConeCandidate> &cones)
  {
    CenterFit fit;
    if (!fitCenterline(cones, fit)) { handleLostTracking(); return; }
    prev_fit_ = fit;
    have_prev_fit_ = true;
    fit_inliers_ = fit.inliers;

    double weighted_error = 0.0, weight_sum = 0.0;
    for (int i = 0; i < 3; ++i) {
      const double weight = rates_[i];
      if (!std::isfinite(weight) || weight <= 0.0) continue;
      double x, y;
      arcPoint(fit, lookahead_[i], x, y);
      weighted_error += 2.0 * x * weight;     // 2*x 與原來 (左x+右x) 量綱一致，PID 參數可沿用
      weight_sum += weight;
    }
    if (!(weight_sum > 0.0) || !std::isfinite(weighted_error)) {
      handleLostTracking(); return;
    }

    lost_tracking_frames_ = 0;
    const double error = clampValue((weighted_error / weight_sum) * 53.33, -60.0, 60.0);
    double angle = pid_.PIDPositional(error);
    if (!std::isfinite(angle)) { handleLostTracking(); return; }
    angle = clampValue(angle, -80.0, 80.0);
    last_angle_ = 0.7 * last_angle_ + 0.3 * angle;
    twist_.angular.z = clampValue(kServoCenter + last_angle_, 0.0, 180.0);
    twist_.linear.x = std::abs(last_angle_) >= 20.0 ? speed_ - 21 : speed_;
  }

  void callback(const sensor_msgs::msg::LaserScan::SharedPtr scan)
  {
    if (!scan) return;
    const auto cones = detectCones(*scan);
    updateDrivingMode(*scan, cones);
    if (is_no_cone_mode_) handleNoConeMode(); else handleTracking(cones);
    pub_->publish(twist_);
  }

  void printStatus()
  {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
      "模式:%s 速度:%.0f 舵机:%.1f 内点:%d 中线x0:%.2f 朝向:%.1fdeg 曲率:%.2f 丢失帧:%d",
      is_no_cone_mode_ ? "无锥" : "循迹", twist_.linear.x, twist_.angular.z,
      fit_inliers_, prev_fit_.x0, prev_fit_.psi * 180.0 / kPi, prev_fit_.kappa, lost_tracking_frames_);
  }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr status_timer_;
  geometry_msgs::msg::Twist twist_;
  Control::PID pid_;

  int speed_ = 1535;
  int no_cone_speed_ = 1525;
  double no_cone_detection_range_ = 2.0;
  double no_cone_sector_angle_ = 60.0 * kPi / 180.0;
  double no_cone_turn_angle_ = 120.0;
  double max_right_dis_ = 2.5;
  double max_left_dis_ = 2.5;
  double lost_tracking_speed_ = 1515.0;
  double cone_cluster_gap_ = 0.20;
  double cone_max_width_ = 0.45;
  double cone_max_depth_ = 0.45;
  double track_width_ = 1.5;
  double half_w_ = 0.75;
  double fit_tau_ = 0.25;
  double fit_arc_len_ = 2.4;
  double fit_max_curv_ = 1.2;
  double reg_psi_ = 0.1;
  double reg_min_scale_ = 0.1;
  double reg_kappa_ = 0.05;
  double block_cap_ = 3.0;
  double lookahead_[3] = {0.6, 1.2, 1.8};
  int lost_hold_frames_ = 10;
  double lost_steer_limit_ = 35.0;
  double rates_[3] = {0.8, 2.2, 0.8};

  bool is_no_cone_mode_ = false;
  int no_cone_frames_ = 0;
  int cone_frames_ = 0;
  int mode_enter_frames_ = 3;
  int mode_exit_frames_ = 2;
  int lost_tracking_frames_ = 0;
  int fit_inliers_ = 0;
  CenterFit prev_fit_;
  bool have_prev_fit_ = false;
  double no_cone_start_time_ = 0.0;
  double last_angle_ = 0.0;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PubAndSub>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
