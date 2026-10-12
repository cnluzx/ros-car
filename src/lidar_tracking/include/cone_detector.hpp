// =============================================================================
// cone_detector.hpp —— cone.py 自适应聚类算法的 C++ 实现
//
// 目的：把这个检测器接到 lidar_tracking.cpp 里，替代原来的
//       "相邻束距离突变 + 平台稳定性" 判据。
//
// 与 cone.py 的对应关系（逐条对齐，便于交叉验证）：
//   cone.py                          C++ (本文件)
//   ------------------------------   ------------------------------------
//   valid = isfinite & rm..rM        same
//   xs = r*cos(theta-off)            same（off = angle_offset）
//   ys = r*sin(theta-off)            same
//   gap = gap_base + gap_slope*r     same（用相邻两点的 max(r) 作为 r）
//   首尾环接 (dist < 0.08)            same
//   n < min_points                   same
//   extent < min_extent/> max_extent  same（包围盒对角长度）
//   rr < min_dist                    same
//   沿视线外推 cone_radius            same
//   位置窗口 + 左右分类               same
//
// 车体坐标约定：x 前、y 左（与 cone.py 一致）。
// =============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

// MinGW 的严格 -std=c++17 不暴露 M_PI（Linux/gcc 的 <math.h> 会定义）
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cone_det {

// ------------------------------------------------------------------ 参数
struct Params {
    double range_min   = 0.15;   // 最近有效距离 [m]
    double range_max   = 5.50;   // 最远有效距离 [m]
    double gap_base    = 0.045;  // 聚类基础间距 [m]
    double gap_slope   = 0.020;  // 聚类距离系数 [m/m]
    int    min_points  = 3;      // 簇最少点数
    double min_extent  = 0.018;  // 簇最小尺寸 [m]
    double max_extent  = 0.25;   // 簇最大尺寸 [m]
    double cone_radius = 0.10;   // 锥桶半径补偿 [m]
    double min_dist    = 0.40;   // 最近保留距离 [m]
    double angle_offset_deg = 0.0;  // 雷达安装角修正 [deg]
    double left_max_y  = 2.50;   // 左侧窗口 [m]
    double right_max_y = 2.50;   // 右侧窗口 [m]
    double fwd_max_x   = 4.00;   // 前方窗口 [m]
    double fwd_min_x   = -0.30;  // 后方容许 [m]
    double min_sep_axis = 0.10;  // 中线死区 [m]
};

// ------------------------------------------------------------------ 输出
struct Cone {
    double x = 0.0;      // 纵向（前为正）[m]
    double y = 0.0;      // 横向（左为正）[m]
    double r = 0.0;      // 距离 [m]
    double a = 0.0;      // 角度 [deg]
    int    n = 0;        // 点数
    double extent = 0.0; // 簇尺寸 [m]
    int    side = 0;     // +1 左 / -1 右 / 0 中线
};

// ------------------------------------------------------------------ 检测器
class Detector {
 public:
    explicit Detector(const Params& p = Params()) : p_(p) {}
    void setParams(const Params& p) { p_ = p; }
    const Params& params() const { return p_; }

    // ranges 允许含 inf/nan 表示无效束
    std::vector<Cone> detect(const std::vector<double>& ranges,
                             double angle_min, double angle_increment) const {
        std::vector<Cone> out;
        const int n = static_cast<int>(ranges.size());
        if (n < 3) return out;

        const double off = p_.angle_offset_deg * M_PI / 180.0;

        // ---- 预先算好每个束的车体坐标（无效束标 valid=false）----
        std::vector<char>   valid(n, 0);
        std::vector<double> xs(n, 0.0), ys(n, 0.0), rr(n, 0.0);
        for (int i = 0; i < n; ++i) {
            double r = ranges[i];
            if (!std::isfinite(r) || r < p_.range_min || r > p_.range_max) continue;
            // 安装角修正：与 cone.py 的 `av = a - radians(angle_offset)` 保持一致
            double th = i * angle_increment + angle_min + off;
            xs[i] = r * std::cos(th);
            ys[i] = r * std::sin(th);
            rr[i] = r;
            valid[i] = 1;
        }

        // ---- 聚类：按束顺序扫描，遇到无效束或间距超限就断开 ----
        std::vector<int> cur;                       // 当前簇的束索引
        std::vector<std::vector<int>> groups;
        auto flush = [&]() {
            if (cur.size() >= 2) groups.push_back(cur);
            cur.clear();
        };
        for (int i = 0; i < n; ++i) {
            if (!valid[i]) { flush(); continue; }
            if (!cur.empty()) {
                int j = cur.back();
                double d = std::hypot(xs[i] - xs[j], ys[i] - ys[j]);
                // cone.py: expect = gap_base + gap_slope * max(r_i, r_j)
                double rmax = std::max(rr[i], rr[j]);
                double expect = p_.gap_base + p_.gap_slope * rmax;
                if (d > expect) flush();
            }
            cur.push_back(i);
        }
        flush();

        // ---- 首尾环接：0° 与 360° 附近可能是同一个物体 ----
        if (groups.size() > 1) {
            auto& f = groups.front();
            auto& l = groups.back();
            double d = std::hypot(xs[f.front()] - xs[l.back()],
                                  ys[f.front()] - ys[l.back()]);
            if (d < 0.08) {
                std::vector<int> merged = l;
                merged.insert(merged.end(), f.begin(), f.end());
                groups.front() = merged;
                groups.pop_back();
            }
        }

        // ---- 尺寸/点数过滤 + 半径补偿 + 分类 ----
        for (const auto& g : groups) {
            int np = static_cast<int>(g.size());
            if (np < p_.min_points) continue;

            double xlo = 1e30, xhi = -1e30, ylo = 1e30, yhi = -1e30;
            double sx = 0.0, sy = 0.0;
            for (int idx : g) {
                xlo = std::min(xlo, xs[idx]); xhi = std::max(xhi, xs[idx]);
                ylo = std::min(ylo, ys[idx]); yhi = std::max(yhi, ys[idx]);
                sx += xs[idx]; sy += ys[idx];
            }
            double extent = std::hypot(xhi - xlo, yhi - ylo);
            if (extent < p_.min_extent || extent > p_.max_extent) continue;

            double cx = sx / np, cy = sy / np;
            double r0 = std::hypot(cx, cy);
            if (r0 < p_.min_dist) continue;
            if (r0 > 1e-6) {                      // 沿视线外推到锥桶表面
                double s = (r0 + p_.cone_radius) / r0;
                cx *= s; cy *= s;
            }

            if (cx < p_.fwd_min_x || cx > p_.fwd_max_x) continue;
            if (cy > p_.left_max_y || cy < -p_.right_max_y) continue;

            Cone c;
            c.x = cx; c.y = cy;
            c.r = std::hypot(cx, cy);
            c.a = std::atan2(cy, cx) * 180.0 / M_PI;
            c.n = np; c.extent = extent;
            c.side = (cy > p_.min_sep_axis) ? 1 : ((cy < -p_.min_sep_axis) ? -1 : 0);
            out.push_back(c);
        }

        // 按距离排序，近的在前
        std::sort(out.begin(), out.end(),
                  [](const Cone& a, const Cone& b) { return a.r < b.r; });
        return out;
    }

 private:
    Params p_;
};

}  // namespace cone_det
