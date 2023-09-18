#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

class runge_kutta_dp54 {
public:
  runge_kutta_dp54(int n_eq) : n_eq_(n_eq) {
    k2_.resize(n_eq_);
    k3_.resize(n_eq_);
    k4_.resize(n_eq_);
    k5_.resize(n_eq_);
    k6_.resize(n_eq_);

    dydx_.resize(n_eq_);
    dydx_new_.resize(n_eq_);
    y_err_.resize(n_eq_);
    y_tmp_.resize(n_eq_);
  }

  ~runge_kutta_dp54() {}

  template <typename F, typename StopCondition>
  std::vector<double> integrate(const F &f, const StopCondition &stop,
                                const double x0, const std::vector<double> &y0,
                                double rtol, double atol) {
    double x = x0;
    double h = 1e-6;
    double hmin = 1e-16;
    double hmax = 1.0;

    std::vector<double> y = y0;
    std::vector<double> y_prev = y0;
    std::vector<double> y_next = y0;
    double err_prev = 1.0;
    dydx_ = f(x, y);
    results_.clear();
    results_x_.clear();
    results_.push_back(y);
    results_x_.push_back(x);

    while (true) {
      // Take a step
      y_next = step(f, h, x, y);

      // Error estimate
      double err = 0.0;
      for (int i = 0; i < n_eq_; i++) {
        double scale =
            atol + std::max(std::abs(y[i]), std::abs(y_next[i])) * rtol;
        err += std::pow(y_err_[i] / scale, 2);
      }
      err = std::max(std::sqrt(err / n_eq_), 1e-10);

      // Accept step if error is below 1, otherwise reject it
      if (err < 1.0) {
        x += h;
        y_prev = y;
        y = y_next;
        dydx_ = dydx_new_;
        results_.push_back(y);
        results_x_.push_back(x);
        // std::cout << "Accepted step, x = " << x << ", y = " << y_next[0]
        //           << ", h = " << h << std::endl;
        if (stop(x, y_prev, y)) {
          break;
        }
      } else {
        // std::cout << "Rejected step, x = " << x << ", h = " << h <<
        // std::endl;
      }

      // Adjust h as needed
      // double err_alpha = 1.0 / 5.0;
      // double err_beta = 0.0;
      double err_alpha = 0.7 / 5.0;
      double err_beta = 0.4 / 5.0;
      // std::cout << "err_prev is " << err_prev << ", err is " << err
      //           << ", factor is "
      //           << 0.9 * h * std::pow(err, -err_alpha) *
      //                  std::pow(err_prev, err_beta)
      //           << std::endl;
      h = std::max(hmin, 0.9 * h * std::pow(err, -err_alpha) *
                             std::pow(err_prev, err_beta));
      h = std::min(hmax, h);
      err_prev = err;
    }
    return y_next;
  }

  template <typename F>
  std::vector<double> step(const F &f, double h, double x,
                           const std::vector<double> &y) {
    std::vector<double> y_next(n_eq_);

    // First step
    for (int i = 0; i < n_eq_; i++) {
      y_tmp_[i] = y[i] + h * a21 * dydx_[i];
    }

    // Second step
    k2_ = f(x + c2 * h, y_tmp_);
    for (int i = 0; i < n_eq_; i++) {
      y_tmp_[i] = y[i] + h * (a31 * dydx_[i] + a32 * k2_[i]);
    }

    // Third step
    k3_ = f(x + c3 * h, y_tmp_);
    for (int i = 0; i < n_eq_; i++) {
      y_tmp_[i] = y[i] + h * (a41 * dydx_[i] + a42 * k2_[i] + a43 * k3_[i]);
    }

    // Fourth step
    k4_ = f(x + c4 * h, y_tmp_);
    for (int i = 0; i < n_eq_; i++) {
      y_tmp_[i] = y[i] + h * (a51 * dydx_[i] + a52 * k2_[i] + a53 * k3_[i] +
                              a54 * k4_[i]);
    }

    // Fifth step
    k5_ = f(x + c5 * h, y_tmp_);
    for (int i = 0; i < n_eq_; i++) {
      y_tmp_[i] = y[i] + h * (a61 * dydx_[i] + a62 * k2_[i] + a63 * k3_[i] +
                              a64 * k4_[i] + a65 * k5_[i]);
    }

    // Sixth step
    k6_ = f(x + h, y_tmp_);
    for (int i = 0; i < n_eq_; i++) {
      y_next[i] = y[i] + h * (a71 * dydx_[i] + a72 * k2_[i] + a73 * k3_[i] +
                              a74 * k4_[i] + a75 * k5_[i] + a76 * k6_[i]);
    }
    dydx_new_ = f(x + h, y_next);

    // Error estimate
    for (int i = 0; i < n_eq_; i++) {
      y_err_[i] = h * (e1 * dydx_[i] + e3 * k3_[i] + e4 * k4_[i] + e5 * k5_[i] +
                       e6 * k6_[i] + e7 * dydx_new_[i]);
    }

    return y_next;
  }

  const std::vector<std::vector<double>> &results() const { return results_; }
  const std::vector<double> &results_x() const { return results_x_; }

private:
  int n_eq_ = 1;
  std::vector<double> k2_, k3_, k4_, k5_, k6_;
  std::vector<double> dydx_, dydx_new_, y_err_, y_tmp_;
  std::vector<std::vector<double>> results_;
  std::vector<double> results_x_;

  const double c2 = 1.0 / 5.0;
  const double c3 = 3.0 / 10.0;
  const double c4 = 4.0 / 5.0;
  const double c5 = 8.0 / 9.0;

  const double a21 = 1.0 / 5.0;
  const double a31 = 3.0 / 40.0;
  const double a32 = 9.0 / 40.0;
  const double a41 = 44.0 / 45.0;
  const double a42 = -56.0 / 15.0;
  const double a43 = 32.0 / 9.0;
  const double a51 = 19372.0 / 6561.0;
  const double a52 = -25360.0 / 2187.0;
  const double a53 = 64448.0 / 6561.0;
  const double a54 = -212.0 / 729.0;
  const double a61 = 9017.0 / 3168.0;
  const double a62 = -355.0 / 33.0;
  const double a63 = 46732.0 / 5247.0;
  const double a64 = 49.0 / 176.0;
  const double a65 = -5103.0 / 18656.0;
  const double a71 = 35.0 / 384.0;
  const double a72 = 0.0;
  const double a73 = 500.0 / 1113.0;
  const double a74 = 125.0 / 192.0;
  const double a75 = -2187.0 / 6784.0;
  const double a76 = 11.0 / 84.0;

  const double e1 = 71.0 / 57600.0;
  const double e2 = 0.0;
  const double e3 = -71.0 / 16695.0;
  const double e4 = 71.0 / 1920.0;
  const double e5 = -17253.0 / 339200.0;
  const double e6 = 22.0 / 525.0;
  const double e7 = -1.0 / 40.0;

  const double d1 = -12715105075.0 / 11282082432.0;
  const double d2 = 0.0;
  const double d3 = 87487479700.0 / 32700410799.0;
  const double d4 = -10690763975.0 / 1880347072.0;
  const double d5 = 701980252875.0 / 199316789632.0;
  const double d6 = -1453857185.0 / 822651844.0;
  const double d7 = 69997945.0 / 29380423.0;
};
