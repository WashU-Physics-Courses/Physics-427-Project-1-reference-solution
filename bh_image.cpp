#include "integrator.h"
// #include "ode_solver.h"
#include <fstream>
#include <iomanip>
#include <omp.h>

double square(double x) { return x * x; }

double cube(double x) { return x * x * x; }

class boyer_lindquist_metric {
public:
  boyer_lindquist_metric(double a0, double M0 = 1.0) : a(a0), M(M0) {}
  ~boyer_lindquist_metric() {}

  void compute_metric(double r, double th) {
    double sth = std::sin(th);
    double cth = std::cos(th);
    double c2th = std::cos(2.0 * th);
    double s2th = std::sin(2.0 * th);
    double cscth = 1.0 / sth;
    double a2 = a * a;
    double a3 = a * a2;
    double a4 = a2 * a2;
    double a5 = a4 * a;
    double a6 = a2 * a4;
    double r2 = r * r;
    double r3 = r2 * r;
    double r4 = r2 * r2;
    double r6 = r2 * r4;

    delta = r * r - 2.0 * M * r + a * a;
    sigma = square(r2 + a2) - a2 * delta * square(sth);
    rho2 = r2 + a * a * cth * cth;

    alpha = std::sqrt(rho2 * delta / sigma);
    beta3 = -2.0 * M * a * r / sigma;

    g_00 = 2.0 * M * r / rho2 - 1.0;
    g_03 = -2.0 * M * a * r / rho2 * square(sth);
    g_11 = rho2 / delta;
    g_22 = rho2;
    g_33 = sigma * square(sth) / rho2;

    gamma11 = delta / rho2;
    gamma22 = 1.0 / rho2;
    gamma33 = rho2 / sigma / square(sth);

    d_alpha_dr = M *
                 (-a6 + 2.0 * r6 + a2 * r3 * (3.0 * r - 4.0 * M) -
                  a2 * (a4 + 2.0 * a2 * r2 + r3 * (r - 4.0 * M)) * c2th) /
                 (2.0 * sigma * sigma * std::sqrt(delta * rho2 / sigma));
    d_beta3_dr = M *
                 (-a5 + 3.0 * a3 * r2 + 6.0 * a * r4 + a3 * (r2 - a2) * c2th) /
                 square(sigma);
    d_gamma11_dr = 2.0 * (r * (M * r - a2) + a2 * (r - 1.0 * M) * square(cth)) /
                   square(rho2);
    d_gamma22_dr = -2.0 * r / square(rho2);
    d_gamma33_dr =
        (-2.0 * a4 * (r - 1.0 * M) * square(cscth) +
         2.0 * (a2 * (2.0 * r - 1.0 * M) + r2 * (2.0 * r + 1.0 * M)) *
             square(a * square(cscth)) -
         2.0 * r * square(a2 + r2) * square(cube(cscth))) /
        square(a4 + a2 * r * (r - 2.0 * M) - square((a2 + r2) * cscth));
    d_alpha_dth = -M * a2 * delta * r * (a2 + r2) * s2th / square(sigma) /
                  std::sqrt(delta * rho2 / sigma);
    d_beta3_dth = -2.0 * M * a3 * r * delta * s2th / square(sigma);
    d_gamma11_dth = a2 * delta * s2th / square(rho2);
    d_gamma22_dth = a2 * s2th / square(rho2);
    d_gamma33_dth =
        2.0 *
        (-a4 * delta + 2.0 * a2 * delta * (a2 + r2) * square(cscth) -
         cube(a2 + r2) * square(square(cscth))) *
        cth / cube(sth) /
        square(a4 + a2 * r * (r - 2.0 * M) - square((a2 + r2) * cscth));

  }

  double u0(double u_1, double u_2, double u_3) {
    return std::sqrt(gamma11 * u_1 * u_1 + gamma22 * u_2 * u_2 +
                     gamma33 * u_3 * u_3) /
           alpha;
  }

  double u_0(double u_1, double u_2, double u_3) {
    return -alpha * alpha * u0(u_1, u_2, u_3) + beta3 * u_3;
  }

  double a = 0.0;
  double M = 1.0;
  double alpha, beta3;
  double gamma11, gamma22, gamma33;
  double g_00, g_11, g_22, g_33, g_03;
  double d_alpha_dr, d_beta3_dr, d_gamma11_dr, d_gamma22_dr, d_gamma33_dr;
  double d_alpha_dth, d_beta3_dth, d_gamma11_dth, d_gamma22_dth, d_gamma33_dth;

  double g00, g11, g22, g33, g03;
  double d_g00_dr, d_g03_dr, d_g11_dr, d_g22_dr, d_g33_dr;
  double d_g00_dth, d_g03_dth, d_g11_dth, d_g22_dth, d_g33_dth;
  double delta, sigma, rho2;
};

int main() {
  const int N_eq = 6;
  const double a = 0.99;
  const double M = 1.0;

  const double D = 500;                   // distance of the observer
  const double Lx = 24;                   // size of the image
  const double Ly = 13.5;                 // size of the image
  const double th0 = 85.0 / 180.0 * M_PI; // inclination angle of the observer
  const double phi0 = 0.0;                // azimuthal angle of the observer
  const double r_in = 5.0;                // inner radius of the disk
  const double r_out = 20.0;              // outer radius of the disk

  // prepare the output array
  const int Nx = 64;
  const int Ny = 36;
  std::vector<std::vector<double>> output;
  output.resize(Ny);
  for (int i = 0; i < Ny; i++) {
    output[i].resize(Nx);
  }

  // define the stopping condition
  auto stop = [&](double x, const std::vector<double> &y_prev,
                  const std::vector<double> &y) {
    double rH = M + std::sqrt(M * M - a * a);
    double z = std::sqrt(y[0] * y[0] + a * a) * std::cos(y[1]);
    double z_prev =
        std::sqrt(y_prev[0] * y_prev[0] + a * a) * std::cos(y_prev[1]);
    if (y[0] < r_out && y[0] > r_in && z * z_prev < 0.0) {
      // crossing the accretion disk
      return true;
    }
    if (y[0] < rH * 1.01 || y[0] > D * 1.01) {
      return true;
    }
    return false;
  };

  // loop over the pixels
  int nthreads = 1;
  std::vector<runge_kutta_dp54> rk_vec;
  // std::vector<ode_solver<ode_method_rk4>> rk_vec;
  std::vector<boyer_lindquist_metric> metric_vec;
  for (int i = 0; i < nthreads; i++) {
    // rk_vec.emplace_back(N_eq, 1e-15, 1e-15);
    rk_vec.emplace_back(N_eq);
    metric_vec.emplace_back(a, M);
  }
  // runge_kutta_dp54 rk(N_eq);
#pragma omp parallel for num_threads(nthreads)
  for (int i = 0; i < Ny; i++) {
    int thread_id = omp_get_thread_num();
    for (int j = 0; j < Nx; j++) {
      std::cout << "i = " << i << ", j = " << j << std::endl;
      double pos_x = (2 * j - Nx) * (Lx / Nx);
      double pos_y = (2 * i - Ny) * (Ly / Ny);
      double dth = pos_y / D;
      double th = th0 - dth;
      double phi = phi0 + pos_x / D;
      auto &metric = metric_vec[thread_id];

      // Set initial conditions
      std::vector<double> y0(N_eq);
      y0[0] = std::sqrt(D * D + pos_x * pos_x + pos_y * pos_y);
      y0[1] = th;
      y0[2] = phi;
      metric.compute_metric(y0[0], th);
      y0[3] = -std::cos(phi) * std::cos(dth) * std::sqrt(metric.g_11);
      y0[4] = std::sin(dth) * std::sqrt(metric.g_22);
      y0[5] = std::sin(phi) * std::cos(dth) * std::sqrt(metric.g_33);

      // Define the equation of motion
      auto f = [&metric](double x, const std::vector<double> &y) {
        std::vector<double> dydx(N_eq);
        metric.compute_metric(y[0], y[1]);
        double u0 = metric.u0(y[3], y[4], y[5]);
        // \dot{r} = \gamma^{rr}u_r/u^0
        dydx[0] = metric.gamma11 * y[3] / u0;
        // \dot{\theta} = \gamma^{\theta\theta}u_\theta/u^0
        dydx[1] = metric.gamma22 * y[4] / u0;
        // \dot{\phi} = \gamma^{\phi\phi}u_\phi/u^0 - \beta^\phi
        dydx[2] = metric.gamma33 * y[5] / u0 - metric.beta3;
        // \dot{u_r} = -\alpha u^0 \alpha_{,r} + u_k \beta^k_{,r} -
        // \frac{u_ju_k}{2u^0}\gamma^{jk}_{,r}
        dydx[3] = -metric.alpha * u0 * metric.d_alpha_dr +
                  y[5] * metric.d_beta3_dr -
                  (square(y[3]) * metric.d_gamma11_dr +
                   square(y[4]) * metric.d_gamma22_dr +
                   square(y[5]) * metric.d_gamma33_dr) /
                      (2.0 * u0);
        // \dot{u_\theta} = -\alpha u^0 \alpha_{,\theta} + u_k \beta^k_{,\theta}
        // - \frac{u_ju_k}{2u^0}\gamma^{jk}_{,\theta}
        dydx[4] = -metric.alpha * u0 * metric.d_alpha_dth +
                  y[5] * metric.d_beta3_dth -
                  (square(y[3]) * metric.d_gamma11_dth +
                   square(y[4]) * metric.d_gamma22_dth +
                   square(y[5]) * metric.d_gamma33_dth) /
                      (2.0 * u0);
        // \dot{u_\phi} = 0
        dydx[5] = 0.0;
        return dydx;
      };
      // integrate
      auto y = rk_vec[thread_id].integrate(f, stop, 0.0, y0, 1e-15, 1e-15);
      // auto y = rk_vec[thread_id].integrate(fi, stop, 0.0, y0, 0.01);
      if (y[0] < r_out && y[0] > r_in) {
        // Compute Doppler factor
        double r = y[0];
        double th = M_PI * 0.5;
        double phi = y[2];

        double Omega = 1.0 / (a + std::sqrt(r * r * r) / std::sqrt(M));
        double u_0 = metric.u_0(-y[3], -y[4], -y[5]);
        double g = (1.0 - Omega * y[5] / u_0) /
                   std::sqrt(-(metric.g_00 + metric.g_33 * Omega * Omega +
                               2.0 * metric.g_03 * Omega));
        output[i][j] = 1.0 / cube(std::abs(g));
        // output[i][j] = 1.0;
      } else {
        output[i][j] = 0.0;
      }
    }
  }

  // write to file
  std::ofstream fout("image.txt");
  // for (auto &y : rk.results()) {
  for (int i = 0; i < output.size(); i++) {
    for (int j = 0; j < output[i].size(); j++) {
      if (j != output[i].size() - 1) {
        fout << output[i][j] << ", ";
      } else {
        fout << output[i][j] << std::endl;
      }
    }
  }
  fout.close();

  // std::ofstream fout2("trajectory.txt");
  // // for (auto &y : rk.results()) {
  // for (int i = 0; i < rk.results().size(); i++) {
  //   auto y = rk.results()[i];
  //   auto x = rk.results_x()[i];
  //   fout2 << std::setprecision(15) << x << "," << y[0] << "," << y[1] << ","
  //         << y[2] << "," << y[3] << "," << y[4] << "," << y[5] << std::endl;
  // }
  // fout2.close();
  return 0;
}
