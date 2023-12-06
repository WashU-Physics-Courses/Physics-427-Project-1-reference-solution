#include "rk45_dormand_prince.h"
#include "metric.h"
#include <fstream>
#include <iomanip>
#include <omp.h>

int main() {
  // define the parameters
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
  const int Nx = 128;
  const int Ny = 72;
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

  // Construct the solver for each thread
  int nthreads = omp_get_max_threads();
  std::cout << "number of threads: " << nthreads << std::endl;

  std::vector<rk45_dormand_prince> rk_vec;
  std::vector<boyer_lindquist_metric> metric_vec;
  for (int i = 0; i < nthreads; i++) {
    rk_vec.emplace_back(N_eq, 1e-12, 1e-12);
    metric_vec.emplace_back(a, M);
  }

  // loop over the pixels
#pragma omp parallel for
  for (int i = 0; i < Ny; i++) {
    // Get the id of this thread
    int thread_id = omp_get_thread_num();
    for (int j = 0; j < Nx; j++) {
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
      auto y = rk_vec[thread_id].integrate(f, stop, 0.01, 0.0, y0);
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
        // std::cout << 1;
      } else {
        output[i][j] = 0.0;
        // std::cout << 0;
      }
    }
    // std::cout << std::endl;
  }

  // write to file
  std::ofstream fout("image.txt");
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

  return 0;
}
