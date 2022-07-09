#ifndef ADS_PROBLEMS_CANNON_CANNON_HPP_
#define ADS_PROBLEMS_CANNON_CANNON_HPP_

#include <cmath>
#include <algorithm>
#include <iostream>
#include "ads/executor/galois.hpp"

#include <cnpy.h>
#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>

#include "ads/simulation.hpp"
#include "ads/output_manager.hpp"

const int element_count = 40;
using namespace indicators;

namespace ads {
namespace problems {


class flow : public simulation_2d {
private:
  using Base = simulation_2d;
  vector_type u, u_prev;

  galois_executor executor{12};

  output_manager<2, FILE_FORMAT::NPY> output;
  ProgressBar bar{
    option::BarWidth{50},
    option::ShowRemainingTime{true},
    option::ShowPercentage{true},
    option::PostfixText{"2d simulation"},
  };

public:
  flow(const config_2d& config)
    : Base{config}
    , u{shape()}
    , u_prev{shape()}
    , output{ x.B, y.B, 100 }
  { 
    bar.set_progress(0);
  }

private:
  struct point_type { 
    point_type(index_type e) : 
      x(e[0] / static_cast<double>(element_count - 1)), 
      y(e[1] / static_cast<double>(element_count - 1)) 
    {}

    double x, y;
  };

  inline double source(const point_type p, const value_type /*u*/, const value_type v, const double /*t*/) const {
    if (p.y < 0.125) {
      return (50 - p.y * 400) * v.val;
    }
    return 0;
  }

  inline double cannon(const point_type /*p*/, const double /*t*/) const {
    return 0;
  }

  inline double thermalInversion(const point_type p, const double /*t*/) const {
    if (p.y > 0.75) return 0.;
    return -5.;
  }

  inline double advection(const point_type p, const value_type u, const value_type v, const double t) const {
    return (thermalInversion(p, t) - cannon(p, t)) * u.dy * v.val;
  }

  inline double diffusion(const point_type /*p*/, const value_type u, const value_type v, const double /*t*/) const {
    const double Kx = 1.0, Ky = 0.1;
    return Kx * u.dx * v.dx + Ky * u.dy * v.dy;
  }


  void before() override {
    prepare_matrices();

    auto init = [](double, double) { return 0; };
    projection(u, init);
    // auto npy = cnpy::npy_load("state12500.npy");
    // std::memcpy(u.data(), npy.data<double>(), sizeof(double) * u.size());

    solve(u);
  }

  void after() override {
    bar.set_progress(100);
  }

  void compute_rhs(int /*iter*/, double t) {
    auto& rhs = u;
    zero(rhs);

    executor.for_each(elements(), [&](index_type e) {
      auto U = element_rhs();
      double J = jacobian(e);

      for (auto q : quad_points()) {
        double w = weight(q);
        const value_type u = eval_fun(u_prev, e, q); // u at position p

        for (auto a : dofs_on_element(e)) {
          auto local_a = dof_global_to_local(e, a);
          const value_type v = eval_basis(e, q, a); // test function
          const double dt = steps.dt;
          const point_type p(e); // position in domain

          double val = (u.val * v.val) +
            dt * (
              - diffusion(p, u, v, t)
              + advection(p, u, v, t)
              + source(p, u, v, t)
            );

          U(local_a[0], local_a[1]) += val * w * J;
        }
      }

      executor.synchronized([&] { update_global_rhs(rhs, U, e); });
    });
  }

  void step(int iter, double t) override {
    compute_rhs(iter, t);
    solve(u);
  }

  void before_step(int /*iter*/, double /*t*/) override {
    std::swap(u, u_prev);
  }

  void after_step(int iter, double /*t*/) override {
    bar.set_progress(100 * iter / steps.step_count);
    // save every 100th step
    if (iter % 100 == 0) {
      output.to_file(u, "out_%d.npy", iter);
    }
  }

  void prepare_matrices() {
    y.fix_left();
    y.fix_right();
    Base::prepare_matrices();
  }
};

}
}

#endif /* ADS_PROBLEMS_CANNON_CANNON_HPP_ */
