#ifndef ADS_PROBLEMS_CANNON_CANNON_HPP_
#define ADS_PROBLEMS_CANNON_CANNON_HPP_

#include <cmath>
#include <algorithm>
#include <iostream>
#include "ads/executor/galois.hpp"

#include <cnpy.h>
#include <indicators/progress_bar.hpp>

#include "ads/simulation.hpp"
#include "ads/output_manager.hpp"

using namespace indicators;

namespace ads {
namespace problems {


class flow : public simulation_2d {
private:
  using Base = simulation_2d;
  vector_type u, u_prev;
  std::string initial_state;

  galois_executor executor{12};

  output_manager<2, FILE_FORMAT::NPY> output;
  ProgressBar bar{
    option::BarWidth{40},
    option::ShowRemainingTime{true},
    option::ShowPercentage{true},
    option::PostfixText{"2d simulation"},
  };


public:
  flow(const config_2d& config, std::string initial_state)
    : Base{config}
    , u{shape()}
    , u_prev{shape()}
    , initial_state{initial_state}
    , output{x.B, y.B, 100}
    , x_element_count{static_cast<double>(config.x.elements-1)}
    , y_element_count{static_cast<double>(config.y.elements-1)}
  { 
    bar.set_progress(0);
  }

private:
  struct point_type { double x, y; };

  double x_element_count, y_element_count; // count - 1
  inline const point_type point_at(index_type e) const {
    return {e[0] / x_element_count, e[1] / y_element_count};
  }

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

  // calculate (u, v)
  inline double metric(const value_type u, const value_type v) const {
    return u.val * v.val;
  }

  double metric(const vector_type& state) const {
    double sum = 0;
    for (auto e : elements()) {
      double J = jacobian(e);
      for (auto q : quad_points()) {
        double w = weight(q);
        for (auto a : dofs_on_element(e)) {
          const value_type v = eval_basis(e, q, a);
          const value_type u = eval_fun(state, e, q);

          sum += metric(u, v) * w * J;
        }
      }
    }
    return sum;
  }

  void before() override {
    prepare_matrices();

    if (initial_state == "") {
      auto init = [](double, double) { return 0; };
      projection(u, init);
    }
    else {
      load_state(initial_state); // example: "initial_state.npy" (u)
    }

    save_projection(0);
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
          const point_type p = point_at(e); // position in domain

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

  // save the state of the simulation (u)
  void save_state(int iter) {
    static boost::format fmt("state_%d.npy");
    auto filename = (fmt % iter).str();
    unsigned long int size = u.size();
    cnpy::npy_save(filename, u.data(), {size});
  }

  void load_state(std::string filename) {
    auto npy = cnpy::npy_load(filename);
    unsigned long int size = u.size();
    if (!(npy.shape.size() == 1 && npy.shape[0] == size)) {
      std::cerr << "Specified file does not contain valid state (invalid shape)" << std::endl;
      exit(1);
    }

    std::memcpy(u.data(), npy.data<double>(), sizeof(double) * size);
  }

  void save_projection(int iter) {
    output.to_file(u, "out_%d.npy", iter);
  }

  void save_metric(int iter) {
    static boost::format fmt("metric_%d.npy");
    auto filename = (fmt % iter).str();

    std::fstream file(filename);
    file << metric(u) << std::endl;
    file.close();
  }

  void step(int iter, double t) override {
    compute_rhs(iter, t);
    solve(u);
  }

  void before_step(int /*iter*/, double /*t*/) override {
    std::swap(u, u_prev);
  }

  void after_step(int iter, double /*t*/) override {
    iter += 1;
    bar.set_progress(100 * iter / steps.step_count);

    // save every 100th step
    if (iter % 100 == 0) {
      save_projection(iter);
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
