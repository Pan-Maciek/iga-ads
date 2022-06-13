#ifndef ADS_PROBLEMS_CANNON_CANNON_HPP_
#define ADS_PROBLEMS_CANNON_CANNON_HPP_

#include <cmath>
#include <iostream>
#include "ads/executor/galois.hpp"
#include <cnpy.h>

#include "ads/simulation.hpp"
#include "ads/output_manager.hpp"

const int iterations = 30000;
const int element_count = 40;
namespace ads {
namespace problems {


class flow : public simulation_3d {
private:
    using Base = simulation_3d;
    vector_type u, u_prev;
    bool stop_gen;

    galois_executor executor{12};

    output_manager<3, FILE_FORMAT::NPY> output;

public:
    flow(const config_3d& config)
    : Base{config}
    , u{shape()}
    , u_prev{shape()}
    , stop_gen(false)
    , output{ x.B, y.B, z.B, 100 }
    { }

private:

    void prepare_matrices() {
        z.fix_left();
        z.fix_right();
        Base::prepare_matrices();
    }

    void before() override {
        prepare_matrices();

        // auto init = [](double, double, double) { return 0; };
        // projection(u, init);
        auto npy = cnpy::npy_load("state12500.npy");
        std::memcpy(u.data(), npy.data<double>(), sizeof(double) * u.size());

        solve(u);
    }

    void before_step(int /* iter */, double /*t*/) override {
        using std::swap;
        swap(u, u_prev);
    }

    void step(int iter, double /*t*/) override {
        compute_rhs(iter);
        solve(u);
    }

    // change in temperature over hight
    double delta_T(double z){
        if(h > 0.75) return 0.;
        return -5.;
    }

    double max(double x, double y){
        return x > y ? x : y;
    }

    // "polution generation"
    double f(double p){
        if (p < 0.125 && stop_gen==0)
            return (50 - p * 400);
        return 0.;
    }

    // control the cannon duration
    template <int t_begin, int t_end>
    double clock(double t) {
        if (t < t_begin || t > t_end) return 0;
        t = (t - t_begin) / (t_end - t_begin);
        return max(sin(2*M_PI*t) * cos(t), 0);
    }

    template <int t_begin, int t_end>
    double cannon(double x, double y, double z, int iter){
      if ((x > 0.2 && x < 0.8) && (y > 0.2 && y < 0.8)) {
        double t = clock<t_begin, t_end>(iter);
        if (t == 0) return 0;
        return 300 * (1 - z)
            * max(-cos(2*M_PI*x), 0)
            * max(-cos(2*M_PI*y), 0)
            * t;
      }
      return 0;
    }

    const double k_x = 1.0, k_y = 1.0, k_z = 0.1;
    void compute_rhs(int iter) {
        auto& rhs = u;

        zero(rhs);
        executor.for_each(elements(), [&](index_type e) {
            auto U = element_rhs();

            double J = jacobian(e);
            for (auto q : quad_points()) {
                double w = weight(q);

                value_type u = eval_fun(u_prev, e, q);

                for (auto a : dofs_on_element(e)) {
                    auto aa = dof_global_to_local(e, a);
                    value_type v = eval_basis(e, q, a);

                    double gradient_prod = (k_x * u.dx * v.dx) + (k_y * u.dy * v.dy) + (k_z * u.dz * v.dz);

                    double x = e[0] / static_cast<double>(element_count);
                    double y = e[1] / static_cast<double>(element_count);
                    double z = e[2] / static_cast<double>(element_count);

                    double val = 
                        (u.val * v.val) 
                        - (steps.dt * gradient_prod) 
                        + (steps.dt * (
                              //  delta_T(z)
                              - cannon<0, 300>(x, y, z, iter)
                              - cannon<200, 500>(x, y, z, iter)
                              - cannon<400, 700>(x, y, z, iter)
                              - cannon<600, 900>(x, y, z, iter)
                              - cannon<800, 1100>(x, y, z, iter)
                              - cannon<1000, 1300>(x, y, z, iter)
                              - cannon<1200, 1500>(x, y, z, iter)
                              - cannon<1400, 1700>(x, y, z, iter)
                              - cannon<1600, 1900>(x, y, z, iter)
                              - cannon<1800, 2100>(x, y, z, iter)
                              - cannon<2000, 2300>(x, y, z, iter)
                            ) * u.dz * v.val)
                        ; //+ steps.dt * f(e[2]) * v.val;

                    U(aa[0], aa[1], aa[2]) += val * w * J;
                }
            }

            executor.synchronized([&] { update_global_rhs(rhs, U, e); });
        });
    }

    void after_step(int iter, double /*t*/) override {
        // save simulation state
        // if (iter == 12500) {
        //   cnpy::npy_save("state12500.npy", u.data(), {u.size()});
        //   exit(0);
        // }
        if (iter > 5000) stop_gen = true;
        if ((iter+1) % 10 == 0) {
            std::cout << iter + 1 << std::endl;
            output.to_file(u, "out_%d.npy", iter + 1);
        }
    }
};

}
}

#endif /* ADS_PROBLEMS_CANNON_CANNON_HPP_ */
