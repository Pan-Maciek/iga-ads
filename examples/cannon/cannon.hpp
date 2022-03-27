#ifndef ADS_PROBLEMS_CANNON_CANNON_HPP_
#define ADS_PROBLEMS_CANNON_CANNON_HPP_

#include <cmath>
#include <iostream>
#include "ads/executor/galois.hpp"

#include "ads/simulation.hpp"
#include "ads/output_manager.hpp"

const int iterations = 30000;
namespace ads {
namespace problems {


class flow : public simulation_3d {
private:
    using Base = simulation_3d;
    vector_type u, u_prev;

    galois_executor executor{4};

    output_manager<3, FILE_FORMAT::NPY> output;

public:
    flow(const config_3d& config)
    : Base{config}
    , u{shape()}
    , u_prev{shape()}
    , output{ x.B, y.B, z.B, 200 }
    { }

    double init_state(double, double, double) {
      // pomyśleć nad innym statnem początkowym, chmura już w powietrzu
        return 0;
    };

private:

    void prepare_matrices() {
        z.fix_left();
        z.fix_right();
        Base::prepare_matrices();
    }

    void before() override {
        prepare_matrices();

        auto init = [this](double x, double y, double z) { return init_state(x, y, z); };
        projection(u, init);
        solve(u);

        output.to_file(u, "out_%d.vti", 0);
    }

    double s;
    void before_step(int /* iter */, double /*t*/) override {
        using std::swap;
        swap(u, u_prev);
        // const double d = 0.7;
        // const double c = 10000;
        // s = std::max(((cos(iter * 3.14159265358979 / c) - d) * 1 / (1-d)), 0.);
        // std::cout << "\r" << iter << "/" << iterations << " (s=" << s << ")                          \r";
    }

    void step(int /*iter*/, double /*t*/) override {
        compute_rhs();
        solve(u);
    }

    double f(double h) {
        if (h <= 0.125) return (150 - 1200 * h);
        // if (h <= 0.125) return (150 - 1200 * h) * s;
        return 0;
    }

    double e2h(double e) {
        return e / 20;
    }

    double dTz(double h) {
      if (h >= 0.8) return 0;
      return -5.2;
    }

    const double k_x = 1.0, k_y = 1.0, k_z = 0.1;
    void compute_rhs() {
        auto& rhs = u;

        zero(rhs);
        executor.for_each(elements(), [&](index_type e) {
            auto U = element_rhs();

            double J = jacobian(e);
            for (auto q : quad_points()) {
                double w = weight(q);

                value_type u = eval_fun(u_prev, e, q);
                double h = e2h(e[2]);

                for (auto a : dofs_on_element(e)) {
                    auto aa = dof_global_to_local(e, a);
                    value_type v = eval_basis(e, q, a);

                    double gradient_prod = (k_x * u.dx * v.dx) + (k_y * u.dy * v.dy) + (k_z * u.dz * v.dz);
                    double val = 
                        (u.val * v.val) 
                        - (steps.dt * gradient_prod) 
                        + (steps.dt * dTz(h) * u.dz * v.val)
                        + steps.dt * f(h) * v.val;

                    U(aa[0], aa[1], aa[2]) += val * w * J;
                }
            }

            executor.synchronized([&] { update_global_rhs(rhs, U, e); });
        });
    }

    void after_step(int iter, double /*t*/) override {
        std::cout << iter << std::endl;
        if ((iter + 1) % 100 == 0) {
            output.to_file(u, "out_%d.npy", iter + 1);
            // output_npy.to_file(u, "out_%d.npy", iter + 1);
        }
    }
};

}
}

#endif /* ADS_PROBLEMS_CANNON_CANNON_HPP_ */
