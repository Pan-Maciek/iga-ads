#ifndef ADS_PROBLEMS_CANNON_CANNON_HPP_
#define ADS_PROBLEMS_CANNON_CANNON_HPP_

#include <cmath>
#include <iostream>
#include "ads/executor/galois.hpp"

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

    output_manager<3, FILE_FORMAT::VTI> output;

public:
    flow(const config_3d& config)
    : Base{config}
    , u{shape()}
    , u_prev{shape()}
    , stop_gen(false)
    , output{ x.B, y.B, z.B, 100 }
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
    }

    void before_step(int /* iter */, double /*t*/) override {
        using std::swap;
        swap(u, u_prev);
    }

    void step(int iter, double /*t*/) override {
        compute_rhs(iter);
        solve(u);
    }

    double delta_T(int p){
        if(p>30) return 0.;
        return -5.;
    }

    double max(double x, double y){
        return x > y ? x : y;
    }

    double f(int p){
        if(p<5&&stop_gen==0)
            return (1.-p/5.) * 50;
        else if(p<35)
            return 0.;
        // else
        //     return 7.-p/5.;
        
        return 0.;
        
    }

    double cannon(double x, double y, double z, double t){
        x=x/40.;
        y=y/40.;
        z=z/40.;

        if((x > 0.3 && x <0.6) && (y > 0.3 && y < 0.6) && t > 10000)
            return 200.*(1.-z)
              * max(sin(10*M_PI*x),0)
              * max(sin(10*M_PI*y),0)
              * max(0,sin(M_PI*(t-8000)/1000));
        else return 0.;
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
                    double val = 
                        (u.val * v.val) 
                        - (steps.dt * gradient_prod) 
                        + (steps.dt * (delta_T(e[2]) - cannon(e[0], e[1], e[2], iter)) * u.dz * v.val)
                        + steps.dt * f(e[2]) * v.val;

                    U(aa[0], aa[1], aa[2]) += val * w * J;
                }
            }

            executor.synchronized([&] { update_global_rhs(rhs, U, e); });
        });
    }

    void after_step(int iter, double /*t*/) override {
        std::cout << iter << std::endl;
        if (iter > 5000) stop_gen = true;
        if ((iter + 1) % 100 == 0) {
            output.to_file(u, "out_%d.vti", iter + 1);
        }
    }
};

}
}

#endif /* ADS_PROBLEMS_CANNON_CANNON_HPP_ */
