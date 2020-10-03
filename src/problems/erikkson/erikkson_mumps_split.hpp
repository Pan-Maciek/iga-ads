#ifndef PROBLEMS_ERIKKSON_ERIKKSON_MUMPS_SPLIT_HPP_
#define PROBLEMS_ERIKKSON_ERIKKSON_MUMPS_SPLIT_HPP_

#include "ads/simulation.hpp"
#include "ads/output_manager.hpp"
#include "ads/executor/galois.hpp"
#include "ads/lin/tensor/view.hpp"
#include "mumps.hpp"


#include "ads/lin/dense_matrix.hpp"
#include "ads/lin/dense_solve.hpp"

#include "problems/erikkson/erikkson_base.hpp"



namespace ads {

class erikkson_mumps_split : public erikkson_base {
private:
    using Base = erikkson_base;
    using vector_view = lin::tensor_view<double, 2>;

    galois_executor executor{8};

    dimension Ux, Uy;
    dimension& Vx;
    dimension& Vy;


    // New stuff
    lin::band_matrix MVx, MVy;
    lin::band_matrix MUx, MUy;

    lin::band_matrix KVx, KVy;
    lin::band_matrix KUx, KUy;


    lin::dense_matrix MUVx, MUVy;
    lin::dense_matrix KUVx, KUVy;
    lin::dense_matrix AUVx, AUVy;

    lin::dense_matrix MUUx, MUUy;
    lin::dense_matrix KUUx, KUUy;
    lin::dense_matrix AUUx, AUUy;

    struct residuum {
        vector_type data;
        const dimension* Vx;
        const dimension* Vy;
    };

    vector_type u;
    residuum r;
    vector_type u_buffer;
    std::vector<double> full_rhs;

    int save_every = 1;

    // double tau = 0.1;
    // double tau = 1;
    // double rho = 0;
    double alpha = 1;
    // double gamma = 1;
    double hh = 1;//(1. / 30 / 30);

    // double bbeta = 0;


    double pecelet = 1e2;
    double epsilon = 1 / pecelet;

    point_type c_diff{{ epsilon, epsilon }};

    // double angle = 0;
    // double len = 1;

    // point_type beta{{ len * cos(angle), len * sin(angle) }};
    point_type beta{{ 1, 0 }};


    mumps::solver solver;

    output_manager<2> output;

public:
    erikkson_mumps_split(dimension trial_x, dimension trial_y, dimension test_x, dimension test_y, const timesteps_config& steps)
    : Base{std::move(test_x), std::move(test_y), steps}
    , Ux{ std::move(trial_x) }
    , Uy{ std::move(trial_y) }
    , Vx{ x }
    , Vy{ y }
    , MVx{Vx.p, Vx.p, Vx.dofs(), Vx.dofs(), 0}
    , MVy{Vy.p, Vy.p, Vy.dofs(), Vy.dofs(), 0}
    , MUx{Ux.p, Ux.p, Ux.dofs(), Ux.dofs(), 0}
    , MUy{Uy.p, Uy.p, Uy.dofs(), Uy.dofs(), 0}
    , KVx{Vx.p, Vx.p, Vx.dofs(), Vx.dofs(), 0}
    , KVy{Vy.p, Vy.p, Vy.dofs(), Vy.dofs(), 0}
    , KUx{Ux.p, Ux.p, Ux.dofs(), Ux.dofs(), 0}
    , KUy{Uy.p, Uy.p, Uy.dofs(), Uy.dofs(), 0}
    , MUVx{ Vx.dofs(), Ux.dofs() }
    , MUVy{ Vy.dofs(), Uy.dofs() }
    , KUVx{ Vx.dofs(), Ux.dofs() }
    , KUVy{ Vy.dofs(), Uy.dofs() }
    , AUVx{ Vx.dofs(), Ux.dofs() }
    , AUVy{ Vy.dofs(), Uy.dofs() }
    , MUUx{ Ux.dofs(), Ux.dofs() }
    , MUUy{ Uy.dofs(), Uy.dofs() }
    , KUUx{ Ux.dofs(), Ux.dofs() }
    , KUUy{ Uy.dofs(), Uy.dofs() }
    , AUUx{ Ux.dofs(), Ux.dofs() }
    , AUUy{ Uy.dofs(), Uy.dofs() }
    , u{{ Ux.dofs(), Uy.dofs() }}
    , r{ {{Vx.dofs(), Vy.dofs()}}, &Vx, &Vy}
    , u_buffer{{ Ux.dofs(), Uy.dofs() }}
    , full_rhs(Vx.dofs() * Vy.dofs() + Ux.dofs() * Uy.dofs())
    , output{ Ux.B, Uy.B, 500 }
    { }

private:

    struct matrix_set {
        using band_matrix_ref = lin::band_matrix&;
        using dense_matrix_ref = lin::dense_matrix&;

        band_matrix_ref MVx, MVy, KVx, KVy;
        dense_matrix_ref MUVx, MUVy, KUVx, KUVy, AUVx, AUVy;
    };

    void assemble_problem(mumps::problem& problem, bool with_x, bool with_y, const dimension& Vx, const dimension& Vy,
                          const matrix_set& M, int k) {
        using std::min;
        using std::max;
        auto N = Vx.dofs() * Vy.dofs();
        vector_type v{{ Vx.dofs(), Vy.dofs() }};

        // auto gamma = 1 - std::pow(bbeta, k);

        // Gram matrix - upper left
        for (auto i : internal_dofs(Vx, Vy)) {
            for (auto j : overlapping_internal_dofs(i, Vx, Vy)) {
                int ii = linear_index(i, Vx, Vy) + 1;
                int jj = linear_index(j, Vx, Vy) + 1;

                double val = alpha * kron(MVx, MVy, i, j);
                // if (with_x)
                    val += hh * kron(KVx, MVy, i, j);
                // if (with_y)
                    val += hh * kron(MVx, KVy, i, j);
                problem.add(ii, jj, val);
            }
        }

        // Dirichlet BC - upper left
        for_boundary_dofs(Vx, Vy, [&](index_type dof) {
            int i = linear_index(dof, Vx, Vy) + 1;
            problem.add(i, i, 1);
        });

        // Dirichlet BC - lower right
        for_boundary_dofs(Ux, Uy, [&](index_type dof) {
            int i = linear_index(dof, Ux, Uy) + 1;
            problem.add(N + i, N + i, 1);
        });

        // B, B^T
        for (auto i : dofs(Vx, Vy)) {
            for (auto j : dofs(Ux, Uy)) {
                double val = kron(M.MUVx, M.MUVy, i, j);
                if (with_x) {
                    val += steps.dt * (c_diff[0] * kron(M.KUVx, M.MUVy, i, j) + beta[0] * kron(M.AUVx, M.MUVy, i, j));
                }
                if (with_y) {
                    val += steps.dt * (c_diff[1] * kron(M.MUVx, M.KUVy, i, j) + beta[1] * kron(M.MUVx, M.AUVy, i, j));
                }
                if (val != 0) {
                    if (! is_boundary(i, Vx, Vy) && ! is_boundary(j, Ux, Uy)) {
                        int ii = linear_index(i, Vx, Vy) + 1;
                        int jj = linear_index(j, Ux, Uy) + 1;

                        problem.add(ii, N + jj, -val);
                        problem.add(N + jj, ii, val);
                    }
                }
            }
        }

        // Mass matrix - lower right
        // for (auto ix = 1; ix < Ux.dofs() - 1; ++ ix) {
        //     for (auto iy = 1; iy < Uy.dofs() - 1; ++ iy) {
        //         int i = &u(ix, iy) - &u(0, 0) + 1;
        //         for (auto jx = max(1, ix - Ux.B.degree); jx < min(Ux.dofs() - 1, ix + Ux.B.degree + 1); ++ jx) {
        //             for (auto jy = max(1, iy - Uy.B.degree); jy < min(Uy.dofs() - 1, iy + Uy.B.degree + 1); ++ jy) {
        //                 int j = &u(jx, jy) - &u(0, 0) + 1;
        //                 double val = gamma * MUx(ix, jx) * MUy(iy, jy);
        //                 problem.add(N + i, N + j, val);
        //             }
        //         }
        //     }
        // }
    }

    matrix_set matrices(bool x_refined, bool y_refined) {
        if (x_refined && y_refined) {
            return { MVx, MVy, KVx, KVy, MUVx, MUVy, KUVx, KUVy, AUVx, AUVy };
        } else if (x_refined) {
            return { MVx, MUy, KVx, KUy, MUVx, MUUy, KUVx, KUUy, AUVx, AUUy };
        } else if (y_refined) {
            return { MUx, MVy, KUx, KVy, MUUx, MUVy, KUUx, KUVy, AUUx, AUVy };
        } else {
            return { MUx, MUy, KUx, KUy, MUUx, MUUy, KUUx, KUUy, AUUx, AUUy };
        }
    }

    void prepare_matrices() {
        gram_matrix_1d(MVx, Vx.basis);
        gram_matrix_1d(MVy, Vy.basis);

        gram_matrix_1d(MUx, Ux.basis);
        gram_matrix_1d(MUy, Uy.basis);

        gram_matrix_1d(MUUx, Ux.basis, Ux.basis);
        gram_matrix_1d(MUUy, Uy.basis, Uy.basis);

        gram_matrix_1d(MUVx, Ux.basis, Vx.basis);
        gram_matrix_1d(MUVy, Uy.basis, Vy.basis);


        stiffness_matrix_1d(KVx, Vx.basis);
        stiffness_matrix_1d(KVy, Vy.basis);

        stiffness_matrix_1d(KUx, Ux.basis);
        stiffness_matrix_1d(KUy, Uy.basis);

        stiffness_matrix_1d(KUVx, Ux.basis, Vx.basis);
        stiffness_matrix_1d(KUVy, Uy.basis, Vy.basis);

        stiffness_matrix_1d(KUUx, Ux.basis, Ux.basis);
        stiffness_matrix_1d(KUUy, Uy.basis, Uy.basis);


        advection_matrix_1d(AUVx, Ux.basis, Vx.basis);
        advection_matrix_1d(AUVy, Uy.basis, Vy.basis);

        advection_matrix_1d(AUUx, Ux.basis, Ux.basis);
        advection_matrix_1d(AUUy, Uy.basis, Uy.basis);
    }

    double init_state(double x, double y) {
        double dx = x - 0.5;
        double dy = y - 0.5;
        double r2 = std::min( (dx * dx + dy * dy), 1.0);
        // return (r2 - 1) * (r2 - 1) * (r2 + 1) * (r2 + 1);
        // return 1;
        return 0;
    };

    void before() override {
        prepare_matrices();
        Ux.factorize_matrix();
        Uy.factorize_matrix();

        auto init = [this](double x, double y) { return init_state(x, y); };
        compute_projection(u, Ux.basis, Uy.basis, init);
        ads_solve(u, u_buffer, Ux.data(), Uy.data());

        zero(r.data);
        zero(u);
        // u(5, 5) = 1;

        output.to_file(u, "out_0.data");
    }

    void copy_solution(const vector_view& u_rhs, vector_type& u, const dimension& Vx, const dimension& Vy) {
        for (auto i = 0; i < Ux.dofs(); ++ i) {
            for (auto j = 0; j < Uy.dofs(); ++ j) {
                u(i, j) = u_rhs(i, j);
            }
        }
        // r = residuum{ {{Vx.dofs(), Vy.dofs()}}, &Vx, &Vy };
        // for (auto i = 0; i < Vx.dofs(); ++ i) {
        //     for (auto j = 0; j < Vy.dofs(); ++ j) {
        //         r.data(i, j) = r_rhs(i, j);
        //     }
        // }
    }

    void substep(vector_type& u, bool x_on_rhs, bool y_on_rhs, bool x_refined, bool y_refined, int k, double t) {
        dimension& Vx = x_refined ? this->Vx : Ux;
        dimension& Vy = y_refined ? this->Vy : Uy;

        vector_view r_rhs{full_rhs.data(), {Vx.dofs(), Vy.dofs()}};
        vector_view u_rhs{full_rhs.data() + r_rhs.size(), {Ux.dofs(), Uy.dofs()}};


        std::fill(begin(full_rhs), end(full_rhs), 0);
        compute_rhs(x_on_rhs, y_on_rhs, Vx, Vy, r_rhs, u_rhs, k, t);

        zero_bc(r_rhs, Vx, Vy);
        zero_bc(u_rhs, Ux, Uy);

        int size = Vx.dofs() * Vy.dofs() + Ux.dofs() * Uy.dofs();
        mumps::problem problem(full_rhs.data(), size);
        assemble_problem(problem, !x_on_rhs, !y_on_rhs, Vx, Vy, matrices(x_refined, y_refined), k + 1);
        solver.solve(problem);

        copy_solution(u_rhs, u, Vx, Vy);
    }

    void step(int iter, double t) override {
        bool xrhs[] = {  true, false };

        bool x_rhs = xrhs[iter % sizeof(xrhs)];
        bool y_rhs = !x_rhs;
        std::cout << iter << ": x " << (x_rhs ? "rhs" : "lhs") << ", y " << (y_rhs ? "rhs" : "lhs") << std::endl;

        bool x_on_rhs = false;//x_rhs;
        bool y_on_rhs = false;//y_rhs;
        bool x_refine = true;//!x_rhs;//true;
        bool y_refine = true;//!y_rhs;//true;

        // substep(u, x_on_rhs, y_on_rhs, x_refine, y_refine, iter);
        // substep(x_rhs, y_rhs, !x_rhs, !y_rhs);
        // substep(false, false, true, true, iter);

        // vector_type uu{{ Ux.dofs(), Uy.dofs() }};
        // double dt = steps.dt;

        // substep(uu, x_on_rhs, y_on_rhs, x_refine, y_refine, iter);
        // double e1 = errorL2(uu);

        // steps.dt = dt * 2;
        // substep(uu, x_on_rhs, y_on_rhs, x_refine, y_refine, iter);
        // double e2 = errorL2(uu);

        // steps.dt = dt / 2;
        // substep(uu, x_on_rhs, y_on_rhs, x_refine, y_refine, iter);
        // double e3 = errorL2(uu);

        // double min = std::min(e1, std::min(e2, e3));
        // if (min == e1) {
        //     std::cout << "dt := dt" << std::endl;
        //     steps.dt = dt;
        // } else if (min == e2) {
        //     std::cout << "dt := 2 dt" << std::endl;
        //     steps.dt = dt * 2;
        // } else if (min == e3) {
        //     std::cout << "dt := dt/2" << std::endl;
        //     steps.dt = dt / 2;
        // }

        // std::cout << "dt: " << e1 << ", 2dt: " << e2 << ", dt/2: " << e3 << std::endl;
        // std::cout << "dt = " << steps.dt << std::endl;

        substep(u, x_on_rhs, y_on_rhs, x_refine, y_refine, iter, t);
    }

    void after_step(int iter, double t) override {
        if ((iter + 1) % save_every == 0) {
            std::cout << "Step " << (iter + 1) << " : " << errorL2(u, t) << " " << errorH1(u, t) << std::endl;
            output.to_file(u, "out_%d.data", (iter + 1) / save_every);
        }
    }

    void after() override {
        plot_middle("final.data", u, Ux, Uy);
        // std::cout << "{ 'L2': '" << errorL2(u) << "', 'H1': '" << errorH1(u) << "'}" << std::endl;

        std::ofstream sol("solution.data");
        for (int i = 0; i < Ux.dofs(); ++ i) {
            for (int j = 0; j < Uy.dofs(); ++ j) {
                sol << i << " " << j << " " << u(i, j) << std::endl;
            }
        }
    }



    void compute_rhs(bool with_x, bool with_y, const dimension& Vx, const dimension& Vy,
                     vector_view& r_rhs, vector_view& u_rhs, int k, double t) {
        // auto gamma = 1 - std::pow(bbeta, k);

        executor.for_each(elements(Vx, Vy), [&](index_type e) {
            auto R = vector_type{{ Vx.basis.dofs_per_element(), Vy.basis.dofs_per_element() }};
            auto U = vector_type{{ Ux.basis.dofs_per_element(), Uy.basis.dofs_per_element() }};

            double J = jacobian(e);
            for (auto q : quad_points(Vx, Vy)) {
                double W = weigth(q);
                double WJ = W * J;
                auto x = point(e, q);
                value_type uu = eval(u, e, q, Ux, Uy);
                // value_type rr = eval(r.data, e, q, *r.Vx, *r.Vy);

                for (auto a : dofs_on_element(e, Vx, Vy)) {
                    auto aa = dof_global_to_local(e, a, Vx, Vy);
                    value_type v = eval_basis(e, q, a, Vx, Vy);
                    // double F = erikkson2_forcing(x[0], x[1], epsilon);
                    double F = erikkson_forcing(x[0], x[1], epsilon, t);

                    double lv = uu.val * v.val;
                    if (with_x) {
                        // lv -= steps.dt * (c_diff[0] * uu.dx * v.dx + beta[0] * uu.dx * v.val);
                    }
                    if (with_y) {
                        // lv -= steps.dt * (c_diff[1] * uu.dy * v.dy + beta[1] * uu.dy * v.val);
                    }
                    lv += steps.dt * F * v.val;
                    double val = -lv;
                    // val += alpha * rr.val * v.val;
                    // if (with_x) val -= hh * rr.dx * v.dx;
                    // if (with_y) val -= hh * rr.dy * v.dy;

                    R(aa[0], aa[1]) += val * WJ;
                }
                // for (auto a : dofs_on_element(e, Ux, Uy)) {
                //     auto aa = dof_global_to_local(e, a, Ux, Uy);
                //     value_type w = eval_basis(e, q, a, Ux, Uy);
                //     double val = 0;
                //     if (with_y) {
                //         val += c_diff[0] * w.dx * rr.dx + beta[0] * w.dx * rr.val;
                //         val += c_diff[1] * w.dy * rr.dy + beta[1] * w.dy * rr.val;
                //     }
                //     val += gamma * w.val * uu.val;
                //     U(aa[0], aa[1]) += val * WJ;
                // }
            }
            executor.synchronized([&]() {
                update_global_rhs(r_rhs, R, e, Vx, Vy);
                update_global_rhs(u_rhs, U, e, Ux, Uy);
            });
        });
    }

    double errorL2(const vector_type& u, double t) const {
        // auto sol = exact(epsilon);
        // auto sol = [&](point_type x) { return erikkson2_exact(x[0], x[1], epsilon); };
        auto sol = [&](point_type x) { return erikkson_nonstationary_exact(x[0], x[1], t); };

        return Base::errorL2(u, Ux, Uy, sol) / normL2(Ux, Uy, sol) * 100;
    }

    double errorH1(const vector_type& u, double t) const {
        // auto sol = exact(epsilon);
        // auto sol = [&](point_type x) { return erikkson2_exact(x[0], x[1], epsilon); };
        auto sol = [&](point_type x) { return erikkson_nonstationary_exact(x[0], x[1], t); };

        return Base::errorH1(u, Ux, Uy, sol) / normH1(Ux, Uy, sol) * 100;
    }
};

}



#endif /* ADS_PROBLEMS_ERIKKSON_ERIKKSON_MUMPS_SPLIT_HPP */
