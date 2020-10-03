#include "problems/validation/validation.hpp"

using namespace ads;
using namespace ads::problems;


int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: validation <p> <n> <nsteps>" << std::endl;
        return 0;
    }
    int p = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int nsteps = std::atoi(argv[3]);

    if (n <= 0) {
        std::cerr << "Invalid value of n: " << argv[1] << std::endl;
    }

    double T = 1.0;
    double dt = T / nsteps;
    nsteps /= 10;
    // nsteps += 1;

    dim_config dim{ p, n };
    timesteps_config steps{ nsteps, dt };
    int ders = 1;

    config_2d c{dim, dim, steps, ders};
    validation sim{c};
    sim.run();
}
