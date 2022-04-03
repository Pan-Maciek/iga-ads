#include "cannon.hpp"

using namespace ads;
using namespace ads::problems;

int main() {
    dim_config dim { 2, element_count };
    timesteps_config steps { iterations, 1e-5 };
    int ders = 1;

    config_3d c {dim, dim, dim, steps, ders};
    flow sim { c };
    sim.run();
}

