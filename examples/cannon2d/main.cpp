#include "cannon.hpp"
#include "argh.h"

using namespace ads;
using namespace ads::problems;

int main(int argc, char** argv) {
  argh::parser cmdl(argc, argv);

  int iterations;
  double dt;

  cmdl({"-i", "--iterations"}) >> iterations;
  cmdl({"-dt", "-s", "--step"}, 1e-5) >> dt;

  std::cerr << "Iterations: " << iterations << std::endl;
  std::cerr << "Time step: " << dt << std::endl;
  std::cerr << "Full time: " << dt * iterations << std::endl;

  dim_config dim { 2, element_count };
  timesteps_config steps { iterations, dt };
  int ders = 1;

  config_2d c {dim, dim, steps, ders};
  flow sim { c };
  sim.run();

  return EXIT_SUCCESS;
}

