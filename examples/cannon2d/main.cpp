#include "cannon.hpp"
#include "argh.h"

using namespace ads;
using namespace ads::problems;

int main(int argc, char** argv) {
  argh::parser cmdl(argc, argv);

  int iterations, element_count;
  double dt;
  std::string initial_state;

  cmdl({"-i", "--iterations"}) >> iterations;
  cmdl({"-dt", "-s", "--step"}, 1e-5) >> dt;
  cmdl({"--initial-state"}) >> initial_state;
  cmdl({"--element-count"}, 40) >> element_count;

  std::cerr << "Simulation: " << std::endl;
  std::cerr << "  Iterations: " << iterations << std::endl;
  std::cerr << "  Time step: " << dt << std::endl;
  std::cerr << "  Full time: " << dt * iterations << std::endl;
  std::cerr << "  Initial state: " << (initial_state == "" ? "empty" : initial_state) << std::endl;
  std::cerr << "  Elements: " << element_count << std::endl;
  std::cerr << std::endl;

  dim_config dim { 2, element_count };
  timesteps_config steps { iterations, dt };
  int ders = 1;

  config_2d c {dim, dim, steps, ders};
  flow sim { c, initial_state };
  sim.run();

  return EXIT_SUCCESS;
}

