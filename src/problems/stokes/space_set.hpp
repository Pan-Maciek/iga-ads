#ifndef PROBLEMS_STOKES_SPACE_SET_HPP_
#define PROBLEMS_STOKES_SPACE_SET_HPP_

#include "ads/simulation.hpp"


namespace ads {

struct space_set {
    dimension U1x, U1y;
    dimension U2x, U2y;
    dimension Px, Py;
};

inline int total_dimension(const space_set& s) {
   auto dimU1 = s.U1x.dofs() * s.U1y.dofs();
   auto dimU2 = s.U2x.dofs() * s.U2y.dofs();
   auto dimP = s.Px.dofs() * s.Py.dofs();
   return dimU1 + dimU2 + dimP;
}

}

#endif /* PROBLEMS_STOKES_SPACE_SET_HPP_ */
