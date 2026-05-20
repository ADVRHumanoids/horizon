#ifndef HORIZON_ILQR_LIE_H
#define HORIZON_ILQR_LIE_H

#include <casadi/casadi.hpp>

namespace horizon::lie
{

casadi::Function jacobian(casadi::Function f,
                          casadi::Function xsum,
                          casadi::Function xdiff,
                          std::vector<std::string> out,
                          std::string name,
                          bool lie_output = false);

}
#endif // LIE_H
