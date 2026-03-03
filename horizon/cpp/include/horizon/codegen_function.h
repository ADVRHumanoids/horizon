#ifndef CODEGEN_FUNCTION_H
#define CODEGEN_FUNCTION_H

#include <casadi/casadi.hpp>

namespace horizon { namespace utils {

casadi::Function codegen(const casadi::Function& f,
                         std::string dir=".");

} }

#endif // CODEGEN_FUNCTION_H
