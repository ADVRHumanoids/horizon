#ifndef PYILQR_HELPERS_H
#define PYILQR_HELPERS_H

#include "src/ilqr.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "src/typedefs.h"

namespace py = pybind11;
using namespace horizon;

casadi::Function to_cpp(py::object pyfn)
{
    // convert python's casadi.Function to cpp's casadi::Function
    #if PYBIND11_VERSION_MINOR > 6
        auto cs = py::module_::import("casadi");
    #else
        auto cs = py::module::import("casadi");
    #endif

    auto Function = cs.attr("Function");
    auto serialize = Function.attr("serialize");
    auto fstr = serialize(pyfn).cast<std::string>();

    return casadi::Function::deserialize(fstr);
}

auto construct(py::object fdyn, int N, IterativeLQR::OptionDict opt)
{
    return std::make_unique<IterativeLQR>(to_cpp(fdyn), N, opt);
}

auto set_inter_cost_wrapper_single(IterativeLQR& self, std::vector<int> k, py::object f)
{
    self.setCost(k, to_cpp(f));
}

auto set_residual_wrapper_single(IterativeLQR& self, std::vector<int> k, py::object f)
{
    self.setResidual(k, to_cpp(f));
}

auto set_final_cost_wrapper(IterativeLQR& self, py::object pyfn)
{
    self.setFinalCost(to_cpp(pyfn));
}

auto set_final_constraint_wrapper(IterativeLQR& self, py::object pyfn)
{
    self.setFinalConstraint(to_cpp(pyfn));
}


auto set_inter_constraint_wrapper_single(IterativeLQR& self,
                                         std::vector<int> k,
                                         py::object f,
                                         std::vector<VectorXr> tgt)
{
    self.setConstraint(k, to_cpp(f), tgt);
}

#endif // PYILQR_HELPERS_H
