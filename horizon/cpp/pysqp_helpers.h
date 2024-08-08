#ifndef PYSQP_HELPERS_H
#define PYSQP_HELPERS_H

#include "src/sqp.h"
#include "pyilqr_helpers.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <typeinfo>
#include "typedefs.h"

namespace py = pybind11;
using namespace horizon;

py::dict get_qpoases_options_mpc()
{
    py::dict opts;
    opts["enableEqualities"] = true;
    opts["initialStatusBounds"] = "inactive";
    opts["numRefinementSteps"] = 0;
    opts["enableDriftCorrection"] = 0;
    opts["terminationTolerance"] = 10e9 * std::numeric_limits<Real>::epsilon();
    opts["enableFlippingBounds"] = false;
    opts["enableNZCTests"] = false;
    opts["enableRamping"] = false;
    opts["enableRegularisation"] = true;
    opts["numRegularisationSteps"] = 2;
    opts["epsRegularisation"] = 5. * 10e3 * std::numeric_limits<Real>::epsilon();
    return opts;
}


py::dict get_qpoases_options_reliable()
{
    py::dict opts;
    opts["enableEqualities"] = false;
    opts["numRefinementSteps"] = 2;
    opts["enableFullLITest"] = true;
    opts["epsLITests"] = 10e5 * std::numeric_limits<Real>::epsilon();
    opts["maxDualJump"] = 10e8;
    opts["enableCholeskyRefactorisation"] = 1;
    return opts;
}


template <typename T>
bool setOption(const std::string& key, const std::string& solver_key, py::handle& value, casadi::Dict& dict)
{
    bool success = false;
    if(key == solver_key)
    {
        dict[key] = value.cast<T>();
        success = true;
    }
    return success;
}


bool checkOptions(const std::string& key, py::handle& value, casadi::Dict& dict)
{
    // -- sqp options --//
    if(setOption    <Real>        (key,   "beta",                             value, dict)) return true;
    if(setOption    <Real>        (key,   "eps_regularization",               value, dict)) return true;
    if(setOption    <Real>        (key,   "alpha_min",                        value, dict)) return true;
    if(setOption    <int>           (key,   "max_iter",                         value, dict)) return true;
    if(setOption    <bool>          (key,   "reinitialize_qpsolver",            value, dict)) return true;
    if(setOption    <Real>        (key,   "merit_derivative_tolerance",       value, dict)) return true;
    if(setOption    <Real>        (key,   "merit_eps",                        value, dict)) return true;
    if(setOption    <Real>        (key,   "constraint_violation_tolerance",   value, dict)) return true;
    if(setOption    <Real>        (key,   "solution_convergence",             value, dict)) return true;
    if(setOption    <bool>          (key,   "use_golden_ratio_update",          value, dict)) return true;
    // -- qpoases options --//
    if(setOption    <bool>          (key,   "sparse",                           value, dict)) return true;
    if(setOption    <bool>          (key,   "schur",                            value, dict)) return true;
    if(setOption    <std::string>   (key,   "hessian_type",                     value, dict)) return true;
    if(setOption    <int>           (key,   "max_schur",                        value, dict)) return true;
    if(setOption    <std::string>   (key,   "linsol_plugin",                    value, dict)) return true;
    if(setOption    <int>           (key,   "nWSR",                             value, dict)) return true;
    if(setOption    <Real>        (key,   "CPUtime",                          value, dict)) return true;
    if(setOption    <std::string>   (key,   "printLevel",                       value, dict)) return true;
    if(setOption    <bool>          (key,   "enableRamping",                    value, dict)) return true;
    if(setOption    <bool>          (key,   "enableFarBounds",                  value, dict)) return true;
    if(setOption    <bool>          (key,   "enableFlippingBounds",             value, dict)) return true;
    if(setOption    <bool>          (key,   "enableRegularisation",             value, dict)) return true;
    if(setOption    <bool>          (key,   "enableFullLITests",                value, dict)) return true;
    if(setOption    <bool>          (key,   "enableNZCTests",                   value, dict)) return true;
    if(setOption    <int>           (key,   "enableDriftCorrection",            value, dict)) return true;
    if(setOption    <int>           (key,   "enableCholeskyRefactorisation",    value, dict)) return true;
    if(setOption    <bool>          (key,   "enableEqualities",                 value, dict)) return true;
    if(setOption    <Real>        (key,   "terminationTolerance",             value, dict)) return true;
    if(setOption    <Real>        (key,   "boundTolerance",                   value, dict)) return true;
    if(setOption    <Real>        (key,   "boundRelaxation",                  value, dict)) return true;
    if(setOption    <Real>        (key,   "epsNum",                           value, dict)) return true;
    if(setOption    <Real>        (key,   "epsDen",                           value, dict)) return true;
    if(setOption    <Real>        (key,   "maxPrimalJump",                    value, dict)) return true;
    if(setOption    <Real>        (key,   "maxDualJump",                      value, dict)) return true;
    if(setOption    <Real>        (key,   "initialRamping",                   value, dict)) return true;
    if(setOption    <Real>        (key,   "finalRamping",                     value, dict)) return true;
    if(setOption    <Real>        (key,   "initialFarBounds",                 value, dict)) return true;
    if(setOption    <Real>        (key,   "growFarBounds",                    value, dict)) return true;
    if(setOption    <std::string>   (key,   "initialStatusBounds",              value, dict)) return true;
    if(setOption    <Real>        (key,   "epsFlipping",                      value, dict)) return true;
    if(setOption    <int>           (key,   "numRegularisationSteps",           value, dict)) return true;
    if(setOption    <Real>        (key,   "epsRegularisation",                value, dict)) return true;
    if(setOption    <int>           (key,   "numRefinementSteps",               value, dict)) return true;
    if(setOption    <Real>        (key,   "epsIterRef",                       value, dict)) return true;
    if(setOption    <Real>        (key,   "epsLITests",                       value, dict)) return true;
    if(setOption    <Real>        (key,   "epsNZCTests",                      value, dict)) return true;
    if(setOption    <bool>          (key,   "enableInertiaCorrection",          value, dict)) return true;
    // -- osqp options --//
    if(setOption    <bool>          (key,   "warm_start_primal",                value, dict)) return true;
    if(setOption    <bool>          (key,   "warm_start_dual",                  value, dict)) return true;
    if(setOption    <Real>        (key,   "osqp.rho",                         value, dict)) return true;
    if(setOption    <Real>        (key,   "osqp.sigma",                       value, dict)) return true;
    if(setOption    <int>           (key,   "osqp.scaling",                     value, dict)) return true;
    if(setOption    <int>           (key,   "osqp.adaptive_rho",                value, dict)) return true;
    if(setOption    <int>           (key,   "osqp.adaptive_rho_interval",       value, dict)) return true;
    if(setOption    <Real>        (key,   "osqp.adaptive_rho_tolerance",      value, dict)) return true;
    if(setOption    <int>           (key,   "osqp.max_iter",                    value, dict)) return true;
    if(setOption    <Real>        (key,   "osqp.eps_abs",                     value, dict)) return true;
    if(setOption    <Real>        (key,   "osqp.eps_rel",                     value, dict)) return true;
    if(setOption    <Real>        (key,   "osqp.eps_prim_inf",                value, dict)) return true;
    if(setOption    <Real>        (key,   "osqp.eps_dual_inf",                value, dict)) return true;
    if(setOption    <Real>        (key,   "osqp.alpha",                       value, dict)) return true;
    if(setOption    <Real>        (key,   "osqp.delta",                       value, dict)) return true;
    if(setOption    <int>           (key,   "osqp.polish",                      value, dict)) return true;
    if(setOption    <int>           (key,   "osqp.polish_refine_iter",          value, dict)) return true;
    if(setOption    <int>           (key,   "osqp.verbose",                     value, dict)) return true;
    if(setOption    <int>           (key,   "osqp.scaled_termination",          value, dict)) return true;
    if(setOption    <int>           (key,   "osqp.check_termination",           value, dict)) return true;


    return false;
}




auto constructMX(const std::string& name, const std::string& qp_solver,
               py::object f, py::object g,
               const py::dict& opt = py::dict())
{
    casadi::Dict casadi_opts = casadi::Dict();

    for (std::pair<py::handle, py::handle> item : opt)
    {
        if(!checkOptions(item.first.cast<std::string>(), item.second, casadi_opts))
            std::cout<<"option: "<<item.first.cast<std::string>()<<"missing from parser!"<<std::endl;
    }

    return std::make_unique< SQPGaussNewton< casadi::MX > >(name, qp_solver, to_cpp(f), to_cpp(g), casadi_opts);
}

auto constructSX(std::string name, std::string qp_solver,
               py::object f, py::object g,
               const py::dict& opt = py::dict())
{
    casadi::Dict casadi_opts = casadi::Dict();

    for (std::pair<py::handle, py::handle> item : opt)
    {
        if(!checkOptions(item.first.cast<std::string>(), item.second, casadi_opts))
            std::cout<<"option: "<<item.first.cast<std::string>()<<"missing from parser!"<<std::endl;
    }

    return std::make_unique< SQPGaussNewton< casadi::SX > >(name, qp_solver, to_cpp(f), to_cpp(g), casadi_opts);
}

bool fMX(SQPGaussNewton<casadi::MX>& self, py::object f, bool reinitialize_qp_solver = true)
{
    return self.f(to_cpp(f), reinitialize_qp_solver);
}

bool fSX(SQPGaussNewton<casadi::SX>& self, py::object f, bool reinitialize_qp_solver = true)
{
    return self.f(to_cpp(f), reinitialize_qp_solver);
}

bool gMX(SQPGaussNewton<casadi::MX>& self, py::object g, bool reinitialize_qp_solver = true)
{
    return self.f(to_cpp(g), reinitialize_qp_solver);
}

bool gSX(SQPGaussNewton<casadi::SX>& self, py::object g, bool reinitialize_qp_solver = true)
{
    return self.f(to_cpp(g), reinitialize_qp_solver);
}

auto callMX(SQPGaussNewton<casadi::MX>& self,
            const VectorXr& x0,
            const VectorXr& p,
            const VectorXr& lbx, const VectorXr& ubx,
            const VectorXr& lbg, const VectorXr& ubg)
{
    casadi::DM _x0_, _p_, _lbx_, _ubx_, _lbg_, _ubg_;
    casadi_utils::toCasadiMatrix(x0, _x0_);
    casadi_utils::toCasadiMatrix(p, _p_);
    casadi_utils::toCasadiMatrix(lbx, _lbx_);
    casadi_utils::toCasadiMatrix(ubx, _ubx_);
    casadi_utils::toCasadiMatrix(lbg, _lbg_);
    casadi_utils::toCasadiMatrix(ubg, _ubg_);
    casadi::DMDict tmp = self.solve(_x0_, _p_, _lbx_, _ubx_, _lbg_, _ubg_);

    py::dict solution;
    VectorXr x;
    casadi_utils::toEigen(tmp.at("x"), x);
    solution["x"] = x;
    solution["f"] = Real(tmp.at("f")(0));
    solution["g"] = Real(tmp.at("g")(0));
    return solution;
}


auto callSX(SQPGaussNewton<casadi::SX>& self,
            const VectorXr& x0,
            const VectorXr& p,
            const VectorXr& lbx,
            const VectorXr& ubx,
            const VectorXr& lbg,
            const VectorXr& ubg)
{
    casadi::DM _x0_, _p_, _lbx_, _ubx_, _lbg_, _ubg_;
    casadi_utils::toCasadiMatrix(x0, _x0_);
    casadi_utils::toCasadiMatrix(p, _p_);
    casadi_utils::toCasadiMatrix(lbx, _lbx_);
    casadi_utils::toCasadiMatrix(ubx, _ubx_);
    casadi_utils::toCasadiMatrix(lbg, _lbg_);
    casadi_utils::toCasadiMatrix(ubg, _ubg_);
    casadi::DMDict tmp = self.solve(_x0_, _p_, _lbx_, _ubx_, _lbg_, _ubg_);

    py::dict solution;
    VectorXr x;
    casadi_utils::toEigen(tmp.at("x"), x);
    solution["x"] = x;
    solution["f"] = Real(tmp.at("f")(0));
    solution["g"] = Real(tmp.at("g")(0));
    return solution;
}

#endif
