#if defined(HORIZON_WITH_KOKKOS) && defined(HORIZON_WITH_PINOCCHIO)
#include "horizon/device_collision.h"
#endif
#include "pyilqr_helpers.h"

void bindIlqrFunctions(py::module_ &m);
void bindIlqrCollision(py::module_ &m);
#ifdef HORIZON_WITH_PINOCCHIO
void bindIlqrPinocchio(py::module_ &m);
#endif

PYBIND11_MODULE(pyilqr, m)
{
    bindIlqrFunctions(m);
    bindIlqrCollision(m);
#ifdef HORIZON_WITH_PINOCCHIO
    bindIlqrPinocchio(m);
#endif

    py::class_<IterativeLQR::ForwardPassResult>(m, "ForwardPassResult")
        .def("print", &IterativeLQR::ForwardPassResult::print, py::arg("N") = 1)
        .def_readonly("xtrj", &IterativeLQR::ForwardPassResult::xtrj)
        .def_readonly("utrj", &IterativeLQR::ForwardPassResult::utrj)
        .def_readonly("accepted", &IterativeLQR::ForwardPassResult::accepted)
        .def_readonly("rho", &IterativeLQR::ForwardPassResult::rho)
        .def_readonly("alpha", &IterativeLQR::ForwardPassResult::alpha)
        .def_readonly("hxx_reg", &IterativeLQR::ForwardPassResult::hxx_reg)
        .def_readonly("derivative_evaluations",
                      &IterativeLQR::ForwardPassResult::derivative_evaluations)
        .def_readonly("value_evaluations",
                      &IterativeLQR::ForwardPassResult::value_evaluations)
        .def_readonly("damping", &IterativeLQR::ForwardPassResult::damping)
        .def_readonly("predicted_reduction",
                      &IterativeLQR::ForwardPassResult::predicted_reduction)
        .def_readonly("actual_reduction",
                      &IterativeLQR::ForwardPassResult::actual_reduction)
        .def_readonly("model_agreement",
                      &IterativeLQR::ForwardPassResult::model_agreement)
        .def_readonly("stationarity",
                      &IterativeLQR::ForwardPassResult::stationarity)
        .def_readonly("trial_count",
                      &IterativeLQR::ForwardPassResult::trial_count)
        .def_readonly("direction_solves",
                      &IterativeLQR::ForwardPassResult::direction_solves)
        .def_readonly("used_correction",
                      &IterativeLQR::ForwardPassResult::used_correction)
        .def_readonly("used_restoration",
                      &IterativeLQR::ForwardPassResult::used_restoration)
        .def_readonly("acceptance_reason",
                      &IterativeLQR::ForwardPassResult::acceptance_reason)
        .def_readonly("bound_violation",
                      &IterativeLQR::ForwardPassResult::bound_violation)
        .def_readonly("constraint_violation",
                      &IterativeLQR::ForwardPassResult::constraint_violation)
        .def_readonly("constraint_values",
                      &IterativeLQR::ForwardPassResult::constraint_values)
        .def_readonly("defect_values",
                      &IterativeLQR::ForwardPassResult::defect_values)
        .def_readonly("cost", &IterativeLQR::ForwardPassResult::cost)
        .def_readonly("defect_norm",
                      &IterativeLQR::ForwardPassResult::defect_norm)
        .def_readonly("merit", &IterativeLQR::ForwardPassResult::merit)
        //            .def_readonly("armijo_merit",
        //            &IterativeLQR::ForwardPassResult::armijo_merit)
        .def_readonly("merit_der", &IterativeLQR::ForwardPassResult::merit_der)
        .def_readonly("mu_f", &IterativeLQR::ForwardPassResult::mu_f)
        .def_readonly("mu_c", &IterativeLQR::ForwardPassResult::mu_c)
        .def_readonly("iter", &IterativeLQR::ForwardPassResult::iter)
        .def_readonly("step_length",
                      &IterativeLQR::ForwardPassResult::step_length);

    py::class_<utils::ProfilingInfo>(m, "ProfilingInfo")
        .def_readonly("timings", &utils::ProfilingInfo::timings);

    py::class_<IterativeLQR>(m, "IterativeLQR")
        .def(py::init(&construct), py::arg("fdyn"), py::arg("N"),
             py::arg("opt") = IterativeLQR::OptionDict{},
             py::arg("space") = nullptr)
        .def(py::init<const horizon::ilqr::FunctionManager &,
                      const std::string &, int, IterativeLQR::OptionDict,
                      std::shared_ptr<const horizon::ilqr::StateSpace>>(),
             py::arg("manager"), py::arg("dynamics"), py::arg("N"),
             py::arg("opt") = IterativeLQR::OptionDict{},
             py::arg("space") = nullptr)
#if defined(HORIZON_WITH_KOKKOS) && defined(HORIZON_WITH_PINOCCHIO)
        .def("setBatchedCost",
             [](IterativeLQR &solver, const std::vector<int> &nodes,
                const std::string &name,
                const horizon::collision::DeviceCollisionCost &cost)
             {
                 solver.setBatchedCost(nodes, name, cost);
             })
#endif
        .def("setCost",
             py::overload_cast<std::vector<int>,
                               const horizon::ilqr::FunctionManager &,
                               const std::string &>(&IterativeLQR::setCost))
        .def("setResidual",
             py::overload_cast<std::vector<int>,
                               const horizon::ilqr::FunctionManager &,
                               const std::string &>(&IterativeLQR::setResidual))
        .def("setConstraint",
             py::overload_cast<
                 std::vector<int>, const horizon::ilqr::FunctionManager &,
                 const std::string &, std::vector<Eigen::VectorXd>>(
                 &IterativeLQR::setConstraint),
             py::arg("indices"), py::arg("manager"), py::arg("name"),
             py::arg("targets") = std::vector<Eigen::VectorXd>{})
        .def("setIntermediateCost", set_inter_cost_wrapper_single)
        .def("setIntermediateResidual", set_residual_wrapper_single)
        .def("setIntermediateConstraint", set_inter_constraint_wrapper_single,
             py::arg("indices"), py::arg("h"), py::arg("target") = py::list())
        .def("setFinalCost", set_final_cost_wrapper)
        .def("setFinalConstraint", set_final_constraint_wrapper)
        .def("setStateBounds", &IterativeLQR::setStateBounds)
        .def(
            "setInputBounds",
            py::overload_cast<const Eigen::MatrixXd &, const Eigen::MatrixXd &>(
                &IterativeLQR::setInputBounds))
        .def("setInputBounds",
             py::overload_cast<double, double>(&IterativeLQR::setInputBounds))
        .def("fixInitialState", &IterativeLQR::fixInitialState)
        .def("setParameterValue", &IterativeLQR::setParameterValue,
             py::arg("pname"), py::arg("value"),
             py::arg("indices") = std::vector<int>{})
        .def("setIndices",
             py::overload_cast<const std::string &, const std::vector<int> &>(
                 &IterativeLQR::setIndices))
        .def("updateIndices", &IterativeLQR::updateIndices)
        .def("solve", &IterativeLQR::solve)
        .def("state", &IterativeLQR::state)
        .def("input", &IterativeLQR::input)
        .def("gain", &IterativeLQR::gain)
        .def("getProfilingInfo", &IterativeLQR::getProfilingInfo)
        .def("getIterationHistory", &IterativeLQR::getIterationHistory)
        .def("setIterationCallback", &IterativeLQR::setIterationCallback)
        .def("getStateTrajectory", &IterativeLQR::getStateTrajectory)
        .def("getInputTrajectory", &IterativeLQR::getInputTrajectory)
        .def("getConstraintsValues", &IterativeLQR::getConstraintsValues)
        .def("getIterationResult", &IterativeLQR::getIterationResult,
             py::return_value_policy::reference_internal)
        .def("getCostsValues", &IterativeLQR::getCostsValues)
        .def("setInitialState", &IterativeLQR::setInitialState)
        .def("setInputInitialGuess", &IterativeLQR::setInputInitialGuess)
        .def("setStateInitialGuess", &IterativeLQR::setStateInitialGuess);
}
