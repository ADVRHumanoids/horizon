#include "pyilqr_helpers.h"

PYBIND11_MODULE(pyilqr, m) {

    py::class_<IterativeLQR::ForwardPassResult>(m, "ForwardPassResult")
            .def("print", &IterativeLQR::ForwardPassResult::print, py::arg("N") = 1)
            .def_readonly("xtrj", &IterativeLQR::ForwardPassResult::xtrj)
            .def_readonly("utrj", &IterativeLQR::ForwardPassResult::utrj)
            .def_readonly("accepted", &IterativeLQR::ForwardPassResult::accepted)
            .def_readonly("alpha", &IterativeLQR::ForwardPassResult::alpha)
            .def_readonly("hxx_reg", &IterativeLQR::ForwardPassResult::hxx_reg)
            .def_readonly("constraint_violation", &IterativeLQR::ForwardPassResult::constraint_violation)
            .def_readonly("constraint_values", &IterativeLQR::ForwardPassResult::constraint_values)
            .def_readonly("defect_values", &IterativeLQR::ForwardPassResult::defect_values)
            .def_readonly("cost", &IterativeLQR::ForwardPassResult::cost)
            .def_readonly("defect_norm", &IterativeLQR::ForwardPassResult::defect_norm)
            .def_readonly("merit", &IterativeLQR::ForwardPassResult::merit)
//            .def_readonly("armijo_merit", &IterativeLQR::ForwardPassResult::armijo_merit)
            .def_readonly("merit_der", &IterativeLQR::ForwardPassResult::merit_der)
            .def_readonly("mu_f", &IterativeLQR::ForwardPassResult::mu_f)
            .def_readonly("mu_c", &IterativeLQR::ForwardPassResult::mu_c)
            .def_readonly("iter", &IterativeLQR::ForwardPassResult::iter)
            .def_readonly("step_length", &IterativeLQR::ForwardPassResult::step_length);

    py::class_<utils::ProfilingInfo>(m, "ProfilingInfo")
            .def_readonly("timings", &utils::ProfilingInfo::timings);

    py::class_<IterativeLQR>(m, "IterativeLQR")
            .def(py::init(&construct))
            .def("setIntermediateCost", set_inter_cost_wrapper_single)
            .def("setIntermediateResidual", set_residual_wrapper_single)
            .def("setIntermediateConstraint",
                 set_inter_constraint_wrapper_single,
                 py::arg("indices"), py::arg("h"), py::arg("target") = py::list())
            .def("setFinalCost", set_final_cost_wrapper)
            .def("setFinalConstraint", set_final_constraint_wrapper)
            .def("setStateBounds", &IterativeLQR::setStateBounds)
            .def("setInputBounds", &IterativeLQR::setInputBounds)
            .def("setParameterValue", &IterativeLQR::setParameterValue)
            .def("setIndices", &IterativeLQR::setIndices)
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
            .def("setInitialState", &IterativeLQR::setInitialState)
            .def("setInputInitialGuess", &IterativeLQR::setInputInitialGuess)
            .def("setStateInitialGuess", &IterativeLQR::setStateInitialGuess)
            ;

}



