#include "pysqp_helpers.h"
#include <casadi/casadi.hpp>


PYBIND11_MODULE(pysqp, m) {

    py::class_<SQPGaussNewton<casadi::SX>>(m, "SQPGaussNewtonSX")
            .def(py::init(&constructSX))
            .def("solve", callSX,
                 py::arg("x0"), py::arg("p"), py::arg("lbx"), py::arg("ubx"), py::arg("lbg"), py::arg("ubg"))
            .def("f", fSX)
            .def("g", gSX)
            .def("__call__", callSX,
                 py::arg("x0"), py::arg("p"), py::arg("lbx"), py::arg("ubx"), py::arg("lbg"), py::arg("ubg"))
            .def("setAlphaMin", &SQPGaussNewton<casadi::SX>::setAlphaMin)
            .def("getAlpha", &SQPGaussNewton<casadi::SX>::getAlpha)
            .def("getBeta", &SQPGaussNewton<casadi::SX>::getBeta)
            .def("setBeta", &SQPGaussNewton<casadi::SX>::setBeta)
            .def("getVariableTrajectory", &SQPGaussNewton<casadi::SX>::getVariableTrajectory)
            .def("getObjectiveIterations", &SQPGaussNewton<casadi::SX>::getObjectiveIterations)
            .def("getConstraintNormIterations", &SQPGaussNewton<casadi::SX>::getConstraintNormIterations)
            .def("getHessianComputationTime", &SQPGaussNewton<casadi::SX>::getHessianComputationTime)
            .def("getQPComputationTime", &SQPGaussNewton<casadi::SX>::getQPComputationTime)
            .def("getLineSearchComputationTime", &SQPGaussNewton<casadi::SX>::getLineSearchComputationTime)
            .def("printConicOptions", &SQPGaussNewton<casadi::SX>::printConicOptions)
            .def("setIterationCallback", &SQPGaussNewton<casadi::SX>::setIterationCallback)
            ;

//    py::class_<SQPGaussNewton<casadi::MX>>(m, "SQPGaussNewtonMX")
//            .def(py::init(&constructMX))
//            .def("solve", callSX,
//                py::arg("x0"), py::arg("lbx"), py::arg("ubx"), py::arg("lbg"), py::arg("ubg"))
//            .def("f", fMX)
//            .def("g", gMX)
//            .def("__call__", callSX,
//                py::arg("x0"), py::arg("lbx"), py::arg("ubx"), py::arg("lbg"), py::arg("ubg"))
//            .def("setAlphaMin", &SQPGaussNewton<casadi::MX>::setAlphaMin)
//            .def("getAlpha", &SQPGaussNewton<casadi::MX>::getAlpha)
//            .def("getBeta", &SQPGaussNewton<casadi::MX>::getBeta)
//            .def("setBeta", &SQPGaussNewton<casadi::MX>::setBeta)
//            .def("getVariableTrajectory", &SQPGaussNewton<casadi::MX>::getVariableTrajectory)
//            .def("getObjectiveIterations", &SQPGaussNewton<casadi::MX>::getObjectiveIterations)
//            .def("getConstraintNormIterations", &SQPGaussNewton<casadi::MX>::getConstraintNormIterations)
//            .def("getHessianComputationTime", &SQPGaussNewton<casadi::MX>::getHessianComputationTime)
//            .def("getQPComputationTime", &SQPGaussNewton<casadi::MX>::getQPComputationTime)
//            .def("getLineSearchComputationTime", &SQPGaussNewton<casadi::MX>::getLineSearchComputationTime)
//            .def("printConicOptions", &SQPGaussNewton<casadi::MX>::printConicOptions)
//            .def("setIterationCallback", &SQPGaussNewton<casadi::MX>::setIterationCallback)
//            .def("setParameterValue", &SQPGaussNewton<casadi::MX>::setParameterValue)
//            ;

}
