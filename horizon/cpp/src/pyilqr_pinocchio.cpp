#include "horizon/pinocchio_collision.h"
#include <pinocchio/parsers/urdf.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace horizon::collision;
namespace
{
struct PythonPinocchioModel
{
    std::shared_ptr<pinocchio::Model> model=std::make_shared<pinocchio::Model>();
    std::shared_ptr<PinocchioConfiguration> space;
    std::shared_ptr<PinocchioModelWorkspace> workspace;
    explicit PythonPinocchioModel(const std::string& urdf)
    {
        pinocchio::urdf::buildModel(urdf,*model);
        space=std::make_shared<PinocchioConfiguration>(model);
        workspace=std::make_shared<PinocchioModelWorkspace>(model);
    }
    std::shared_ptr<horizon::ilqr::Function> centers(std::vector<LinkSphere> spheres) const
    { return std::make_shared<PinocchioSphereCenters>(model,space,SphereModel(std::move(spheres)),workspace); }
};
}

void bindIlqrPinocchio(py::module_& m)
{
    py::class_<LinkSphere>(m,"LinkSphere")
        .def(py::init<std::string,Eigen::Vector3d,double>(),py::arg("link"),py::arg("center"),py::arg("radius"));
    py::class_<PythonPinocchioModel>(m,"PinocchioModel")
        .def(py::init<const std::string&>(),py::arg("urdf"))
        .def_property_readonly("nq",[](const PythonPinocchioModel& r){return r.model->nq;})
        .def_property_readonly("nv",[](const PythonPinocchioModel& r){return r.model->nv;})
        .def_property_readonly("configuration_space",[](const PythonPinocchioModel& r)->std::shared_ptr<horizon::ilqr::StateSpace>{return r.space;})
        .def_property_readonly("lower_position_limit",[](const PythonPinocchioModel& r){return r.model->lowerPositionLimit;})
        .def_property_readonly("upper_position_limit",[](const PythonPinocchioModel& r){return r.model->upperPositionLimit;})
        .def("step",[](const PythonPinocchioModel& r,double dt)->std::shared_ptr<horizon::ilqr::Function>{return std::make_shared<PinocchioStep>(r.space,dt);},py::arg("dt"))
        .def("sphereCenters",&PythonPinocchioModel::centers)
        .def("framePositions",[](const PythonPinocchioModel& r,const std::vector<std::string>& frames){
            std::vector<LinkSphere> spheres;
            for(const auto& frame:frames)spheres.push_back({frame,Eigen::Vector3d::Zero(),.001});
            return r.centers(std::move(spheres));
        },py::arg("frames"));
}
