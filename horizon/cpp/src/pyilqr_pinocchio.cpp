#include "horizon/pinocchio_collision.h"
#include "horizon/pinocchio_robot.h"
#ifdef HORIZON_WITH_KOKKOS
#include "horizon/device_collision.h"
#endif
#include <pinocchio/parsers/urdf.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace horizon::collision;
using namespace horizon::kinematics;
namespace
{
struct PythonPinocchioModel
{
    std::shared_ptr<pinocchio::Model> model =
        std::make_shared<pinocchio::Model>();
    std::shared_ptr<PinocchioConfiguration> space;
    std::shared_ptr<PinocchioModelWorkspace> workspace;
    explicit PythonPinocchioModel(const std::string &urdf)
    {
        pinocchio::urdf::buildModel(urdf, *model);
        space = std::make_shared<PinocchioConfiguration>(model);
        workspace = std::make_shared<PinocchioModelWorkspace>(model);
    }
    std::shared_ptr<horizon::ilqr::Function> function(
        RobotQuantity quantity, std::string frame = {},
        pinocchio::ReferenceFrame reference = pinocchio::LOCAL_WORLD_ALIGNED,
        bool symbolic_gravity = false) const
    {
        return std::make_shared<PinocchioRobotFunction>(
            model, space, quantity, std::move(frame), reference,
            symbolic_gravity, workspace);
    }
    std::shared_ptr<horizon::ilqr::Function>
    centers(std::vector<LinkSphere> spheres) const
    {
        return std::make_shared<PinocchioSphereCenters>(
            model, space, SphereModel(std::move(spheres)), workspace);
    }
};
std::vector<ContactSpec> parseContacts(const py::iterable &input)
{
    std::vector<ContactSpec> result;
    for (auto contact : input)
    {
        if (py::isinstance<py::str>(contact))
        {
            result.emplace_back(py::cast<std::string>(contact), 3);
        }
        else
        {
            result.push_back(py::cast<ContactSpec>(contact));
        }
    }
    return result;
}
} // namespace

void bindIlqrPinocchio(py::module_ &m)
{
    py::class_<LinkSphere>(m, "LinkSphere")
        .def(py::init<std::string, Eigen::Vector3d, double>(), py::arg("link"),
             py::arg("center"), py::arg("radius"));
    py::enum_<pinocchio::ReferenceFrame>(m, "ReferenceFrame")
        .value("LOCAL", pinocchio::LOCAL)
        .value("WORLD", pinocchio::WORLD)
        .value("LOCAL_WORLD_ALIGNED", pinocchio::LOCAL_WORLD_ALIGNED)
        .export_values();
    py::class_<ContactSpec>(m, "Contact")
        .def(py::init<std::string, int>(), py::arg("frame"),
             py::arg("dimension") = 3)
        .def_readwrite("frame", &ContactSpec::frame)
        .def_readwrite("dimension", &ContactSpec::dimension);
    auto cls = py::class_<PythonPinocchioModel>(m, "PinocchioModel");
#ifdef HORIZON_WITH_KOKKOS
    cls.def(
        "deviceCollisionCost",
        [](const PythonPinocchioModel &robot,
           const std::vector<LinkSphere> &spheres,
           const std::vector<std::pair<int, int>> &pairs,
           std::shared_ptr<SdfGrid> grid, double weight, double margin,
           double smoothing)
        {
            return std::make_shared<DeviceCollisionCost>(
                exportDeviceRobot(*robot.model, SphereModel(spheres)),
                std::move(grid), pairs, weight, margin, smoothing);
        },
        py::arg("spheres"), py::arg("pairs"), py::arg("grid") = nullptr,
        py::arg("weight") = 1., py::arg("margin") = .02,
        py::arg("smoothing") = .005);
#endif
    cls.def(py::init<const std::string &>(), py::arg("urdf"))
        .def_property_readonly("nq",
                               [](const PythonPinocchioModel &r)
                               {
                                   return r.model->nq;
                               })
        .def_property_readonly("nv",
                               [](const PythonPinocchioModel &r)
                               {
                                   return r.model->nv;
                               })
        .def_property_readonly("configuration_space",
                               [](const PythonPinocchioModel &r)
                                   -> std::shared_ptr<horizon::ilqr::StateSpace>
                               {
                                   return r.space;
                               })
        .def_property_readonly("lower_position_limit",
                               [](const PythonPinocchioModel &r)
                               {
                                   return r.model->lowerPositionLimit;
                               })
        .def_property_readonly("upper_position_limit",
                               [](const PythonPinocchioModel &r)
                               {
                                   return r.model->upperPositionLimit;
                               })
        .def(
            "step",
            [](const PythonPinocchioModel &r,
               double dt) -> std::shared_ptr<horizon::ilqr::Function>
            {
                return std::make_shared<PinocchioStep>(r.space, dt);
            },
            py::arg("dt"))
        .def("sphereCenters", &PythonPinocchioModel::centers)
        .def(
            "sphereClearance",
            [](const PythonPinocchioModel &r,
               const std::vector<LinkSphere> &spheres,
               const std::vector<std::pair<int, int>> &pairs,
               std::shared_ptr<DistanceField> environment)
                -> std::shared_ptr<horizon::ilqr::Function>
            {
                std::vector<std::shared_ptr<const DistanceField>> fields;
                if (environment)
                {
                    fields.push_back(std::move(environment));
                }
                return std::make_shared<PinocchioSphereClearance>(
                    r.model, r.space, SphereModel(spheres), std::move(fields),
                    pairs, r.workspace);
            },
            py::arg("spheres"), py::arg("pairs"),
            py::arg("environment") = nullptr)
        .def(
            "framePositions",
            [](const PythonPinocchioModel &r,
               const std::vector<std::string> &frames)
                -> std::shared_ptr<horizon::ilqr::Function>
            {
                return std::make_shared<PinocchioFramePositions>(
                    r.model, r.space, frames, r.workspace);
            },
            py::arg("frames"))
        .def(
            "framePose",
            [](const PythonPinocchioModel &r, const std::string &frame)
                -> std::shared_ptr<horizon::ilqr::Function>
            {
                return std::make_shared<PinocchioFramePose>(r.model, r.space,
                                                            frame, r.workspace);
            },
            py::arg("frame"))
        .def(
            "framePoseError",
            [](const PythonPinocchioModel &r, const std::string &frame)
                -> std::shared_ptr<horizon::ilqr::Function>
            {
                return std::make_shared<PinocchioFramePoseError>(
                    r.model, r.space, frame, r.workspace);
            },
            py::arg("frame"))
        .def("integrate",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return std::make_shared<PinocchioStep>(r.space, 1.);
             })
        .def("qdot",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::QDot);
             })
        .def("difference",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::Difference);
             })
        .def("crba",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::MassMatrix);
             })
        .def("massMatrix",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::MassMatrix);
             })
        .def("aba",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::ForwardDynamics);
             })
        .def("forwardDynamics",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::ForwardDynamics);
             })
        .def("gravity",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::Gravity);
             })
        .def("nonlinearEffects",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::NonlinearEffects);
             })
        .def("kineticEnergy",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::KineticEnergy);
             })
        .def("potentialEnergy",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::PotentialEnergy);
             })
        .def("centerOfMass",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::CenterOfMass);
             })
        .def("comPosition",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::ComPosition);
             })
        .def("comVelocity",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::ComVelocity);
             })
        .def("comAcceleration",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::ComAcceleration);
             })
        .def("centroidalMomentum",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::CentroidalMomentum);
             })
        .def("computeCentroidalDynamics",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::CentroidalDynamics);
             })
        .def("centroidalDynamics",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::CentroidalDynamics);
             })
        .def("ccrba",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::CentroidalMap);
             })
        .def("centroidalMap",
             [](const PythonPinocchioModel &r)
                 -> std::shared_ptr<horizon::ilqr::Function>
             {
                 return r.function(RobotQuantity::CentroidalMap);
             })
        .def(
            "fk",
            [](const PythonPinocchioModel &r, const std::string &frame)
            {
                return r.function(RobotQuantity::FrameTransform, frame);
            },
            py::arg("frame"))
        .def(
            "jacobian",
            [](const PythonPinocchioModel &r, const std::string &frame,
               pinocchio::ReferenceFrame reference)
            {
                return r.function(RobotQuantity::FrameJacobian, frame,
                                  reference);
            },
            py::arg("frame"),
            py::arg("reference") = pinocchio::LOCAL_WORLD_ALIGNED)
        .def(
            "frameVelocity",
            [](const PythonPinocchioModel &r, const std::string &frame,
               pinocchio::ReferenceFrame reference)
            {
                return r.function(RobotQuantity::FrameVelocity, frame,
                                  reference);
            },
            py::arg("frame"),
            py::arg("reference") = pinocchio::LOCAL_WORLD_ALIGNED)
        .def(
            "frameAcceleration",
            [](const PythonPinocchioModel &r, const std::string &frame,
               pinocchio::ReferenceFrame reference)
            {
                return r.function(RobotQuantity::FrameAcceleration, frame,
                                  reference);
            },
            py::arg("frame"),
            py::arg("reference") = pinocchio::LOCAL_WORLD_ALIGNED)
        .def(
            "frameSpatialAcceleration",
            [](const PythonPinocchioModel &r, const std::string &frame,
               pinocchio::ReferenceFrame reference)
            {
                return r.function(RobotQuantity::FrameSpatialAcceleration,
                                  frame, reference);
            },
            py::arg("frame"),
            py::arg("reference") = pinocchio::LOCAL_WORLD_ALIGNED)
        .def(
            "rnea",
            [](const PythonPinocchioModel &r, bool sym_g)
            {
                return r.function(RobotQuantity::InverseDynamics, {},
                                  pinocchio::LOCAL_WORLD_ALIGNED, sym_g);
            },
            py::arg("sym_g") = false)
        .def(
            "inverseDynamics",
            [](const PythonPinocchioModel &r, const py::iterable &input)
                -> std::shared_ptr<horizon::ilqr::Function>
            {
                auto contacts = parseContacts(input);
                if (contacts.empty())
                {
                    return r.function(RobotQuantity::InverseDynamics);
                }
                return std::make_shared<PinocchioContactDynamics>(
                    r.model, r.space, std::move(contacts),
                    horizon::ilqr::Matrix(), true, r.workspace);
            },
            py::arg("contacts") = py::tuple())
        .def(
            "contactDynamics",
            [](const PythonPinocchioModel &r, const py::iterable &input,
               const horizon::ilqr::Matrix &actuation)
                -> std::shared_ptr<horizon::ilqr::Function>
            {
                return std::make_shared<PinocchioContactDynamics>(
                    r.model, r.space, parseContacts(input), actuation, false,
                    r.workspace);
            },
            py::arg("contacts"),
            py::arg("actuation_matrix") = horizon::ilqr::Matrix())
        .def_property_readonly("velocity_limits",
                               [](const PythonPinocchioModel &r)
                               {
                                   return r.model->velocityLimit;
                               })
        .def_property_readonly("effort_limits",
                               [](const PythonPinocchioModel &r)
                               {
                                   return r.model->effortLimit;
                               })
        .def_property_readonly("joint_names",
                               [](const PythonPinocchioModel &r)
                               {
                                   return std::vector<std::string>(
                                       r.model->names.begin() + 1,
                                       r.model->names.end());
                               })
        .def_property_readonly("mass",
                               [](const PythonPinocchioModel &r)
                               {
                                   double mass = 0;
                                   for (const auto &inertia : r.model->inertias)
                                   {
                                       mass += inertia.mass();
                                   }
                                   return mass;
                               })
        .def("joint_nq",
             [](const PythonPinocchioModel &r, const std::string &name)
             {
                 if (!r.model->existJointName(name))
                 {
                     throw std::invalid_argument("unknown joint: " + name);
                 }
                 return r.model->joints[r.model->getJointId(name)].nq();
             })
        .def("joint_iq",
             [](const PythonPinocchioModel &r, const std::string &name)
             {
                 if (!r.model->existJointName(name))
                 {
                     throw std::invalid_argument("unknown joint: " + name);
                 }
                 return r.model->joints[r.model->getJointId(name)].idx_q();
             })
        .def("joint_nv",
             [](const PythonPinocchioModel &r, const std::string &name)
             {
                 if (!r.model->existJointName(name))
                 {
                     throw std::invalid_argument("unknown joint: " + name);
                 }
                 return r.model->joints[r.model->getJointId(name)].nv();
             })
        .def("joint_iv",
             [](const PythonPinocchioModel &r, const std::string &name)
             {
                 if (!r.model->existJointName(name))
                 {
                     throw std::invalid_argument("unknown joint: " + name);
                 }
                 return r.model->joints[r.model->getJointId(name)].idx_v();
             });
    cls.attr("LOCAL") = py::cast(pinocchio::LOCAL);
    cls.attr("WORLD") = py::cast(pinocchio::WORLD);
    cls.attr("LOCAL_WORLD_ALIGNED") = py::cast(pinocchio::LOCAL_WORLD_ALIGNED);
}
