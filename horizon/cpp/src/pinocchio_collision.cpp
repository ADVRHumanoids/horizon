#include "horizon/pinocchio_collision.h"
#include <Eigen/Cholesky>
#include <cstring>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/spatial/skew.hpp>
#include <stdexcept>

namespace horizon::collision
{
DeviceRobotModel exportDeviceRobot(const pinocchio::Model &model,
                                   const SphereModel &spheres)
{
    DeviceRobotModel result;
    result.nq = model.nq;
    result.nv = model.nv;
    result.joints.resize(model.njoints);
    const Eigen::VectorXd neutral = pinocchio::neutral(model);
    for (pinocchio::JointIndex i = 1; i < model.njoints; ++i)
    {
        const auto &joint = model.joints[i];
        auto &out = result.joints[i];
        out.parent = model.parents[i];
        out.q = joint.idx_q();
        out.v = joint.idx_v();
        const auto &p = model.jointPlacements[i];
        for (int a = 0; a < 3; ++a)
        {
            out.placement[9 + a] = p.translation()[a];
            for (int b = 0; b < 3; ++b)
            {
                out.placement[3 * a + b] = p.rotation()(a, b);
            }
        }
        const auto name = joint.shortname();
        if (name.find("Mimic") != std::string::npos)
        {
            throw std::invalid_argument(
                "device collision does not support mimic joints: " + name);
        }
        if (joint.nv() == 6 && joint.nq() == 7 &&
            name.find("FreeFlyer") != std::string::npos)
        {
            out.type = DeviceRobotModel::FreeFlyer;
        }
        else if (joint.nv() == 1 && (joint.nq() == 1 || joint.nq() == 2))
        {
            auto data = joint.createData();
            joint.calc(data, neutral);
            const Eigen::MatrixXd S = data.S().matrix();
            const bool angular = S.bottomRows(3).norm() > .5;
            out.type = angular ? (joint.nq() == 2 ? DeviceRobotModel::Continuous
                                                  : DeviceRobotModel::Revolute)
                               : DeviceRobotModel::Prismatic;
            for (int a = 0; a < 3; ++a)
            {
                out.axis[a] = S(a + (angular ? 3 : 0), 0);
            }
        }
        else
        {
            throw std::invalid_argument("unsupported device collision joint: " +
                                        name);
        }
    }
    for (const auto &sphere : spheres.spheres())
    {
        if (!model.existFrame(sphere.link))
        {
            throw std::invalid_argument("unknown sphere frame: " + sphere.link);
        }
        const auto &frame = model.frames[model.getFrameId(sphere.link)];
        DeviceRobotModel::Sphere out;
        out.joint = frame.parent;
        const Eigen::Vector3d center = frame.placement.act(sphere.center);
        for (int a = 0; a < 3; ++a)
        {
            out.center[a] = center[a];
        }
        out.radius = sphere.radius;
        result.spheres.push_back(out);
    }
    return result;
}

namespace
{
std::shared_ptr<const pinocchio::Model>
checkedModel(std::shared_ptr<const pinocchio::Model> model)
{
    if (!model)
    {
        throw std::invalid_argument("null Pinocchio model");
    }
    return model;
}
std::shared_ptr<const ilqr::StateSpace>
checkedSpace(std::shared_ptr<const ilqr::StateSpace> space,
             const std::shared_ptr<const pinocchio::Model> &model)
{
    if (!space || space->stateSize() != model->nq ||
        space->tangentSize() != model->nv)
    {
        throw std::invalid_argument(
            "Pinocchio model and state-space dimensions differ");
    }
    return space;
}
} // namespace

PinocchioSphereCenters::PinocchioSphereCenters(
    std::shared_ptr<const pinocchio::Model> model,
    std::shared_ptr<const ilqr::StateSpace> space, const SphereModel &spheres,
    std::shared_ptr<PinocchioModelWorkspace> workspace)
    : TangentFunction(checkedSpace(space, checkedModel(model)),
                      static_cast<int>(3 * spheres.spheres().size())),
      _model(checkedModel(std::move(model))),
      _state_space(checkedSpace(std::move(space), _model)),
      _workspace_prototype(
          workspace ? std::move(workspace)
                    : std::make_shared<PinocchioModelWorkspace>(_model)),
      _workspace(_workspace_prototype), _spheres(spheres.spheres()),
      _frame_jacobian(6, _model->nv)
{
    if (&_workspace_prototype->model() != _model.get())
    {
        throw std::invalid_argument(
            "Pinocchio workspace belongs to a different model");
    }
    _frames.reserve(_spheres.size());
    for (const auto &sphere : _spheres)
    {
        const auto id = _model->getFrameId(sphere.link);
        if (id >= _model->frames.size())
        {
            throw std::invalid_argument(
                "sphere references unknown Pinocchio frame: " + sphere.link);
        }
        _frames.push_back(id);
        _frame_spheres[id].push_back(_frames.size() - 1);
    }
}

std::unique_ptr<ilqr::Function> PinocchioSphereCenters::clone() const
{
    return std::make_unique<PinocchioSphereCenters>(
        _model, _state_space, SphereModel(_spheres),
        _workspace_prototype->cloneEmpty());
}

void PinocchioSphereCenters::setEvaluationContext(
    const std::shared_ptr<ilqr::EvaluationContext> &context)
{
    _workspace = context ? _workspace_prototype->forContext(context)
                         : _workspace_prototype;
}

void PinocchioSphereCenters::evaluateImpl(ilqr::Input q, ilqr::VectorOutput out)
{
    _workspace->ensureFramePlacements(q);
    fillCenters(out);
}

void PinocchioSphereCenters::fillCenters(ilqr::VectorOutput out) const
{
    const auto count = static_cast<Eigen::Index>(_spheres.size());
    for (std::size_t i = 0; i < _spheres.size(); ++i)
    {
        const Eigen::Vector3d p =
            _workspace->data().oMf[_frames[i]].act(_spheres[i].center);
        out[i] = p.x();
        out[count + i] = p.y();
        out[2 * count + i] = p.z();
    }
}

void PinocchioSphereCenters::tangentJacobian(ilqr::Input q,
                                             ilqr::MatrixOutput jac)
{
    _workspace->ensureJointJacobians(q);
    _workspace->ensureFramePlacements(q);
    fillJacobian(jac);
}

void PinocchioSphereCenters::valueAndTangentJacobian(
    ilqr::Input q, ilqr::VectorOutput centers, ilqr::MatrixOutput jacobian)
{
    // The combined path performs FK and joint Jacobians once for every sphere
    // and shares the resulting Pinocchio Data across the output rows.
    _workspace->ensureJointJacobians(q);
    _workspace->ensureFramePlacements(q);
    fillCenters(centers);
    fillJacobian(jacobian);
}

void PinocchioSphereCenters::fillJacobian(ilqr::MatrixOutput jac)
{
    const auto count = static_cast<Eigen::Index>(_spheres.size());
    for (const auto &[frame, indices] : _frame_spheres)
    {
        _frame_jacobian.setZero();
        pinocchio::getFrameJacobian(*_model, _workspace->data(), frame,
                                    pinocchio::LOCAL_WORLD_ALIGNED,
                                    _frame_jacobian);
        for (const auto i : indices)
        {
            const Eigen::Vector3d offset =
                _workspace->data().oMf[_frames[i]].rotation() *
                _spheres[i].center;
            for (int col = 0; col < _model->nv; ++col)
            {
                const Eigen::Vector3d point =
                    _frame_jacobian.col(col).head<3>() +
                    _frame_jacobian.col(col).tail<3>().cross(offset);
                jac(i, col) = point.x();
                jac(count + i, col) = point.y();
                jac(2 * count + i, col) = point.z();
            }
        }
    }
}
} // namespace horizon::collision
