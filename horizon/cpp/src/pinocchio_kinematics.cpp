#include "horizon/pinocchio_kinematics.h"
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <cmath>
#include <cstring>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <pinocchio/spatial/skew.hpp>
#include <stdexcept>

namespace horizon::kinematics
{
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
void normalizeConfiguration(const pinocchio::Model &model, ilqr::Input q,
                            ilqr::Vector &normalized)
{
    if (q.size() != model.nq || !q.allFinite())
    {
        throw std::invalid_argument("invalid Pinocchio configuration");
    }
    normalized = q;
    pinocchio::normalize(model, normalized);
    if (!normalized.allFinite() ||
        !pinocchio::isNormalized(model, normalized, 1e-10))
    {
        throw std::invalid_argument(
            "invalid Pinocchio configuration quaternion");
    }
}

// Composite joints retain absolute coordinate indices in their child joints.
void scaleUnitCoordinates(const pinocchio::JointModel &joint, ilqr::Input q,
                          ilqr::MatrixOutput jacobian)
{
    if (const auto *composite =
            boost::get<pinocchio::JointModelComposite>(&joint.toVariant()))
    {
        for (const auto &child : composite->joints)
        {
            scaleUnitCoordinates(child, q, jacobian);
        }
        return;
    }
    const auto name = joint.shortname();
    int coordinates = 0;
    if (name.find("FreeFlyer") != std::string::npos ||
        name.find("Spherical") != std::string::npos)
    {
        coordinates = joint.nq() == joint.nv() + 1 ? 4 : 0;
    }
    else if (name.find("Planar") != std::string::npos ||
             (joint.nq() == 2 && joint.nv() == 1))
    {
        coordinates = 2;
    }
    if (coordinates)
    {
        const int offset = joint.idx_q() + joint.nq() - coordinates;
        jacobian.middleRows(offset, coordinates) *=
            q.segment(offset, coordinates).norm();
    }
}
} // namespace

PinocchioConfiguration::PinocchioConfiguration(
    std::shared_ptr<const pinocchio::Model> m)
    : model(std::move(m))
{
    if (!model)
    {
        throw std::invalid_argument("null Pinocchio model");
    }
}
int PinocchioConfiguration::stateSize() const
{
    return model->nq;
}
int PinocchioConfiguration::tangentSize() const
{
    return model->nv;
}
void PinocchioConfiguration::neutral(ilqr::VectorOutput q) const
{
    q = pinocchio::neutral(*model);
}
void PinocchioConfiguration::integrate(ilqr::Input q, ilqr::Input v,
                                       ilqr::VectorOutput out) const
{
    out = pinocchio::integrate(*model, q, v);
}
void PinocchioConfiguration::difference(ilqr::Input a, ilqr::Input b,
                                        ilqr::VectorOutput out) const
{
    out = pinocchio::difference(*model, a, b);
}
void PinocchioConfiguration::retractionJacobian(ilqr::Input q,
                                                ilqr::MatrixOutput j) const
{
    // Extend the tangent chart to nonunit quaternion/complex coordinates.
    // Pinocchio's coefficient-wise Jacobian itself requires a normalized q.
    ilqr::Vector normalized;
    normalizeConfiguration(*model, q, normalized);
    j.setZero();
    pinocchio::integrateCoeffWiseJacobian(*model, normalized, j);
    for (pinocchio::JointIndex i = 1; i < model->njoints; ++i)
    {
        scaleUnitCoordinates(model->joints[i], q, j);
    }
}
void PinocchioConfiguration::tangentProjection(
    ilqr::Input q, ilqr::MatrixOutput projection) const
{
    // Configuration integration is a product over joints. Its coefficient
    // Jacobian is block diagonal, so solve each joint Gram matrix separately
    // instead of factoring a robot-wide nv x nv matrix at every native call.
    ilqr::Matrix R(model->nq, model->nv);
    retractionJacobian(q, R);
    projection.setZero();
    for (pinocchio::JointIndex i = 1; i < model->njoints; ++i)
    {
        const auto &joint = model->joints[i];
        if (joint.nv() == 0)
        {
            continue;
        }
        auto block =
            R.block(joint.idx_q(), joint.idx_v(), joint.nq(), joint.nv());
        auto out = projection.block(joint.idx_v(), joint.idx_q(), joint.nv(),
                                    joint.nq());
        if (joint.nv() == 1)
        {
            out = block.transpose() / block.squaredNorm();
        }
        else
        {
            out = (block.transpose() * block).ldlt().solve(block.transpose());
        }
    }
}
void PinocchioConfiguration::differenceJacobians(ilqr::Input a, ilqr::Input b,
                                                 ilqr::MatrixOutput A,
                                                 ilqr::MatrixOutput B) const
{
    A.setZero();
    B.setZero();
    pinocchio::dDifference(*model, a, b, A, pinocchio::ARG0);
    pinocchio::dDifference(*model, a, b, B, pinocchio::ARG1);
}

PinocchioStep::PinocchioStep(std::shared_ptr<const PinocchioConfiguration> s,
                             double dt)
    : TangentFunction(
          std::make_shared<ilqr::ProductSpace>(
              std::vector<std::shared_ptr<const ilqr::StateSpace>>{
                  s, std::make_shared<ilqr::EuclideanSpace>(s->tangentSize())}),
          s->stateSize()),
      _space(std::move(s)), _dt(dt), _A(_space->model->nv, _space->model->nv),
      _B(_A), _R(_space->model->nq, _space->model->nv),
      _prediction(_space->model->nq), _configuration(_space->model->nq)
{
    if (!std::isfinite(dt) || dt <= 0)
    {
        throw std::invalid_argument(
            "integration dt must be finite and positive");
    }
}
std::unique_ptr<ilqr::Function> PinocchioStep::clone() const
{
    return std::make_unique<PinocchioStep>(_space, _dt);
}
void PinocchioStep::evaluateImpl(ilqr::Input z, ilqr::VectorOutput out)
{
    normalizeConfiguration(*_space->model, z.head(_space->model->nq),
                           _configuration);
    _space->integrate(_configuration, _dt * z.tail(_space->model->nv), out);
}
void PinocchioStep::tangentJacobian(ilqr::Input z, ilqr::MatrixOutput j)
{
    const auto &m = *_space->model;
    ilqr::Vector delta = _dt * z.tail(m.nv);
    _A.setZero();
    _B.setZero();
    evaluateImpl(z, _prediction);
    pinocchio::dIntegrate(m, _configuration, delta, _A, pinocchio::ARG0);
    pinocchio::dIntegrate(m, _configuration, delta, _B, pinocchio::ARG1);
    _space->retractionJacobian(_prediction, _R);
    j.leftCols(m.nv).noalias() = _R * _A;
    j.rightCols(m.nv).noalias() = _dt * _R * _B;
}

PinocchioModelWorkspace::PinocchioModelWorkspace(
    std::shared_ptr<const pinocchio::Model> model)
    : PinocchioModelWorkspace(checkedModel(std::move(model)),
                              std::make_shared<const int>(0))
{
}

PinocchioModelWorkspace::PinocchioModelWorkspace(
    std::shared_ptr<const pinocchio::Model> model,
    std::shared_ptr<const int> identity)
    : _model(checkedModel(std::move(model))), _identity(std::move(identity)),
      _data(*_model)
{
}

std::shared_ptr<PinocchioModelWorkspace>
PinocchioModelWorkspace::cloneEmpty() const
{
    return std::shared_ptr<PinocchioModelWorkspace>(
        new PinocchioModelWorkspace(_model, _identity));
}

std::shared_ptr<PinocchioModelWorkspace> PinocchioModelWorkspace::forContext(
    const std::shared_ptr<ilqr::EvaluationContext> &context) const
{
    if (!context)
    {
        throw std::invalid_argument(
            "Pinocchio workspace requires an evaluation context");
    }
    return context->shareObject<PinocchioModelWorkspace>(
        identity(),
        [this]
        {
            return cloneEmpty();
        });
}

bool PinocchioModelWorkspace::same(const Eigen::VectorXd &cached,
                                   ilqr::Input value, bool valid) const
{
    return valid && cached.size() == value.size() &&
        std::memcmp(cached.data(), value.data(),
                    value.size() * sizeof(double)) == 0;
}

void PinocchioModelWorkspace::invalidate()
{
    _has_q = _positions = _velocity = _acceleration = _frames = _jacobians =
        _derivatives = false;
}

void PinocchioModelWorkspace::resetForConfiguration(ilqr::Input q)
{
    if (q.size() != _model->nq || !q.allFinite())
    {
        throw std::invalid_argument("invalid Pinocchio configuration");
    }
    if (!same(_input_q, q, _has_q))
    {
        normalizeConfiguration(*_model, q, _candidate_q);
        _q = _candidate_q;
        _input_q = q;
        _has_q = true;
        _positions = _velocity = _acceleration = _frames = _jacobians =
            _derivatives = false;
    }
}

void PinocchioModelWorkspace::ensurePositions(ilqr::Input q)
{
    resetForConfiguration(q);
    if (!_positions)
    {
        pinocchio::forwardKinematics(*_model, _data, _q);
        _positions = true;
        _velocity = _acceleration = _frames = _jacobians = _derivatives = false;
    }
}

void PinocchioModelWorkspace::ensureVelocity(ilqr::Input q, ilqr::Input v)
{
    resetForConfiguration(q);
    if (v.size() != _model->nv || !v.allFinite())
    {
        throw std::invalid_argument("invalid Pinocchio velocity");
    }
    if (!_velocity || !same(_v, v, true))
    {
        pinocchio::forwardKinematics(*_model, _data, _q, v);
        _v = v;
        _positions = _velocity = true;
        _acceleration = _frames = _jacobians = _derivatives = false;
    }
}

void PinocchioModelWorkspace::ensureAcceleration(ilqr::Input q, ilqr::Input v,
                                                 ilqr::Input a)
{
    resetForConfiguration(q);
    if (v.size() != _model->nv || a.size() != _model->nv || !v.allFinite() ||
        !a.allFinite())
    {
        throw std::invalid_argument(
            "invalid Pinocchio velocity or acceleration");
    }
    if (!_acceleration || !same(_v, v, true) || !same(_a, a, true))
    {
        pinocchio::forwardKinematics(*_model, _data, _q, v, a);
        _v = v;
        _a = a;
        _positions = _velocity = _acceleration = true;
        _frames = _jacobians = _derivatives = false;
    }
}

void PinocchioModelWorkspace::ensureFramePlacements(ilqr::Input q)
{
    ensurePositions(q);
    if (!_frames)
    {
        pinocchio::updateFramePlacements(*_model, _data);
        _frames = true;
    }
}

void PinocchioModelWorkspace::ensureJointJacobians(ilqr::Input q)
{
    ensurePositions(q);
    if (!_jacobians)
    {
        // This overload uses the already populated joint placements and avoids
        // repeating forward kinematics when a prior native call used the same
        // q.
        pinocchio::computeJointJacobians(*_model, _data);
        _jacobians = true;
        _derivatives = false;
    }
}

void PinocchioModelWorkspace::ensureKinematicsDerivatives(ilqr::Input q,
                                                           ilqr::Input v,
                                                           ilqr::Input a)
{
    resetForConfiguration(q);
    if (v.size() != _model->nv || a.size() != _model->nv || !v.allFinite() ||
        !a.allFinite())
    {
        throw std::invalid_argument("invalid Pinocchio velocity or acceleration");
    }
    if (!_derivatives || !same(_v, v, true) || !same(_a, a, true))
    {
        pinocchio::computeForwardKinematicsDerivatives(*_model, _data, _q, v, a);
        _v = v;
        _a = a;
        _positions = _velocity = _acceleration = _derivatives = true;
        _frames = _jacobians = false;
    }
}

PinocchioFrameFunction::PinocchioFrameFunction(
    std::shared_ptr<const pinocchio::Model> model,
    std::shared_ptr<const ilqr::StateSpace> configuration,
    std::shared_ptr<const ilqr::StateSpace> input_space,
    std::vector<std::string> frames, int output_size,
    std::shared_ptr<PinocchioModelWorkspace> workspace)
    : TangentFunction(std::move(input_space), output_size),
      _model(checkedModel(std::move(model))),
      _configuration(checkedSpace(std::move(configuration), _model)),
      _workspace_prototype(
          workspace ? std::move(workspace)
                    : std::make_shared<PinocchioModelWorkspace>(_model)),
      _workspace(_workspace_prototype), _names(std::move(frames)),
      _frame_jacobian(6, _model->nv)
{
    if (&_workspace_prototype->model() != _model.get())
    {
        throw std::invalid_argument(
            "Pinocchio workspace belongs to a different model");
    }
    if (_names.empty())
    {
        throw std::invalid_argument("at least one frame is required");
    }
    for (const auto &frame : _names)
    {
        if (!_model->existFrame(frame))
        {
            throw std::invalid_argument("unknown Pinocchio frame: " + frame);
        }
        _frames.push_back(_model->getFrameId(frame));
    }
}

void PinocchioFrameFunction::setEvaluationContext(
    const std::shared_ptr<ilqr::EvaluationContext> &context)
{
    _workspace = context ? _workspace_prototype->forContext(context)
                         : _workspace_prototype;
}
void PinocchioFrameFunction::prepare(ilqr::Input q, bool derivatives)
{
    if (derivatives)
    {
        _workspace->ensureJointJacobians(q);
    }
    _workspace->ensureFramePlacements(q);
}
void PinocchioFrameFunction::frameJacobian(pinocchio::FrameIndex frame)
{
    _frame_jacobian.setZero();
    pinocchio::getFrameJacobian(*_model, _workspace->data(), frame,
                                pinocchio::LOCAL_WORLD_ALIGNED,
                                _frame_jacobian);
}

PinocchioFramePositions::PinocchioFramePositions(
    std::shared_ptr<const pinocchio::Model> model,
    std::shared_ptr<const ilqr::StateSpace> configuration,
    std::vector<std::string> frames,
    std::shared_ptr<PinocchioModelWorkspace> workspace)
    : PinocchioFrameFunction(model, configuration,
                             checkedSpace(configuration, checkedModel(model)),
                             frames, static_cast<int>(3 * frames.size()),
                             std::move(workspace))
{
}
std::unique_ptr<ilqr::Function> PinocchioFramePositions::clone() const
{
    return std::make_unique<PinocchioFramePositions>(
        _model, _configuration, _names, _workspace_prototype->cloneEmpty());
}
void PinocchioFramePositions::evaluateImpl(ilqr::Input q,
                                           ilqr::VectorOutput out)
{
    prepare(q, false);
    const int count = _frames.size();
    for (int i = 0; i < count; ++i)
    {
        const auto &p = _workspace->data().oMf[_frames[i]].translation();
        out[i] = p.x();
        out[count + i] = p.y();
        out[2 * count + i] = p.z();
    }
}
void PinocchioFramePositions::tangentJacobian(ilqr::Input q,
                                              ilqr::MatrixOutput jac)
{
    prepare(q, true);
    const int count = _frames.size();
    for (int i = 0; i < count; ++i)
    {
        frameJacobian(_frames[i]);
        for (int axis = 0; axis < 3; ++axis)
        {
            jac.row(axis * count + i) = _frame_jacobian.row(axis);
        }
    }
}

PinocchioFramePose::PinocchioFramePose(
    std::shared_ptr<const pinocchio::Model> model,
    std::shared_ptr<const ilqr::StateSpace> configuration, std::string frame,
    std::shared_ptr<PinocchioModelWorkspace> workspace)
    : PinocchioFrameFunction(model, configuration,
                             checkedSpace(configuration, checkedModel(model)),
                             {std::move(frame)}, 7, std::move(workspace))
{
}
std::unique_ptr<ilqr::Function> PinocchioFramePose::clone() const
{
    return std::make_unique<PinocchioFramePose>(
        _model, _configuration, _names[0], _workspace_prototype->cloneEmpty());
}
void PinocchioFramePose::evaluateImpl(ilqr::Input q, ilqr::VectorOutput out)
{
    prepare(q, false);
    const auto &pose = _workspace->data().oMf[_frames[0]];
    out.head<3>() = pose.translation();
    out.tail<4>() = Eigen::Quaterniond(pose.rotation()).coeffs();
}
void PinocchioFramePose::tangentJacobian(ilqr::Input q, ilqr::MatrixOutput jac)
{
    prepare(q, true);
    frameJacobian(_frames[0]);
    const Eigen::Quaterniond orientation(
        _workspace->data().oMf[_frames[0]].rotation());
    Eigen::Matrix<double, 4, 3> derivative;
    derivative.topRows<3>() = .5 *
        (orientation.w() * Eigen::Matrix3d::Identity() -
         pinocchio::skew(orientation.vec()));
    derivative.bottomRows<1>() = -.5 * orientation.vec().transpose();
    jac.topRows<3>() = _frame_jacobian.topRows<3>();
    jac.bottomRows<4>().noalias() =
        derivative * _frame_jacobian.bottomRows<3>();
}

namespace
{
std::shared_ptr<const ilqr::StateSpace>
poseErrorInput(const std::shared_ptr<const pinocchio::Model> &model,
               const std::shared_ptr<const ilqr::StateSpace> &configuration)
{
    return std::make_shared<ilqr::ProductSpace>(
        std::vector<std::shared_ptr<const ilqr::StateSpace>>{
            checkedSpace(configuration, checkedModel(model)),
            std::make_shared<ilqr::EuclideanSpace>(7)});
}
Eigen::Quaterniond targetQuaternion(ilqr::Input target)
{
    if (!target.allFinite())
    {
        throw std::invalid_argument("pose target must be finite");
    }
    Eigen::Quaterniond quat;
    quat.coeffs() = target.tail<4>();
    if (quat.norm() < 1e-12)
    {
        throw std::invalid_argument("pose target quaternion must be nonzero");
    }
    quat.normalize();
    return quat;
}
} // namespace

PinocchioFramePoseError::PinocchioFramePoseError(
    std::shared_ptr<const pinocchio::Model> model,
    std::shared_ptr<const ilqr::StateSpace> configuration, std::string frame,
    std::shared_ptr<PinocchioModelWorkspace> workspace)
    : PinocchioFrameFunction(model, configuration,
                             poseErrorInput(model, configuration),
                             {std::move(frame)}, 6, std::move(workspace))
{
}
std::unique_ptr<ilqr::Function> PinocchioFramePoseError::clone() const
{
    return std::make_unique<PinocchioFramePoseError>(
        _model, _configuration, _names[0], _workspace_prototype->cloneEmpty());
}
void PinocchioFramePoseError::evaluateImpl(ilqr::Input z,
                                           ilqr::VectorOutput out)
{
    const auto target = z.tail<7>();
    const auto orientation = targetQuaternion(target);
    prepare(z.head(_model->nq), false);
    const auto &pose = _workspace->data().oMf[_frames[0]];
    out.head<3>() = pose.translation() - target.head<3>();
    const Eigen::Matrix3d error =
        orientation.toRotationMatrix().transpose() * pose.rotation();
    out.tail<3>() = pinocchio::log3(error);
}
void PinocchioFramePoseError::tangentJacobian(ilqr::Input z,
                                              ilqr::MatrixOutput jac)
{
    const auto target = z.tail<7>();
    const auto orientation = targetQuaternion(target);
    prepare(z.head(_model->nq), true);
    frameJacobian(_frames[0]);
    const auto &rotation = _workspace->data().oMf[_frames[0]].rotation();
    const Eigen::Matrix3d error =
        orientation.toRotationMatrix().transpose() * rotation;
    Eigen::Matrix3d log_jacobian;
    pinocchio::Jlog3(error, log_jacobian);
    jac.setZero();
    jac.topLeftCorner(3, _model->nv) = _frame_jacobian.topRows<3>();
    jac.block(3, 0, 3, _model->nv).noalias() =
        log_jacobian * rotation.transpose() * _frame_jacobian.bottomRows<3>();
    jac.block<3, 3>(0, _model->nv) = -Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 3, 4> quaternion_projection;
    quaternion_projection.leftCols<3>() =
        orientation.w() * Eigen::Matrix3d::Identity() -
        pinocchio::skew(orientation.vec());
    quaternion_projection.rightCols<1>() = -orientation.vec();
    quaternion_projection *= 2. / target.tail<4>().norm();
    jac.block(3, _model->nv + 3, 3, 4).noalias() =
        -log_jacobian * error.transpose() * quaternion_projection;
}
} // namespace horizon::kinematics
