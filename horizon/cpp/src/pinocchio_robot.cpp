#include "pinocchio_robot_internal.h"
#include <pinocchio/algorithm/jacobian.hpp>

namespace horizon::kinematics
{
namespace
{
int additional(const pinocchio::Model &m, RobotQuantity kind, bool gravity)
{
    switch (kind)
    {
    case RobotQuantity::QDot:
    case RobotQuantity::FrameVelocity:
    case RobotQuantity::KineticEnergy:
    case RobotQuantity::ComVelocity:
    case RobotQuantity::NonlinearEffects:
    case RobotQuantity::CentroidalMomentum:
        return m.nv;
    case RobotQuantity::FrameAcceleration:
    case RobotQuantity::FrameSpatialAcceleration:
    case RobotQuantity::CenterOfMass:
    case RobotQuantity::ComAcceleration:
    case RobotQuantity::CentroidalDynamics:
    case RobotQuantity::ForwardDynamics:
        return 2 * m.nv;
    case RobotQuantity::InverseDynamics:
        return 2 * m.nv + (gravity ? 3 : 0);
    default:
        return 0;
    }
}
std::shared_ptr<const ilqr::StateSpace>
inputSpace(const std::shared_ptr<const pinocchio::Model> &model,
           const std::shared_ptr<const PinocchioConfiguration> &space,
           RobotQuantity kind, bool gravity)
{
    if (!model)
    {
        throw std::invalid_argument("null Pinocchio model");
    }
    if (kind == RobotQuantity::Difference)
    {
        detail::product(model, space, 0);
        return std::make_shared<ilqr::ProductSpace>(
            std::vector<std::shared_ptr<const ilqr::StateSpace>>{space, space});
    }
    return detail::product(model, space, additional(*model, kind, gravity));
}
int robotOutputSize(const std::shared_ptr<const pinocchio::Model> &model,
                    RobotQuantity kind)
{
    if (!model)
    {
        throw std::invalid_argument("null Pinocchio model");
    }
    switch (kind)
    {
    case RobotQuantity::QDot:
        return model->nq;
    case RobotQuantity::Difference:
    case RobotQuantity::InverseDynamics:
    case RobotQuantity::ForwardDynamics:
    case RobotQuantity::Gravity:
    case RobotQuantity::NonlinearEffects:
        return model->nv;
    case RobotQuantity::FrameJacobian:
    case RobotQuantity::CentroidalMap:
        return 6 * model->nv;
    case RobotQuantity::MassMatrix:
        return model->nv * model->nv;
    case RobotQuantity::FrameTransform:
    case RobotQuantity::CentroidalDynamics:
        return 12;
    case RobotQuantity::KineticEnergy:
    case RobotQuantity::PotentialEnergy:
        return 1;
    case RobotQuantity::CenterOfMass:
        return 9;
    case RobotQuantity::ComPosition:
    case RobotQuantity::ComVelocity:
    case RobotQuantity::ComAcceleration:
        return 3;
    default:
        return 6;
    }
}
void quaternionQdotDerivative(int qidx, int vidx, ilqr::Input v,
                              const ilqr::Matrix &R, ilqr::MatrixOutput jac)
{
    const Eigen::Vector3d omega = v.segment<3>(vidx);
    for (int k = 0; k < 3; ++k)
    {
        const Eigen::Vector4d dq = R.block<4, 1>(qidx, vidx + k);
        jac.block<3, 1>(qidx, vidx + k) = .5 *
            (dq[3] * Eigen::Matrix3d::Identity() +
             pinocchio::skew(dq.head<3>())) *
            omega;
        jac(qidx + 3, vidx + k) = -.5 * dq.head<3>().dot(omega);
    }
}
void qdotDerivative(const pinocchio::JointModel &joint, ilqr::Input q,
                    ilqr::Input v, const ilqr::Matrix &R,
                    ilqr::MatrixOutput jac)
{
    if (const auto *composite =
            boost::get<pinocchio::JointModelComposite>(&joint.toVariant()))
    {
        for (const auto &child : composite->joints)
        {
            qdotDerivative(child, q, v, R, jac);
        }
        return;
    }
    const int iq = joint.idx_q(), iv = joint.idx_v();
    const auto name = joint.shortname();
    if (name.find("FreeFlyer") != std::string::npos)
    {
        Eigen::Quaterniond orientation;
        orientation.coeffs() = q.segment<4>(iq + 3);
        jac.block<3, 3>(iq, iv + 3) =
            -orientation.toRotationMatrix() * pinocchio::skew(v.segment<3>(iv));
        quaternionQdotDerivative(iq + 3, iv + 3, v, R, jac);
    }
    else if (joint.nq() == 4 && joint.nv() == 3 &&
             name.find("Planar") == std::string::npos)
    {
        quaternionQdotDerivative(iq, iv, v, R, jac);
    }
    else if (name.find("Planar") != std::string::npos)
    {
        Eigen::Matrix2d rotation;
        rotation << q[iq + 2], -q[iq + 3], q[iq + 3], q[iq + 2];
        Eigen::Matrix2d skew;
        skew << 0, -1, 1, 0;
        jac.block<2, 1>(iq, iv + 2) = rotation * skew * v.segment<2>(iv);
        jac.block<2, 1>(iq + 2, iv + 2) = -q.segment<2>(iq + 2) * v[iv + 2];
    }
    else if (joint.nq() == 2 && joint.nv() == 1)
    {
        jac.block<2, 1>(iq, iv) = -q.segment<2>(iq) * v[iv];
    }
    else if (joint.nq() != joint.nv())
    {
        throw std::invalid_argument("qdot derivatives unsupported for joint: " +
                                    name);
    }
}
} // namespace

PinocchioRobotFunction::Impl::Impl(
    std::shared_ptr<const pinocchio::Model> model,
    std::shared_ptr<const PinocchioConfiguration> space, RobotQuantity quantity,
    std::string frame, pinocchio::ReferenceFrame reference,
    bool symbolic_gravity, std::shared_ptr<PinocchioModelWorkspace> workspace)
    : PinocchioRobotImplBase(std::move(model), std::move(space), std::move(workspace)),
      quantity(quantity), frame(std::move(frame)), reference(reference),
      symbolic_gravity(symbolic_gravity)
{
    if (reference != pinocchio::LOCAL && reference != pinocchio::WORLD &&
        reference != pinocchio::LOCAL_WORLD_ALIGNED)
    {
        throw std::invalid_argument("unknown frame reference convention");
    }
    if (!this->frame.empty())
    {
        frame_id = detail::frameId(this->model, this->frame);
    }
    else if (quantity == RobotQuantity::FrameJacobian ||
             quantity == RobotQuantity::FrameVelocity ||
             quantity == RobotQuantity::FrameAcceleration ||
             quantity == RobotQuantity::FrameSpatialAcceleration ||
             quantity == RobotQuantity::FrameTransform)
    {
        throw std::invalid_argument("a frame name is required");
    }
    if (symbolic_gravity && quantity != RobotQuantity::InverseDynamics)
    {
        throw std::invalid_argument(
            "symbolic gravity is supported by inverse dynamics only");
    }
}
bool PinocchioRobotFunction::Impl::isDynamics() const
{
    return quantity >= RobotQuantity::InverseDynamics;
}
PinocchioRobotFunction::PinocchioRobotFunction(
    std::shared_ptr<const pinocchio::Model> model,
    std::shared_ptr<const PinocchioConfiguration> space, RobotQuantity quantity,
    std::string frame, pinocchio::ReferenceFrame reference,
    bool symbolic_gravity, std::shared_ptr<PinocchioModelWorkspace> workspace)
    : TangentFunction(inputSpace(model, space, quantity, symbolic_gravity),
                      robotOutputSize(model, quantity)),
      _impl(std::make_unique<Impl>(std::move(model), std::move(space), quantity,
                                   std::move(frame), reference,
                                   symbolic_gravity, std::move(workspace)))
{
}
PinocchioRobotFunction::~PinocchioRobotFunction() = default;
std::unique_ptr<ilqr::Function> PinocchioRobotFunction::clone() const
{
    return std::make_unique<PinocchioRobotFunction>(
        _impl->source, _impl->space, _impl->quantity, _impl->frame,
        _impl->reference, _impl->symbolic_gravity, _impl->workspace_prototype);
}
void PinocchioRobotFunction::setEvaluationContext(
    const std::shared_ptr<ilqr::EvaluationContext> &context)
{
    _impl->setEvaluationContext(context);
}
void PinocchioRobotFunction::evaluateImpl(ilqr::Input z, ilqr::VectorOutput out)
{
    _impl->prepare(z);
    if (_impl->isDynamics())
    {
        _impl->workspace->invalidate();
        _impl->evaluateDynamics(z, out);
    }
    else
    {
        _impl->evaluateKinematics(z, out);
    }
}
void PinocchioRobotFunction::tangentJacobian(ilqr::Input z,
                                             ilqr::MatrixOutput jac)
{
    _impl->prepare(z);
    if (_impl->isDynamics())
    {
        _impl->workspace->invalidate();
        _impl->jacobianDynamics(z, jac);
    }
    else
    {
        _impl->jacobianKinematics(z, jac);
    }
}
void PinocchioRobotFunction::Impl::evaluateKinematics(ilqr::Input z,
                                                      ilqr::VectorOutput out)
{
    using Q = RobotQuantity;
    if (quantity == Q::Difference)
    {
        ilqr::Vector other;
        detail::normalize(model, z.tail(model.nq), other);
        out = pinocchio::difference(model, q, other);
        return;
    }
    if (quantity == Q::QDot)
    {
        ilqr::Matrix R(model.nq, model.nv);
        space->retractionJacobian(q, R);
        out.noalias() = R * z.tail(model.nv);
        return;
    }
    if (quantity == Q::FrameJacobian)
    {
        workspace->ensureJointJacobians(q);
        workspace->ensureFramePlacements(q);
        ilqr::Matrix J = ilqr::Matrix::Zero(6, model.nv);
        pinocchio::getFrameJacobian(model, data(), frame_id, reference, J);
        out = Eigen::Map<const ilqr::Vector>(J.data(), J.size());
        return;
    }
    if (quantity == Q::FrameTransform)
    {
        workspace->ensureFramePlacements(q);
        const auto &pose = data().oMf[frame_id];
        out.head<3>() = pose.translation();
        out.tail<9>() =
            Eigen::Map<const ilqr::Vector>(pose.rotation().data(), 9);
        return;
    }
    const auto v = z.segment(model.nq, model.nv);
    if (quantity == Q::FrameVelocity)
    {
        workspace->ensureVelocity(q, v);
    }
    else
    {
        workspace->ensureAcceleration(q, v, z.tail(model.nv));
    }
    workspace->ensureFramePlacements(q);
    if (quantity == Q::FrameVelocity)
    {
        out = pinocchio::getFrameVelocity(model, data(), frame_id, reference)
                  .toVector();
    }
    else if (quantity == Q::FrameAcceleration &&
             reference == pinocchio::LOCAL_WORLD_ALIGNED)
    {
        out = pinocchio::getFrameClassicalAcceleration(model, data(), frame_id,
                                                       reference)
                  .toVector();
    }
    else
    {
        out = pinocchio::getFrameAcceleration(model, data(), frame_id, reference)
                  .toVector();
    }
}
void PinocchioRobotFunction::Impl::jacobianKinematics(ilqr::Input z,
                                                      ilqr::MatrixOutput jac)
{
    using Q = RobotQuantity;
    jac.setZero();
    if (quantity == Q::Difference)
    {
        ilqr::Vector other;
        detail::normalize(model, z.tail(model.nq), other);
        ilqr::Matrix A = ilqr::Matrix::Zero(model.nv, model.nv), B = A;
        pinocchio::dDifference(model, q, other, A, pinocchio::ARG0);
        pinocchio::dDifference(model, q, other, B, pinocchio::ARG1);
        jac.leftCols(model.nv) = A;
        jac.rightCols(model.nv) = B;
        return;
    }
    if (quantity == Q::QDot)
    {
        ilqr::Matrix R(model.nq, model.nv);
        space->retractionJacobian(q, R);
        jac.rightCols(model.nv) = R;
        for (pinocchio::JointIndex i = 1; i < model.njoints; ++i)
        {
            qdotDerivative(model.joints[i], q, z.tail(model.nv), R, jac);
        }
        return;
    }
    if (quantity == Q::FrameTransform)
    {
        workspace->ensureJointJacobians(q);
        workspace->ensureFramePlacements(q);
        ilqr::Matrix J = ilqr::Matrix::Zero(6, model.nv);
        pinocchio::getFrameJacobian(model, data(), frame_id,
                                    pinocchio::LOCAL_WORLD_ALIGNED, J);
        jac.topRows(3) = J.topRows(3);
        for (int k = 0; k < model.nv; ++k)
        {
            const Eigen::Matrix3d derivative =
                pinocchio::skew(J.col(k).tail<3>()) *
                data().oMf[frame_id].rotation();
            jac.col(k).tail<9>() =
                Eigen::Map<const ilqr::Vector>(derivative.data(), 9);
        }
        return;
    }
    if (quantity == Q::FrameJacobian)
    {
        workspace->invalidate();
        ilqr::Vector basis = zero;
        for (int k = 0; k < model.nv; ++k)
        {
            basis[k] = 1;
            pinocchio::computeForwardKinematicsDerivatives(model, data(), q,
                                                           basis, zero);
            pinocchio::updateFramePlacements(model, data());
            ilqr::Matrix dq = ilqr::Matrix::Zero(6, model.nv), dv = dq;
            detail::frameVelocityDerivatives(model, data(), frame_id, reference,
                                             dq, dv);
            jac.middleRows(6 * k, 6) = dq;
            basis[k] = 0;
        }
        return;
    }
    const ilqr::Vector v = z.segment(model.nq, model.nv);
    const ilqr::Vector a =
        quantity == Q::FrameVelocity ? zero : ilqr::Vector(z.tail(model.nv));
    workspace->ensureKinematicsDerivatives(q, v, a);
    workspace->ensureFramePlacements(q);
    ilqr::Matrix dq, dv, da;
    if (quantity == Q::FrameVelocity)
    {
        dq.setZero(6, model.nv);
        dv = dq;
        detail::frameVelocityDerivatives(model, data(), frame_id, reference, dq,
                                         dv);
        jac.leftCols(model.nv) = dq;
        jac.rightCols(model.nv) = dv;
    }
    else
    {
        detail::classicalDerivatives(
            model, data(), frame_id, reference, dq, dv, da,
            quantity == Q::FrameAcceleration &&
                reference == pinocchio::LOCAL_WORLD_ALIGNED);
        jac.leftCols(model.nv) = dq;
        jac.middleCols(model.nv, model.nv) = dv;
        jac.rightCols(model.nv) = da;
    }
}
} // namespace horizon::kinematics
