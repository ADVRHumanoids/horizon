#pragma once
#include "horizon/pinocchio_robot.h"
#include <Eigen/Geometry>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/spatial/skew.hpp>

namespace horizon::kinematics::detail
{
inline void normalize(const pinocchio::Model &model, ilqr::Input input,
                      ilqr::Vector &q)
{
    if (input.size() != model.nq || !input.allFinite())
    {
        throw std::invalid_argument("invalid Pinocchio configuration");
    }
    q = input;
    pinocchio::normalize(model, q);
    if (!q.allFinite() || !pinocchio::isNormalized(model, q, 1e-10))
    {
        throw std::invalid_argument(
            "invalid Pinocchio configuration quaternion");
    }
}
inline void finite(ilqr::Input value)
{
    if (!value.allFinite())
    {
        throw std::invalid_argument("robot input must be finite");
    }
}
inline void symmetric(ilqr::Matrix &matrix)
{
    matrix.template triangularView<Eigen::StrictlyLower>() =
        matrix.transpose().template triangularView<Eigen::StrictlyLower>();
}
inline std::shared_ptr<const ilqr::StateSpace>
product(const std::shared_ptr<const pinocchio::Model> &model,
        const std::shared_ptr<const PinocchioConfiguration> &space,
        int additional)
{
    if (!model || !space || space->model.get() != model.get())
    {
        throw std::invalid_argument(
            "robot function and configuration must share a model");
    }
    if (!additional)
    {
        return space;
    }
    return std::make_shared<ilqr::ProductSpace>(
        std::vector<std::shared_ptr<const ilqr::StateSpace>>{
            space, std::make_shared<ilqr::EuclideanSpace>(additional)});
}
inline pinocchio::FrameIndex frameId(const pinocchio::Model &model,
                                     const std::string &name)
{
    if (!model.existFrame(name))
    {
        throw std::invalid_argument("unknown Pinocchio frame: " + name);
    }
    return model.getFrameId(name);
}

// Transform LOCAL derivatives explicitly. This includes the derivative of the
// moving frame origin/basis, which older Pinocchio LWA derivative overloads
// omit.
inline void rotateRows(const Eigen::Matrix3d &rotation, ilqr::Matrix &matrix)
{
    matrix.topRows(3) = (rotation * matrix.topRows(3)).eval();
    matrix.bottomRows(3) = (rotation * matrix.bottomRows(3)).eval();
}
inline void frameVelocityDerivatives(const pinocchio::Model &model,
                                     pinocchio::Data &data,
                                     pinocchio::FrameIndex frame,
                                     pinocchio::ReferenceFrame reference,
                                     ilqr::Matrix &dq, ilqr::Matrix &dv)
{
    dq.setZero(6, model.nv);
    dv.setZero(6, model.nv);
    if (reference != pinocchio::LOCAL_WORLD_ALIGNED)
    {
        pinocchio::getFrameVelocityDerivatives(model, data, frame, reference,
                                               dq, dv);
        return;
    }
    pinocchio::getFrameVelocityDerivatives(model, data, frame, pinocchio::LOCAL,
                                           dq, dv);
    const Eigen::Matrix3d rotation = data.oMf[frame].rotation();
    const auto velocity =
        pinocchio::getFrameVelocity(model, data, frame, pinocchio::LOCAL);
    const ilqr::Matrix angular = dv.bottomRows(3);
    rotateRows(rotation, dq);
    dq.topRows(3) -= rotation * pinocchio::skew(velocity.linear()) * angular;
    dq.bottomRows(3) -=
        rotation * pinocchio::skew(velocity.angular()) * angular;
    rotateRows(rotation, dv);
}

// Classical acceleration is spatial acceleration plus omega x linear velocity.
inline void classicalDerivatives(const pinocchio::Model &model,
                                 pinocchio::Data &data,
                                 pinocchio::FrameIndex frame,
                                 pinocchio::ReferenceFrame reference,
                                 ilqr::Matrix &dq, ilqr::Matrix &dv,
                                 ilqr::Matrix &da, bool classical)
{
    ilqr::Matrix velocity_q = ilqr::Matrix::Zero(6, model.nv),
                 velocity_v = velocity_q;
    dq.setZero(6, model.nv);
    dv.setZero(6, model.nv);
    da.setZero(6, model.nv);
    const auto derivative_reference =
        reference == pinocchio::LOCAL_WORLD_ALIGNED ? pinocchio::LOCAL
                                                    : reference;
    pinocchio::getFrameAccelerationDerivatives(model, data, frame,
                                               derivative_reference, velocity_q,
                                               velocity_v, dq, dv, da);
    if (reference == pinocchio::LOCAL_WORLD_ALIGNED)
    {
        const Eigen::Matrix3d rotation = data.oMf[frame].rotation();
        const auto acceleration = pinocchio::getFrameAcceleration(
            model, data, frame, pinocchio::LOCAL);
        const ilqr::Matrix angular = velocity_v.bottomRows(3);
        rotateRows(rotation, dq);
        rotateRows(rotation, dv);
        rotateRows(rotation, da);
        dq.topRows(3) -=
            rotation * pinocchio::skew(acceleration.linear()) * angular;
        dq.bottomRows(3) -=
            rotation * pinocchio::skew(acceleration.angular()) * angular;
        frameVelocityDerivatives(model, data, frame, reference, velocity_q,
                                 velocity_v);
    }
    if (classical)
    {
        const auto motion =
            pinocchio::getFrameVelocity(model, data, frame, reference);
        dq.topRows(3) +=
            pinocchio::skew(motion.angular()) * velocity_q.topRows(3) -
            pinocchio::skew(motion.linear()) * velocity_q.bottomRows(3);
        dv.topRows(3) +=
            pinocchio::skew(motion.angular()) * velocity_v.topRows(3) -
            pinocchio::skew(motion.linear()) * velocity_v.bottomRows(3);
    }
}

struct PinocchioRobotImplBase
{
    std::shared_ptr<const pinocchio::Model> source;
    std::shared_ptr<const PinocchioConfiguration> space;
    pinocchio::Model model;
    pinocchio::Data data;
    ilqr::Vector q, zero;
    PinocchioRobotImplBase(std::shared_ptr<const pinocchio::Model> source,
                           std::shared_ptr<const PinocchioConfiguration> space)
        : source(std::move(source)), space(std::move(space)),
          model(*this->source), data(model), q(model.nq),
          zero(ilqr::Vector::Zero(model.nv))
    {
    }
    void prepare(ilqr::Input z)
    {
        finite(z);
        normalize(model, z.head(model.nq), q);
    }
};
} // namespace horizon::kinematics::detail

namespace horizon::kinematics
{
struct PinocchioRobotFunction::Impl : detail::PinocchioRobotImplBase
{
    RobotQuantity quantity;
    std::string frame;
    pinocchio::ReferenceFrame reference;
    pinocchio::FrameIndex frame_id = 0;
    bool symbolic_gravity;
    Impl(std::shared_ptr<const pinocchio::Model> model,
         std::shared_ptr<const PinocchioConfiguration> space,
         RobotQuantity quantity, std::string frame,
         pinocchio::ReferenceFrame reference, bool symbolic_gravity);
    bool isDynamics() const;
    void evaluateKinematics(ilqr::Input z, ilqr::VectorOutput out);
    void jacobianKinematics(ilqr::Input z, ilqr::MatrixOutput jac);
    void evaluateDynamics(ilqr::Input z, ilqr::VectorOutput out);
    void jacobianDynamics(ilqr::Input z, ilqr::MatrixOutput jac);
};
} // namespace horizon::kinematics
