#include "spot_contact_projection.h"
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/spatial/skew.hpp>

namespace spot_test
{
struct ContactProjection::Impl
{
    std::shared_ptr<const Robot> robot;
    pin::Data data;
    Matrix J, frame, Vq, Vv, velocity_q;
    Vector error;
    Eigen::PartialPivLU<Matrix> lu;
    explicit Impl(std::shared_ptr<const Robot> r)
        : robot(std::move(r)), data(*robot->model), J(nf, nv), frame(6, nv),
          Vq(6, nv), Vv(6, nv), velocity_q(nf, nv), error(nf)
    {
    }

    void geometry(Input q)
    {
        pin::computeJointJacobians(*robot->model, data, q);
        pin::updateFramePlacements(*robot->model, data);
        for (int i = 0; i < 4; ++i)
        {
            frame.setZero();
            pin::getFrameJacobian(*robot->model, data, robot->feet[i],
                                  pin::LOCAL_WORLD_ALIGNED, frame);
            J.middleRows<3>(3 * i) = frame.topRows(3);
            error.segment<3>(3 * i) = data.oMf[robot->feet[i]].translation() -
                robot->anchors.segment<3>(3 * i);
        }
        lu.compute(J.rightCols(12));
    }
    void evaluate(Input raw, VectorOutput out)
    {
        out = raw;
        for (int iter = 0; iter < 40; ++iter)
        {
            geometry(out.head(nq));
            if (error.lpNorm<Eigen::Infinity>() < 1e-13)
            {
                break;
            }
            Eigen::Matrix<double, 12, 1> correction = lu.solve(-error);
            if (!correction.allFinite())
            {
                throw EvaluationError("contact projection IK failed");
            }
            // Stay on the same local leg branch during trial evaluations.
            double scale = std::min(1.0, .3 / correction.cwiseAbs().maxCoeff());
            out.segment(7, 12) += scale * correction;
        }
        if (error.lpNorm<Eigen::Infinity>() > 1e-11)
        {
            throw EvaluationError("contact projection IK did not converge");
        }
        out.tail(12) = lu.solve(-J.leftCols(6) * raw.segment(nq, 6));
        if (!out.allFinite())
        {
            throw EvaluationError("contact velocity projection failed");
        }
    }
};
ContactProjection::ContactProjection(std::shared_ptr<const Robot> r)
    : impl(std::make_unique<Impl>(std::move(r)))
{
}
ContactProjection::~ContactProjection() = default;
void ContactProjection::evaluate(Input raw, VectorOutput out)
{
    impl->evaluate(raw, out);
}
void ContactProjection::valueAndJacobian(Input raw, VectorOutput out,
                                         MatrixOutput jac)
{
    auto &p = *impl;
    p.evaluate(raw, out);
    Eigen::Matrix<double, nv, nv> tangent =
        Eigen::Matrix<double, nv, nv>::Zero();
    tangent.topLeftCorner(6, 6).setIdentity();
    tangent.bottomLeftCorner(12, 6) = p.lu.solve(-p.J.leftCols(6));
    Eigen::Matrix<double, nq, nv> R;
    p.robot->configuration->retractionJacobian(out.head(nq), R);
    jac.setZero();
    jac.topLeftCorner(nq, nv).noalias() = R * tangent;
    jac.bottomRightCorner(nv, nv) = tangent;
    pin::computeForwardKinematicsDerivatives(
        *p.robot->model, p.data, out.head(nq), out.tail(nv), Vector::Zero(nv));
    pin::updateFramePlacements(*p.robot->model, p.data);
    for (int i = 0; i < 4; ++i)
    {
        p.Vq.setZero();
        p.Vv.setZero();
        pin::getFrameVelocityDerivatives(
            *p.robot->model, p.data, p.robot->feet[i], pin::LOCAL, p.Vq, p.Vv);
        auto vel = pin::getFrameVelocity(*p.robot->model, p.data,
                                         p.robot->feet[i], pin::LOCAL);
        auto rotation = p.data.oMf[p.robot->feet[i]].rotation();
        p.velocity_q.middleRows<3>(3 * i) = rotation *
            (p.Vq.topRows(3) - pin::skew(vel.linear()) * p.Vv.bottomRows(3));
    }
    jac.bottomLeftCorner(12, nv) = p.lu.solve(-p.velocity_q * tangent);
}
} // namespace spot_test
