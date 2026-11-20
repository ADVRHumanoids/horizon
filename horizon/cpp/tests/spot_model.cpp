#include "spot_model.h"
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/skew.hpp>

namespace spot_test
{
constexpr double kp = 25, kd = 10;
void ConfigurationSpace::neutral(VectorOutput x) const
{
    x = pin::neutral(*model);
}
void ConfigurationSpace::integrate(Input x, Input dx, VectorOutput out) const
{
    out = pin::integrate(*model, x, dx);
}
void ConfigurationSpace::difference(Input a, Input b, VectorOutput out) const
{
    out = pin::difference(*model, a, b);
}
void ConfigurationSpace::retractionJacobian(Input x, MatrixOutput j) const
{
    j.setZero();
    pin::integrateCoeffWiseJacobian(*model, x, j);
}
void ConfigurationSpace::tangentProjection(Input x,
                                           MatrixOutput projection) const
{
    Eigen::Matrix<double, nq, nv> R;
    retractionJacobian(x, R);
    projection = R.transpose();
    projection.middleRows<3>(3) *= 4;
}
void ConfigurationSpace::projectDirections(Input x, MatrixInput storage,
                                           MatrixOutput tangent) const
{
    Eigen::Matrix<double, nq, nv> R;
    retractionJacobian(x, R);
    tangent.topRows<3>().noalias() =
        R.topLeftCorner<3, 3>().transpose() * storage.topRows<3>();
    tangent.middleRows<3>(3).noalias() =
        4 * R.block<4, 3>(3, 3).transpose() * storage.middleRows<4>(3);
    tangent.bottomRows<12>() = storage.bottomRows<12>();
}
void ConfigurationSpace::differenceJacobians(Input a, Input b, MatrixOutput A,
                                             MatrixOutput B) const
{
    A.setZero();
    B.setZero();
    pin::dDifference(*model, a, b, A, pin::ARG0);
    pin::dDifference(*model, a, b, B, pin::ARG1);
}
void ConfigurationSpace::retractionHessian(Input q, Input w,
                                           MatrixOutput h) const
{
    // Spot is SE(3) x R^12, using body-frame free-flyer increments.
    Eigen::Quaterniond rotation(q(6), q(3), q(4), q(5));
    Eigen::Vector3d c = rotation.toRotationMatrix().transpose() * w.head(3);
    h.setZero();
    h.block<3, 3>(0, 3) = .5 * pin::skew(c);
    h.block<3, 3>(3, 0) = h.block<3, 3>(0, 3).transpose();
    h.block<3, 3>(3, 3).diagonal().setConstant(
        -.25 * q.segment(3, 4).dot(w.segment(3, 4)));
}

Robot::Robot(const std::string &urdf)
{
    auto loaded = std::make_shared<pin::Model>();
    pin::urdf::buildModel(urdf, *loaded);
    if (loaded->nq != nq || loaded->nv != nv)
    {
        throw std::runtime_error("expected Spot free-flyer plus twelve joints");
    }
    model = loaded;
    configuration = std::make_shared<ConfigurationSpace>(model);
    state = std::make_shared<ProductSpace>(
        std::vector<std::shared_ptr<const StateSpace>>{
            configuration, std::make_shared<EuclideanSpace>(nv)});
    standing = pin::neutral(*model);
    const std::array<std::string, 4> legs{"lf", "rf", "lh", "rh"};
    for (int i = 0; i < 4; ++i)
    {
        feet[i] = model->getFrameId(legs[i] + "_foot");
        if (feet[i] >= model->frames.size())
        {
            throw std::runtime_error("missing Spot foot frame");
        }
        for (auto joint : std::vector<std::pair<std::string, double>>{
                 {"haa", 0}, {"hfe", .9}, {"kfe", -1.52}})
        {
            standing(model->idx_qs[model->getJointId(
                legs[i] + "_" + joint.first + "_joint")]) = joint.second;
        }
    }
    pin::Data data(*model);
    pin::framesForwardKinematics(*model, data, standing);
    double z = 0;
    for (auto foot : feet)
    {
        z += data.oMf[foot].translation().z() / 4;
    }
    standing(2) -= z;
    pin::computeJointJacobians(*model, data, standing);
    pin::updateFramePlacements(*model, data);
    anchors.resize(nf);
    Matrix A(6, 4), J6 = Matrix::Zero(6, nv);
    for (int i = 0; i < 4; ++i)
    {
        anchors.segment<3>(3 * i) = data.oMf[feet[i]].translation();
        J6.setZero();
        pin::getFrameJacobian(*model, data, feet[i], pin::LOCAL_WORLD_ALIGNED,
                              J6);
        A.col(i) = J6.topRows(3).leftCols(6).row(2).transpose();
    }
    Vector gravity = pin::computeGeneralizedGravity(*model, data, standing);
    Vector normal = A.colPivHouseholderQr().solve(gravity.head(6));
    support = Vector::Zero(nf);
    for (int i = 0; i < 4; ++i)
    {
        support(3 * i + 2) = normal(i);
    }
    if ((A * normal - gravity.head(6)).norm() > 1e-7 || normal.minCoeff() <= 0)
    {
        throw std::runtime_error(
            "Spot standing pose has no positive static support");
    }
    weight = 0;
    for (const auto &inertia : model->inertias)
    {
        weight += 9.81 * inertia.mass();
    }
}

Engine::Engine(std::shared_ptr<const Robot> r)
    : robot(std::move(r)), dynamics(*robot->model), kinematics(*robot->model),
      external_forces(robot->model->njoints, pin::Force::Zero()), J(6, nv),
      Vq(6, nv), Vv(6, nv), Aq(6, nv), Av(6, nv), Aa(6, nv), M(nv, nv),
      K(nv + nf, nv + nf), residualJac(nv + nf, 3 * nv + nf),
      zero(Vector::Zero(nv)), terms(nv + 3 * nf)
{
}

void Engine::evaluate(Input q, Input v, Input a, Input f, VectorOutput out)
{
    out.head(nv) = pin::rnea(*robot->model, dynamics, q, v, a);
    pin::computeJointJacobians(*robot->model, kinematics, q);
    pin::forwardKinematics(*robot->model, kinematics, q, v, a);
    pin::updateFramePlacements(*robot->model, kinematics);
    for (int i = 0; i < 4; ++i)
    {
        const auto frame = robot->feet[i];
        J.setZero();
        pin::getFrameJacobian(*robot->model, kinematics, frame,
                              pin::LOCAL_WORLD_ALIGNED, J);
        out.head(nv).noalias() -=
            J.topRows(3).transpose() * f.segment<3>(3 * i);
        const auto p = kinematics.oMf[frame].translation();
        const auto vel = pin::getFrameVelocity(*robot->model, kinematics, frame,
                                               pin::LOCAL_WORLD_ALIGNED);
        const auto acc = pin::getFrameClassicalAcceleration(
            *robot->model, kinematics, frame, pin::LOCAL_WORLD_ALIGNED);
        out.segment<3>(nv + 3 * i) = acc.linear() +
            kp * (p - robot->anchors.segment<3>(3 * i)) + kd * vel.linear();
        out.segment<3>(nv + nf + 3 * i) = p;
        out.segment<3>(nv + 2 * nf + 3 * i) = vel.linear();
    }
}

void Engine::linearize(Input q, Input v, Input a, Input f, MatrixOutput jac)
{
    jac.setZero();
    dynamics.dtau_dq.setZero();
    dynamics.dtau_dv.setZero();
    dynamics.M.setZero();
    pin::computeForwardKinematicsDerivatives(*robot->model, kinematics, q, v,
                                             a);
    pin::updateFramePlacements(*robot->model, kinematics);
    for (auto &force : external_forces)
    {
        force.setZero();
    }
    for (int i = 0; i < 4; ++i)
    {
        const auto frame = robot->feet[i];
        const auto &placement = robot->model->frames[frame];
        Eigen::Vector3d local_force =
            kinematics.oMf[frame].rotation().transpose() * f.segment<3>(3 * i);
        external_forces[placement.parent] += placement.placement.act(
            pin::Force(local_force, Eigen::Vector3d::Zero()));
    }
    // RNEA differentiates the contracted external wrench in one native sweep.
    // Its external forces are fixed in joint-local coordinates. Correct below
    // for our contact forces being fixed in world coordinates instead.
    pin::computeRNEADerivatives(*robot->model, dynamics, q, v, a,
                                external_forces);
    terms.head(nv) = dynamics.tau;
    jac.block(0, 0, nv, nv) = dynamics.dtau_dq;
    jac.block(0, nv, nv, nv) = dynamics.dtau_dv;
    jac.block(0, 2 * nv, nv, nv) = dynamics.M.selfadjointView<Eigen::Upper>();
    for (int i = 0; i < 4; ++i)
    {
        const auto frame = robot->feet[i];
        Vq.setZero();
        Vv.setZero();
        Aq.setZero();
        Av.setZero();
        Aa.setZero();
        // Compute in the fixed local frame, then differentiate the rotation
        // into world coordinates explicitly. This also accounts for the
        // rotating foot offset in Pinocchio versions with partial LWA helpers.
        pin::getFrameAccelerationDerivatives(*robot->model, kinematics, frame,
                                             pin::LOCAL, Vq, Aq, Av, Aa);
        pin::getFrameVelocityDerivatives(*robot->model, kinematics, frame,
                                         pin::LOCAL, Vq, Vv);
        auto vel =
            pin::getFrameVelocity(*robot->model, kinematics, frame, pin::LOCAL);
        auto acc = pin::getFrameClassicalAcceleration(*robot->model, kinematics,
                                                      frame, pin::LOCAL);
        Eigen::Matrix3d R = kinematics.oMf[frame].rotation();
        Eigen::Matrix<double, 3, nv> Jworld = R * Vv.topRows(3);
        Eigen::Vector3d local_force = R.transpose() * f.segment<3>(3 * i);
        jac.topLeftCorner(nv, nv).noalias() -= Vv.topRows(3).transpose() *
            pin::skew(local_force) * Vv.bottomRows(3);
        terms.segment<3>(nv + 3 * i) = R * acc.linear() +
            kp *
                (kinematics.oMf[frame].translation() -
                 robot->anchors.segment<3>(3 * i)) +
            kd * R * vel.linear();
        terms.segment<3>(nv + nf + 3 * i) = kinematics.oMf[frame].translation();
        terms.segment<3>(nv + 2 * nf + 3 * i) = R * vel.linear();
        Eigen::Matrix<double, 3, nv> velocity_q =
            R * (Vq.topRows(3) - pin::skew(vel.linear()) * Vv.bottomRows(3));
        Eigen::Matrix<double, 3, nv> cq = R *
            (Aq.topRows(3) - pin::skew(vel.linear()) * Vq.bottomRows(3) +
             pin::skew(vel.angular()) * Vq.topRows(3) -
             pin::skew(acc.linear()) * Vv.bottomRows(3));
        Eigen::Matrix<double, 3, nv> cv = R *
            (Av.topRows(3) - pin::skew(vel.linear()) * Vv.bottomRows(3) +
             pin::skew(vel.angular()) * Vv.topRows(3));
        jac.block(nv + 3 * i, 0, 3, nv) = cq + kp * Jworld + kd * velocity_q;
        jac.block(nv + 3 * i, nv, 3, nv) = cv + kd * Jworld;
        jac.block(nv + 3 * i, 2 * nv, 3, nv).noalias() = R * Aa.topRows(3);
        jac.block(nv + nf + 3 * i, 0, 3, nv) = Jworld;
        jac.block(nv + 2 * nf + 3 * i, 0, 3, nv) = velocity_q;
        jac.block(nv + 2 * nf + 3 * i, nv, 3, nv) = Jworld;
        jac.block(0, 3 * nv + 3 * i, nv, 3) = -Jworld.transpose();
    }
}

void Engine::forward(Input q, Input v, Input tau, VectorOutput af)
{
    evaluate(q, v, zero, Vector::Zero(nf), terms);
    pin::crba(*robot->model, dynamics, q);
    M = dynamics.M.selfadjointView<Eigen::Upper>();
    K.setZero();
    K.topLeftCorner(nv, nv) = M;
    for (int i = 0; i < 4; ++i)
    {
        J.setZero();
        pin::getFrameJacobian(*robot->model, kinematics, robot->feet[i],
                              pin::LOCAL_WORLD_ALIGNED, J);
        K.block(0, nv + 3 * i, nv, 3) = -J.topRows(3).transpose();
        K.block(nv + 3 * i, 0, 3, nv) = J.topRows(3);
    }
    Eigen::Matrix<double, nv + nf, 1> rhs;
    rhs.head(nv) = -terms.head(nv);
    rhs.segment(6, 12) += tau;
    rhs.tail(nf) = -terms.segment(nv, nf);
    lu.compute(K);
    af = lu.solve(rhs);
    if (!af.allFinite() || (K * af - rhs).norm() > 1e-7 * (1 + rhs.norm()))
    {
        throw EvaluationError("contact KKT solve failed");
    }
}
void Engine::forwardJacobian(Input q, Input v, Input tau, VectorOutput af,
                             MatrixOutput jac, MatrixOutput full)
{
    forward(q, v, tau, af);
    linearize(q, v, af.head(nv), af.tail(nf), full);
    residualJac.topRows(nv) = full.topRows(nv);
    residualJac.bottomRows(nf) = full.middleRows(nv, nf);
    // Implicit derivative: R_y dy = -R_(q,v) d(q,v) + [S^T;0] d tau.
    Eigen::Matrix<double, nv + nf, 2 * nv + 12> rhs =
        Eigen::Matrix<double, nv + nf, 2 * nv + 12>::Zero();
    rhs.leftCols(2 * nv) = -residualJac.leftCols(2 * nv);
    rhs.block(6, 2 * nv, 12, 12).setIdentity();
    jac = lu.solve(rhs);
}
} // namespace spot_test
