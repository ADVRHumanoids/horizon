#include "pinocchio_robot_internal.h"
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal-derivatives.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/energy.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>

namespace horizon::kinematics
{
void PinocchioRobotFunction::Impl::evaluateDynamics(ilqr::Input z,
                                                    ilqr::VectorOutput out)
{
    using Q = RobotQuantity;
    if (symbolic_gravity)
    {
        model.gravity.linear() = z.tail<3>();
    }
    switch (quantity)
    {
    case Q::InverseDynamics:
        out = pinocchio::rnea(model, data, q, z.segment(model.nq, model.nv),
                              z.segment(model.nq + model.nv, model.nv));
        return;
    case Q::ForwardDynamics:
        out = pinocchio::aba(model, data, q, z.segment(model.nq, model.nv),
                             z.tail(model.nv));
        return;
    case Q::MassMatrix:
    {
        ilqr::Matrix M = pinocchio::crba(model, data, q);
        detail::symmetric(M);
        out = Eigen::Map<const ilqr::Vector>(M.data(), M.size());
        return;
    }
    case Q::Gravity:
        out = pinocchio::computeGeneralizedGravity(model, data, q);
        return;
    case Q::NonlinearEffects:
        out = pinocchio::nonLinearEffects(model, data, q, z.tail(model.nv));
        return;
    case Q::KineticEnergy:
        out[0] =
            pinocchio::computeKineticEnergy(model, data, q, z.tail(model.nv));
        return;
    case Q::PotentialEnergy:
        out[0] = pinocchio::computePotentialEnergy(model, data, q);
        return;
    case Q::CentroidalMap:
    {
        ilqr::Matrix A = pinocchio::ccrba(model, data, q, zero);
        out = Eigen::Map<const ilqr::Vector>(A.data(), A.size());
        return;
    }
    default:
        break;
    }
    const ilqr::Vector v = (quantity == Q::ComPosition)
        ? zero
        : ilqr::Vector(z.segment(model.nq, model.nv));
    const ilqr::Vector a =
        (quantity == Q::CenterOfMass || quantity == Q::ComAcceleration ||
         quantity == Q::CentroidalDynamics)
        ? ilqr::Vector(z.tail(model.nv))
        : zero;
    if (quantity == Q::CentroidalMomentum || quantity == Q::CentroidalDynamics)
    {
        pinocchio::computeCentroidalMomentumTimeVariation(model, data, q, v, a);
        out.head<6>() = data.hg.toVector();
        if (quantity == Q::CentroidalDynamics)
        {
            out.tail<6>() = data.dhg.toVector();
        }
        return;
    }
    if (pinocchio::computeTotalMass(model) <= 0)
    {
        throw ilqr::EvaluationError(
            "center of mass requires positive robot mass");
    }
    pinocchio::centerOfMass(model, data, q, v, a);
    if (quantity == Q::CenterOfMass)
    {
        out.head<3>() = data.com[0];
        out.segment<3>(3) = data.vcom[0];
        out.tail<3>() = data.acom[0];
    }
    else if (quantity == Q::ComPosition)
    {
        out = data.com[0];
    }
    else if (quantity == Q::ComVelocity)
    {
        out = data.vcom[0];
    }
    else if (quantity == Q::ComAcceleration)
    {
        out = data.acom[0];
    }
    else
    {
        throw std::logic_error("unknown native robot dynamics quantity");
    }
}
void PinocchioRobotFunction::Impl::jacobianDynamics(ilqr::Input z,
                                                    ilqr::MatrixOutput jac)
{
    using Q = RobotQuantity;
    jac.setZero();
    if (symbolic_gravity)
    {
        model.gravity.linear() = z.tail<3>();
    }
    if (quantity == Q::ForwardDynamics)
    {
        pinocchio::computeABADerivatives(
            model, data, q, z.segment(model.nq, model.nv), z.tail(model.nv));
        ilqr::Matrix inverse = data.Minv;
        detail::symmetric(inverse);
        jac.leftCols(model.nv) = data.ddq_dq;
        jac.middleCols(model.nv, model.nv) = data.ddq_dv;
        jac.rightCols(model.nv) = inverse;
        return;
    }
    if (quantity == Q::InverseDynamics || quantity == Q::NonlinearEffects ||
        quantity == Q::Gravity)
    {
        const ilqr::Vector v = quantity == Q::Gravity
            ? zero
            : ilqr::Vector(z.segment(model.nq, model.nv));
        const ilqr::Vector a = quantity == Q::InverseDynamics
            ? ilqr::Vector(z.segment(model.nq + model.nv, model.nv))
            : zero;
        pinocchio::computeRNEADerivatives(model, data, q, v, a);
        jac.leftCols(model.nv) = data.dtau_dq;
        if (quantity != Q::Gravity)
        {
            jac.middleCols(model.nv, model.nv) = data.dtau_dv;
        }
        if (quantity == Q::InverseDynamics)
        {
            ilqr::Matrix M = data.M;
            detail::symmetric(M);
            jac.middleCols(2 * model.nv, model.nv) = M;
        }
        if (symbolic_gravity)
        {
            pinocchio::jacobianCenterOfMass(model, data, q);
            jac.rightCols(3) =
                -pinocchio::computeTotalMass(model) * data.Jcom.transpose();
        }
        return;
    }
    if (quantity == Q::MassMatrix || quantity == Q::KineticEnergy)
    {
        pinocchio::computeRNEADerivatives(model, data, q, zero, zero);
        const ilqr::Matrix gravity_q = data.dtau_dq;
        if (quantity == Q::MassMatrix)
        {
            ilqr::Vector basis = zero;
            for (int k = 0; k < model.nv; ++k)
            {
                basis[k] = 1;
                pinocchio::computeRNEADerivatives(model, data, q, zero, basis);
                jac.middleRows(k * model.nv, model.nv) =
                    data.dtau_dq - gravity_q;
                basis[k] = 0;
            }
        }
        else
        {
            const ilqr::Vector v = z.tail(model.nv);
            pinocchio::computeRNEADerivatives(model, data, q, zero, v);
            jac.leftCols(model.nv) =
                .5 * v.transpose() * (data.dtau_dq - gravity_q);
            ilqr::Matrix M = data.M;
            detail::symmetric(M);
            jac.rightCols(model.nv) = v.transpose() * M;
        }
        return;
    }
    if (quantity == Q::PotentialEnergy)
    {
        jac = pinocchio::computeGeneralizedGravity(model, data, q).transpose();
        return;
    }
    Eigen::Matrix<double, 6, Eigen::Dynamic>
        momentum_q =
            Eigen::Matrix<double, 6, Eigen::Dynamic>::Zero(6, model.nv),
        rate_q = momentum_q, rate_v = momentum_q, A = momentum_q;
    if (quantity == Q::CentroidalMap)
    {
        ilqr::Vector basis = zero;
        for (int k = 0; k < model.nv; ++k)
        {
            basis[k] = 1;
            pinocchio::computeCentroidalDynamicsDerivatives(
                model, data, q, basis, zero, momentum_q, rate_q, rate_v, A);
            jac.middleRows(6 * k, 6) = momentum_q;
            basis[k] = 0;
        }
        return;
    }
    const ilqr::Vector v = quantity == Q::ComPosition
        ? zero
        : ilqr::Vector(z.segment(model.nq, model.nv));
    const ilqr::Vector a =
        (quantity == Q::CenterOfMass || quantity == Q::ComAcceleration ||
         quantity == Q::CentroidalDynamics)
        ? ilqr::Vector(z.tail(model.nv))
        : zero;
    pinocchio::computeCentroidalDynamicsDerivatives(
        model, data, q, v, a, momentum_q, rate_q, rate_v, A);
    if (quantity == Q::CentroidalMomentum || quantity == Q::CentroidalDynamics)
    {
        jac.block(0, 0, 6, model.nv) = momentum_q;
        jac.block(0, model.nv, 6, model.nv) = A;
        if (quantity == Q::CentroidalDynamics)
        {
            jac.block(6, 0, 6, model.nv) = rate_q;
            jac.block(6, model.nv, 6, model.nv) = rate_v;
            jac.block(6, 2 * model.nv, 6, model.nv) = A;
        }
        return;
    }
    const double mass = pinocchio::computeTotalMass(model);
    if (mass <= 0)
    {
        throw ilqr::EvaluationError(
            "center of mass requires positive robot mass");
    }
    if (quantity == Q::ComPosition)
    {
        jac = A.topRows(3) / mass;
    }
    else if (quantity == Q::ComVelocity)
    {
        jac.leftCols(model.nv) = momentum_q.topRows(3) / mass;
        jac.rightCols(model.nv) = A.topRows(3) / mass;
    }
    else if (quantity == Q::ComAcceleration)
    {
        jac.leftCols(model.nv) = rate_q.topRows(3) / mass;
        jac.middleCols(model.nv, model.nv) = rate_v.topRows(3) / mass;
        jac.rightCols(model.nv) = A.topRows(3) / mass;
    }
    else if (quantity == Q::CenterOfMass)
    {
        jac.block(0, 0, 3, model.nv) = A.topRows(3) / mass;
        jac.block(3, 0, 3, model.nv) = momentum_q.topRows(3) / mass;
        jac.block(3, model.nv, 3, model.nv) = A.topRows(3) / mass;
        jac.block(6, 0, 3, model.nv) = rate_q.topRows(3) / mass;
        jac.block(6, model.nv, 3, model.nv) = rate_v.topRows(3) / mass;
        jac.block(6, 2 * model.nv, 3, model.nv) = A.topRows(3) / mass;
    }
    else
    {
        throw std::logic_error("unknown native dynamics derivative quantity");
    }
}
} // namespace horizon::kinematics
