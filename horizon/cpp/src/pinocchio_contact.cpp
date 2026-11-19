#include "pinocchio_robot_internal.h"
#include <Eigen/LU>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <set>

namespace horizon::kinematics
{
namespace
{
int contactSize(const std::vector<ContactSpec> &contacts)
{
    int result = 0;
    std::set<std::string> names;
    for (const auto &contact : contacts)
    {
        if (contact.dimension != 3 && contact.dimension != 6)
        {
            throw std::invalid_argument("contact dimension must be 3 or 6");
        }
        if (!names.insert(contact.frame).second)
        {
            throw std::invalid_argument("duplicate contact frame: " +
                                        contact.frame);
        }
        result += contact.dimension;
    }
    return result;
}
int torqueSize(const std::shared_ptr<const pinocchio::Model> &model,
               const ilqr::Matrix &actuation)
{
    if (!model)
    {
        throw std::invalid_argument("null Pinocchio model");
    }
    if (actuation.rows() == 0 && actuation.cols() == 0)
    {
        return model->nv;
    }
    if (actuation.rows() != model->nv || !actuation.allFinite())
    {
        throw std::invalid_argument(
            "actuation must be a finite nv by ntau matrix");
    }
    return actuation.cols();
}
std::shared_ptr<const ilqr::StateSpace>
contactInput(const std::shared_ptr<const pinocchio::Model> &model,
             const std::shared_ptr<const PinocchioConfiguration> &space,
             const std::vector<ContactSpec> &contacts,
             const ilqr::Matrix &actuation, bool inverse)
{
    const int ntau = torqueSize(model, actuation);
    return detail::product(model, space,
                           inverse ? 2 * model->nv + contactSize(contacts)
                                   : model->nv + ntau);
}
int contactOutput(const std::shared_ptr<const pinocchio::Model> &model,
                  const std::vector<ContactSpec> &contacts, bool inverse)
{
    if (!model)
    {
        throw std::invalid_argument("null Pinocchio model");
    }
    return model->nv + (inverse ? 0 : contactSize(contacts));
}

} // namespace
struct PinocchioContactDynamics::Impl : detail::PinocchioRobotImplBase
{
    std::vector<ContactSpec> contacts;
    std::vector<pinocchio::FrameIndex> frames;
    ilqr::Matrix actuation, J, K;
    ilqr::Vector gamma, solution;
    Eigen::FullPivLU<ilqr::Matrix> factor;
    bool inverse;
    int nc;
    Impl(std::shared_ptr<const pinocchio::Model> model,
         std::shared_ptr<const PinocchioConfiguration> space,
         std::vector<ContactSpec> contacts, ilqr::Matrix actuation,
         bool inverse)
        : PinocchioRobotImplBase(std::move(model), std::move(space)),
          contacts(std::move(contacts)), actuation(std::move(actuation)),
          inverse(inverse), nc(contactSize(this->contacts))
    {
        if (this->actuation.rows() == 0 && this->actuation.cols() == 0)
        {
            this->actuation =
                ilqr::Matrix::Identity(this->model.nv, this->model.nv);
        }
        if (inverse &&
            (this->actuation.rows() != this->actuation.cols() ||
             !this->actuation.isIdentity()))
        {
            throw std::invalid_argument(
                "inverse contact dynamics returns generalized torques; "
                "actuation is forward-only");
        }
        for (const auto &contact : this->contacts)
        {
            frames.push_back(detail::frameId(this->model, contact.frame));
        }
        J.resize(nc, this->model.nv);
        gamma.resize(nc);
        K.resize(this->model.nv + nc, this->model.nv + nc);
        solution.resize(K.rows());
    }
    void constraints(ilqr::Input v)
    {
        pinocchio::forwardKinematics(model, data, q, v, zero);
        pinocchio::computeJointJacobians(model, data);
        pinocchio::updateFramePlacements(model, data);
        int offset = 0;
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            ilqr::Matrix frame_jac = ilqr::Matrix::Zero(6, model.nv);
            pinocchio::getFrameJacobian(model, data, frames[i],
                                        pinocchio::LOCAL_WORLD_ALIGNED,
                                        frame_jac);
            const auto acceleration =
                pinocchio::getFrameClassicalAcceleration(
                    model, data, frames[i], pinocchio::LOCAL_WORLD_ALIGNED)
                    .toVector()
                    .eval();
            const int n = contacts[i].dimension;
            J.middleRows(offset, n) = frame_jac.topRows(n);
            gamma.segment(offset, n) = acceleration.head(n);
            offset += n;
        }
    }
    void solve(ilqr::Input v, ilqr::Input tau)
    {
        constraints(v);
        ilqr::Matrix M = pinocchio::crba(model, data, q);
        detail::symmetric(M);
        const ilqr::Vector h = pinocchio::nonLinearEffects(model, data, q, v);
        K.setZero();
        K.topLeftCorner(model.nv, model.nv) = M;
        K.topRightCorner(model.nv, nc) = -J.transpose();
        K.bottomLeftCorner(nc, model.nv) = J;
        factor.compute(K);
        factor.setThreshold(1e-12);
        if (!factor.isInvertible())
        {
            throw ilqr::EvaluationError(
                "singular contact dynamics: contact rows must be independent "
                "and mass matrix invertible");
        }
        ilqr::Vector rhs(K.rows());
        rhs.head(model.nv) = actuation * tau - h;
        rhs.tail(nc) = -gamma;
        solution = factor.solve(rhs);
        if (!solution.allFinite() ||
            (K * solution - rhs).norm() > 1e-8 * (1 + rhs.norm()))
        {
            throw ilqr::EvaluationError("contact dynamics KKT solve failed");
        }
    }
    ilqr::Matrix generalizedForceDerivative(ilqr::Input force)
    {
        ilqr::Matrix derivative = ilqr::Matrix::Zero(model.nv, model.nv);
        if (nc == 0)
        {
            return derivative;
        }
        ilqr::Vector basis = zero;
        for (int k = 0; k < model.nv; ++k)
        {
            basis[k] = 1;
            pinocchio::computeForwardKinematicsDerivatives(model, data, q,
                                                           basis, zero);
            pinocchio::updateFramePlacements(model, data);
            int offset = 0;
            for (std::size_t i = 0; i < contacts.size(); ++i)
            {
                ilqr::Matrix dq = ilqr::Matrix::Zero(6, model.nv), dv = dq;
                detail::frameVelocityDerivatives(model, data, frames[i],
                                                 pinocchio::LOCAL_WORLD_ALIGNED,
                                                 dq, dv);
                const int n = contacts[i].dimension;
                derivative.row(k).noalias() +=
                    force.segment(offset, n).transpose() * dq.topRows(n);
                offset += n;
            }
            basis[k] = 0;
        }
        return derivative;
    }
};
PinocchioContactDynamics::PinocchioContactDynamics(
    std::shared_ptr<const pinocchio::Model> model,
    std::shared_ptr<const PinocchioConfiguration> space,
    std::vector<ContactSpec> contacts, ilqr::Matrix actuation, bool inverse)
    : TangentFunction(contactInput(model, space, contacts, actuation, inverse),
                      contactOutput(model, contacts, inverse)),
      _impl(std::make_unique<Impl>(std::move(model), std::move(space),
                                   std::move(contacts), std::move(actuation),
                                   inverse))
{
}
PinocchioContactDynamics::~PinocchioContactDynamics() = default;
std::unique_ptr<ilqr::Function> PinocchioContactDynamics::clone() const
{
    return std::make_unique<PinocchioContactDynamics>(
        _impl->source, _impl->space, _impl->contacts, _impl->actuation,
        _impl->inverse);
}
void PinocchioContactDynamics::evaluateImpl(ilqr::Input z,
                                            ilqr::VectorOutput out)
{
    auto &w = *_impl;
    w.prepare(z);
    const auto v = z.segment(w.model.nq, w.model.nv);
    if (w.inverse)
    {
        w.constraints(v);
        out = pinocchio::rnea(w.model, w.data, w.q, v,
                              z.segment(w.model.nq + w.model.nv, w.model.nv)) -
            w.J.transpose() * z.tail(w.nc);
    }
    else
    {
        w.solve(v, z.tail(w.actuation.cols()));
        out = w.solution;
    }
}
void PinocchioContactDynamics::tangentJacobian(ilqr::Input z,
                                               ilqr::MatrixOutput jac)
{
    auto &w = *_impl;
    w.prepare(z);
    const int nv = w.model.nv;
    const ilqr::Vector v = z.segment(w.model.nq, nv);
    ilqr::Vector a(nv), force(w.nc);
    if (w.inverse)
    {
        w.constraints(v);
        a = z.segment(w.model.nq + nv, nv);
        force = z.tail(w.nc);
    }
    else
    {
        w.solve(v, z.tail(w.actuation.cols()));
        a = w.solution.head(nv);
        force = w.solution.tail(w.nc);
    }
    pinocchio::computeRNEADerivatives(w.model, w.data, w.q, v, a);
    ilqr::Matrix torque_q = w.data.dtau_dq, torque_v = w.data.dtau_dv,
                 M = w.data.M;
    detail::symmetric(M);
    torque_q -= w.generalizedForceDerivative(force);
    jac.setZero();
    if (w.inverse)
    {
        jac.leftCols(nv) = torque_q;
        jac.middleCols(nv, nv) = torque_v;
        jac.middleCols(2 * nv, nv) = M;
        jac.rightCols(w.nc) = -w.J.transpose();
        return;
    }
    ilqr::Matrix rhs =
        ilqr::Matrix::Zero(nv + w.nc, 2 * nv + w.actuation.cols());
    rhs.topLeftCorner(nv, nv) = -torque_q;
    rhs.block(0, nv, nv, nv) = -torque_v;
    rhs.topRightCorner(nv, w.actuation.cols()) = w.actuation;
    pinocchio::computeForwardKinematicsDerivatives(w.model, w.data, w.q, v, a);
    pinocchio::updateFramePlacements(w.model, w.data);
    int offset = 0;
    for (std::size_t i = 0; i < w.contacts.size(); ++i)
    {
        ilqr::Matrix dq, dv, da;
        detail::classicalDerivatives(w.model, w.data, w.frames[i],
                                     pinocchio::LOCAL_WORLD_ALIGNED, dq, dv, da,
                                     true);
        const int n = w.contacts[i].dimension;
        rhs.block(nv + offset, 0, n, nv) = -dq.topRows(n);
        rhs.block(nv + offset, nv, n, nv) = -dv.topRows(n);
        offset += n;
    }
    jac = w.factor.solve(rhs);
}
} // namespace horizon::kinematics
