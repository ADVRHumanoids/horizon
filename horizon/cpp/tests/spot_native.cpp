#include "spot_contact_projection.h"
#include "spot_model.h"
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace spot_test
{
namespace
{
std::shared_ptr<const StateSpace>
inputSpace(const std::shared_ptr<const Robot> &robot, Formulation f)
{
    return std::make_shared<ProductSpace>(
        std::vector<std::shared_ptr<const StateSpace>>{
            robot->configuration,
            std::make_shared<EuclideanSpace>(nv + controlSize(f))});
}

class Dynamics final : public TangentFunction
{
  public:
    Dynamics(std::shared_ptr<const Robot> r, Formulation f, double dt,
             bool project_contacts)
        : TangentFunction(inputSpace(r, f), nx), robot(std::move(r)), form(f),
          dt(dt), af(nv + nf), delta(nv), prediction(nq),
          Da(nv, 2 * nv + controlSize(f)), Ddelta(Da), config(Da), A(nv, nv),
          B(nv, nv), R(nq, nv), dy(nv + nf, 2 * nv + 12),
          full(nv + 3 * nf, 3 * nv + nf), raw(nx), projected(nx),
          rawJac(2 * nv, 2 * nv + controlSize(f)), projectionJac(nx, 2 * nv)
    {
        if (f == Formulation::Torque)
        {
            engine = std::make_unique<Engine>(robot);
        }
        if (project_contacts)
        {
            projection = std::make_unique<ContactProjection>(robot);
        }
    }
    std::unique_ptr<Function> clone() const override
    {
        return std::make_unique<Dynamics>(robot, form, dt, bool(projection));
    }

  private:
    void acceleration(Input z)
    {
        if (form == Formulation::Torque)
        {
            engine->forward(z.head(nq), z.segment(nq, nv), z.tail(12), af);
        }
        else
        {
            af = z.tail(nv + nf);
        }
        delta = dt * z.segment(nq, nv) + .5 * dt * dt * af.head(nv);
    }
    void evaluateImpl(Input z, VectorOutput out) override
    {
        acceleration(z);
        robot->configuration->integrate(z.head(nq), delta, out.head(nq));
        out.tail(nv) = z.segment(nq, nv) + dt * af.head(nv);
        if (projection)
        {
            raw = out;
            projection->evaluate(raw, out);
        }
    }
    void tangentJacobian(Input z, MatrixOutput jac) override
    {
        Da.setZero();
        if (form == Formulation::Torque)
        {
            engine->forwardJacobian(z.head(nq), z.segment(nq, nv), z.tail(12),
                                    af, dy, full);
            Da = dy.topRows(nv);
        }
        else
        {
            af = z.tail(nv + nf);
            Da.block(0, 2 * nv, nv, nv).setIdentity();
        }
        delta = dt * z.segment(nq, nv) + .5 * dt * dt * af.head(nv);
        Ddelta = .5 * dt * dt * Da;
        Ddelta.block(0, nv, nv, nv).diagonal().array() += dt;
        A.setZero();
        B.setZero();
        pin::dIntegrate(*robot->model, z.head(nq), delta, A, pin::ARG0);
        pin::dIntegrate(*robot->model, z.head(nq), delta, B, pin::ARG1);
        robot->configuration->integrate(z.head(nq), delta, prediction);
        robot->configuration->retractionJacobian(prediction, R);
        if (form == Formulation::AccelerationForce)
        {
            config.setZero();
            config.leftCols(nv) = A;
            config.middleCols(nv, nv) = dt * B;
            config.middleCols(2 * nv, nv) = .5 * dt * dt * B;
        }
        else
        {
            config.noalias() = B * Ddelta;
            config.leftCols(nv) += A;
        }
        jac.topRows(nq).noalias() = R * config;
        jac.bottomRows(nv) = dt * Da;
        jac.block(nq, nv, nv, nv).diagonal().array() += 1;
        if (projection)
        {
            raw << prediction, z.segment(nq, nv) + dt * af.head(nv);
            rawJac.topRows(nv) = config;
            rawJac.bottomRows(nv) = jac.bottomRows(nv);
            projection->valueAndJacobian(raw, projected, projectionJac);
            jac.noalias() = projectionJac.leftCols(6) * rawJac.topRows(6);
            jac.noalias() +=
                projectionJac.middleCols(nv, 6) * rawJac.middleRows(nv, 6);
        }
    }
    void valueAndTangentJacobian(Input z, VectorOutput value,
                                 MatrixOutput jac) override
    {
        tangentJacobian(z, jac);
        value.head(nq) = prediction;
        value.tail(nv) = z.segment(nq, nv) + dt * af.head(nv);
        if (projection)
        {
            value = projected;
        }
    }
    std::shared_ptr<const Robot> robot;
    std::unique_ptr<Engine> engine;
    Formulation form;
    double dt;
    Vector af, delta, prediction;
    Matrix Da, Ddelta, config, A, B, R, dy, full;
    Vector raw, projected;
    Matrix rawJac, projectionJac;
    std::unique_ptr<ContactProjection> projection;
};

class Stage final : public TangentFunction
{
  public:
    Stage(std::shared_ptr<const Robot> r, Formulation f)
        : TangentFunction(inputSpace(r, f), 84), engine(std::move(r)), form(f),
          af(nv + nf), terms(nv + 3 * nf), full(nv + 3 * nf, 3 * nv + nf),
          map(3 * nv + nf, 2 * nv + controlSize(f)), dy(nv + nf, 2 * nv + 12),
          total(nv + 3 * nf, 2 * nv + controlSize(f))
    {
    }
    std::unique_ptr<Function> clone() const override
    {
        return std::make_unique<Stage>(engine.robot, form);
    }

  private:
    void controls(Input z)
    {
        if (form == Formulation::Torque)
        {
            engine.forward(z.head(nq), z.segment(nq, nv), z.tail(12), af);
        }
        else
        {
            af = z.tail(nv + nf);
        }
    }
    void evaluateImpl(Input z, VectorOutput out) override
    {
        controls(z);
        engine.evaluate(z.head(nq), z.segment(nq, nv), af.head(nv), af.tail(nf),
                        terms);
        pack(terms, out);
    }
    void pack(Input values, VectorOutput out)
    {
        out.head(6) = values.head(6);
        out.segment(6, nf) = values.segment(nv, nf);
        out.segment(18, nv) = af.head(nv);
        out.segment(36, nf) = af.tail(nf);
        out.segment(48, 12) = values.segment(6, 12);
        out.segment(60, nf) = values.segment(nv + nf, nf);
        out.segment(72, nf) = values.segment(nv + 2 * nf, nf);
    }
    void tangentJacobian(Input z, MatrixOutput jac) override
    {
        map.setZero();
        map.topLeftCorner(2 * nv, 2 * nv).setIdentity();
        if (form == Formulation::Torque)
        {
            engine.forwardJacobian(z.head(nq), z.segment(nq, nv), z.tail(12),
                                   af, dy, full);
            map.bottomRows(nv + nf) = dy;
        }
        else
        {
            af = z.tail(nv + nf);
            engine.linearize(z.head(nq), z.segment(nq, nv), af.head(nv),
                             af.tail(nf), full);
            map.bottomRightCorner(nv + nf, nv + nf).setIdentity();
        }
        if (form == Formulation::AccelerationForce)
        {
            total = full;
        }
        else
        {
            total.noalias() = full.rightCols(nv + nf) * dy;
            total.leftCols(2 * nv) += full.leftCols(2 * nv);
        }
        jac.topRows(6) = total.topRows(6);
        jac.middleRows(6, nf) = total.middleRows(nv, nf);
        jac.middleRows(18, nv) = map.middleRows(2 * nv, nv);
        jac.middleRows(36, nf) = map.bottomRows(nf);
        jac.middleRows(48, 12) = total.middleRows(6, 12);
        jac.middleRows(60, nf) = total.middleRows(nv + nf, nf);
        jac.middleRows(72, nf) = total.bottomRows(nf);
    }
    void valueAndTangentJacobian(Input z, VectorOutput value,
                                 MatrixOutput jac) override
    {
        tangentJacobian(z, jac);
        pack(engine.lastLinearizedTerms(), value);
    }
    Engine engine;
    Formulation form;
    Vector af, terms;
    Matrix full, map, dy, total;
};
} // namespace
std::shared_ptr<const Function> makeDynamics(std::shared_ptr<const Robot> r,
                                             Formulation f, double dt,
                                             bool project_contacts)
{
    return std::make_shared<Dynamics>(std::move(r), f, dt, project_contacts);
}
std::shared_ptr<const Function> makeStage(std::shared_ptr<const Robot> r,
                                          Formulation f)
{
    return std::make_shared<Stage>(std::move(r), f);
}
} // namespace spot_test
