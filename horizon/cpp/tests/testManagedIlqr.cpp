#include "ilqr/casadi_function.h"
#include "ilqr/ilqr_impl.h"
#include <Eigen/Geometry>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <unistd.h>

using namespace horizon::ilqr;
using horizon::IterativeLQR;

namespace
{
class AmbientOnlyFunction final : public Function
{
  public:
    explicit AmbientOnlyFunction(std::unique_ptr<Function> source)
        : Function(source->inputSize(), source->outputSize()),
          _source(std::move(source))
    {
    }
    std::unique_ptr<Function> clone() const override
    {
        return std::make_unique<AmbientOnlyFunction>(_source->clone());
    }
    bool hasHessian() const override
    {
        return _source->hasHessian();
    }
    bool hasJacobianProduct() const override
    {
        return _source->hasJacobianProduct();
    }
    void setEvaluationContext(
        const std::shared_ptr<EvaluationContext> &context) override
    {
        _source->setEvaluationContext(context);
    }

  private:
    void evaluateImpl(Input x, VectorOutput value) override
    {
        _source->evaluate(x, value);
    }
    void jacobianImpl(Input x, MatrixOutput jac) override
    {
        _source->jacobian(x, jac);
    }
    void valueAndJacobianImpl(Input x, VectorOutput value,
                              MatrixOutput jac) override
    {
        _source->valueAndJacobian(x, value, jac);
    }
    void weightedHessianImpl(Input x, Input weights, MatrixOutput hess) override
    {
        _source->weightedHessian(x, weights, hess);
    }
    void valueAndJacobianProductImpl(Input x, MatrixInput directions,
                                     VectorOutput value,
                                     MatrixOutput product) override
    {
        _source->valueAndJacobianProduct(x, directions, value, product);
    }
    std::unique_ptr<Function> _source;
};
} // namespace

namespace horizon
{
struct IterativeLQRTestAccess
{
    using Dynamics = IterativeLQR::Dynamics;
    using FunctionWorkspace = IterativeLQR::FunctionWorkspace;
    using ManagedCost = IterativeLQR::ManagedCostEntity;
    using Constraint = IterativeLQR::ConstraintEntity;
    using CasadiCost = IterativeLQR::IntermediateCostEntity;
    using Cost = IterativeLQR::IntermediateCost;
    using Constraints = IterativeLQR::Constraint;
    static Dynamics &dynamics(IterativeLQR &solver)
    {
        return solver._dyn.at(0);
    }
    static std::shared_ptr<ManagedCost> cost(IterativeLQR &solver,
                                             const std::string &name)
    {
        return std::dynamic_pointer_cast<ManagedCost>(
            solver._cost_map.at(name));
    }
    static Constraint::Ptr constraint(IterativeLQR &solver,
                                      const std::string &name)
    {
        return solver._constr_map.at(name);
    }
    static void useAmbientCompatibility(IterativeLQR &solver)
    {
        for (auto &d : solver._dyn)
        {
            if (d.native)
            {
                d.native->local = casadi::Function();
                d.native->function = std::make_unique<AmbientOnlyFunction>(
                    std::move(d.native->function));
            }
        }
        for (auto &cost : solver._cost)
        {
            for (auto &item : cost.items)
            {
                if (auto managed = std::dynamic_pointer_cast<ManagedCost>(item))
                {
                    managed->workspace->local = casadi::Function();
                    managed->workspace->function =
                        std::make_unique<AmbientOnlyFunction>(
                            std::move(managed->workspace->function));
                }
            }
        }
        for (auto &constraint : solver._constraint)
        {
            for (auto &item : constraint.items)
            {
                if (item->native)
                {
                    item->native->local = casadi::Function();
                    item->native->function =
                        std::make_unique<AmbientOnlyFunction>(
                            std::move(item->native->function));
                }
            }
        }
    }
    static void sweep(IterativeLQR &solver)
    {
        solver.linearize_quadratize();
    }
    static void compareSweeps(IterativeLQR &a, IterativeLQR &b)
    {
        ASSERT_EQ(a._dyn.size(), b._dyn.size());
        for (std::size_t k = 0; k < a._dyn.size(); ++k)
        {
            EXPECT_LT((a._dyn[k].A() - b._dyn[k].A()).norm(), 1e-10);
            EXPECT_LT((a._dyn[k].B() - b._dyn[k].B()).norm(), 1e-10);
            EXPECT_LT((a._dyn[k].d - b._dyn[k].d).norm(), 1e-10);
        }
        ASSERT_EQ(a._cost.size(), b._cost.size());
        for (std::size_t k = 0; k < a._cost.size(); ++k)
        {
            EXPECT_LT((a._cost[k].Q() - b._cost[k].Q()).norm(), 1e-10);
            EXPECT_LT((a._cost[k].P() - b._cost[k].P()).norm(), 1e-10);
            EXPECT_LT((a._cost[k].R() - b._cost[k].R()).norm(), 1e-10);
            EXPECT_LT((a._cost[k].q() - b._cost[k].q()).norm(), 1e-10);
            EXPECT_LT((a._cost[k].r() - b._cost[k].r()).norm(), 1e-10);
            EXPECT_LT((a._constraint[k].C() - b._constraint[k].C()).norm(),
                      1e-10);
            EXPECT_LT((a._constraint[k].D() - b._constraint[k].D()).norm(),
                      1e-10);
            EXPECT_LT((a._constraint[k].h() - b._constraint[k].h()).norm(),
                      1e-10);
        }
    }
    static void compareSteps(IterativeLQR &a, IterativeLQR &b)
    {
        ASSERT_EQ(a._bp_res.size(), b._bp_res.size());
        for (std::size_t k = 0; k < a._bp_res.size(); ++k)
        {
            const auto &left = a._bp_res[k], &right = b._bp_res[k];
            ASSERT_EQ(left.Lu.rows(), right.Lu.rows());
            ASSERT_EQ(left.Lu.cols(), right.Lu.cols());
            ASSERT_EQ(left.lu.size(), right.lu.size());
            ASSERT_EQ(left.glam.size(), right.glam.size());
            ASSERT_EQ(left.dx_lam.size(), right.dx_lam.size());
            EXPECT_LT((left.Lu - right.Lu).norm(), 1e-8);
            EXPECT_LT((left.lu - right.lu).norm(), 1e-8);
            EXPECT_LT((left.glam - right.glam).norm(), 1e-8);
            EXPECT_LT((left.dx_lam - right.dx_lam).norm(), 1e-8);
        }
        const auto &left = a.getIterationHistory();
        const auto &right = b.getIterationHistory();
        ASSERT_EQ(left.size(), right.size());
        for (std::size_t k = 0; k < left.size(); ++k)
        {
            EXPECT_EQ(left[k].accepted, right[k].accepted);
            EXPECT_NEAR(left[k].alpha, right[k].alpha, 1e-10);
            EXPECT_NEAR(left[k].cost, right[k].cost, 1e-10);
            EXPECT_NEAR(left[k].constraint_violation,
                        right[k].constraint_violation, 1e-10);
            EXPECT_NEAR(left[k].defect_norm, right[k].defect_norm, 1e-10);
        }
    }
};
} // namespace horizon

namespace
{
constexpr double inf = std::numeric_limits<double>::infinity();

class MultiplicativeDynamics final : public TangentDynamics
{
  public:
    explicit MultiplicativeDynamics(std::shared_ptr<const StateSpace> space)
        : TangentDynamics(space, 2), _space(std::move(space))
    {
    }
    std::unique_ptr<Function> clone() const override
    {
        return std::make_unique<MultiplicativeDynamics>(_space);
    }

  private:
    void evaluateImpl(Input z, VectorOutput out) override
    {
        out(0) = z(0) * std::exp(z(1) + z(2));
    }
    void tangentOutputLinearization(Input z, VectorOutput out,
                                    MatrixOutput jac) override
    {
        evaluateImpl(z, out);
        jac.setOnes();
    }
    std::shared_ptr<const StateSpace> _space;
};

class QuaternionScalarCost final : public TangentFunction
{
  public:
    explicit QuaternionScalarCost(std::shared_ptr<const StateSpace> input,
                                  std::shared_ptr<int> direct_calls = {})
        : TangentFunction(input, 1), _input(std::move(input)),
          _directCalls(std::move(direct_calls))
    {
    }
    std::unique_ptr<Function> clone() const override
    {
        return std::make_unique<QuaternionScalarCost>(_input, _directCalls);
    }
    bool hasHessian() const override
    {
        return true;
    }
    void weightedTangentHessian(Input x, Input w, MatrixOutput hess) override
    {
        if (_directCalls)
        {
            ++*_directCalls;
        }
        TangentFunction::weightedTangentHessian(x, w, hess);
    }

  private:
    void evaluateImpl(Input z, VectorOutput out) override
    {
        out(0) = z(3) * (1 + z(4) * z(5)) + z(4) * z(4);
    }
    void tangentJacobian(Input z, MatrixOutput jac) override
    {
        jac.leftCols(3) = -.5 * (1 + z(4) * z(5)) * z.head(3).transpose();
        jac(0, 3) = z(5) * z(3) + 2 * z(4);
        jac(0, 4) = z(4) * z(3);
    }
    void tangentWeightedHessian(Input z, Input w, MatrixOutput hess) override
    {
        hess.setZero();
        hess.topLeftCorner(3, 3).diagonal().setConstant(-.25 * z(3) *
                                                        (1 + z(4) * z(5)));
        hess.block(0, 3, 3, 1) = -.5 * z(5) * z.head(3);
        hess.block(3, 0, 1, 3) = hess.block(0, 3, 3, 1).transpose();
        hess.block(0, 4, 3, 1) = -.5 * z(4) * z.head(3);
        hess.block(4, 0, 1, 3) = hess.block(0, 4, 3, 1).transpose();
        hess(3, 3) = 2;
        hess(3, 4) = hess(4, 3) = z(3);
        hess *= w(0);
    }
    std::shared_ptr<const StateSpace> _input;
    std::shared_ptr<int> _directCalls;
};

void fixEndpoints(IterativeLQR &solver, const Vector &start, const Vector &goal,
                  int N)
{
    Matrix lb = Matrix::Constant(start.size(), N + 1, -inf), ub = -lb;
    lb.col(0) = ub.col(0) = start;
    lb.col(N) = ub.col(N) = goal;
    solver.setStateBounds(lb, ub);
}
} // namespace

TEST(ManagedIlqr, TangentOutputDynamicsSkipsAmbientConversions)
{
    int projections = 0;
    auto makeSpace = [&projections]
    {
        return std::make_shared<CallbackStateSpace>(
            Vector::Ones(1), 1,
            [](Input x, Input dx, VectorOutput y)
            {
                y(0) = x(0) * std::exp(dx(0));
            },
            [](Input x, Input y, VectorOutput d)
            {
                d(0) = std::log(y(0) / x(0));
            },
            [&projections](Input x, MatrixOutput jac)
            {
                ++projections;
                jac(0, 0) = x(0);
            },
            CallbackStateSpace::Hessian{},
            [](Input, Input, MatrixOutput from, MatrixOutput to)
            {
                from(0, 0) = -1;
                to(0, 0) = 1;
            });
    };
    auto space = makeSpace();
    auto dynamics = std::make_shared<MultiplicativeDynamics>(space);
    FunctionSpec spec{dynamics, {{"p", 1}}};
    using Access = horizon::IterativeLQRTestAccess;
    Access::Dynamics direct(1, 1), fallback(1, 1);
    direct.space = space;
    fallback.space = makeSpace();
    direct.native = std::make_shared<Access::FunctionWorkspace>(spec, 1, 1);
    fallback.native = std::make_shared<Access::FunctionWorkspace>(spec, 1, 1);
    direct.param = fallback.param =
        std::make_shared<std::map<std::string, Matrix>>();
    (*direct.param)["p"] = Matrix::Constant(1, 1, .2);
    Vector x = Vector::Constant(1, 2), u = Vector::Constant(1, -.3);
    Vector next = Vector::Constant(1, 1.3);
    projections = 0;
    direct.linearize(x, u, next, 0);
    EXPECT_EQ(projections, 0);
    EXPECT_EQ(direct.Rx.size(), 0);
    EXPECT_EQ(direct.projection.size(), 0);
    fallback.linearize(x, u, next, 0);
    EXPECT_GT(projections, 0);
    EXPECT_NEAR(direct.A()(0, 0), fallback.A()(0, 0), 1e-12);
    EXPECT_NEAR(direct.B()(0, 0), fallback.B()(0, 0), 1e-12);
    EXPECT_NEAR(direct.d(0), fallback.d(0), 1e-12);
    EXPECT_NEAR(direct.d(0), std::log(2 * std::exp(-.1) / 1.3), 1e-12);

    Matrix retraction(1, 1);
    space->retractionJacobian(x, retraction);
    direct.native->setInput(x, u, 0, direct.param);
    fallback.native->setInput(x, u, 0, fallback.param);
    projections = 0;
    direct.native->linearizeLocal(space.get(), retraction, 1);
    const int direct_retractions = projections;
    fallback.native->linearizeLocal(fallback.space.get(), retraction, 1);
    EXPECT_GT(projections - direct_retractions, direct_retractions);
    EXPECT_LT((direct.native->localJac - fallback.native->localJac).norm(),
              1e-12);
    EXPECT_LT((direct.native->value - fallback.native->value).norm(), 1e-12);

    Access::ManagedCost cost(spec, "residual", true, 1, 1);
    cost.param = direct.param;
    Matrix Q = Matrix::Zero(1, 1), R = Q, P = Q;
    Vector q = Vector::Zero(1), r = q;
    ASSERT_TRUE(
        cost.quadratizeLocal(x, u, 0, space.get(), retraction, Q, R, P, q, r));
    const double value = 2 * std::exp(-.1);
    EXPECT_NEAR(Q(0, 0), value * value, 1e-12);
    EXPECT_NEAR(R(0, 0), value * value, 1e-12);
    EXPECT_NEAR(P(0, 0), value * value, 1e-12);
    EXPECT_NEAR(q(0), value * value, 1e-12);
    EXPECT_NEAR(r(0), value * value, 1e-12);

    Access::Constraint constraint;
    constraint.native = std::make_shared<Access::FunctionWorkspace>(spec, 1, 1);
    constraint.param = direct.param;
    constraint.setTargetValue(Vector::Constant(1, .4));
    constraint.linearizeLocal(x, u, 0, space.get(), retraction);
    EXPECT_NEAR(constraint.C()(0, 0), value, 1e-12);
    EXPECT_NEAR(constraint.D()(0, 0), value, 1e-12);
    EXPECT_NEAR(constraint.h()(0), value - .4, 1e-12);
}

TEST(ManagedIlqr, TangentOutputDynamicsMatchesFallbackShooting)
{
    auto makeSpace = []
    {
        return std::make_shared<CallbackStateSpace>(
            Vector::Ones(1), 1,
            [](Input x, Input dx, VectorOutput y)
            {
                y(0) = x(0) * std::exp(dx(0));
            },
            [](Input x, Input y, VectorOutput d)
            {
                d(0) = std::log(y(0) / x(0));
            },
            [](Input x, MatrixOutput jac)
            {
                jac(0, 0) = x(0);
            },
            CallbackStateSpace::Hessian{},
            [](Input, Input, MatrixOutput from, MatrixOutput to)
            {
                from(0, 0) = -1;
                to(0, 0) = 1;
            });
    };
    auto dynamics_space = makeSpace();
    FunctionManager fm;
    fm.add("dyn", std::make_shared<MultiplicativeDynamics>(dynamics_space),
           {{"p", 1}});
    fm.add("effort",
           std::make_shared<CallbackFunction>(
               3, 1,
               [](Input z, VectorOutput y)
               {
                   y(0) = z(1);
               },
               [](Input, MatrixOutput j)
               {
                   j << 0, 1, 0;
               }),
           {{"p", 1}});
    for (int threads : {0, 2})
    {
        IterativeLQR direct(fm, "dyn", 4, {{"ilqr.n_threads", threads}},
                            dynamics_space);
        IterativeLQR fallback(fm, "dyn", 4, {{"ilqr.n_threads", threads}},
                              makeSpace());
        for (auto *solver : {&direct, &fallback})
        {
            solver->setParameterValue("p", Matrix::Constant(1, 5, .05));
            solver->setResidual({0, 1, 2, 3}, fm, "effort");
            fixEndpoints(*solver, Vector::Ones(1),
                         Vector::Constant(1, std::exp(.8)), 4);
            solver->setStateInitialGuess(Matrix::Constant(1, 5, 1.2));
            ASSERT_TRUE(solver->solve(30));
        }
        EXPECT_LT((direct.getStateTrajectory() - fallback.getStateTrajectory())
                      .norm(),
                  1e-10);
        EXPECT_LT((direct.getInputTrajectory() - fallback.getInputTrajectory())
                      .norm(),
                  1e-10);
        horizon::IterativeLQRTestAccess::compareSteps(direct, fallback);
    }
}

TEST(ManagedIlqr, CasadiLocalCostsAndConstraintsMatchAmbientFallback)
{
    using Access = horizon::IterativeLQRTestAccess;
    auto space = std::make_shared<QuaternionSpace>();
    auto x = casadi::MX::sym("x", 4), u = casadi::MX::sym("u"),
         p = casadi::MX::sym("p");
    auto residual = casadi::MX::vertcat({x(0) + p * u, x(3) - .2 * u});
    auto exact = x(0) + x(3) * x(3) + p * u * x(1) + u * u;
    casadi::Function scalar("scalar_local_test", {x, u, p}, {exact},
                            {"x", "u", "p"}, {"l"});
    casadi::Function res("residual_local_test", {x, u, p}, {residual},
                         {"x", "u", "p"}, {"r"});
    auto Jx = casadi::MX::jacobian(residual, x);
    auto Ju = casadi::MX::jacobian(residual, u);
    auto mul = [](const casadi::MX &a, const casadi::MX &b)
    {
        return casadi::MX::mtimes(a, b);
    };
    casadi::Function cost("gn_cost_test", {x, u, p},
                          {.5 * casadi::MX::sumsqr(residual)}, {"x", "u", "p"},
                          {"f"});
    casadi::Function grad("gn_grad_test", {x, u, p},
                          {mul(Jx.T(), residual), mul(Ju.T(), residual)},
                          {"x", "u", "p"}, {"gx", "gu"});
    casadi::Function hess("gn_hess_test", {x, u, p},
                          {mul(Jx.T(), Jx), mul(Ju.T(), Ju), mul(Ju.T(), Jx)},
                          {"x", "u", "p"}, {"xx", "uu", "ux"});
    auto params = std::make_shared<std::map<std::string, Matrix>>();
    (*params)["p"] = Matrix::Constant(1, 1, .7);
    Vector state(4), identity(4), rotation(3),
        control = Vector::Constant(1, .3);
    space->neutral(identity);
    rotation << .4, -.2, .3;
    space->integrate(identity, rotation, state);
    for (bool gn : {false, true})
    {
        auto ambient = std::make_shared<Access::CasadiCost>();
        ambient->param = params;
        ambient->gauss_newton = gn;
        if (gn)
        {
            ambient->setCost(cost, grad, hess);
        }
        else
        {
            auto g = Access::CasadiCost::Gradient(scalar);
            ambient->setCost(scalar, g, Access::CasadiCost::Hessian(g));
        }
        auto local = std::make_shared<Access::CasadiCost>(*ambient);
        local->local =
            compileLocalDerivatives(gn ? res : scalar, *space,
                                    gn ? LocalDerivativeKind::GaussNewton
                                       : LocalDerivativeKind::ExactScalar);
        Access::Cost a(4, 1), b(4, 1);
        a.space = b.space = space;
        a.addCost(ambient);
        b.addCost(local);
        a.quadratize(state, control, 0);
        b.quadratize(state, control, 0);
        EXPECT_LT((a.Q() - b.Q()).norm(), 1e-10);
        EXPECT_LT((a.P() - b.P()).norm(), 1e-10);
        EXPECT_LT((a.R() - b.R()).norm(), 1e-10);
        EXPECT_LT((a.q() - b.q()).norm(), 1e-10);
        EXPECT_LT((a.r() - b.r()).norm(), 1e-10);
    }

    casadi::Function h("constraint_local_test", {x, u, p}, {residual},
                       {"x", "u", "p"}, {"h"});
    auto ambient = std::make_shared<Access::Constraint>();
    ambient->setConstraint(h);
    ambient->param = params;
    auto local = std::make_shared<Access::Constraint>(*ambient);
    local->local =
        compileLocalDerivatives(h, *space, LocalDerivativeKind::Jacobian);
    local->nativeC.setZero(2, 3);
    local->nativeD.setZero(2, 1);
    Access::Constraints a(4, 1), b(4, 1);
    a.space = b.space = space;
    a.addConstraint(ambient);
    b.addConstraint(local);
    a.linearize(state, control, 0);
    b.linearize(state, control, 0);
    EXPECT_LT((a.C() - b.C()).norm(), 1e-10);
    EXPECT_LT((a.D() - b.D()).norm(), 1e-10);
    EXPECT_LT((a.h() - b.h()).norm(), 1e-10);
}

TEST(ManagedIlqr, NativeExactCostUsesMatchingTangentHessian)
{
    using Access = horizon::IterativeLQRTestAccess;
    auto space = std::make_shared<QuaternionSpace>();
    auto input = std::make_shared<ProductSpace>(
        std::vector<std::shared_ptr<const StateSpace>>{
            space, std::make_shared<EuclideanSpace>(2)});
    FunctionSpec spec{std::make_shared<QuaternionScalarCost>(input),
                      {{"p", 1}}};
    auto native =
        std::make_shared<Access::ManagedCost>(spec, "native", false, 4, 1);
    auto params = std::make_shared<std::map<std::string, Matrix>>();
    (*params)["p"] = Matrix::Constant(1, 1, .7);
    native->param = params;
    auto x = casadi::MX::sym("x", 4), u = casadi::MX::sym("u"),
         p = casadi::MX::sym("p");
    casadi::Function scalar("native_exact_reference", {x, u, p},
                            {x(3) * (1 + u * p) + u * u}, {"x", "u", "p"},
                            {"l"});
    auto reference = std::make_shared<Access::CasadiCost>();
    reference->param = params;
    auto gradient = Access::CasadiCost::Gradient(scalar);
    reference->setCost(scalar, gradient, Access::CasadiCost::Hessian(gradient));
    Vector state(4), identity(4), rotation(3),
        control = Vector::Constant(1, .3);
    space->neutral(identity);
    rotation << .4, -.2, .3;
    space->integrate(identity, rotation, state);
    Matrix retract(4, 3), Q = Matrix::Zero(3, 3), R = Matrix::Zero(1, 1),
                          P = Matrix::Zero(1, 3);
    Vector q = Vector::Zero(3), r = Vector::Zero(1);
    space->retractionJacobian(state, retract);
    EXPECT_TRUE(native->quadratizeLocal(state, control, 0, space.get(), retract,
                                        Q, R, P, q, r));
    QuaternionSpace different_space;
    EXPECT_FALSE(native->quadratizeLocal(state, control, 0, &different_space,
                                         retract, Q, R, P, q, r));
    Access::Cost a(4, 1);
    a.space = space;
    a.addCost(reference);
    a.quadratize(state, control, 0);
    EXPECT_LT((a.Q() - Q).norm(), 1e-10);
    EXPECT_LT((a.P() - P).norm(), 1e-10);
    EXPECT_LT((a.R() - R).norm(), 1e-10);
    EXPECT_LT((a.q() - q).norm(), 1e-10);
    EXPECT_LT((a.r() - r).norm(), 1e-10);
}

TEST(ManagedIlqr, MixedGraphExactCostMatchesAmbientHessian)
{
    using Access = horizon::IterativeLQRTestAccess;
    auto space = std::make_shared<QuaternionSpace>();
    auto input = std::make_shared<ProductSpace>(
        std::vector<std::shared_ptr<const StateSpace>>{
            space, std::make_shared<EuclideanSpace>(2)});
    SymbolicFunctionManager manager;
    auto direct_calls = std::make_shared<int>(0);
    auto leaf = manager.addNative(
        "exact_leaf",
        std::make_shared<QuaternionScalarCost>(input, direct_calls));
    auto x = casadi::MX::sym("x", 4), u = casadi::MX::sym("u"),
         p = casadi::MX::sym("p");
    auto matched = leaf(casadi::MX::vertcat({x, u, p}));
    auto transformed = leaf(casadi::MX::vertcat({x, u + p, p}));
    manager.addCost("mixed_exact", {x, u, p},
                    sin(matched) + transformed * transformed + x(0) * u);
    auto native = std::make_shared<Access::ManagedCost>(
        manager.at("mixed_exact"), "mixed_exact", false, 4, 1);
    native->workspace->function->setEvaluationContext(
        std::make_shared<EvaluationContext>());
    auto params = std::make_shared<std::map<std::string, Matrix>>();
    (*params)["p"] = Matrix::Constant(1, 1, .7);
    native->param = params;
    auto f = x(3) * (1 + u * p) + u * u;
    auto g = x(3) * (1 + (u + p) * p) + (u + p) * (u + p);
    casadi::Function scalar("mixed_exact_reference", {x, u, p},
                            {sin(f) + g * g + x(0) * u}, {"x", "u", "p"},
                            {"l"});
    auto reference = std::make_shared<Access::CasadiCost>();
    reference->param = params;
    auto gradient = Access::CasadiCost::Gradient(scalar);
    reference->setCost(scalar, gradient, Access::CasadiCost::Hessian(gradient));
    Vector state(4), identity(4), rotation(3),
        control = Vector::Constant(1, .3);
    space->neutral(identity);
    rotation << .4, -.2, .3;
    space->integrate(identity, rotation, state);
    Access::Cost a(4, 1), b(4, 1);
    a.space = b.space = space;
    a.addCost(reference);
    b.addCost(native);
    a.quadratize(state, control, 0);
    b.quadratize(state, control, 0);
    EXPECT_EQ(*direct_calls, 1);
    EXPECT_LT((a.Q() - b.Q()).norm(), 1e-10);
    EXPECT_LT((a.P() - b.P()).norm(), 1e-10);
    EXPECT_LT((a.R() - b.R()).norm(), 1e-10);
    EXPECT_LT((a.q() - b.q()).norm(), 1e-10);
    EXPECT_LT((a.r() - b.r()).norm(), 1e-10);
}

TEST(ManagedIlqr, CasadiTangentDynamicsMatchesAmbientDefectTransport)
{
    using Access = horizon::IterativeLQRTestAccess;
    auto space = std::make_shared<QuaternionSpace>();
    auto x = casadi::MX::sym("x", 4), u = casadi::MX::sym("u"),
         p = casadi::MX::sym("p");
    auto raw = casadi::MX::vertcat(
        {x(0) + .1 * u, x(1) + .2 * p * u, x(2) - .1 * u, x(3)});
    auto prediction = raw / sqrt(casadi::MX::dot(raw, raw));
    casadi::Function f("local_dyn_test", {x, u, p}, {prediction},
                       {"x", "u", "p"}, {"f"});
    Access::Dynamics direct(4, 1), fallback(4, 1);
    direct.space = fallback.space = space;
    direct.setDynamics(f);
    fallback.setDynamics(f);
    direct.local =
        compileLocalDerivatives(f, *space, LocalDerivativeKind::Dynamics);
    auto params = std::make_shared<std::map<std::string, Matrix>>();
    (*params)["p"] = Matrix::Constant(1, 1, .7);
    direct.param = fallback.param = params;
    Vector state(4), identity(4), rotation(3), uval = Vector::Constant(1, .3);
    space->neutral(identity);
    rotation << .4, -.2, .3;
    space->integrate(identity, rotation, state);
    Vector predicted = direct.integrate(state, uval, 0), next(4);
    const double pi = std::acos(-1.0);
    for (double offset : {0., .3, -1.2, (pi - 1e-5) / std::sqrt(3.),
                          (pi + 1e-5) / std::sqrt(3.)})
    {
        space->integrate(predicted, Vector::Constant(3, offset), next);
        for (double sign : {1., -1.})
        {
            Vector signed_next = sign * next;
            direct.linearize(state, uval, signed_next, 0);
            fallback.linearize(state, uval, signed_next, 0);
            EXPECT_LT((direct.A() - fallback.A()).norm(), 1e-10);
            EXPECT_LT((direct.B() - fallback.B()).norm(), 1e-10);
            EXPECT_LT((direct.d - fallback.d).norm(), 1e-10);
        }
    }
    Vector invalid = 2 * state;
    EXPECT_THROW(direct.linearize(invalid, uval, next, 0),
                 std::invalid_argument);
    EXPECT_THROW(fallback.linearize(invalid, uval, next, 0),
                 std::invalid_argument);
}

TEST(ManagedIlqr, PureSymbolicGraphUsesLocalDerivatives)
{
    using Access = horizon::IterativeLQRTestAccess;
    auto space = std::make_shared<QuaternionSpace>();
    auto x = casadi::MX::sym("x", 4), u = casadi::MX::sym("u"),
         p = casadi::MX::sym("p");
    auto raw = casadi::MX::vertcat(
        {x(0) + .1 * u, x(1) + .2 * p * u, x(2) - .1 * u, x(3)});
    auto prediction = raw / sqrt(casadi::MX::dot(raw, raw));
    auto residual = casadi::MX::vertcat({x(0) + p * u, x(3) - .2 * u});
    auto scalar = x(0) + x(3) * x(3) + p * u * x(1) + u * u;
    SymbolicFunctionManager manager;
    manager.addExpression("pure_dynamics", {x, u, p}, prediction, false);
    manager.addResidual("pure_residual", {x, u, p}, residual);
    manager.addCost("pure_cost", {x, u, p}, scalar);
    manager.addExpression("pure_constraint", {x, u, p}, residual, false);
    IterativeLQR solver(manager, "pure_dynamics", 2, {}, space);
    solver.setParameterValue("p", Matrix::Constant(1, 3, .7));
    solver.setResidual({0}, manager, "pure_residual");
    solver.setCost({0}, manager, "pure_cost");
    solver.setConstraint({0}, manager, "pure_constraint");
    Vector state(4), identity(4), rotation(3),
        control = Vector::Constant(1, .3);
    space->neutral(identity);
    rotation << .4, -.2, .3;
    space->integrate(identity, rotation, state);

    auto &direct = Access::dynamics(solver);
    ASSERT_TRUE(direct.native->local.is_valid());
    auto fallback = direct;
    fallback.native =
        std::make_shared<Access::FunctionWorkspace>(*direct.native);
    fallback.native->local = casadi::Function();
    Vector predicted = direct.integrate(state, control, 0), next(4);
    space->integrate(predicted, Vector::Constant(3, .3), next);
    direct.linearize(state, control, next, 0);
    fallback.linearize(state, control, next, 0);
    EXPECT_LT((direct.A() - fallback.A()).norm(), 1e-10);
    EXPECT_LT((direct.B() - fallback.B()).norm(), 1e-10);
    EXPECT_LT((direct.d - fallback.d).norm(), 1e-10);
    Vector invalid = 2 * state;
    EXPECT_THROW(direct.linearize(invalid, control, next, 0),
                 std::invalid_argument);
    EXPECT_THROW(fallback.linearize(invalid, control, next, 0),
                 std::invalid_argument);

    for (const std::string name : {"pure_cost", "pure_residual"})
    {
        auto local = Access::cost(solver, name);
        ASSERT_NE(local, nullptr);
        ASSERT_TRUE(local->workspace->local.is_valid());
        auto ambient =
            std::dynamic_pointer_cast<Access::ManagedCost>(local->clone());
        ambient->workspace->local = casadi::Function();
        Access::Cost a(4, 1), b(4, 1);
        a.space = b.space = space;
        a.addCost(ambient);
        b.addCost(local);
        a.quadratize(state, control, 0);
        b.quadratize(state, control, 0);
        EXPECT_EQ(local->workspace->jac.size(), 0);
        EXPECT_EQ(local->workspace->hess.size(), 0);
        EXPECT_LT((a.Q() - b.Q()).norm(), 1e-10);
        EXPECT_LT((a.P() - b.P()).norm(), 1e-10);
        EXPECT_LT((a.R() - b.R()).norm(), 1e-10);
        EXPECT_LT((a.q() - b.q()).norm(), 1e-10);
        EXPECT_LT((a.r() - b.r()).norm(), 1e-10);
    }
    auto local = Access::constraint(solver, "pure_constraint");
    ASSERT_TRUE(local->native->local.is_valid());
    auto ambient = local->clone();
    ambient->native->local = casadi::Function();
    Access::Constraints a(4, 1), b(4, 1);
    a.space = b.space = space;
    a.addConstraint(ambient);
    b.addConstraint(local);
    a.linearize(state, control, 0);
    b.linearize(state, control, 0);
    EXPECT_LT((a.C() - b.C()).norm(), 1e-10);
    EXPECT_LT((a.D() - b.D()).norm(), 1e-10);
    EXPECT_LT((a.h() - b.h()).norm(), 1e-10);
}

TEST(ManagedIlqr, PureSymbolicCodegenMatchesInterpreted)
{
    using Access = horizon::IterativeLQRTestAccess;
    char directory[] = "/tmp/horizon-managed-codegen-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    struct Cleanup
    {
        const char *path;
        ~Cleanup()
        {
            std::filesystem::remove_all(path);
        }
    } cleanup{directory};
    auto space = std::make_shared<QuaternionSpace>();
    auto x = casadi::MX::sym("x", 4), u = casadi::MX::sym("u"),
         p = casadi::MX::sym("p");
    auto raw = casadi::MX::vertcat({x(0) + u * p, x(1), x(2), x(3)});
    SymbolicFunctionManager manager;
    manager.addExpression("generated_dynamics", {x, u, p},
                          raw / sqrt(casadi::MX::dot(raw, raw)), false);
    manager.addCost("generated_cost", {x, u, p}, x(0) + p * u * x(1) + u * u);
    manager.addExpression("generated_constraint", {x, u, p}, x(0) + p * u,
                          false);
    IterativeLQR interpreted(manager, "generated_dynamics", 1, {}, space);
    IterativeLQR generated(manager, "generated_dynamics", 1,
                           {{"ilqr.codegen_enabled", 1},
                            {"ilqr.codegen_workdir", std::string(directory)}},
                           space);
    for (auto *solver : {&interpreted, &generated})
    {
        solver->setParameterValue("p", Matrix::Constant(1, 2, .7));
        solver->setCost({0}, manager, "generated_cost");
        solver->setConstraint({0}, manager, "generated_constraint");
    }
    Vector state(4), control = Vector::Constant(1, .2), next(4);
    space->neutral(state);
    space->integrate(state, Vector::Constant(3, .15), next);
    auto &a = Access::dynamics(interpreted);
    auto &b = Access::dynamics(generated);
    a.linearize(state, control, next, 0);
    b.linearize(state, control, next, 0);
    EXPECT_LT((a.A() - b.A()).norm(), 1e-10);
    EXPECT_LT((a.B() - b.B()).norm(), 1e-10);
    EXPECT_LT((a.d - b.d).norm(), 1e-10);
    Access::Cost ca(4, 1), cb(4, 1);
    ca.space = cb.space = space;
    ca.addCost(Access::cost(interpreted, "generated_cost"));
    cb.addCost(Access::cost(generated, "generated_cost"));
    ca.quadratize(state, control, 0);
    cb.quadratize(state, control, 0);
    EXPECT_LT((ca.Q() - cb.Q()).norm(), 1e-10);
    EXPECT_LT((ca.P() - cb.P()).norm(), 1e-10);
    EXPECT_LT((ca.R() - cb.R()).norm(), 1e-10);
    EXPECT_LT((ca.q() - cb.q()).norm(), 1e-10);
    EXPECT_LT((ca.r() - cb.r()).norm(), 1e-10);
    Access::Constraints ha(4, 1), hb(4, 1);
    ha.space = hb.space = space;
    ha.addConstraint(Access::constraint(interpreted, "generated_constraint"));
    hb.addConstraint(Access::constraint(generated, "generated_constraint"));
    ha.linearize(state, control, 0);
    hb.linearize(state, control, 0);
    EXPECT_LT((ha.C() - hb.C()).norm(), 1e-10);
    EXPECT_LT((ha.D() - hb.D()).norm(), 1e-10);
    EXPECT_LT((ha.h() - hb.h()).norm(), 1e-10);
    EXPECT_FALSE(std::filesystem::is_empty(directory));
}

namespace
{
void compareDerivativeSweeps(bool benchmark)
{
    using Access = horizon::IterativeLQRTestAccess;
    constexpr int N = 16;
    auto space = std::make_shared<QuaternionSpace>();
    auto x = casadi::MX::sym("x", 4), u = casadi::MX::sym("u"),
         p = casadi::MX::sym("p");
    auto raw =
        casadi::MX::vertcat({x(0) + p * u, x(1) - .2 * u, x(2) + .1 * u, x(3)});
    SymbolicFunctionManager manager;
    manager.addExpression("sweep_dynamics", {x, u, p},
                          raw / sqrt(casadi::MX::dot(raw, raw)), false);
    manager.addResidual("sweep_residual", {x, u, p},
                        casadi::MX::vertcat({x(0) + p * u, x(3) - .2 * u}));
    manager.addCost("sweep_cost", {x, u, p},
                    x(0) + x(3) * x(3) + p * u * x(1) + u * u);
    manager.addExpression("sweep_constraint", {x, u, p}, x(1) + p * u, false);
    std::vector<int> nodes;
    for (int k = 0; k < N; ++k)
    {
        nodes.push_back(k);
    }
    Matrix states(4, N + 1), controls = Matrix::Constant(1, N, .03);
    Vector identity(4);
    space->neutral(identity);
    for (int k = 0; k <= N; ++k)
    {
        space->integrate(identity, Vector::Constant(3, .01 * k), states.col(k));
    }
    for (int threads : {0, 2})
    {
        IterativeLQR direct(manager, "sweep_dynamics", N,
                            {{"ilqr.n_threads", threads}}, space);
        IterativeLQR fallback(manager, "sweep_dynamics", N,
                              {{"ilqr.n_threads", threads}}, space);
        for (auto *solver : {&direct, &fallback})
        {
            solver->setParameterValue("p", Matrix::Constant(1, N + 1, .4));
            solver->setResidual(nodes, manager, "sweep_residual");
            solver->setCost(nodes, manager, "sweep_cost");
            solver->setConstraint(nodes, manager, "sweep_constraint");
            solver->setStateInitialGuess(states);
            solver->setInputInitialGuess(controls);
        }
        Access::useAmbientCompatibility(fallback);
        Access::sweep(direct);
        Access::sweep(fallback);
        Access::compareSweeps(direct, fallback);
        if (benchmark)
        {
            double checksum = 0;
            auto run = [&](IterativeLQR &solver)
            {
                const auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < 1000; ++i)
                {
                    Access::sweep(solver);
                    checksum += Access::dynamics(solver).A()(0, 0);
                }
                return std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - start)
                           .count() /
                    1000;
            };
            const char *selected_threads =
                std::getenv("HORIZON_BENCH_SWEEP_THREADS");
            if (selected_threads && std::stoi(selected_threads) != threads)
            {
                continue;
            }
            if (const char *mode = std::getenv("HORIZON_BENCH_SWEEP_PATH"))
            {
                const std::string path(mode);
                ASSERT_TRUE(path == "direct" || path == "fallback");
                std::cout << path << " sweep threads=" << threads
                          << " us/call: "
                          << run(path == "direct" ? direct : fallback) << '\n';
            }
            else
            {
                const double a = run(direct), b = run(fallback);
                std::cout << "sweep threads=" << threads
                          << " us/call: direct=" << a << " fallback=" << b
                          << '\n';
            }
            EXPECT_TRUE(std::isfinite(checksum));
        }
    }
}
} // namespace

TEST(ManagedIlqr, LocalAndCompatibilityDerivativeSweepsAgree)
{
    compareDerivativeSweeps(false);
}

TEST(ManagedIlqr, DISABLED_ProfileLocalDerivativeSweeps)
{
    compareDerivativeSweeps(true);
}

namespace
{
void compareSymbolicShooting(bool benchmark)
{
    using Access = horizon::IterativeLQRTestAccess;
    constexpr int N = 5;
    auto space = std::make_shared<QuaternionSpace>();
    auto x = casadi::MX::sym("x", 4), u = casadi::MX::sym("u", 3),
         p = casadi::MX::sym("p");
    auto vector = .5 * p * u;
    auto scale = sqrt(1 + casadi::MX::dot(vector, vector));
    auto q = x(casadi::Slice(0, 3));
    auto next =
        casadi::MX::vertcat({q + x(3) * vector + casadi::MX::cross(q, vector),
                             x(3) - casadi::MX::dot(q, vector)}) /
        scale;
    SymbolicFunctionManager manager;
    manager.addExpression("symbolic_shooting", {x, u, p}, next, false);
    manager.addResidual("shooting_effort", {x, u}, u);
    Vector start(4), goal(4), rotation(3);
    space->neutral(start);
    rotation << .7, -.5, .4;
    space->integrate(start, rotation, goal);
    Matrix guess(4, N + 1);
    for (int k = 0; k <= N; ++k)
    {
        Vector d(3);
        d << .2 * std::sin(k), .3 * std::cos(k), -.1 * k;
        space->integrate(start, d, guess.col(k));
    }
    for (int threads : {0, 2})
    {
        for (int sample = 0; sample < (benchmark ? 3 : 1); ++sample)
        {
            IterativeLQR direct(manager, "symbolic_shooting", N,
                                {{"ilqr.n_threads", threads}}, space);
            IterativeLQR fallback(manager, "symbolic_shooting", N,
                                  {{"ilqr.n_threads", threads}}, space);
            for (auto *solver : {&direct, &fallback})
            {
                solver->setParameterValue("p", Matrix::Ones(1, N + 1));
                solver->setResidual({0, 1, 2, 3, 4}, manager,
                                    "shooting_effort");
                fixEndpoints(*solver, start, goal, N);
                solver->setStateInitialGuess(guess);
            }
            Access::useAmbientCompatibility(fallback);
            auto start_time = std::chrono::steady_clock::now();
            ASSERT_TRUE(direct.solve(80));
            const auto direct_time =
                std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start_time)
                    .count();
            start_time = std::chrono::steady_clock::now();
            ASSERT_TRUE(fallback.solve(80));
            const auto fallback_time =
                std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start_time)
                    .count();
            EXPECT_LT(
                (direct.getStateTrajectory() - fallback.getStateTrajectory())
                    .norm(),
                1e-8);
            EXPECT_LT(
                (direct.getInputTrajectory() - fallback.getInputTrajectory())
                    .norm(),
                1e-8);
            Access::compareSteps(direct, fallback);
            if (benchmark)
            {
                std::cout << "solve threads=" << threads
                          << " us: direct=" << direct_time
                          << " fallback=" << fallback_time << '\n';
            }
        }
    }
}
} // namespace

TEST(ManagedIlqr, PureSymbolicDynamicsMatchesCompatibilityShooting)
{
    compareSymbolicShooting(false);
}

TEST(ManagedIlqr, DISABLED_ProfileSymbolicShooting)
{
    compareSymbolicShooting(true);
}

TEST(ManagedIlqr, NativeAndCasadiAgreeWithParametersAndThreads)
{
    auto x = casadi::SX::sym("x"), u = casadi::SX::sym("u"),
         p = casadi::SX::sym("p");
    casadi::Function dyn("dyn", {x, u, p}, {x + u + p}, {"x", "u", "p"}, {"f"});
    casadi::Function res("res", {x, u}, {casadi::SX::vertcat({x, 0.3 * u})},
                         {"x", "u"}, {"res"});
    FunctionManager fm;
    fm.add("dyn",
           std::make_shared<CallbackFunction>(
               3, 1,
               [](Input z, VectorOutput y)
               {
                   y(0) = z.sum();
               },
               [](Input, MatrixOutput j)
               {
                   j.setOnes();
               }),
           {{"p", 1}});
    fm.add("res", std::make_shared<CasadiFunction>(res, false));
    Matrix reference;
    for (int mode = 0; mode < 4; ++mode)
    {
        IterativeLQR::OptionDict opt{{"ilqr.n_threads", mode % 2 ? 2 : 0},
                                     {"ilqr.use_kkt_solver", mode / 2},
                                     {"ilqr.merit_der_threshold", 1e-10}};
        IterativeLQR native(fm, "dyn", 4, opt), symbolic(dyn, 4, opt);
        native.setResidual({0, 1, 2, 3}, fm, "res");
        symbolic.setResidual({0, 1, 2, 3}, res);
        for (auto *solver : {&native, &symbolic})
        {
            solver->setParameterValue("p", Matrix::Constant(1, 5, .1));
            fixEndpoints(*solver, Vector::Constant(1, 1), Vector::Zero(1), 4);
            solver->solve(10);
            EXPECT_NEAR(solver->state(0)(0), 1, 1e-7);
            EXPECT_NEAR(solver->state(4)(0), 0, 1e-7);
            for (int k = 0; k < 4; ++k)
            {
                EXPECT_NEAR(solver->state(k + 1)(0),
                            solver->state(k)(0) + solver->input(k)(0) + .1,
                            1e-7);
            }
        }
        EXPECT_LT((native.getStateTrajectory() - symbolic.getStateTrajectory())
                      .norm(),
                  1e-7);
        if (mode == 0)
        {
            reference = native.getInputTrajectory();
        }
        else
        {
            EXPECT_LT((native.getInputTrajectory() - reference).norm(), 1e-7);
        }
    }
}

TEST(ManagedIlqr, TargetsIndicesAndValidation)
{
    FunctionManager fm;
    fm.add("dyn",
           std::make_shared<CallbackFunction>(
               2, 1,
               [](Input z, VectorOutput y)
               {
                   y(0) = z.sum();
               },
               [](Input, MatrixOutput j)
               {
                   j.setOnes();
               }));
    fm.add("state",
           std::make_shared<CallbackFunction>(
               2, 1,
               [](Input z, VectorOutput y)
               {
                   y(0) = z(0);
               },
               [](Input, MatrixOutput j)
               {
                   j << 1, 0;
               }));
    IterativeLQR solver(fm, "dyn", 3, {{"ilqr.n_threads", 2}});
    solver.setConstraint({0, 3}, fm, "state",
                         {Vector::Constant(1, 1), Vector::Constant(1, 2)});
    EXPECT_THROW(solver.setIndices("state", {4}), std::invalid_argument);
    EXPECT_THROW(solver.setCost({0}, fm, "state"), std::invalid_argument);
    solver.solve(5);
    EXPECT_NEAR(solver.state(0)(0), 1, 1e-7);
    EXPECT_NEAR(solver.state(3)(0), 2, 1e-7);
    EXPECT_NEAR(solver.getConstraintsValues().at("state")(0, 0), 0, 1e-7);
    EXPECT_NEAR(solver.getConstraintsValues().at("state")(0, 3), 0, 1e-7);
    solver.setIndices("state", {3});
    solver.updateIndices();
    solver.solve(3);
    EXPECT_TRUE(std::isnan(solver.getConstraintsValues().at("state")(0, 0)));
}

TEST(ManagedIlqr, ArbitraryMultiplicativeStateOperations)
{
    auto space = std::make_shared<CallbackStateSpace>(
        Vector::Ones(1), 1,
        [](Input x, Input dx, VectorOutput y)
        {
            y(0) = x(0) * std::exp(dx(0));
        },
        [](Input a, Input b, VectorOutput d)
        {
            d(0) = std::log(b(0) / a(0));
        });
    FunctionManager fm;
    fm.add("dyn",
           std::make_shared<CallbackFunction>(
               2, 1,
               [](Input z, VectorOutput y)
               {
                   y(0) = z(0) * std::exp(z(1));
               },
               [](Input z, MatrixOutput j)
               {
                   j << std::exp(z(1)), z(0) * std::exp(z(1));
               }));
    fm.add("effort",
           std::make_shared<CallbackFunction>(
               2, 1,
               [](Input z, VectorOutput y)
               {
                   y(0) = z(1);
               },
               [](Input, MatrixOutput j)
               {
                   j << 0, 1;
               }));
    for (int kkt : {0, 1})
    {
        IterativeLQR solver(fm, "dyn", 4,
                            {{"ilqr.n_threads", 2},
                             {"ilqr.use_kkt_solver", kkt},
                             {"ilqr.merit_der_threshold", 1e-10}},
                            space);
        solver.setResidual({0, 1, 2, 3}, fm, "effort");
        fixEndpoints(solver, Vector::Ones(1),
                     Vector::Constant(1, std::exp(1.2)), 4);
        solver.solve(20);
        for (int k = 0; k < 4; ++k)
        {
            EXPECT_NEAR(solver.input(k)(0), .3, 1e-6);
            EXPECT_NEAR(solver.state(k + 1)(0), std::exp(.3 * (k + 1)), 1e-6);
        }
    }
}

TEST(ManagedIlqr, QuaternionShootingWithInfeasibleNodes)
{
    auto space = std::make_shared<QuaternionSpace>();
    auto skew = [](const Eigen::Vector3d &v)
    {
        Eigen::Matrix3d m;
        m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
        return m;
    };
    auto dynamics = std::make_shared<CallbackFunction>(
        7, 4,
        [](Input z, VectorOutput y)
        {
            Eigen::Quaterniond q(z(3), z(0), z(1), z(2));
            Eigen::Quaterniond dq(1, .5 * z(4), .5 * z(5), .5 * z(6));
            y = (q * dq.normalized()).coeffs();
        },
        [skew](Input z, MatrixOutput j)
        {
            Eigen::Vector4d dq;
            dq << .5 * z.tail(3), 1;
            double t = dq.norm();
            dq /= t;
            j.topLeftCorner(3, 3) =
                dq(3) * Eigen::Matrix3d::Identity() - skew(dq.head<3>());
            j.block(0, 3, 3, 1) = dq.head<3>();
            j.block(3, 0, 1, 3) = -dq.head<3>().transpose();
            j(3, 3) = dq(3);
            Eigen::Matrix4d Q;
            Q.topLeftCorner<3, 3>() =
                z(3) * Eigen::Matrix3d::Identity() + skew(z.head<3>());
            Q.topRightCorner<3, 1>() = z.head<3>();
            Q.bottomLeftCorner<1, 3>() = -z.head<3>().transpose();
            Q(3, 3) = z(3);
            j.rightCols(3) = Q *
                (Eigen::Matrix4d::Identity() - dq * dq.transpose())
                    .leftCols<3>() /
                (2 * t);
        });
    FunctionManager fm;
    fm.add("dyn", dynamics);
    fm.add("effort",
           std::make_shared<CallbackFunction>(
               7, 3,
               [](Input z, VectorOutput y)
               {
                   y = z.tail(3);
               },
               [](Input, MatrixOutput j)
               {
                   j.setZero();
                   j.rightCols(3).setIdentity();
               }));
    Vector start(4), goal(4), rotation(3);
    space->neutral(start);
    rotation << .7, -.5, .4;
    space->integrate(start, rotation, goal);
    for (const std::string strategy : {"legacy", "lm_merit", "filter_sqp"})
    {
        Matrix reference;
        for (int kkt : {0, 1})
        {
            IterativeLQR solver(fm, "dyn", 5,
                                {{"ilqr.globalization", strategy},
                                 {"ilqr.n_threads", 2},
                                 {"ilqr.use_kkt_solver", kkt},
                                 {"ilqr.merit_der_threshold", 1e-10}},
                                space);
            solver.setResidual({0, 1, 2, 3, 4}, fm, "effort");
            fixEndpoints(solver, start, goal, 5);
            Matrix guess(4, 6);
            for (int k = 0; k <= 5; ++k)
            {
                Vector d(3);
                d << .2 * std::sin(k), .3 * std::cos(k), -.1 * k;
                space->integrate(start, d, guess.col(k));
            }
            solver.setStateInitialGuess(guess);
            ASSERT_TRUE(solver.solve(80))
                << strategy << " kkt=" << kkt
                << " reason=" << solver.getIterationResult().acceptance_reason
                << " stationarity=" << solver.getIterationResult().stationarity;
            Vector gap(3);
            space->difference(goal, solver.state(5), gap);
            EXPECT_LT(gap.norm(), 1e-7);
            for (int k = 0; k < 5; ++k)
            {
                Vector z(7), pred(4);
                z << solver.state(k), solver.input(k);
                dynamics->evaluate(z, pred);
                space->difference(pred, solver.state(k + 1), gap);
                EXPECT_LT(gap.norm(), 1e-7);
                EXPECT_NEAR(solver.state(k).norm(), 1, 1e-12);
            }
            EXPECT_EQ(solver.gain(0).cols(), 3);
            if (!kkt)
            {
                reference = solver.getInputTrajectory();
            }
            else
            {
                EXPECT_LT((reference - solver.getInputTrajectory()).norm(),
                          1e-5);
            }
        }
    }
}

TEST(ManagedIlqr, WorkerExceptionsReachCallerAndWorkersStop)
{
    auto x = casadi::SX::sym("x"), u = casadi::SX::sym("u");
    FunctionManager fm;
    fm.add("dyn",
           std::make_shared<CallbackFunction>(
               2, 1,
               [](Input z, VectorOutput y)
               {
                   y(0) = z.sum();
               },
               [](Input, MatrixOutput)
               {
                   throw std::runtime_error("native derivative failed");
               }));
    IterativeLQR solver(fm, "dyn", 4, {{"ilqr.n_threads", 2}});
    EXPECT_THROW(solver.solve(1), std::runtime_error);
    EXPECT_THROW(solver.solve(1), std::runtime_error);
}

TEST(ManagedIlqr, ExactScalarCostMatchesResidualAndNormalization)
{
    auto x = casadi::SX::sym("x"), u = casadi::SX::sym("u");
    casadi::Function dyn("dyn", {x, u}, {x + u}, {"x", "u"}, {"f"});
    casadi::Function residual("res", {x, u}, {casadi::SX::vertcat({x, .3 * u})},
                              {"x", "u"}, {"res"});
    casadi::Function cost("cost", {x, u}, {.5 * (x * x + .09 * u * u)},
                          {"x", "u"}, {"l"});
    casadi::Function final("final", {x, u}, {3 * x * x}, {"x", "u"}, {"l"});
    FunctionManager fm;
    fm.add("cost", std::make_shared<CasadiFunction>(cost));
    fm.add("final", std::make_shared<CasadiFunction>(final));
    horizon::IterativeLQR scalar(dyn, 4), res(dyn, 4), legacy(dyn, 4);
    scalar.setCost({0, 1, 2, 3}, fm, "cost");
    scalar.setCost({4}, fm, "final");
    res.setResidual({0, 1, 2, 3}, residual);
    res.setFinalCost(final);
    legacy.setCost({0}, cost);
    legacy.setCost({0, 1, 2, 3}, cost);
    legacy.setFinalCost(final);
    for (auto *solver : {&scalar, &res, &legacy})
    {
        Matrix lb = Matrix::Constant(1, 5, -inf), ub = -lb;
        lb(0, 0) = ub(0, 0) = 1;
        solver->setStateBounds(lb, ub);
        solver->solve(10);
    }
    EXPECT_LT((scalar.getInputTrajectory() - res.getInputTrajectory()).norm(),
              1e-7);
    EXPECT_LT(
        (scalar.getInputTrajectory() - legacy.getInputTrajectory()).norm(),
        1e-7);
    // Independent dense LQ solution for the averaged running cost plus final
    // cost.
    Matrix S = Matrix::Zero(5, 4);
    for (int k = 1; k <= 4; ++k)
    {
        S.row(k).head(k).setOnes();
    }
    Vector weights = Vector::Constant(5, .25);
    weights(4) = 6;
    Matrix H = S.transpose() * weights.asDiagonal() * S +
        .0225 * Matrix::Identity(4, 4);
    Vector expected = -H.ldlt().solve(S.transpose() * weights);
    EXPECT_LT((scalar.getInputTrajectory().transpose() - expected).norm(),
              1e-7);
}

TEST(ManagedIlqr, AugmentedInputBoundsIncludeTerminalNodeSafely)
{
    auto x = casadi::SX::sym("x"), u = casadi::SX::sym("u");
    casadi::Function dyn("dyn", {x, u}, {x + u}, {"x", "u"}, {"f"});
    casadi::Function final("final", {x, u}, {(x - 2) * (x - 2)}, {"x", "u"},
                           {"l"});
    horizon::IterativeLQR solver(dyn, 1,
                                 {{"ilqr.enable_auglag", 1},
                                  {"ilqr.rho_base", 10.0},
                                  {"ilqr.merit_der_threshold", 1e-10}});
    solver.setFinalCost(final);
    Matrix lb = Matrix::Constant(1, 2, -inf), ub = -lb;
    lb(0, 0) = ub(0, 0) = 0;
    solver.setStateBounds(lb, ub);
    solver.setInputBounds(Matrix::Constant(1, 1, -.5),
                          Matrix::Constant(1, 1, .5));
    solver.solve(60);
    EXPECT_NEAR(solver.input(0)(0), .5, 1e-6);
    EXPECT_NEAR(solver.state(1)(0), .5, 1e-6);
}

TEST(ManagedIlqr, LegacyIntegratorMayReturnQuadrature)
{
    auto x = casadi::SX::sym("x"), u = casadi::SX::sym("u"),
         dt = casadi::SX::sym("dt");
    casadi::Function dyn("rk", {x, u, dt}, {x + dt * u, dt * u * u},
                         {"x", "u", "dt"}, {"f", "qf"});
    horizon::IterativeLQR solver(dyn, 2);
    solver.setParameterValue("dt", Matrix::Constant(1, 3, .1));
    fixEndpoints(solver, Vector::Zero(1), Vector::Ones(1), 2);
    solver.solve(5);
    EXPECT_NEAR(solver.state(2)(0), 1, 1e-7);
}

TEST(ManagedIlqr, RejectedLineSearchRestoresAcceptedCostValues)
{
    auto x = casadi::SX::sym("x"), u = casadi::SX::sym("u");
    casadi::Function dyn("dyn", {x, u}, {x + u}, {"x", "u"}, {"f"});
    casadi::Function cost("quartic", {x, u}, {.25 * u * u * u * u - u},
                          {"x", "u"}, {"l"});
    horizon::IterativeLQR solver(dyn, 1, {{"ilqr.alpha_min", 1.0}});
    solver.setCost({0}, cost);
    Matrix guess(1, 2);
    guess << 0, .1;
    solver.setStateInitialGuess(guess);
    solver.setInputInitialGuess(Matrix::Constant(1, 1, .1));
    Matrix lb = Matrix::Constant(1, 2, -inf), ub = -lb;
    lb(0, 0) = ub(0, 0) = 0;
    solver.setStateBounds(lb, ub);
    EXPECT_FALSE(solver.solve(1));
    EXPECT_DOUBLE_EQ(solver.input(0)(0), .1);
    EXPECT_NEAR(solver.getCostsValues().at("quartic")(0),
                .25 * std::pow(.1, 4) - .1, 1e-14);
}

TEST(ManagedIlqr, LineSearchBacktracksAfterNumericalEvaluationFailure)
{
    auto failures = std::make_shared<int>(0);
    FunctionManager fm;
    fm.add("dyn",
           std::make_shared<CallbackFunction>(
               2, 1,
               [](Input z, VectorOutput y)
               {
                   y(0) = z.sum();
               },
               [](Input, MatrixOutput j)
               {
                   j.setOnes();
               }));
    fm.add("log_control",
           std::make_shared<CallbackFunction>(
               2, 1,
               [failures](Input z, VectorOutput y)
               {
                   if (z(1) <= 0)
                   {
                       ++*failures;
                   }
                   y(0) = std::log(z(1));
               },
               [](Input z, MatrixOutput j)
               {
                   j << 0, 1 / z(1);
               }));
    IterativeLQR solver(fm, "dyn", 1, {{"ilqr.merit_der_threshold", 1e-12}});
    solver.setResidual({0}, fm, "log_control");
    Matrix lb = Matrix::Constant(1, 2, -inf), ub = -lb;
    lb(0, 0) = ub(0, 0) = 0;
    solver.setStateBounds(lb, ub);
    Matrix guess(1, 2);
    guess << 0, 10;
    solver.setStateInitialGuess(guess);
    solver.setInputInitialGuess(Matrix::Constant(1, 1, 10));
    EXPECT_TRUE(solver.solve(30));
    EXPECT_GE(*failures, 2);
    EXPECT_NEAR(solver.input(0)(0), 1, 1e-6);
    EXPECT_NEAR(solver.state(1)(0), 1, 1e-6);

    // An exhausted search reports the accepted iterate, even when the only
    // candidate failed before all of its metrics could be evaluated.
    IterativeLQR exhausted(fm, "dyn", 1, {{"ilqr.alpha_min", 1.0}});
    exhausted.setResidual({0}, fm, "log_control");
    exhausted.setStateBounds(lb, ub);
    exhausted.setStateInitialGuess(guess);
    exhausted.setInputInitialGuess(Matrix::Constant(1, 1, 10));
    bool saw_rejection = false;
    exhausted.setIterationCallback(
        [&](const auto &r)
        {
            if (!r.accepted)
            {
                saw_rejection = true;
                EXPECT_TRUE(r.xtrj.allFinite());
                EXPECT_TRUE(r.utrj.allFinite());
                EXPECT_NEAR(r.cost, .5 * std::pow(std::log(10.), 2), 1e-12);
                EXPECT_TRUE(std::isfinite(r.merit));
            }
            return true;
        });
    EXPECT_FALSE(exhausted.solve(1));
    EXPECT_TRUE(saw_rejection);
    EXPECT_DOUBLE_EQ(exhausted.input(0)(0), 10);
}
