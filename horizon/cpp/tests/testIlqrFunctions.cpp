#include "ilqr/casadi_function.h"
#include "ilqr/state_space.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <gtest/gtest.h>
#include <iostream>

using namespace horizon::ilqr;

namespace
{
class ProductOnly final : public Function
{
  public:
    ProductOnly() : Function(2, 1)
    {
    }
    bool hasJacobianProduct() const override
    {
        return true;
    }
    std::unique_ptr<Function> clone() const override
    {
        return std::make_unique<ProductOnly>();
    }

  private:
    void evaluateImpl(Input x, VectorOutput y) override
    {
        y[0] = x.squaredNorm();
    }
    void jacobianImpl(Input, MatrixOutput) override
    {
        throw std::logic_error("dense Jacobian must not be requested");
    }
    void valueAndJacobianProductImpl(Input x, MatrixInput d, VectorOutput y,
                                     MatrixOutput p) override
    {
        evaluateImpl(x, y);
        p.noalias() = 2 * x.transpose() * d;
    }
};

class LocalLeaf final : public TangentFunction
{
  public:
    LocalLeaf(std::shared_ptr<const StateSpace> space,
              std::shared_ptr<std::atomic<int>> calls)
        : TangentFunction(space, 1), _space(std::move(space)),
          _calls(std::move(calls))
    {
    }
    std::unique_ptr<Function> clone() const override
    {
        return std::make_unique<LocalLeaf>(_space, _calls);
    }

  private:
    void evaluateImpl(Input x, VectorOutput y) override
    {
        y(0) = x(0) * x(0) + 2 * x(1);
    }
    void tangentJacobian(Input x, MatrixOutput jac) override
    {
        ++*_calls;
        jac << 2 * x(0) * x(0), 2 * x(1);
    }
    std::shared_ptr<const StateSpace> _space;
    std::shared_ptr<std::atomic<int>> _calls;
};

class CombinedLeaf final : public TangentFunction
{
  public:
    explicit CombinedLeaf(std::shared_ptr<const StateSpace> space)
        : TangentFunction(space, 1), _space(std::move(space))
    {
    }
    std::unique_ptr<Function> clone() const override
    {
        return std::make_unique<CombinedLeaf>(_space);
    }

  private:
    void evaluateImpl(Input x, VectorOutput y) override
    {
        y(0) = x(0) * x(0) + 2 * x(1) + x(2) * x(3) + x(4) * x(4);
    }
    void tangentJacobian(Input x, MatrixOutput jac) override
    {
        jac << 2 * x(0) * x(0), 2 * x(1), x(3), x(2), 2 * x(4);
    }
    std::shared_ptr<const StateSpace> _space;
};
} // namespace

TEST(IlqrFunctions, GraphCombinesNativeManifoldAndEuclideanInputs)
{
    auto projections = std::make_shared<std::atomic<int>>(0);
    auto local = std::make_shared<CallbackStateSpace>(
        Vector::Ones(2), 2,
        [](Input x, Input d, VectorOutput y)
        {
            y = x.array() * d.array().exp();
        },
        [](Input x, Input y, VectorOutput d)
        {
            d = (y.array() / x.array()).log();
        },
        [projections](Input x, MatrixOutput jac)
        {
            ++*projections;
            jac.setZero();
            jac.diagonal() = x;
        });
    ProductSpace state({local, std::make_shared<EuclideanSpace>(1)});
    auto input_space = std::make_shared<ProductSpace>(
        std::vector<std::shared_ptr<const StateSpace>>{
            local, std::make_shared<EuclideanSpace>(3)});
    SymbolicFunctionManager manager;
    auto leaf = manager.addNative("combined_leaf",
                                  std::make_shared<CombinedLeaf>(input_space));
    auto x = casadi::MX::sym("x", 3), u = casadi::MX::sym("u"),
         p = casadi::MX::sym("p");
    manager.addResidual("combined", {x, u, p},
                        sin(leaf(casadi::MX::vertcat({x, u, p}))));
    manager.addResidual("transformed", {x, u, p},
                        sin(leaf(casadi::MX::vertcat({x, u + 1, p}))));
    auto direct = manager.at("combined").function->clone();
    auto fallback = manager.at("transformed").function->clone();
    Vector z(5), value(1), ambient_value(1);
    z << 1.2, .8, .3, -.2, .4;
    Matrix retract(3, 3), jac(1, 4), ambient(1, 4), seed = Matrix::Zero(5, 4);
    state.retractionJacobian(z.head(3), retract);
    seed.topLeftCorner(3, 3) = retract;
    seed(3, 3) = 1;
    for (auto *function : {direct.get(), fallback.get()})
    {
        auto *stage = dynamic_cast<StageTangentLinearization *>(function);
        ASSERT_NE(stage, nullptr);
        projections->store(0);
        stage->valueAndStageLinearization(z, state, 1, retract, value, jac);
        if (function == direct.get())
        {
            EXPECT_EQ(projections->load(), 0);
        }
        else
        {
            EXPECT_GT(projections->load(), 0);
        }
        function->valueAndJacobianProduct(z, seed, ambient_value, ambient);
        EXPECT_LT((jac - ambient).norm(), 1e-12);
        EXPECT_LT((value - ambient_value).norm(), 1e-12);
        for (int i = 0; i < 4; ++i)
        {
            Vector a = z, b = z, va(1), vb(1), d = Vector::Zero(3);
            if (i < 3)
            {
                d(i) = 1e-6;
                state.integrate(z.head(3), d, a.head(3));
                state.integrate(z.head(3), -d, b.head(3));
            }
            else
            {
                a(3) += 1e-6;
                b(3) -= 1e-6;
            }
            function->evaluate(a, va);
            function->evaluate(b, vb);
            EXPECT_NEAR(jac(0, i), (va(0) - vb(0)) / 2e-6, 1e-8);
        }
    }
}

TEST(IlqrFunctions, DISABLED_ProfileGraphTangentPropagation)
{
    auto calls = std::make_shared<std::atomic<int>>(0);
    auto local = std::make_shared<CallbackStateSpace>(
        Vector::Ones(2), 2,
        [](Input x, Input d, VectorOutput y)
        {
            y = x.array() * d.array().exp();
        },
        [](Input x, Input y, VectorOutput d)
        {
            d = (y.array() / x.array()).log();
        },
        [](Input x, MatrixOutput jac)
        {
            jac.setZero();
            jac.diagonal() = x;
        });
    ProductSpace state({std::make_shared<EuclideanSpace>(1), local});
    SymbolicFunctionManager manager;
    auto leaf = manager.addNative("profile_leaf",
                                  std::make_shared<LocalLeaf>(local, calls));
    auto x = casadi::MX::sym("x", 3), u = casadi::MX::sym("u"),
         p = casadi::MX::sym("p");
    manager.addResidual("profile_graph", {x, u, p},
                        sin(leaf(x(casadi::Slice(1, 3)))) + x(0) + u + p);
    auto direct = manager.at("profile_graph").function->clone();
    auto fallback = manager.at("profile_graph").function->clone();
    auto *stage = dynamic_cast<StageTangentLinearization *>(direct.get());
    ASSERT_NE(stage, nullptr);
    Vector z(5), value(1), other(1);
    z << .3, 1.2, .8, -.2, .4;
    Matrix retract(3, 3), jac(1, 4), reference(1, 4), seed = Matrix::Zero(5, 4);
    state.retractionJacobian(z.head(3), retract);
    seed.topLeftCorner(3, 3) = retract;
    seed(3, 3) = 1;
    stage->valueAndStageLinearization(z, state, 1, retract, value, jac);
    fallback->valueAndJacobianProduct(z, seed, other, reference);
    ASSERT_LT((jac - reference).norm() + (value - other).norm(), 1e-12);
    double checksum = 0;
    auto run = [&](bool fast)
    {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 10000; ++i)
        {
            if (fast)
            {
                stage->valueAndStageLinearization(z, state, 1, retract, value,
                                                  jac);
            }
            else
            {
                fallback->valueAndJacobianProduct(z, seed, value, jac);
            }
            checksum += jac(0, 0);
        }
        return std::chrono::duration<double, std::micro>(
                   std::chrono::steady_clock::now() - start)
                   .count() /
            10000;
    };
    if (const char *mode = std::getenv("HORIZON_BENCH_GRAPH_PATH"))
    {
        const std::string path(mode);
        ASSERT_TRUE(path == "direct" || path == "fallback");
        std::cout << path << " graph us/call: " << run(path == "direct")
                  << '\n';
    }
    else
    {
        for (int i = 0; i < 3; ++i)
        {
            const double a = run(true), b = run(false);
            std::cout << "graph us/call: direct=" << a << " fallback=" << b
                      << '\n';
        }
    }
    EXPECT_TRUE(std::isfinite(checksum));
}

TEST(IlqrFunctions, DISABLED_ProfileCasadiLocalDerivatives)
{
    QuaternionSpace space;
    auto x = casadi::MX::sym("x", 4), u = casadi::MX::sym("u"),
         p = casadi::MX::sym("p");
    auto y = casadi::MX::vertcat({sin(x(0) + p * u), x(3) * x(1) + u * u});
    casadi::Function source("profile_local", {x, u, p}, {y}, {"x", "u", "p"},
                            {"f"});
    casadi_utils::WrappedFunction local(
        compileLocalDerivatives(source, space, LocalDerivativeKind::Jacobian));
    CasadiFunction fallback(source, false);
    Vector z(6), q0(4), value(2), other(2);
    space.neutral(q0);
    space.integrate(q0, Vector::Constant(3, .2), z.head(4));
    z.tail(2) << .3, .7;
    Matrix retract(4, 3), seed = Matrix::Zero(6, 4), jac(2, 4), reference(2, 4);
    space.retractionJacobian(z.head(4), retract);
    seed.topLeftCorner(4, 3) = retract;
    seed(4, 3) = 1;
    local.setInput(0, z.head(4));
    local.setInput(1, z.segment(4, 1));
    local.setInput(2, z.tail(1));
    local.call();
    value = local.getOutput(0);
    jac = local.getOutput(1);
    fallback.valueAndJacobianProduct(z, seed, other, reference);
    ASSERT_LT((jac - reference).norm() + (value - other).norm(), 1e-12);
    double checksum = 0;
    auto run = [&](bool fast)
    {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 10000; ++i)
        {
            if (fast)
            {
                local.setInput(0, z.head(4));
                local.setInput(1, z.segment(4, 1));
                local.setInput(2, z.tail(1));
                local.call();
                value = local.getOutput(0);
                jac = local.getOutput(1);
            }
            else
            {
                fallback.valueAndJacobianProduct(z, seed, value, jac);
            }
            checksum += jac(0, 0);
        }
        return std::chrono::duration<double, std::micro>(
                   std::chrono::steady_clock::now() - start)
                   .count() /
            10000;
    };
    if (const char *mode = std::getenv("HORIZON_BENCH_CASADI_PATH"))
    {
        const std::string path(mode);
        ASSERT_TRUE(path == "direct" || path == "fallback");
        std::cout << path << " CasADi us/call: " << run(path == "direct")
                  << '\n';
    }
    else
    {
        for (int i = 0; i < 3; ++i)
        {
            const double a = run(true), b = run(false);
            std::cout << "CasADi us/call: direct=" << a << " fallback=" << b
                      << '\n';
        }
    }
    EXPECT_TRUE(std::isfinite(checksum));
}

TEST(IlqrFunctions, GraphUsesMatchingNativeTangentAndFallsBackForReordering)
{
    auto projections = std::make_shared<std::atomic<int>>(0);
    auto calls = std::make_shared<std::atomic<int>>(0);
    auto local = std::make_shared<CallbackStateSpace>(
        Vector::Ones(2), 2,
        [](Input x, Input d, VectorOutput y)
        {
            y = x.array() * d.array().exp();
        },
        [](Input x, Input y, VectorOutput d)
        {
            d = (y.array() / x.array()).log();
        },
        [projections](Input x, MatrixOutput jac)
        {
            ++*projections;
            jac.setZero();
            jac.diagonal() = x;
        });
    ProductSpace state({std::make_shared<EuclideanSpace>(1), local});
    SymbolicFunctionManager manager;
    auto leaf = manager.addNative("local_leaf",
                                  std::make_shared<LocalLeaf>(local, calls));
    auto x = casadi::MX::sym("x", 3), u = casadi::MX::sym("u"),
         p = casadi::MX::sym("p");
    manager.addResidual("direct", {x, u, p},
                        sin(leaf(x(casadi::Slice(1, 3)))) + x(0) + u + p);
    manager.addResidual("reordered", {x, u, p},
                        sin(leaf(casadi::MX::vertcat({x(2), x(1)}))) + u + p);
    auto direct = manager.at("direct").function->clone();
    auto reordered = manager.at("reordered").function->clone();
    auto context = std::make_shared<EvaluationContext>();
    direct->setEvaluationContext(context);
    reordered->setEvaluationContext(context);
    auto *stage = dynamic_cast<StageTangentLinearization *>(direct.get());
    auto *other = dynamic_cast<StageTangentLinearization *>(reordered.get());
    ASSERT_NE(stage, nullptr);
    ASSERT_NE(other, nullptr);
    Vector z(5), value(1), ambient_value(1);
    z << .3, 1.2, .8, -.2, .4;
    Matrix retract(3, 3), jac(1, 4), ambient(1, 4), seed = Matrix::Zero(5, 4);
    state.retractionJacobian(z.head(3), retract);
    seed.topLeftCorner(3, 3) = retract;
    seed(3, 3) = 1;
    projections->store(0);
    stage->valueAndStageLinearization(z, state, 1, retract, value, jac);
    EXPECT_EQ(projections->load(), 0);
    EXPECT_EQ(calls->load(), 1);
    direct->valueAndJacobianProduct(z, seed, ambient_value, ambient);
    EXPECT_LT((jac - ambient).norm(), 1e-12);
    EXPECT_LT((value - ambient_value).norm(), 1e-12);
    z(4) += .2;
    stage->valueAndStageLinearization(z, state, 1, retract, value, jac);
    EXPECT_EQ(calls->load(), 1);
    context->invalidate();
    stage->valueAndStageLinearization(z, state, 1, retract, value, jac);
    EXPECT_EQ(calls->load(), 2);

    projections->store(0);
    other->valueAndStageLinearization(z, state, 1, retract, value, jac);
    EXPECT_GT(projections->load(), 0);
    reordered->valueAndJacobianProduct(z, seed, ambient_value, ambient);
    EXPECT_LT((jac - ambient).norm(), 1e-12);
    EXPECT_LT((value - ambient_value).norm(), 1e-12);
    for (int i = 0; i < 4; ++i)
    {
        Vector a = z, b = z, va(1), vb(1), d = Vector::Zero(3);
        if (i < 3)
        {
            d(i) = 1e-6;
            state.integrate(z.head(3), d, a.head(3));
            state.integrate(z.head(3), -d, b.head(3));
        }
        else
        {
            a(3) += 1e-6;
            b(3) -= 1e-6;
        }
        reordered->evaluate(a, va);
        reordered->evaluate(b, vb);
        EXPECT_NEAR(jac(0, i), (va(0) - vb(0)) / 2e-6, 1e-8);
    }
}

TEST(IlqrFunctions, CasadiLocalExpansionPreservesEmbeddedCalls)
{
    QuaternionSpace space;
    auto x = casadi::MX::sym("x", 4), u = casadi::MX::sym("u");
    casadi::Function simple("simple_expansion_test", {x, u},
                            {sin(x(0)) + x(3) * x(1) + u});
    EXPECT_TRUE(
        compileLocalDerivatives(simple, space, LocalDerivativeKind::Jacobian)
            .is_a("SXFunction"));
    auto large = x(0) * casadi::MX::ones(100, 100);
    casadi::Function large_source(
        "large_expansion_test", {x, u},
        {casadi::MX::sum1(casadi::MX::sum2(large)) + u});
    EXPECT_TRUE(compileLocalDerivatives(large_source, space,
                                        LocalDerivativeKind::Jacobian)
                    .is_a("MXFunction"));
    casadi::Function inner("inner_expansion_test", {x},
                           {sin(x(0)) + x(3) * x(1)});
    std::vector<casadi::MX> result;
    inner.call(std::vector<casadi::MX>{x}, result, false, true);
    auto nested = result.at(0);
    casadi::Function source("nested_expansion_test", {x, u}, {nested + u});
    bool has_call = false;
    for (casadi_int i = 0; i < source.n_instructions(); ++i)
    {
        has_call = has_call || source.instruction_id(i) == casadi::OP_CALL;
    }
    ASSERT_TRUE(has_call);
    auto compiled =
        compileLocalDerivatives(source, space, LocalDerivativeKind::Jacobian);
    EXPECT_TRUE(compiled.is_a("MXFunction"));
    casadi_utils::WrappedFunction local(compiled);
    CasadiFunction ambient(source, false);
    Vector z(5), identity(4), value(1), other(1);
    space.neutral(identity);
    space.integrate(identity, Vector::Constant(3, .2), z.head(4));
    z(4) = .3;
    Matrix retract(4, 3), seed = Matrix::Zero(5, 4), reference(1, 4);
    space.retractionJacobian(z.head(4), retract);
    seed.topLeftCorner(4, 3) = retract;
    seed(4, 3) = 1;
    local.setInput(0, z.head(4));
    local.setInput(1, z.tail(1));
    local.call();
    ambient.valueAndJacobianProduct(z, seed, other, reference);
    EXPECT_LT((local.getOutput(1) - reference).norm(), 1e-12);
    EXPECT_LT((local.getOutput(0).col(0) - other).norm(), 1e-12);
}

TEST(IlqrFunctions, CasadiPerturbationDerivativesIncludeRetractionCurvature)
{
    auto quaternion = std::make_shared<QuaternionSpace>();
    ProductSpace state({quaternion, std::make_shared<EuclideanSpace>(1)});
    auto x = casadi::MX::sym("x", 5), u = casadi::MX::sym("u"),
         p = casadi::MX::sym("p");
    auto scalar = x(0) + x(3) * x(3) + .3 * u * x(1) + p * u * u + x(4);
    casadi::Function f("local_scalar", {x, u, p}, {scalar}, {"x", "u", "p"},
                       {"f"});
    casadi_utils::WrappedFunction local(
        compileLocalDerivatives(f, state, LocalDerivativeKind::ExactScalar));
    Vector base(5), identity(4), rotation(3), control(1), parameter(1);
    quaternion->neutral(identity);
    rotation << .3, -.4, .2;
    quaternion->integrate(identity, rotation, base.head(4));
    base(4) = .7;
    control(0) = .4;
    parameter(0) = 1.3;
    local.setInput(0, base);
    local.setInput(1, control);
    local.setInput(2, parameter);
    local.call();
    Matrix jac = local.getOutput(1), hess = local.getOutput(2), numeric(5, 5);
    EXPECT_TRUE(jac.allFinite());
    EXPECT_TRUE(hess.allFinite());
    casadi_utils::WrappedFunction original(f);
    auto evaluate = [&](const Vector &d)
    {
        Vector shifted(5), shifted_u = control + d.tail(1);
        state.integrate(base, d.head(4), shifted);
        original.setInput(0, shifted);
        original.setInput(1, shifted_u);
        original.setInput(2, parameter);
        original.call();
        return original.getOutput(0)(0, 0);
    };
    const double eps = 1e-4;
    for (int i = 0; i < 5; ++i)
    {
        Vector a = Vector::Zero(5);
        a(i) = eps;
        EXPECT_NEAR(jac(0, i), (evaluate(a) - evaluate(-a)) / (2 * eps), 1e-8);
        for (int j = 0; j < 5; ++j)
        {
            Vector b = Vector::Zero(5);
            b(j) = eps;
            numeric(i, j) = (evaluate(a + b) - evaluate(a - b) -
                             evaluate(-a + b) + evaluate(-a - b)) /
                (4 * eps * eps);
        }
    }
    EXPECT_LT((hess - numeric).norm(), 1e-6);
    casadi_utils::WrappedFunction gn(
        compileLocalDerivatives(f, state, LocalDerivativeKind::GaussNewton));
    gn.setInput(0, base);
    gn.setInput(1, control);
    gn.setInput(2, parameter);
    gn.call();
    EXPECT_LT((gn.getOutput(2) - jac.transpose() * jac).norm(), 1e-12);
    CallbackStateSpace unsupported(
        Vector::Ones(1), 1,
        [](Input x, Input d, VectorOutput y)
        {
            y = x + d;
        },
        [](Input x, Input y, VectorOutput d)
        {
            d = y - x;
        });
    auto a = casadi::MX::sym("a"), b = casadi::MX::sym("b");
    casadi::Function fallback("fallback", {a, b}, {a + b});
    EXPECT_TRUE(compileLocalDerivatives(fallback, unsupported,
                                        LocalDerivativeKind::Jacobian)
                    .is_null());
}

TEST(IlqrFunctions, SharedCachePreservesNativeDirectionalKernel)
{
    auto context = std::make_shared<EvaluationContext>();
    auto f = context->share(std::make_shared<ProductOnly>());
    Vector x(2), v(1);
    x << .3, -.2;
    Matrix directions(2, 3), p(1, 3);
    directions << 1, 2, 3, 4, 5, 6;
    f->valueAndJacobianProduct(x, directions, v, p);
    EXPECT_NEAR(v[0], x.squaredNorm(), 1e-14);
    EXPECT_LT((p - 2 * x.transpose() * directions).norm(), 1e-14);
    x[0] = .6;
    context->invalidate();
    f->valueAndJacobianProduct(x, directions, v, p);
    EXPECT_LT((p - 2 * x.transpose() * directions).norm(), 1e-14);
    f->evaluate(x, v);
    EXPECT_NEAR(v[0], x.squaredNorm(), 1e-14);
}

TEST(IlqrFunctions, NativeSymbolicCompositionDerivatives)
{
    CallbackFunction inner(
        2, 2,
        [](Input x, VectorOutput y)
        {
            y << x(0) * x(1), x(0) * x(0) + x(1);
        },
        [](Input x, MatrixOutput j)
        {
            j << x(1), x(0), 2 * x(0), 1;
        },
        [](Input, Input w, MatrixOutput h)
        {
            h << 2 * w(1), w(0), w(0), 0;
        });
    auto y = casadi::MX::sym("y", 2);
    CasadiFunction outer(casadi::Function(
        "outer", {y},
        {casadi::MX::vertcat({sin(y(0)) + y(1) * y(1), y(0) * y(1)})}));
    ComposedFunction f(outer, inner);
    Vector x(2), w(2), value(2);
    x << .4, -.7;
    w << .8, -.3;
    Matrix j(2, 2), h(2, 2), numericalJ(2, 2), numericalH(2, 2);
    f.evaluate(x, value);
    f.jacobian(x, j);
    f.weightedHessian(x, w, h);
    const double eps = 1e-6;
    for (int i = 0; i < 2; ++i)
    {
        Vector a = x, b = x, va(2), vb(2);
        Matrix ja(2, 2), jb(2, 2);
        a(i) += eps;
        b(i) -= eps;
        f.evaluate(a, va);
        f.evaluate(b, vb);
        numericalJ.col(i) = (va - vb) / (2 * eps);
        f.jacobian(a, ja);
        f.jacobian(b, jb);
        numericalH.col(i) = (ja - jb).transpose() * w / (2 * eps);
    }
    EXPECT_LT((j - numericalJ).norm(), 1e-8);
    EXPECT_LT((h - numericalH).norm(), 1e-8);
    auto clone = f.clone();
    auto task = std::async(std::launch::async,
                           [&]
                           {
                               Vector result(2);
                               for (int i = 0; i < 100; ++i)
                               {
                                   clone->evaluate(x, result);
                               }
                               return result;
                           });
    for (int i = 0; i < 100; ++i)
    {
        f.jacobian(x, j);
    }
    EXPECT_LT((task.get() - value).norm(), 1e-14);
}

TEST(IlqrFunctions, CasadiConcatenationAndClone)
{
    auto x = casadi::SX::sym("x", 2), u = casadi::SX::sym("u"),
         p = casadi::SX::sym("p");
    CasadiFunction f(casadi::Function(
        "sparse_output", {x, u, p},
        {casadi::SX::vertcat({x(0) * p, casadi::SX(0), u * x(1)})}));
    Vector z(4), value(3), w(3);
    z << 2, 3, 4, 5;
    w << 1, 2, 3;
    Matrix j(3, 4), h(4, 4);
    f.evaluate(z, value);
    f.jacobian(z, j);
    f.weightedHessian(z, w, h);
    EXPECT_EQ(value(0), 10);
    EXPECT_EQ(value(1), 0);
    EXPECT_EQ(value(2), 12);
    EXPECT_EQ(j(0, 3), 2);
    EXPECT_EQ(j(2, 1), 4);
    EXPECT_EQ(j(1, 1), 0);
    EXPECT_EQ(h(0, 3), 1);
    EXPECT_EQ(h(1, 2), 3);
    f.clone()->evaluate(z, value);
    EXPECT_EQ(value(0), 10);
}

TEST(IlqrFunctions, ValidationAndMissingHessian)
{
    auto f = std::make_shared<CallbackFunction>(
        2, 1,
        [](Input x, VectorOutput y)
        {
            y(0) = x.sum();
        },
        [](Input, MatrixOutput j)
        {
            j.setOnes();
        });
    Vector x = Vector::Ones(2), w = Vector::Ones(1), value(1);
    Matrix h(2, 2);
    EXPECT_THROW(f->weightedHessian(x, w, h), std::logic_error);
    EXPECT_THROW(f->evaluate(w, value), std::invalid_argument);
    FunctionManager manager;
    manager.add("f", f, {{"p", 1}});
    EXPECT_THROW(manager.add("f", f), std::invalid_argument);
    EXPECT_THROW(manager.add("bad", f, {{"p", 1}, {"p", 1}}),
                 std::invalid_argument);
    EXPECT_THROW(manager.at("missing"), std::out_of_range);
}

TEST(IlqrFunctions, SymbolicNativeCallsWithBranchesParametersAndExactHessian)
{
    SymbolicFunctionManager manager;
    auto native = std::make_shared<CallbackFunction>(
        2, 2,
        [](Input x, VectorOutput y)
        {
            y << x(0) * x(1), x(0) * x(0) + x(1);
        },
        [](Input x, MatrixOutput j)
        {
            j << x(1), x(0), 2 * x(0), 1;
        },
        [](Input, Input w, MatrixOutput h)
        {
            h << 2 * w(1), w(0), w(0), 0;
        });
    auto fk = manager.addNative("fk", native);
    auto q = casadi::MX::sym("q", 2), u = casadi::MX::sym("u"),
         ref = casadi::MX::sym("ref", 2);
    auto a = fk(q);
    auto b = fk(a + u);
    auto residual = sin(b - ref) + a;
    auto expression = casadi::MX::sumsqr(residual) + u * u;
    manager.addCost("tracking", {q, u, ref}, expression);
    auto function = manager.at("tracking").function->clone();
    function->setEvaluationContext(std::make_shared<EvaluationContext>());
    Vector z(5), w = Vector::Ones(1), value(1);
    z << .3, -.4, .2, .1, .7;
    Matrix j(1, 5), h(5, 5), nj(1, 5), nh(5, 5);
    function->evaluate(z, value);
    function->jacobian(z, j);
    function->weightedHessian(z, w, h);
    const double eps = 1e-6;
    for (int k = 0; k < 5; ++k)
    {
        Vector plus = z, minus = z, vp(1), vm(1);
        Matrix jp(1, 5), jm(1, 5);
        plus(k) += eps;
        minus(k) -= eps;
        function->evaluate(plus, vp);
        function->evaluate(minus, vm);
        nj(0, k) = (vp(0) - vm(0)) / (2 * eps);
        function->jacobian(plus, jp);
        function->jacobian(minus, jm);
        nh.col(k) = (jp - jm).transpose() / (2 * eps);
    }
    EXPECT_LT((j - nj).norm(), 1e-7);
    EXPECT_LT((h - nh).norm(), 1e-7);
    EXPECT_EQ(manager.at("tracking").parameters.at(0).name, "ref");
    manager.addResidual("residual", {q, u, ref}, residual);
    EXPECT_FALSE(manager.at("residual").function->hasHessian());
    auto clone = function->clone();
    clone->evaluate(z, value);
    EXPECT_LT((h - h.transpose()).norm(), 1e-12);
}

namespace
{
class QuaternionVector final : public TangentFunction
{
  public:
    QuaternionVector() : TangentFunction(std::make_shared<QuaternionSpace>(), 3)
    {
    }
    std::unique_ptr<Function> clone() const override
    {
        return std::make_unique<QuaternionVector>();
    }
    bool hasHessian() const override
    {
        return true;
    }

  private:
    void evaluateImpl(Input q, VectorOutput y) override
    {
        y = q.head(3);
    }
    void tangentJacobian(Input q, MatrixOutput j) override
    {
        QuaternionSpace space;
        Matrix R(4, 3);
        space.retractionJacobian(q, R);
        j = R.topRows(3);
    }
    void tangentWeightedHessian(Input q, Input w, MatrixOutput h) override
    {
        h.setIdentity();
        h *= -.25 * q.head(3).dot(w);
    }
};
} // namespace

TEST(IlqrFunctions, NativeTangentJacobianThroughSymbolicGraph)
{
    SymbolicFunctionManager manager;
    auto fk = manager.addNative("fk", std::make_shared<QuaternionVector>());
    auto q = casadi::MX::sym("q", 4), u = casadi::MX::sym("u"),
         ref = casadi::MX::sym("ref", 3);
    auto error = fk(q) - ref;
    manager.addCost("cost", {q, u, ref}, casadi::MX::sumsqr(error));
    auto function = manager.at("cost").function->clone();
    QuaternionSpace space;
    Vector identity(4), z = Vector::Zero(8), rotation(3), w = Vector::Ones(1);
    space.neutral(identity);
    rotation << .3, -.5, .8;
    space.integrate(identity, rotation, z.head(4));
    z.tail(3) << .2, -.1, .3;
    Matrix j(1, 8), h(8, 8), R(4, 3), curvature(3, 3), localH(3, 3), numH(3, 3);
    function->jacobian(z, j);
    function->weightedHessian(z, w, h);
    space.retractionJacobian(z.head(4), R);
    space.retractionHessian(z.head(4), j.leftCols(4).transpose(), curvature);
    localH = R.transpose() * h.topLeftCorner(4, 4) * R + curvature;
    const double eps = 1e-4;
    for (int a = 0; a < 3; ++a)
    {
        for (int b = 0; b < 3; ++b)
        {
            Vector da = Vector::Zero(3), db = da;
            da(a) = eps;
            db(b) = eps;
            Vector zz = z, v(1);
            double values[4];
            space.integrate(z.head(4), da + db, zz.head(4));
            function->evaluate(zz, v);
            values[0] = v(0);
            space.integrate(z.head(4), da - db, zz.head(4));
            function->evaluate(zz, v);
            values[1] = v(0);
            space.integrate(z.head(4), -da + db, zz.head(4));
            function->evaluate(zz, v);
            values[2] = v(0);
            space.integrate(z.head(4), -da - db, zz.head(4));
            function->evaluate(zz, v);
            values[3] = v(0);
            numH(a, b) = (values[0] - values[1] - values[2] + values[3]) /
                (4 * eps * eps);
        }
    }
    EXPECT_LT((localH - numH).norm(), 1e-7);
}

TEST(IlqrFunctions, JointLinearizationEvaluatesSharedNativeCallOnce)
{
    auto calls = std::make_shared<std::atomic<int>>(0);
    SymbolicFunctionManager manager;
    auto fk = manager.addNative("fk",
                                std::make_shared<CallbackFunction>(
                                    1, 1,
                                    [calls](Input q, VectorOutput y)
                                    {
                                        ++*calls;
                                        y(0) = q(0) * q(0);
                                    },
                                    [](Input q, MatrixOutput j)
                                    {
                                        j(0, 0) = 2 * q(0);
                                    }));
    auto q = casadi::MX::sym("q"), u = casadi::MX::sym("u");
    auto result = fk(q);
    manager.addResidual("res", {q, u}, result + sin(result) + u);
    auto f = manager.at("res").function->clone();
    Vector z(2), value(1);
    z << .3, .2;
    Matrix jac(1, 2);
    f->valueAndJacobian(z, value, jac);
    EXPECT_EQ(calls->load(), 1);
    EXPECT_NEAR(jac(0, 0), .6 * (1 + std::cos(.09)), 1e-12);
}

TEST(IlqrFunctions, SeededTangentGraphMatchesManifoldFiniteDifferences)
{
    SymbolicFunctionManager manager;
    auto fk = manager.addNative("fk", std::make_shared<QuaternionVector>());
    auto q = casadi::MX::sym("q", 4), u = casadi::MX::sym("u"),
         p = casadi::MX::sym("p", 3);
    manager.addCost("cost", {q, u, p},
                    casadi::MX::sumsqr(sin(fk(q) - p)) + u * u);
    auto f = manager.at("cost").function->clone();
    f->setEvaluationContext(std::make_shared<EvaluationContext>());
    QuaternionSpace space;
    Vector identity(4), z = Vector::Zero(8), rotation(3), value(1);
    space.neutral(identity);
    rotation << .2, -.3, .4;
    space.integrate(identity, rotation, z.head(4));
    z(4) = .7;
    z.tail(3) << .1, .2, -.1;
    Matrix R(4, 3), seed = Matrix::Zero(8, 4), product(1, 4), numeric(1, 4),
                    h(8, 8), j(1, 8);
    space.retractionJacobian(z.head(4), R);
    seed.topLeftCorner(4, 3) = R;
    seed(4, 3) = 1;
    f->valueAndJacobianProduct(z, seed, value, product);
    for (int k = 0; k < 4; ++k)
    {
        Vector a = z, b = z, va(1), vb(1), d = Vector::Zero(3);
        if (k < 3)
        {
            d(k) = 1e-6;
            space.integrate(z.head(4), d, a.head(4));
            space.integrate(z.head(4), -d, b.head(4));
        }
        else
        {
            a(4) += 1e-6;
            b(4) -= 1e-6;
        }
        f->evaluate(a, va);
        f->evaluate(b, vb);
        numeric.col(k) = (va - vb) / 2e-6;
    }
    EXPECT_LT((product - numeric).norm(), 1e-8);
    // Mixing seeded, full-Jacobian, and exact-Hessian calls must retain the
    // original ambient-coordinate API and its independently sized buffers.
    f->jacobian(z, j);
    f->weightedHessian(z, Vector::Ones(1), h);
    EXPECT_LT((product - j * seed).norm(), 1e-12);
    EXPECT_TRUE(h.allFinite());
    Matrix shorter = seed.leftCols(1), one(1, 1);
    f->valueAndJacobianProduct(z, shorter, value, one);
    EXPECT_NEAR(one(0, 0), product(0, 0), 1e-12);
}

TEST(IlqrFunctions, SharedNativeCallsAcrossGraphsInvalidateExactly)
{
    auto values = std::make_shared<std::atomic<int>>(0),
         jacs = std::make_shared<std::atomic<int>>(0);
    SymbolicFunctionManager manager;
    auto native = manager.addNative("kernel",
                                    std::make_shared<CallbackFunction>(
                                        2, 2,
                                        [values](Input x, VectorOutput y)
                                        {
                                            ++*values;
                                            y << x(0) * x(0) + x(1),
                                                x(0) * x(1);
                                        },
                                        [jacs](Input x, MatrixOutput j)
                                        {
                                            ++*jacs;
                                            j << 2 * x(0), 1, x(1), x(0);
                                        }));
    auto x = casadi::MX::sym("x"), u = casadi::MX::sym("u"),
         p = casadi::MX::sym("p");
    auto y = native(casadi::MX::vertcat({x, u}));
    manager.addResidual("r", {x, u, p}, y + p);
    manager.addExpression("c", {x, u}, y(0) - 3 * y(1), false);
    auto a = manager.at("r").function->clone(),
         b = manager.at("c").function->clone();
    auto context = std::make_shared<EvaluationContext>();
    a->setEvaluationContext(context);
    b->setEvaluationContext(context);
    Vector z(3), v(2), w(1);
    z << .3, .2, .7;
    Matrix seed = Matrix::Zero(3, 2), ja(2, 2), dir(2, 1), jb(1, 1);
    seed.topRows(2).setIdentity();
    dir << 2, -1;
    a->valueAndJacobianProduct(z, seed, v, ja);
    b->valueAndJacobianProduct(z.head(2), dir, w, jb);
    EXPECT_EQ(values->load(), 1);
    EXPECT_EQ(jacs->load(), 1);
    EXPECT_NEAR(v(0), .99, 1e-14);
    EXPECT_NEAR(w(0), .11, 1e-14);
    EXPECT_NEAR(jb(0, 0), -.1, 1e-14);
    z(2) += .1;
    a->valueAndJacobianProduct(z, seed, v, ja);
    EXPECT_EQ(values->load(), 1);
    EXPECT_NEAR(v(0), 1.09, 1e-14);
    z(0) = std::nextafter(z(0), 1.);
    a->valueAndJacobianProduct(z, seed, v, ja);
    EXPECT_EQ(values->load(), 2);
    EXPECT_EQ(jacs->load(), 2);
    context->invalidate();
    b->valueAndJacobianProduct(z.head(2), dir, w, jb);
    EXPECT_EQ(values->load(), 3);
    auto independent = manager.at("r").function->clone();
    independent->setEvaluationContext(std::make_shared<EvaluationContext>());
    independent->valueAndJacobianProduct(z, seed, v, ja);
    EXPECT_EQ(values->load(), 4);
    a->setEvaluationContext(nullptr);
    a->valueAndJacobianProduct(z, seed, v, ja);
    EXPECT_EQ(values->load(), 5);
}

TEST(IlqrFunctions, NumericalFailureDoesNotPoisonSharedNativeCache)
{
    auto calls = std::make_shared<int>(0), attempts = std::make_shared<int>(0);
    auto source = std::make_shared<CallbackFunction>(
        1, 1,
        [calls](Input x, VectorOutput y)
        {
            ++*calls;
            y(0) = x(0) * x(0);
        },
        [attempts](Input x, MatrixOutput j)
        {
            j(0, 0) = 123;
            if (++*attempts == 1)
            {
                throw EvaluationError("failed derivative");
            }
            j(0, 0) = 2 * x(0);
        });
    EvaluationContext context;
    auto a = context.share(source), b = context.share(source);
    Vector x = Vector::Constant(1, .3), value(1);
    Matrix jac(1, 1);
    EXPECT_THROW(a->valueAndJacobian(x, value, jac), EvaluationError);
    b->valueAndJacobian(x, value, jac);
    EXPECT_NEAR(value(0), .09, 1e-14);
    EXPECT_NEAR(jac(0, 0), .6, 1e-14);
    EXPECT_EQ(*calls, 2);
    EXPECT_EQ(*attempts, 2);
}

TEST(IlqrFunctions, SharedNativeCacheKeysFunctionIdentityNotSharedOwnership)
{
    struct Pair
    {
        std::shared_ptr<Function> a, b;
    };
    auto pair = std::make_shared<Pair>();
    auto make = [](double scale)
    {
        return std::make_shared<CallbackFunction>(
            1, 1,
            [scale](Input x, VectorOutput y)
            {
                y(0) = scale * x(0);
            },
            [scale](Input, MatrixOutput j)
            {
                j(0, 0) = scale;
            });
    };
    pair->a = make(1);
    pair->b = make(2);
    std::shared_ptr<const Function> a(pair, pair->a.get()),
        b(pair, pair->b.get());
    EvaluationContext context;
    auto ca = context.share(a), cb = context.share(b);
    Vector x = Vector::Ones(1), va(1), vb(1);
    Matrix ja(1, 1), jb(1, 1);
    ca->valueAndJacobian(x, va, ja);
    cb->valueAndJacobian(x, vb, jb);
    EXPECT_DOUBLE_EQ(va(0), 1);
    EXPECT_DOUBLE_EQ(vb(0), 2);
    EXPECT_DOUBLE_EQ(ja(0, 0), 1);
    EXPECT_DOUBLE_EQ(jb(0, 0), 2);
}

TEST(IlqrFunctions, SparseCasadiDirectionalOutputsAndStructuralZeros)
{
    auto x = casadi::SX::sym("x", 3);
    CasadiFunction f(casadi::Function(
        "sparse", {x},
        {casadi::SX::vertcat({x(0) * x(2), casadi::SX(0), x(1) + 2 * x(2)})}));
    Vector z(3), value(3);
    z << 2, 3, 4;
    Matrix seed(3, 2), product(3, 2);
    seed << 1, 2, 3, 4, 5, 6;
    f.valueAndJacobianProduct(z, seed, value, product);
    Matrix expected(3, 2);
    expected << 14, 20, 0, 0, 13, 16;
    EXPECT_LT((product - expected).norm(), 1e-14);
    EXPECT_DOUBLE_EQ(value(1), 0);
    seed.setZero();
    f.valueAndJacobianProduct(z, seed, value, product);
    EXPECT_DOUBLE_EQ(product.norm(), 0);
}
