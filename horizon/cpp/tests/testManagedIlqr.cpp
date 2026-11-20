#include "ilqr/casadi_function.h"
#include "ilqr/ilqr.h"
#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <limits>

using namespace horizon::ilqr;
using horizon::IterativeLQR;

namespace
{
constexpr double inf = std::numeric_limits<double>::infinity();
void fixEndpoints(IterativeLQR &solver, const Vector &start, const Vector &goal,
                  int N)
{
    Matrix lb = Matrix::Constant(start.size(), N + 1, -inf), ub = -lb;
    lb.col(0) = ub.col(0) = start;
    lb.col(N) = ub.col(N) = goal;
    solver.setStateBounds(lb, ub);
}
} // namespace

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
