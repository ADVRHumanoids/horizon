#include "ilqr/ilqr_impl.h"
#include <cmath>
#include <gtest/gtest.h>
#include <tuple>

namespace horizon
{
struct IterativeLQRTestAccess
{
    using BoundCost = IterativeLQR::BoundAuglagCostEntity;
    static const Eigen::MatrixXd &stateDual(const IterativeLQR &s)
    {
        return s._lam_bound_x;
    }
    static const Eigen::MatrixXd &inputDual(const IterativeLQR &s)
    {
        return s._lam_bound_u;
    }
    static double rho(const IterativeLQR &s)
    {
        return s._rho;
    }
    static double violation(IterativeLQR &s)
    {
        return s.compute_bound_penalty(s._xtrj, s._utrj);
    }
    static void stateEquality(const IterativeLQR &s, int k, Eigen::MatrixXd &C,
                              Eigen::VectorXd &h)
    {
        s.stateEquality(k, C, h);
    }
};
} // namespace horizon
using Access = horizon::IterativeLQRTestAccess;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
constexpr double inf = std::numeric_limits<double>::infinity();

TEST(IlqrBoundCost, EmptyStateEqualityDoesNotEvaluateRetraction)
{
    int calls = 0;
    auto space = std::make_shared<horizon::ilqr::CallbackStateSpace>(
        Vector::Zero(2), 2,
        [](horizon::ilqr::Input x, horizon::ilqr::Input d,
           horizon::ilqr::VectorOutput y)
        {
            y = x + d;
        },
        [](horizon::ilqr::Input x, horizon::ilqr::Input y,
           horizon::ilqr::VectorOutput d)
        {
            d = y - x;
        },
        [&calls](horizon::ilqr::Input, horizon::ilqr::MatrixOutput j)
        {
            ++calls;
            j.setIdentity();
        });
    auto x = casadi::SX::sym("x", 2), u = casadi::SX::sym("u");
    horizon::IterativeLQR s(
        casadi::Function("bound_dyn", {x, u}, {x}, {"x", "u"}, {"f"}), 1, {},
        space);
    Matrix lb = Matrix::Constant(2, 2, -inf), ub = -lb, C;
    Vector h;
    s.setStateBounds(lb, ub);
    calls = 0;
    Access::stateEquality(s, 1, C, h);
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(C.rows(), 0);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_EQ(h.size(), 0);
    lb(0, 1) = ub(0, 1) = .4;
    s.setStateBounds(lb, ub);
    s.setStateInitialGuess(Matrix::Zero(2, 2));
    Access::stateEquality(s, 1, C, h);
    EXPECT_EQ(calls, 1);
    ASSERT_EQ(C.rows(), 1);
    EXPECT_DOUBLE_EQ(C(0, 0), 1);
    EXPECT_DOUBLE_EQ(C(0, 1), 0);
    EXPECT_NEAR(h(0), -.4, 1e-14);
}

TEST(IlqrBoundCost, ProjectedDerivativesAndMultiplierRelease)
{
    Vector lb = Vector::Constant(1, -1), ub = Vector::Constant(1, 1), x(1),
           u(1);
    Access::BoundCost c(1, lb, ub, lb, ub);
    c.setRho(4);
    x << 1.5;
    u << -1.5;
    c.update_lam(x, u, 0);
    EXPECT_DOUBLE_EQ(c.getStateMultiplier()(0), 2);
    EXPECT_DOUBLE_EQ(c.getInputMultiplier()(0), -2);
    // Shifted activity persists inside a bound; the derivative must agree
    // with the cost even when the unshifted violation is zero.
    for (double v : {-1.7, -.9, -.1, .7, 1.4})
    {
        x << v;
        u << -v;
        Matrix Q = Matrix::Zero(1, 1), R = Q, P = Q;
        c.quadratize(x, u, 0, Q, R, P);
        const double eps = 1e-5;
        const double f = c.evaluate(x, u, 0);
        Vector xp = x, xm = x, up = u, um = u;
        xp(0) += eps;
        xm(0) -= eps;
        up(0) += eps;
        um(0) -= eps;
        const double fp = c.evaluate(xp, u, 0), fm = c.evaluate(xm, u, 0);
        const double gp = c.evaluate(x, up, 0), gm = c.evaluate(x, um, 0);
        EXPECT_NEAR(c.q()(0), (fp - fm) / (2 * eps), 1e-8);
        EXPECT_NEAR(c.r()(0), (gp - gm) / (2 * eps), 1e-8);
        EXPECT_NEAR(Q(0, 0), (fp - 2 * f + fm) / (eps * eps), 1e-4);
        EXPECT_NEAR(R(0, 0), (gp - 2 * f + gm) / (eps * eps), 1e-4);
        EXPECT_DOUBLE_EQ(P(0, 0), 0);
    }
    x << 0;
    u << 0;
    c.update_lam(x, u, 0);
    EXPECT_DOUBLE_EQ(c.getStateMultiplier()(0), 0);
    EXPECT_DOUBLE_EQ(c.getInputMultiplier()(0), 0);
    x << -2;
    u << 2;
    c.update_lam(x, u, 0);
    EXPECT_DOUBLE_EQ(c.getStateMultiplier()(0), -4);
    EXPECT_DOUBLE_EQ(c.getInputMultiplier()(0), 4);
    EXPECT_GT(c.multiplierResidual(x, u, 0), 0);
}

TEST(IlqrBoundCost, InfiniteEqualityAndTerminalBounds)
{
    Vector lb(4), ub(4), x(4), u(4);
    lb << -inf, 0, -inf, 2;
    ub << inf, inf, 1, 2;
    x << 20, -.5, 1.5, 10;
    u = x;
    Access::BoundCost c(1, lb, ub, lb, ub);
    c.setRho(2);
    Matrix Q = Matrix::Zero(4, 4), R = Q, P = Q;
    EXPECT_DOUBLE_EQ(c.evaluate(x, u, 0), 1);
    c.quadratize(x, u, 0, Q, R, P);
    Vector expected(4);
    expected << 0, -1, 1, 0;
    EXPECT_LT((c.q() - expected).norm(), 1e-14);
    EXPECT_LT((c.r() - expected).norm(), 1e-14);
    EXPECT_DOUBLE_EQ(Q(0, 0), 0);
    EXPECT_DOUBLE_EQ(Q(3, 3), 0);
    c.update_lam(x, u, 0);
    // Terminal cost includes only state bounds and resets the input gradient.
    Q.setZero();
    R.setZero();
    c.quadratize(x, u, 1, Q, R, P);
    EXPECT_DOUBLE_EQ(R.norm(), 0);
    EXPECT_DOUBLE_EQ(c.r().norm(), 0);
    EXPECT_DOUBLE_EQ(c.evaluate(x, u, 1), 1.5);
    c.setRho(0);
    Q.setZero();
    R.setZero();
    c.quadratize(x, u, 0, Q, R, P);
    EXPECT_DOUBLE_EQ(c.evaluate(x, u, 0), 0);
    EXPECT_DOUBLE_EQ(c.q().norm() + c.r().norm() + Q.norm() + R.norm(), 0);
}

// Independent exhaustive active-set QP oracle. The horizon is deliberately
// small so every combination of active lower/upper bounds can be checked.
Vector boxQp(const Matrix &H, const Vector &g, const Matrix &E,
             const Vector &lb, const Vector &ub)
{
    const int n = g.size(), m = lb.size();
    int combinations = 1;
    for (int i = 0; i < m; ++i)
    {
        combinations *= 3;
    }
    Vector best;
    double best_cost = inf;
    for (int code = 0; code < combinations; ++code)
    {
        std::vector<int> rows, signs;
        int digits = code;
        for (int j = 0; j < m; ++j, digits /= 3)
        {
            if (digits % 3)
            {
                rows.push_back(j);
                signs.push_back(digits % 3 == 1 ? -1 : 1);
            }
        }
        const int r = rows.size();
        if (r > n)
        {
            continue;
        }
        Matrix K = Matrix::Zero(n + r, n + r);
        K.topLeftCorner(n, n) = H;
        Vector rhs(n + r);
        rhs.head(n) = -g;
        for (int j = 0; j < r; ++j)
        {
            K.block(n + j, 0, 1, n) = signs[j] * E.row(rows[j]);
            K.block(0, n + j, n, 1) = signs[j] * E.row(rows[j]).transpose();
            rhs(n + j) = signs[j] * (signs[j] < 0 ? lb(rows[j]) : ub(rows[j]));
        }
        Eigen::FullPivLU<Matrix> lu(K);
        if (lu.rank() != K.rows())
        {
            continue;
        }
        Vector solution = lu.solve(rhs), v = solution.head(n), y = E * v;
        if ((y.array() < lb.array() - 1e-9).any() ||
            (y.array() > ub.array() + 1e-9).any())
        {
            continue;
        }
        if (r && solution.tail(r).minCoeff() < -1e-9)
        {
            continue;
        }
        const double cost = .5 * v.dot(H * v) + g.dot(v);
        if (cost < best_cost)
        {
            best_cost = cost;
            best = v;
        }
    }
    EXPECT_EQ(best.size(), n);
    return best;
}

class IlqrBounds : public testing::TestWithParam<
                       std::tuple<double, bool, bool, bool, std::string>>
{
};
TEST_P(IlqrBounds, MatchesExhaustiveBoxQp)
{
    const auto [target, kkt, filter, infeasible, strategy] = GetParam();
    constexpr int N = 3;
    const double initial = .2;
    auto x = casadi::SX::sym("x"), u = casadi::SX::sym("u");
    horizon::IterativeLQR s(
        casadi::Function("dyn", {x, u}, {x + u}, {"x", "u"}, {"f"}), N,
        {{"ilqr.globalization", strategy},
         {"ilqr.stationarity_threshold", 1e-9},
         {"ilqr.enable_auglag", 1},
         {"ilqr.use_kkt_solver", int(kkt)},
         {"ilqr.use_filter", int(filter)},
         {"ilqr.n_threads", filter ? 2 : 0},
         {"ilqr.constraint_violation_threshold", 1e-8},
         {"ilqr.merit_der_threshold", 1e-12},
         {"ilqr.auglag_inner_tolerance", 1e-13}});
    s.setCost({0, 1, 2},
              casadi::Function(
                  "cost", {x, u},
                  {.5 * ((x - .1) * (x - .1) + .3 * (u + .05) * (u + .05))},
                  {"x", "u"}, {"l"}));
    s.setFinalCost(casadi::Function(
        "final", {x, u}, {2 * (x - target) * (x - target)}, {"x", "u"}, {"l"}));
    Matrix xlb = Matrix::Constant(1, N + 1, -.45),
           xub = Matrix::Constant(1, N + 1, .65);
    xlb(0, 0) = xub(0, 0) = initial;
    s.setStateBounds(xlb, xub);
    s.setInputBounds(Matrix::Constant(1, N, -.3), Matrix::Constant(1, N, .4));
    if (infeasible)
    {
        s.setStateInitialGuess(Matrix::Constant(1, N + 1, 2));
        s.setInputInitialGuess(Matrix::Constant(1, N, -2));
    }
    ASSERT_TRUE(s.solve(300))
        << "x=" << s.getStateTrajectory() << " u=" << s.getInputTrajectory()
        << " rho=" << Access::rho(s) << " lx=" << Access::stateDual(s)
        << " lu=" << Access::inputDual(s);
    Matrix S = Matrix::Zero(N + 1, N);
    for (int k = 1; k <= N; ++k)
    {
        S.row(k).head(k).setOnes();
    }
    Vector w = Vector::Constant(N + 1, 1. / N);
    w(N) = 4;
    Vector ref = Vector::Constant(N + 1, .1);
    ref(N) = target;
    Matrix H =
        S.transpose() * w.asDiagonal() * S + .3 / N * Matrix::Identity(N, N);
    Vector g = S.transpose() * w.asDiagonal() *
            (Vector::Constant(N + 1, initial) - ref) +
        Vector::Constant(N, .3 / N * .05);
    Matrix E(2 * N, N);
    E.topRows(N) = S.bottomRows(N);
    E.bottomRows(N).setIdentity();
    Vector lb(2 * N), ub(2 * N);
    lb.head(N).setConstant(-.45 - initial);
    ub.head(N).setConstant(.65 - initial);
    lb.tail(N).setConstant(-.3);
    ub.tail(N).setConstant(.4);
    Vector expected = boxQp(H, g, E, lb, ub);
    ASSERT_EQ(expected.size(), N);
    EXPECT_LT((s.getInputTrajectory().transpose() - expected).norm(), 2e-6);
    EXPECT_LT((s.getStateTrajectory().transpose() -
               (Vector::Constant(N + 1, initial) + S * expected))
                  .norm(),
              2e-6);
    EXPECT_LE(Access::violation(s), 1e-8);
    const Vector lx = Access::stateDual(s).row(0).tail(N).transpose();
    const Vector lu = Access::inputDual(s).row(0).transpose();
    EXPECT_LT((H * expected + g + S.bottomRows(N).transpose() * lx + lu).norm(),
              3e-6);
    Vector y = E * s.getInputTrajectory().transpose(), dual(2 * N);
    dual << lx, lu;
    for (int j = 0; j < 2 * N; ++j)
    {
        if (dual(j) > 0)
        {
            EXPECT_NEAR(dual(j) * (y(j) - ub(j)), 0, 1e-7);
        }
        else
        {
            EXPECT_NEAR(dual(j) * (y(j) - lb(j)), 0, 1e-7);
        }
    }
    EXPECT_LE(Access::rho(s), 1e8);
    // Identical bounds preserve warm-start duals.
    const Matrix active_dual = Access::stateDual(s);
    s.setStateBounds(xlb, xub);
    EXPECT_DOUBLE_EQ((active_dual - Access::stateDual(s)).norm(), 0);
    // A new objective with unchanged bounds must release stale active duals.
    s.setFinalCost(
        casadi::Function("final", {x, u}, {2 * x * x}, {"x", "u"}, {"l"}));
    ASSERT_TRUE(s.solve(150));
    ref(N) = 0;
    g = S.transpose() * w.asDiagonal() *
            (Vector::Constant(N + 1, initial) - ref) +
        Vector::Constant(N, .3 / N * .05);
    EXPECT_LT((s.getInputTrajectory().transpose() + H.ldlt().solve(g)).norm(),
              2e-6);
    EXPECT_LT(Access::stateDual(s).norm() + Access::inputDual(s).norm(), 1e-7);
    // Identical bounds preserve warm-start duals; removed bounds clear them.
    const Matrix old = Access::stateDual(s);
    s.setStateBounds(xlb, xub);
    EXPECT_DOUBLE_EQ((old - Access::stateDual(s)).norm(), 0);
    xlb.setConstant(-inf);
    xub.setConstant(inf);
    xlb(0, 0) = xub(0, 0) = initial;
    s.setStateBounds(xlb, xub);
    s.setInputBounds(Matrix::Constant(1, N, -inf), Matrix::Constant(1, N, inf));
    EXPECT_DOUBLE_EQ(Access::stateDual(s).norm() + Access::inputDual(s).norm(),
                     0);
    ASSERT_TRUE(s.solve(100));
    EXPECT_LT((s.getInputTrajectory().transpose() + H.ldlt().solve(g)).norm(),
              2e-6);
}
INSTANTIATE_TEST_SUITE_P(
    Backends, IlqrBounds,
    testing::Combine(testing::Values(-1.3, .3, 1.4), testing::Bool(),
                     testing::Bool(), testing::Bool(),
                     testing::Values(std::string("legacy"),
                                     std::string("lm_merit"),
                                     std::string("filter_sqp"))));

TEST(IlqrBounds, RejectsInvalidBoundsAndPenaltyOptions)
{
    auto x = casadi::SX::sym("x"), u = casadi::SX::sym("u");
    auto dyn = casadi::Function("dyn", {x, u}, {x + u}, {"x", "u"}, {"f"});
    for (const auto &option :
         horizon::IterativeLQR::OptionDict{{"ilqr.rho_base", 0.},
                                           {"ilqr.rho_max", .5},
                                           {"ilqr.rho_growth_factor", 1.},
                                           {"ilqr.auglag_progress_ratio", 1.},
                                           {"ilqr.auglag_inner_tolerance", 0.}})
    {
        horizon::IterativeLQR::OptionDict options{{"ilqr.enable_auglag", 1}};
        options.insert(option);
        EXPECT_THROW(horizon::IterativeLQR(dyn, 1, options),
                     std::invalid_argument);
    }
    horizon::IterativeLQR s(dyn, 1);
    EXPECT_THROW(s.setStateBounds(Matrix::Ones(1, 2), Matrix::Zero(1, 2)),
                 std::invalid_argument);
    EXPECT_THROW(s.setInputBounds(Matrix::Constant(1, 1, inf),
                                  Matrix::Constant(1, 1, inf)),
                 std::invalid_argument);
    EXPECT_THROW(
        s.setInputBounds(Matrix::Constant(1, 1, NAN), Matrix::Zero(1, 1)),
        std::invalid_argument);
}

TEST(IlqrBounds, InfeasibleInequalitiesRespectPenaltyCapAndDoNotReportSuccess)
{
    auto x = casadi::SX::sym("x"), u = casadi::SX::sym("u");
    horizon::IterativeLQR s(
        casadi::Function("dyn", {x, u}, {x + u}, {"x", "u"}, {"f"}), 1,
        {{"ilqr.enable_auglag", 1}, {"ilqr.rho_max", 4.0}});
    Matrix lb(1, 2), ub(1, 2);
    lb << 0, 1;
    ub << 0, inf;
    s.setStateBounds(lb, ub);
    s.setInputBounds(Matrix::Zero(1, 1), Matrix::Zero(1, 1));
    EXPECT_FALSE(s.solve(30));
    EXPECT_DOUBLE_EQ(Access::rho(s), 4);
    EXPECT_DOUBLE_EQ(Access::violation(s), 1);
    EXPECT_TRUE(s.getStateTrajectory().allFinite());
    EXPECT_TRUE(Access::stateDual(s).allFinite());
}

TEST(IlqrBounds, ChangingBoundsClearsOnlyAffectedMultipliers)
{
    auto x = casadi::SX::sym("x"), u = casadi::SX::sym("u");
    horizon::IterativeLQR s(
        casadi::Function("dyn", {x, u}, {x + u}, {"x", "u"}, {"f"}), 2,
        {{"ilqr.enable_auglag", 1},
         {"ilqr.constraint_violation_threshold", 1e-8},
         {"ilqr.merit_der_threshold", 1e-12}});
    s.setCost(
        {0, 1},
        casadi::Function("effort", {x, u}, {.5 * u * u}, {"x", "u"}, {"l"}));
    s.setFinalCost(casadi::Function("final", {x, u}, {10 * (x - 2) * (x - 2)},
                                    {"x", "u"}, {"l"}));
    Matrix lb = Matrix::Constant(1, 3, -inf), ub = -lb;
    lb(0, 0) = ub(0, 0) = 0;
    ub(0, 2) = .5;
    s.setStateBounds(lb, ub);
    Matrix ulb = Matrix::Constant(1, 2, -inf), uub = -ulb;
    uub(0, 0) = .2;
    s.setInputBounds(ulb, uub);
    ASSERT_TRUE(s.solve(200))
        << "x=" << s.getStateTrajectory() << " u=" << s.getInputTrajectory()
        << " rho=" << Access::rho(s) << " lx=" << Access::stateDual(s)
        << " lu=" << Access::inputDual(s);
    ASSERT_GT(Access::stateDual(s)(0, 2), 1);
    ASSERT_GT(Access::inputDual(s)(0, 0), .01);
    const Matrix input_dual = Access::inputDual(s);
    ub(0, 2) = inf;
    s.setStateBounds(lb, ub);
    EXPECT_DOUBLE_EQ(Access::stateDual(s).norm(), 0);
    EXPECT_DOUBLE_EQ((Access::inputDual(s) - input_dual).norm(), 0);
    uub(0, 0) = inf;
    s.setInputBounds(ulb, uub);
    EXPECT_DOUBLE_EQ(Access::inputDual(s).norm(), 0);
}

TEST(IlqrModelingAPI, SelectionsBroadcastingAndFixedInitialState)
{
    using casadi::SX;
    using horizon::IterativeLQR;
    using horizon::Nodes;
    constexpr int N = 3;
    auto x = SX::sym("x"), u = SX::sym("u");
    casadi::Function dynamics("api_dynamics", {x, u}, {x + u}, {"x", "u"},
                              {"f"});
    casadi::Function effort("api_effort", {x, u}, {.2 * u}, {"x", "u"},
                            {"res"});
    casadi::Function target("api_target", {x, u}, {x - .8}, {"x", "u"},
                            {"res"});
    IterativeLQR explicit_api(dynamics, N), concise_api(dynamics, N);
    for (auto *solver : {&explicit_api, &concise_api})
    {
        solver->setIterationCallback(
            [](const auto &)
            {
                return true;
            });
    }
    explicit_api.setResidual({0, 1, 2}, effort);
    explicit_api.setResidual({3}, target);
    concise_api.setResidual(Nodes::running(), effort);
    concise_api.setResidual(Nodes::terminal(), target);
    concise_api.setIndices("api_effort", Nodes::running());
    EXPECT_EQ(Nodes::all().resolve(N), (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(Nodes::running().resolve(N), (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(Nodes::terminal().resolve(N), (std::vector<int>{3}));
    Matrix lower = Matrix::Constant(1, N + 1, -1);
    Matrix upper = Matrix::Constant(1, N + 1, 1);
    lower(0, 0) = upper(0, 0) = .2;
    explicit_api.setStateBounds(lower, upper);
    concise_api.setStateBounds(Vector::Constant(1, -1), Vector::Constant(1, 1));
    concise_api.fixInitialState(Vector::Constant(1, .2));
    explicit_api.setInputBounds(Matrix::Constant(1, N, -.3),
                                Matrix::Constant(1, N, .3));
    concise_api.setInputBounds(-.3, .3);
    explicit_api.setStateInitialGuess(Matrix::Constant(1, N + 1, .4));
    concise_api.setStateInitialGuess(Vector::Constant(1, .4));
    explicit_api.setInputInitialGuess(Matrix::Zero(1, N));
    concise_api.setInputInitialGuess(Vector::Zero(1));
    EXPECT_TRUE(explicit_api.getStateTrajectory().isApprox(
        concise_api.getStateTrajectory()));
    // setInitialState remains a guess operation; fixed bounds survive it.
    explicit_api.setInitialState(Vector::Constant(1, .5));
    concise_api.setInitialState(Vector::Constant(1, .5));
    EXPECT_DOUBLE_EQ(concise_api.state(0)[0], .5);
    EXPECT_EQ(explicit_api.solve(40), concise_api.solve(40));
    EXPECT_NEAR(concise_api.state(0)[0], .2, 1e-8);
    EXPECT_TRUE(explicit_api.getStateTrajectory().isApprox(
        concise_api.getStateTrajectory(), 1e-10));
    EXPECT_TRUE(explicit_api.getInputTrajectory().isApprox(
        concise_api.getInputTrajectory(), 1e-10));
    EXPECT_THROW(concise_api.fixInitialState(Vector::Zero(2)),
                 std::invalid_argument);
    EXPECT_THROW(concise_api.setStateBounds(Vector::Zero(2), Vector::Ones(2)),
                 std::invalid_argument);
}
