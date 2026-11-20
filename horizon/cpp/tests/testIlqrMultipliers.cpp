#include "ilqr/ilqr_impl.h"
#include <gtest/gtest.h>

namespace horizon
{
struct IterativeLQRTestAccess
{
    struct Step
    {
        Eigen::MatrixXd dx, du, costate;
        std::vector<Eigen::VectorXd> equality;
        double bound_max;
    };

    static Step step(IterativeLQR &s, bool baseline)
    {
        s.linearize_quadratize();
        if (baseline)
        {
            s.kkt_solve();
        }
        else
        {
            s.backward_pass();
        }
        return {s._dx, s._du, s._lam_x, s._lam_g, s._bound_multiplier_max};
    }

    // Check the original, untransformed QP equations independently of the
    // constraint decomposition and value-function recursion.
    static void stationarity(IterativeLQR &s, const Step &p)
    {
        double bound_max = 0;
        for (int k = 0; k <= s._N; ++k)
        {
            SCOPED_TRACE(k);
            Eigen::VectorXd gx = s._cost[k].q() + s._cost[k].Q() * p.dx.col(k) +
                s._hxx_reg * p.dx.col(k);
            if (p.equality[k].size())
            {
                gx += s._constraint[k].C().transpose() * p.equality[k];
            }
            if (k < s._N)
            {
                gx += s._cost[k].P().transpose() * p.du.col(k) +
                    s._dyn[k].A().transpose() * p.costate.col(k);
                Eigen::VectorXd gu = s._cost[k].r() +
                    s._cost[k].R() * p.du.col(k) + s._huu_reg * p.du.col(k) +
                    s._cost[k].P() * p.dx.col(k) +
                    s._dyn[k].B().transpose() * p.costate.col(k);
                if (p.equality[k].size())
                {
                    gu += s._constraint[k].D().transpose() * p.equality[k];
                }
                for (int j = 0; j < s._nu; ++j)
                {
                    if (s._u_lb(j, k) == s._u_ub(j, k))
                    {
                        EXPECT_NEAR(p.du(j, k), s._u_lb(j, k) - s._utrj(j, k),
                                    1e-9);
                        bound_max = std::max(bound_max, std::abs(gu(j)));
                        gu(j) = 0;
                    }
                }
                EXPECT_LT(gu.norm(), 1e-9);
                EXPECT_LT((s._dyn[k].A() * p.dx.col(k) +
                           s._dyn[k].B() * p.du.col(k) + s._dyn[k].d -
                           p.dx.col(k + 1))
                              .norm(),
                          1e-9);
            }
            if (k > 0)
            {
                gx -= p.costate.col(k - 1);
            }
            Eigen::MatrixXd C;
            Eigen::VectorXd h;
            s.stateEquality(k, C, h);
            if (h.size())
            {
                EXPECT_LT((C * p.dx.col(k) + h).norm(), 1e-9);
                Eigen::VectorXd bound_dual =
                    C.transpose().colPivHouseholderQr().solve(-gx);
                gx += C.transpose() * bound_dual;
                bound_max =
                    std::max(bound_max, bound_dual.cwiseAbs().maxCoeff());
            }
            EXPECT_LT(gx.norm(), 1e-9);
            if (p.equality[k].size())
            {
                Eigen::VectorXd h =
                    s._constraint[k].h() + s._constraint[k].C() * p.dx.col(k);
                if (k < s._N)
                {
                    h += s._constraint[k].D() * p.du.col(k);
                }
                EXPECT_LT(h.norm(), 1e-9);
            }
        }
        EXPECT_NEAR(p.bound_max, bound_max, 1e-9);
    }
};
} // namespace horizon

class IlqrMultipliers
    : public testing::TestWithParam<
          std::tuple<int, std::string, std::string, bool, double>>
{
};

TEST_P(IlqrMultipliers, MatchesBaselineKkt)
{
    const auto [scenario, constraint_decomp, kkt_decomp, fixed, reg] =
        GetParam();
    constexpr int N = 5;
    auto x = casadi::SX::sym("x", 2), u = casadi::SX::sym("u", 2);
    // Rank-one B forces terminal position constraints to propagate two stages.
    auto dyn = casadi::Function(
        "dyn", {x, u},
        {casadi::SX::vertcat(
            {x(0) + .7 * x(1) + .13, .8 * x(1) + u(0) + .3 * u(1) - .21})},
        {"x", "u"}, {"f"});
    horizon::IterativeLQR s(dyn, N,
                            {{"ilqr.constr_decomp_type", constraint_decomp},
                             {"ilqr.kkt_decomp_type", kkt_decomp},
                             {"ilqr.hxx_reg", reg},
                             {"ilqr.huu_reg", reg * .7}});
    auto cost = casadi::Function(
        "cost", {x, u},
        {.5 * (casadi::SX::sumsqr(x) + casadi::SX::sumsqr(u)) +
         .12 * x(0) * u(1) + .4 * x(0) - .3 * x(1) + .6 * u(0) - .2 * u(1)},
        {"x", "u"}, {"l"});
    s.setCost({0, 1, 2, 3, 4}, cost);
    s.setFinalCost(
        casadi::Function("terminal", {x, u},
                         {1.3 * casadi::SX::sumsqr(x) + .7 * x(0) - .8 * x(1)},
                         {"x", "u"}, {"l"}));
    auto constrain = [&](std::vector<int> nodes, const casadi::SX &h,
                         const std::string &name)
    {
        s.setConstraint(nodes,
                        casadi::Function(name, {x, u}, {h}, {"x", "u"}, {"h"}));
    };
    if (scenario == 1)
    {
        constrain({0, 2},
                  casadi::SX::vertcat({x(0) + 2 * u(0) + u(1) - .3,
                                       .4 * x(1) - u(0) + 3 * u(1) + .5}),
                  "mixed");
    }
    if (scenario == 2)
    {
        constrain({N}, x - casadi::SX::vertcat({.2, -.4}), "terminal_eq");
        constrain({2}, x(0) + .17, "state_eq");
    }
    if (scenario == 3)
    {
        // Nonzero, dependent D rows exercise the Q2 propagation path.
        constrain({2},
                  casadi::SX::vertcat({x(0) + u(0) + .3 * u(1) - .2,
                                       x(1) + 2 * u(0) + .6 * u(1) + .4}),
                  "rank_deficient");
        constrain({N}, x(0) - .3, "terminal_eq");
    }
    if (scenario == 4)
    {
        Eigen::MatrixXd lb = Eigen::MatrixXd::Constant(2, N + 1, -INFINITY),
                        ub = -lb;
        lb(0, N) = ub(0, N) = .2;
        s.setStateBounds(lb, ub);
        lb = Eigen::MatrixXd::Constant(2, N, -INFINITY);
        ub = -lb;
        lb(1, 1) = ub(1, 1) = -.15;
        s.setInputBounds(lb, ub);
        constrain({3}, x(1) + u(0) - .1, "mixed");
    }
    if (fixed)
    {
        Eigen::MatrixXd lb = Eigen::MatrixXd::Constant(2, N + 1, -INFINITY),
                        ub = -lb;
        if (scenario == 4)
        {
            lb(0, N) = ub(0, N) = .2;
        }
        lb.col(0) = ub.col(0) = Eigen::Vector2d(.6, -.7);
        s.setStateBounds(lb, ub);
    }
    Eigen::MatrixXd xs(2, N + 1), us(2, N);
    for (int k = 0; k <= N; ++k)
    {
        xs.col(k) << .1 + .13 * k, -.2 + .07 * k;
    }
    for (int k = 0; k < N; ++k)
    {
        us.col(k) << -.1 + .04 * k, .2 - .09 * k;
    }
    s.setStateInitialGuess(xs);
    s.setInputInitialGuess(us);
    using Access = horizon::IterativeLQRTestAccess;
    const auto actual = Access::step(s, false),
               reference = Access::step(s, true);
    EXPECT_LT((actual.dx - reference.dx).norm(), 1e-9);
    EXPECT_LT((actual.du - reference.du).norm(), 1e-9);
    EXPECT_LT((actual.costate - reference.costate).norm(), 1e-9);
    ASSERT_EQ(actual.equality.size(), reference.equality.size());
    for (int k = 0; k <= N; ++k)
    {
        SCOPED_TRACE(k);
        ASSERT_EQ(actual.equality[k].size(), reference.equality[k].size());
        EXPECT_LT((actual.equality[k] - reference.equality[k]).norm(), 1e-9);
    }
    EXPECT_NEAR(actual.bound_max, reference.bound_max, 1e-9);
    Access::stationarity(s, reference);
    Access::stationarity(s, actual);
    // Reuse workspaces with different affine terms and constraint row counts.
    if (scenario == 1 || scenario == 3)
    {
        s.setIndices(scenario == 1 ? "mixed" : "rank_deficient", {1});
    }
    s.updateIndices();
    xs.array() += .19;
    us.array() -= .11;
    s.setStateInitialGuess(xs);
    s.setInputInitialGuess(us);
    const auto repeated = Access::step(s, false),
               repeated_reference = Access::step(s, true);
    EXPECT_LT((repeated.costate - repeated_reference.costate).norm(), 1e-9);
    for (int k = 0; k <= N; ++k)
    {
        ASSERT_EQ(repeated.equality[k].size(),
                  repeated_reference.equality[k].size());
        EXPECT_LT(
            (repeated.equality[k] - repeated_reference.equality[k]).norm(),
            1e-9);
    }
    Access::stationarity(s, repeated);
}

INSTANTIATE_TEST_SUITE_P(Decompositions, IlqrMultipliers,
                         testing::Combine(testing::Values(0, 1, 2, 3, 4),
                                          testing::Values("qr", "svd", "cod"),
                                          testing::Values("lu", "qr", "ldlt"),
                                          testing::Bool(),
                                          testing::Values(0., .08)));
