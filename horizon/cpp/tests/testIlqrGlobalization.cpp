#include <gtest/gtest.h>
#include "ilqr/ilqr_impl.h"
#include <cmath>

namespace horizon {
struct IterativeLQRTestAccess
{
    static std::pair<double,double> correction(IterativeLQR& s)
    {
        s._fp_res->cost=s.compute_cost(s._xtrj,s._utrj);
        s.linearize_quadratize();s.prepareGlobalization();
        Eigen::VectorXd step=Eigen::VectorXd::Zero(6);
        step(2)=step(4)=.2;
        s.evaluateGlobalizationTrial(step);
        auto residual=s.globalizationResidual(s._fp_res->xtrj,s._fp_res->utrj);
        auto correction=s.feasibilityStep(*s._global_model,residual,true);
        EXPECT_EQ(correction.size(),step.size());
        s.evaluateGlobalizationTrial(step+correction);
        return {residual.norm(),s.globalizationResidual(s._fp_res->xtrj,s._fp_res->utrj).norm()};
    }
    static bool restoration(IterativeLQR& s)
    {
        s._fp_res->cost=s.compute_cost(s._xtrj,s._utrj);
        s._fp_res->constraint_violation=s.compute_constr(s._xtrj,s._utrj);
        s._fp_res->defect_norm=s.compute_defect(s._xtrj,s._utrj);
        s._filter_violation_limit=10*(s._fp_res->constraint_violation+s._fp_res->defect_norm);
        s.linearize_quadratize();s.prepareGlobalization();return s.restoreFeasibility();
    }
};
}
using horizon::IterativeLQR;
using Vector=Eigen::VectorXd;
using Matrix=Eigen::MatrixXd;
namespace cs=casadi;

class Globalization : public testing::TestWithParam<std::tuple<std::string,bool>> {};
TEST_P(Globalization, QuadraticAgreementAndBackendParity)
{
    auto [strategy,kkt]=GetParam();
    auto x=cs::SX::sym("x"),u=cs::SX::sym("u");
    IterativeLQR s(cs::Function("dyn",{x,u},{x+u},{"x","u"},{"f"}),3,
        {{"ilqr.globalization",strategy},{"ilqr.use_kkt_solver",int(kkt)},
         {"ilqr.stationarity_threshold",1e-8},{"ilqr.merit_der_threshold",1e-12}});
    s.setCost({0,1,2},cs::Function("effort",{x,u},{.15*u*u},{"x","u"},{"l"}));
    s.setFinalCost(cs::Function("target",{x,u},{.5*(x-1)*(x-1)},{"x","u"},{"l"}));
    s.fixInitialState(Vector::Zero(1));
    int accepted=0;
    s.setIterationCallback([&](const auto& r) {
        if(r.accepted && r.predicted_reduction>1e-12)
        { ++accepted; EXPECT_NEAR(r.model_agreement,1.,1e-7); }
        return true;
    });
    ASSERT_TRUE(s.solve(80));
    EXPECT_GT(accepted,0);
    EXPECT_NEAR(s.input(0)[0],1./3.1,1e-7);
    EXPECT_NEAR(s.state(3)[0],3./3.1,1e-7);
}

TEST_P(Globalization, DampingDoesNotCauseFalseConvergence)
{
    auto [strategy,kkt]=GetParam();
    auto x=cs::SX::sym("x"),u=cs::SX::sym("u");
    IterativeLQR s(cs::Function("dyn",{x,u},{x+u},{"x","u"},{"f"}),2,
        {{"ilqr.globalization",strategy},{"ilqr.use_kkt_solver",int(kkt)},
         {"ilqr.globalization_damping",1e12},{"ilqr.stationarity_threshold",1e-8}});
    s.fixInitialState(Vector::Zero(1));
    s.setFinalCost(cs::Function("target",{x,u},{(x-1)*(x-1)},{"x","u"},{"l"}));
    EXPECT_FALSE(s.solve(1));
    EXPECT_LT(s.state(2)[0],.1);
}

TEST_P(Globalization, BoundActivationAndRelease)
{
    auto [strategy,kkt]=GetParam();
    auto x=cs::SX::sym("x"),u=cs::SX::sym("u");
    IterativeLQR s(cs::Function("dyn",{x,u},{x+u},{"x","u"},{"f"}),2,
        {{"ilqr.globalization",strategy},{"ilqr.use_kkt_solver",int(kkt)},
         {"ilqr.enable_auglag",1},{"ilqr.constraint_violation_threshold",1e-8},
         {"ilqr.stationarity_threshold",1e-8},{"ilqr.merit_der_threshold",1e-12}});
    s.fixInitialState(Vector::Zero(1));s.setInputBounds(-.2,.2);
    s.setCost({0,1},cs::Function("effort",{x,u},{.1*u*u},{"x","u"},{"l"}));
    s.setFinalCost(cs::Function("target",{x,u},{(x-1)*(x-1)},{"x","u"},{"l"}));
    ASSERT_TRUE(s.solve(300));EXPECT_NEAR(s.state(2)[0],.4,1e-7);
    s.setFinalCost(cs::Function("target",{x,u},{x*x},{"x","u"},{"l"}));
    ASSERT_TRUE(s.solve(300));EXPECT_NEAR(s.state(2)[0],0,1e-7);
    EXPECT_LT(s.getInputTrajectory().norm(),1e-7);
}

TEST_P(Globalization, InconsistentEqualitiesFailAndPreserveAcceptedState)
{
    auto [strategy,kkt]=GetParam();
    auto x=cs::SX::sym("x"),u=cs::SX::sym("u");
    IterativeLQR s(cs::Function("dyn",{x,u},{x+u},{"x","u"},{"f"}),2,
        {{"ilqr.globalization",strategy},{"ilqr.use_kkt_solver",int(kkt)}});
    s.fixInitialState(Vector::Zero(1));
    s.setConstraint({2},cs::Function("impossible",{x,u},{cs::SX::vertcat({x-1,x-2})},{"x","u"},{"h"}));
    EXPECT_FALSE(s.solve(20));EXPECT_TRUE(s.getStateTrajectory().allFinite());
    EXPECT_NEAR(s.state(0)[0],0.,1e-12);
}

TEST_P(Globalization, BacktracksNonfiniteTrials)
{
    auto [strategy,kkt]=GetParam();
    auto x=cs::SX::sym("x"),u=cs::SX::sym("u");
    IterativeLQR s(cs::Function("dyn",{x,u},{x+u},{"x","u"},{"f"}),1,
        {{"ilqr.globalization",strategy},{"ilqr.use_kkt_solver",int(kkt)},
         {"ilqr.stationarity_threshold",1e-7}});
    s.fixInitialState(Vector::Ones(1));
    s.setStateInitialGuess(Vector::Ones(1));
    s.setFinalCost(cs::Function("target",{x,u},{cs::SX::pow(cs::SX::sqrt(x)-.1,2)}, {"x","u"},{"l"}));
    int trials=0;
    s.setIterationCallback([&](const auto& r){trials=std::max(trials,r.trial_count);return true;});
    ASSERT_TRUE(s.solve(80));
    EXPECT_GT(trials,1);EXPECT_NEAR(s.state(1)[0],.01,1e-6);
}

INSTANTIATE_TEST_SUITE_P(StrategiesAndBackends,Globalization,
    testing::Combine(testing::Values(std::string("lm_merit"),std::string("filter_sqp")),testing::Bool()));

TEST(GlobalizationFilter, RestorationReducesNonlinearViolation)
{
    auto x=cs::SX::sym("x"),u=cs::SX::sym("u");
    IterativeLQR s(cs::Function("dyn",{x,u},{x+u},{"x","u"},{"f"}),1,
        {{"ilqr.globalization",std::string("filter_sqp")}});
    s.fixInitialState(Vector::Zero(1));
    s.setConstraint({1},cs::Function("curve",{x,u},{x*x-1},{"x","u"},{"h"}));
    Matrix guess(1,2);guess<<0,2;s.setStateInitialGuess(guess);
    s.setInputInitialGuess(Vector::Constant(1,2));
    ASSERT_TRUE(horizon::IterativeLQRTestAccess::restoration(s));
    EXPECT_LT(std::abs(s.state(1)[0]*s.state(1)[0]-1),2.7);
    EXPECT_NEAR(s.state(0)[0],0.,1e-12);
    EXPECT_NEAR(s.state(1)[0],s.input(0)[0],1e-5);
}

TEST(GlobalizationOptions, RejectsUnknownStrategyAndInvalidDamping)
{
    auto x=cs::SX::sym("x"),u=cs::SX::sym("u");
    cs::Function dyn("dyn",{x,u},{x+u},{"x","u"},{"f"});
    EXPECT_THROW(IterativeLQR(dyn,2,{{"ilqr.globalization",std::string("unknown")}}),std::invalid_argument);
    EXPECT_THROW(IterativeLQR(dyn,2,{{"ilqr.globalization_damping",0.}}),std::invalid_argument);
}

TEST(GlobalizationFilter, SecondOrderCorrectionRepairsCurvature)
{
    auto x=cs::SX::sym("x",2),u=cs::SX::sym("u",2);
    IterativeLQR s(cs::Function("dyn",{x,u},{x+u},{"x","u"},{"f"}),1,
        {{"ilqr.globalization",std::string("filter_sqp")}});
    Vector start(2);start<<0,1;s.fixInitialState(start);s.setStateInitialGuess(start);
    s.setConstraint({1},cs::Function("circle",{x,u},{cs::SX::sumsqr(x)-1},{"x","u"},{"h"}));
    const auto [before,after]=horizon::IterativeLQRTestAccess::correction(s);
    EXPECT_NEAR(before,.04,1e-10);EXPECT_LT(after,.001);
}
