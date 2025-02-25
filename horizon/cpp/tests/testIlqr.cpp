#include <gtest/gtest.h>

#include "../src/ilqr.h"


class testIlqr : public ::testing::Test
{
protected:

    testIlqr(){

    }

    virtual ~testIlqr() {

    }

    virtual void SetUp() {

        x = casadi::SX::sym("x", 2);
        u = casadi::SX::sym("u", 1);

        auto xnext = casadi::SX::vertcat(
                         {
                             x(0) + x(1),
                             x(0) - x(1) + u
                         }
                         );

        dyn = casadi::Function("dyn", {x, u}, {xnext},
                               {"x", "u"}, {"f"});
    }

    virtual void TearDown() {

    }

    casadi::SX x, u;
    casadi::Function dyn;

};

TEST_F(testIlqr, checkResiduals)
{
    casadi::SX res = casadi::SX::vertcat(
                         {x(0) - u + 1, u - 1});

    auto res_fn = casadi::Function("res", {x, u},
                                   {res},
                                   {"x", "u"}, {"res"});

    auto cost_fn = casadi::Function("cost", {x, u},
                                   {casadi::SX::sumsqr(res)},
                                   {"x", "u"}, {"l"});

    casadi::SX xf = x;
    auto fc = casadi::Function("fc", {x, u},
                               {xf},
                               {"x", "u"}, {"h"});

    horizon::IterativeLQR ilqr_res(dyn, 10);

    horizon::IterativeLQR ilqr_cost(dyn, 10);

    ilqr_res.setResidual({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, res_fn);

    ilqr_cost.setCost({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, cost_fn);

    for(auto ilqr : {&ilqr_res, &ilqr_cost})
    {
        ilqr->setConstraint({10}, fc);

        Eigen::Vector2d x0(1, 1);

        ilqr->setInitialState(x0);

        ilqr->setIterationCallback([](const auto& fpres) { fpres.print(); return true; });

        ilqr->solve(1);

        std::cout << ilqr->getStateTrajectory() << "\n\n";
    }




}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
