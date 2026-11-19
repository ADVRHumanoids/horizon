// Fixed-foot whole-body inverse kinematics on a floating-base robot.
#include "cli.h"
#include "ilqr/casadi_function.h"
#include "ilqr/ilqr.h"
#include "support.h"
#include <iostream>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>

int main(int argc, char **argv)
{
    // Choose the horizon, solver settings and optional asset paths.
    const auto options = examples::parseOptions(argc, argv, 30, 80, 0., false);
    using namespace horizon::ilqr;
    using namespace horizon::kinematics;
    using casadi::MX;
    const int N = options.nodes;
    auto model = std::make_shared<pinocchio::Model>();
    pinocchio::urdf::buildModel(options.urdf, *model);
    auto space = std::make_shared<PinocchioConfiguration>(model);
    Vector start = examples::standing(*model);
    auto frames = std::make_shared<PinocchioFramePositions>(
        model, space,
        std::vector<std::string>{"contact_1", "contact_2", "contact_3",
                                 "contact_4", "dagana_1_base"});
    Vector initial(15);
    frames->evaluate(start, initial);

    // Compose native robot operations with symbolic dynamics and costs.
    SymbolicFunctionManager functions;
    auto positions = functions.addNative("frames", frames);
    auto q = MX::sym("q", model->nq), v = MX::sym("v", model->nv);

    // Frame coordinates use SoA order: all x, then all y, then all z.
    auto p = positions(q);
    std::vector<MX> feet;
    for (int i = 0; i < 4; ++i)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            feet.push_back(p(axis * 5 + i) - initial[axis * 5 + i]);
        }
    }
    auto tool = MX::vertcat({p(4), p(9), p(14)});
    Vector target(3);
    target << initial[4] + .04, initial[9] + .02, initial[14] + .01;
    MX reference(
        casadi::DM(std::vector<double>(target.data(), target.data() + 3)));
    functions.add("dynamics",
                  std::make_shared<PinocchioStep>(space, options.dt));
    functions.addExpression("feet", {q, v}, MX::vertcat(feet), false);
    functions.addResidual("effort", {q, v}, .05 * v);
    functions.addResidual("tracking", {q, v}, 100 * (tool - reference));

    // Configure the solver and choose which nodes carry each term.
    horizon::IterativeLQR solver(
        functions, "dynamics", N,
        {{"ilqr.n_threads", options.threads},
         {"ilqr.constr_decomp_type", std::string("svd")},
         {"ilqr.huu_reg", .001 / N},
         {"ilqr.constraint_violation_threshold", 1e-7},
         {"ilqr.defect_norm_threshold", 1e-7}},
        space);
    solver.setResidual(horizon::Nodes::running(), functions, "effort");
    solver.setResidual(horizon::Nodes::terminal(), functions, "tracking");

    // Anchor all four feet while the tool moves.
    solver.setConstraint(horizon::Nodes::all(), functions, "feet");
    solver.setStateBounds(model->lowerPositionLimit, model->upperPositionLimit);
    solver.fixInitialState(start);
    solver.setInputBounds(-1, 1);

    // Solve and report convergence.
    examples::reportIterations(solver, options);
    solver.solve(options.iterations);

    // Check the resulting tool motion and foot drift.
    double drift = 0;
    Vector actual(15);
    for (int k = 0; k <= N; ++k)
    {
        frames->evaluate(solver.state(k), actual);
        for (int i = 0; i < 4; ++i)
        {
            Eigen::Vector3d delta(actual[i] - initial[i],
                                  actual[5 + i] - initial[5 + i],
                                  actual[10 + i] - initial[10 + i]);
            drift = std::max(drift, delta.norm());
        }
    }
    Eigen::Vector3d final_tool(actual[4], actual[9], actual[14]);
    double error = (final_tool - target).norm();
    std::cout << "Tool error: " << error << "; foot drift: " << drift << '\n';
    return error < .005 && drift < 1e-4 ? 0 : 1;
}
