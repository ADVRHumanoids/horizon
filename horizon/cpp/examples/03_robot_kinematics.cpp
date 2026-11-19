// Replace symbolic kinematics with a native Pinocchio frame position.
#include "cli.h"
#include "horizon/pinocchio_kinematics.h"
#include "ilqr/casadi_function.h"
#include "ilqr/ilqr.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <pinocchio/parsers/urdf.hpp>

int main(int argc, char **argv)
{
    // Choose the horizon, solver settings and optional asset paths.
    const auto options = examples::parseOptions(argc, argv, 30, 50, .1, true);
    using namespace horizon::ilqr;
    using namespace horizon::kinematics;
    using casadi::MX;
    auto model = std::make_shared<pinocchio::Model>();
    pinocchio::urdf::buildModel(options.urdf, *model);
    auto space = std::make_shared<PinocchioConfiguration>(model);

    // The native frame evaluator provides both tool position and its Jacobian.
    auto position = std::make_shared<PinocchioFramePositions>(
        model, space, std::vector<std::string>{"tool"});

    // Compose native robot operations with symbolic dynamics and costs.
    SymbolicFunctionManager functions;
    auto fk = functions.addNative("tool_position", position);
    auto q = MX::sym("q", model->nq), u = MX::sym("u", model->nv);
    auto target = MX::sym("target", 3);
    functions.add("dynamics",
                  std::make_shared<PinocchioStep>(space, options.dt));
    functions.addResidual("effort", {q, u}, 0.1 * u);
    functions.addResidual("tracking", {q, u, target}, 10 * (fk(q) - target));
    const int N = options.nodes;

    // Configure the solver and choose which nodes carry each term.
    horizon::IterativeLQR solver(functions, "dynamics", N,
                                 {{"ilqr.n_threads", options.threads}}, space);
    std::vector<int> running;
    for (int k = 0; k < N; ++k)
    {
        running.push_back(k);
    }
    solver.setResidual(running, functions, "effort");
    solver.setResidual({N}, functions, "tracking");
    Vector goal(2), destination(3);
    goal << 0.5, 0.5;
    position->evaluate(goal, destination);
    solver.setParameterValue("target", destination.replicate(1, N + 1));
    Matrix lower = Matrix::Constant(model->nq, N + 1,
                                    -std::numeric_limits<double>::infinity());
    Matrix upper = -lower;
    lower.col(0).setZero();
    upper.col(0).setZero();
    solver.setStateBounds(lower, upper);

    // Solve and report convergence.
    examples::reportIterations(solver, options);
    solver.solve(options.iterations);

    // Check the final tool-position error.
    Vector result(3);
    position->evaluate(solver.state(N), result);
    std::cout << "Tool error: " << (result - destination).norm() << '\n';
    return (result - destination).norm() < 0.01 ? 0 : 1;
}
