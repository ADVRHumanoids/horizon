// Add a parameter so the destination can change without rebuilding the model.
#include "cli.h"
#include "ilqr/casadi_function.h"
#include "ilqr/ilqr.h"
#include <cmath>
#include <iostream>
#include <limits>

int main(int argc, char **argv)
{
    // Choose the horizon, solver settings and optional asset paths.
    const auto options = examples::parseOptions(argc, argv, 30, 30, .1, false);
    using casadi::MX;
    using namespace horizon::ilqr;
    const int N = options.nodes;
    auto x = MX::sym("x", 3), u = MX::sym("u", 2);
    auto target = MX::sym("target", 3);

    // x = [x, y, heading], u = [forward speed, yaw rate]; use Euler
    // integration.
    auto velocity = MX::vertcat({u(0) * cos(x(2)), u(0) * sin(x(2)), u(1)});

    // Define the discrete dynamics and objectives with CasADi.
    SymbolicFunctionManager functions;
    functions.addExpression("dynamics", {x, u}, x + options.dt * velocity,
                            false);
    functions.addResidual("effort", {x, u}, 0.1 * u);
    functions.addResidual("tracking", {x, u, target}, 10 * (x - target));

    // Configure the solver and choose which nodes carry each term.
    horizon::IterativeLQR solver(functions, "dynamics", N,
                                 {{"ilqr.n_threads", options.threads}});
    std::vector<int> running;
    for (int k = 0; k < N; ++k)
    {
        running.push_back(k);
    }
    solver.setResidual(running, functions, "effort");
    solver.setResidual({N}, functions, "tracking");
    Vector goal(3);
    goal << 1, 0, 0;
    solver.setParameterValue("target", goal.replicate(1, N + 1));
    Matrix lower =
        Matrix::Constant(3, N + 1, -std::numeric_limits<double>::infinity());
    Matrix upper = -lower;

    // Only the initial state is fixed; later states remain free.
    lower.col(0).setZero();
    upper.col(0).setZero();
    solver.setStateBounds(lower, upper);

    // Solve and report convergence.
    examples::reportIterations(solver, options);
    solver.solve(options.iterations);
    std::cout << "Final pose: " << solver.state(N).transpose() << '\n';
    return std::abs(solver.state(N)(0) - goal[0]) < 0.01 ? 0 : 1;
}
