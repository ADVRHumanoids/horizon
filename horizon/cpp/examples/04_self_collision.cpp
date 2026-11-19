// Add a self-collision clearance penalty to the arm reaching problem.
#include "cli.h"
#include "horizon/pinocchio_collision.h"
#include "horizon/pinocchio_kinematics.h"
#include "ilqr/casadi_function.h"
#include "ilqr/ilqr.h"
#include <algorithm>
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
    horizon::collision::SphereModel spheres(
        {{"base", Eigen::Vector3d::Zero(), 0.05},
         {"tool", Eigen::Vector3d::Zero(), 0.05}});
    auto clearance_function = std::make_shared<
        horizon::collision::PinocchioSphereClearance>(
        model, space, spheres,
        std::vector<std::shared_ptr<const horizon::collision::DistanceField>>{},
        std::vector<std::pair<int, int>>{{0, 1}});
    auto clearance = functions.addNative("clearance", clearance_function);
    auto q = MX::sym("q", model->nq), u = MX::sym("u", model->nv);
    auto target = MX::sym("target", 3);
    functions.add("dynamics",
                  std::make_shared<PinocchioStep>(space, options.dt));
    functions.addResidual("effort", {q, u}, 0.1 * u);

    // A hinge residual penalizes clearance below the margin; it is a soft cost.
    functions.addResidual("collision", {q, u},
                          100 * MX::fmax(1.9 - clearance(q), 0));
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
    running.push_back(N);
    solver.setResidual(running, functions, "collision");
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

    // Check the resulting motion and clearance.
    Vector result(3);
    position->evaluate(solver.state(N), result);
    Vector distance(1);
    double minimum = std::numeric_limits<double>::infinity();
    for (int k = 0; k <= N; ++k)
    {
        clearance_function->evaluate(solver.state(k), distance);
        minimum = std::min(minimum, distance(0));
    }
    std::cout << "Tool error: " << (result - destination).norm()
              << "; minimum knot clearance: " << minimum << '\n';
    return minimum > 1.895 && (result - destination).norm() < .1 ? 0 : 1;
}
