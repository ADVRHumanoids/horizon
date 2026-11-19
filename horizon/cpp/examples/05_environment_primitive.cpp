// Move a tool around a primitive sphere on its direct route.
#include "cli.h"
#include "horizon/pinocchio_collision.h"
#include "horizon/pinocchio_kinematics.h"
#include "ilqr/casadi_function.h"
#include "ilqr/ilqr.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <pinocchio/parsers/urdf.hpp>

int main(int argc, char **argv)
{
    using namespace horizon::ilqr;
    using namespace horizon::kinematics;
    using casadi::MX;
    const auto options = examples::parseOptions(argc, argv, 30, 50, .1, true);
    const int N = options.nodes;

    // Load the arm and define the tool sphere and mid-route obstacle.
    auto model = std::make_shared<pinocchio::Model>();
    pinocchio::urdf::buildModel(options.urdf, *model);
    auto space = std::make_shared<PinocchioConfiguration>(model);
    auto position = std::make_shared<PinocchioFramePositions>(
        model, space, std::vector<std::string>{"tool"});
    horizon::collision::SphereModel spheres(
        {{"tool", Eigen::Vector3d::Zero(), .05}});
    auto obstacle = std::make_shared<horizon::collision::SphereField>(
        Eigen::Vector3d(1.89, .72, 0), .10);
    auto clearance_function = std::make_shared<
        horizon::collision::PinocchioSphereClearance>(
        model, space, spheres,
        std::vector<std::shared_ptr<const horizon::collision::DistanceField>>{
            obstacle},
        std::vector<std::pair<int, int>>{});

    // Compose native kinematics and clearance with symbolic objectives.
    SymbolicFunctionManager functions;
    auto fk = functions.addNative("tool_position", position);
    auto clearance = functions.addNative("clearance", clearance_function);
    auto step = functions.addNative(
        "sample_step", std::make_shared<PinocchioStep>(space, options.dt));
    auto q = MX::sym("q", model->nq), u = MX::sym("u", model->nv);
    auto target = MX::sym("target", 3);
    functions.add("dynamics",
                  std::make_shared<PinocchioStep>(space, options.dt));
    functions.addResidual("effort", {q, u}, .1 * u);
    functions.addResidual("tracking", {q, u, target}, 10 * (fk(q) - target));
    functions.addResidual("collision", {q, u},
                          100 * MX::fmax(.03 - clearance(q), 0));

    // Sample three interior configurations as soft collision costs per
    // interval.
    std::vector<MX> samples;
    for (double fraction : {.25, .5, .75})
    {
        samples.push_back(clearance(step(MX::vertcat({q, fraction * u}))));
    }
    functions.addResidual("between_knots", {q, u},
                          100 * MX::fmax(.03 - MX::vertcat(samples), 0));

    // Use a feasible detour seed to avoid a symmetric collision initialization.
    Vector waypoint(2), goal(2), destination(3);
    waypoint << -.15, 1.15;
    goal << .5, .5;
    position->evaluate(goal, destination);
    Matrix guess(2, N + 1), controls(2, N);
    for (int k = 0; k <= N; ++k)
    {
        const double t = static_cast<double>(k) / N;
        if (t <= .5)
        {
            guess.col(k) = 2 * t * waypoint;
        }
        else
        {
            guess.col(k) = waypoint + (2 * t - 1) * (goal - waypoint);
        }
    }
    for (int k = 0; k < N; ++k)
    {
        controls.col(k) = (guess.col(k + 1) - guess.col(k)) / options.dt;
    }

    horizon::IterativeLQR solver(functions, "dynamics", N,
                                 {{"ilqr.n_threads", options.threads}}, space);
    solver.setResidual(horizon::Nodes::running(), functions, "effort");
    solver.setResidual(horizon::Nodes::running(), functions, "between_knots");
    solver.setResidual(horizon::Nodes::all(), functions, "collision");
    solver.setResidual(horizon::Nodes::terminal(), functions, "tracking");
    solver.setParameterValue("target", destination.replicate(1, N + 1));
    solver.fixInitialState(Vector::Zero(model->nq));
    solver.setInputBounds(-1., 1.);
    solver.setStateInitialGuess(guess);
    solver.setInputInitialGuess(controls);
    examples::reportIterations(solver, options);
    solver.solve(options.iterations);

    // Check the target, a denser sampled path, and the obstructed direct route.
    Vector result(3), distance(1), delta(model->nv), sample(model->nq);
    position->evaluate(solver.state(N), result);
    double minimum = std::numeric_limits<double>::infinity();
    double max_defect = 0;
    for (int k = 0; k < N; ++k)
    {
        space->integrate(solver.state(k), options.dt * solver.input(k), sample);
        space->difference(sample, solver.state(k + 1), delta);
        max_defect = std::max(max_defect, delta.norm());
        space->difference(solver.state(k), solver.state(k + 1), delta);
        for (int j = 0; j <= 10; ++j)
        {
            space->integrate(solver.state(k), (j / 10.) * delta, sample);
            clearance_function->evaluate(sample, distance);
            minimum = std::min(minimum, distance(0));
        }
    }
    clearance_function->evaluate(Vector::Zero(model->nq), distance);
    const double start_clearance = distance(0);
    clearance_function->evaluate(goal, distance);
    const double goal_clearance = distance(0);
    clearance_function->evaluate(.5 * goal, distance);
    const double direct_midpoint = distance(0);
    const double error = (result - destination).norm();
    std::cout << "Tool error: " << error
              << "; minimum sampled clearance: " << minimum
              << "; direct-route midpoint clearance: " << direct_midpoint
              << "\nStart clearance: " << start_clearance
              << "; target clearance: " << goal_clearance
              << "; maximum dynamics defect: " << max_defect << '\n';
    return error < .01 && minimum > .02 && direct_midpoint < 0 &&
            std::min(start_clearance, goal_clearance) > .2 && max_defect < 1e-5
        ? 0
        : 1;
}
