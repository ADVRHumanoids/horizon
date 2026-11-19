// Use joint torques as controls with native constrained forward dynamics.
#include "cli.h"
#include "contact_support.h"
#include "ilqr/casadi_function.h"
#include "ilqr/ilqr.h"
#include <iostream>

int main(int argc, char **argv)
{
    // Choose the horizon, solver settings and optional asset paths.
    const auto options = examples::parseOptions(argc, argv, 12, 20, .05, false);
    using namespace horizon::ilqr;
    using namespace horizon::kinematics;
    using casadi::MX;
    namespace pin = pinocchio;
    const int N = options.nodes;
    const double dt = options.dt;
    auto model = std::make_shared<pin::Model>();
    pin::urdf::buildModel(options.urdf, *model);
    auto space = std::make_shared<PinocchioConfiguration>(model);
    auto state_space = std::make_shared<ProductSpace>(
        std::vector<std::shared_ptr<const StateSpace>>{
            space, std::make_shared<EuclideanSpace>(model->nv)});
    Vector q0 = examples::standing(*model, true);
    const int nq = model->nq, nv = model->nv, na = nv - 6;

    // The floating base occupies the first six velocity coordinates and is
    // unactuated.
    Matrix selection = Matrix::Identity(nv, nv).rightCols(na);
    std::vector<ContactSpec> contacts{
        {"contact_1"}, {"contact_2"}, {"contact_3"}, {"contact_4"}};
    Vector equilibrium = examples::equilibrium(*model, q0, selection, contacts);
    Vector tau0 = equilibrium.head(na);
    auto forward = std::make_shared<PinocchioContactDynamics>(
        model, space, contacts, selection);
    auto inverse = std::make_shared<PinocchioContactDynamics>(
        model, space, contacts, Matrix{}, true);

    // Combine native contact dynamics with the symbolic integration step.
    SymbolicFunctionManager functions;
    auto contact = functions.addNative("contact", forward);
    auto integrate = functions.addNative(
        "integrate", std::make_shared<PinocchioStep>(space, 1.));
    auto x = MX::sym("x", nq + nv), tau = MX::sym("tau", na);
    auto q = x(casadi::Slice(0, nq)), v = x(casadi::Slice(nq, nq + nv));
    auto terms = contact(MX::vertcat({q, v, tau}));
    auto acceleration = terms(casadi::Slice(0, nv));

    // Integrate configuration on its manifold and velocity in tangent space.
    auto next_q =
        integrate(MX::vertcat({q, dt * v + .5 * dt * dt * acceleration}));
    functions.addExpression("dynamics", {x, tau},
                            MX::vertcat({next_q, v + dt * acceleration}),
                            false);
    MX tau_reference(
        casadi::DM(std::vector<double>(tau0.data(), tau0.data() + na)));
    functions.addResidual("velocity", {x, tau}, v);
    functions.addResidual("torque", {x, tau}, .01 * (tau - tau_reference));
    int joint = model->joints[model->getJointId("shoulder_pitch_1")].idx_q();
    functions.addResidual("target", {x, tau},
                          50 * (q(joint) - q0[joint] - .02));
    functions.addResidual("terminal_velocity", {x, tau}, 100 * v);

    // Configure the solver and choose which nodes carry each term.
    horizon::IterativeLQR solver(
        functions, "dynamics", N,
        {{"ilqr.n_threads", options.threads}, {"ilqr.huu_reg", 1e-4}},
        state_space);
    solver.setResidual(horizon::Nodes::running(), functions, "velocity");
    solver.setResidual(horizon::Nodes::running(), functions, "torque");
    solver.setResidual(horizon::Nodes::terminal(), functions,
                       "terminal_velocity");
    solver.setResidual(horizon::Nodes::terminal(), functions, "target");
    Vector x0(nq + nv);
    x0 << q0, Vector::Zero(nv);
    solver.fixInitialState(x0);
    solver.setStateInitialGuess(x0.replicate(1, N + 1));
    solver.setInputInitialGuess(tau0.replicate(1, N));

    // Solve and report convergence.
    examples::reportIterations(solver, options);
    solver.solve(options.iterations);

    // Check dynamics and contact drift at the solution.
    Vector input(nq + nv + na);
    input << solver.state(N), solver.input(N - 1);
    Vector out(nv + 12);
    forward->evaluate(input, out);
    Vector inverse_input(nq + 2 * nv + 12);
    inverse_input << solver.state(N), out;
    Vector generalized(nv);
    inverse->evaluate(inverse_input, generalized);
    double residual = (generalized - selection * solver.input(N - 1)).norm();
    double speed = solver.state(N).tail(nv).norm();
    PinocchioFramePositions feet(
        model, space, {"contact_1", "contact_2", "contact_3", "contact_4"});
    Vector initial_feet(12), actual_feet(12);
    feet.evaluate(q0, initial_feet);

    // Check the resulting tool motion and foot drift.
    double drift = 0;
    for (int k = 0; k <= N; ++k)
    {
        feet.evaluate(solver.state(k).head(nq), actual_feet);
        for (int i = 0; i < 4; ++i)
        {
            Eigen::Vector3d delta(actual_feet[i] - initial_feet[i],
                                  actual_feet[4 + i] - initial_feet[4 + i],
                                  actual_feet[8 + i] - initial_feet[8 + i]);
            drift = std::max(drift, delta.norm());
        }
    }
    double displacement = solver.state(N)[joint] - q0[joint];
    std::cout << "Joint displacement: " << displacement
              << "; terminal speed: " << speed
              << "; dynamics residual: " << residual
              << "; foot drift: " << drift << '\n';
    return displacement > .015 && speed < 1e-3 && residual < 1e-7 &&
            drift < .001
        ? 0
        : 1;
}
