// Use acceleration and contact forces as controls, constraining floating-base
// dynamics.
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
    Vector force0 = equilibrium.tail(12);
    auto inverse = std::make_shared<PinocchioContactDynamics>(
        model, space, contacts, Matrix{}, true);

    // Combine native contact dynamics with the symbolic integration step.
    SymbolicFunctionManager functions;
    auto inverse_dynamics = functions.addNative("inverse_dynamics", inverse);
    auto integrate = functions.addNative(
        "integrate", std::make_shared<PinocchioStep>(space, 1.));
    auto x = MX::sym("x", nq + nv), control = MX::sym("control", nv + 12);
    auto q = x(casadi::Slice(0, nq)), v = x(casadi::Slice(nq, nq + nv));
    auto acceleration = control(casadi::Slice(0, nv));
    auto forces = control(casadi::Slice(nv, nv + 12));
    auto generalized =
        inverse_dynamics(MX::vertcat({q, v, acceleration, forces}));
    std::vector<MX> contact_acceleration;
    for (const auto &contact : contacts)
    {
        auto frame_acceleration = functions.addNative(
            "acc_" + contact.frame,
            std::make_shared<PinocchioRobotFunction>(
                model, space, RobotQuantity::FrameAcceleration, contact.frame));
        contact_acceleration.push_back(frame_acceleration(
            MX::vertcat({q, v, acceleration}))(casadi::Slice(0, 3)));
    }

    // The first six generalized forces belong to the unactuated floating base.
    functions.addExpression("underactuation", {x, control},
                            generalized(casadi::Slice(0, 6)), false);
    functions.addExpression("contact_acceleration", {x, control},
                            MX::vertcat(contact_acceleration), false);

    // Integrate configuration on its manifold and velocity in tangent space.
    auto next_q =
        integrate(MX::vertcat({q, dt * v + .5 * dt * dt * acceleration}));
    functions.addExpression("dynamics", {x, control},
                            MX::vertcat({next_q, v + dt * acceleration}),
                            false);
    MX force_reference(
        casadi::DM(std::vector<double>(force0.data(), force0.data() + 12)));
    functions.addResidual("velocity", {x, control}, v);
    functions.addResidual("acceleration", {x, control}, .01 * acceleration);
    functions.addResidual("forces", {x, control},
                          .001 * (forces - force_reference));
    int joint = model->joints[model->getJointId("shoulder_pitch_1")].idx_q();
    functions.addResidual("target", {x, control},
                          50 * (q(joint) - q0[joint] - .02));
    functions.addResidual("terminal_velocity", {x, control}, 100 * v);

    // Configure the solver and choose which nodes carry each term.
    horizon::IterativeLQR solver(
        functions, "dynamics", N,
        {{"ilqr.n_threads", options.threads},
         {"ilqr.huu_reg", 1e-4},
         {"ilqr.constr_decomp_type", std::string("svd")},
         {"ilqr.constraint_violation_threshold", 1e-7}},
        state_space);
    solver.setResidual(horizon::Nodes::running(), functions, "velocity");
    solver.setResidual(horizon::Nodes::running(), functions, "acceleration");
    solver.setResidual(horizon::Nodes::running(), functions, "forces");
    solver.setConstraint(horizon::Nodes::running(), functions,
                         "underactuation");
    solver.setConstraint(horizon::Nodes::running(), functions,
                         "contact_acceleration");
    solver.setResidual(horizon::Nodes::terminal(), functions,
                       "terminal_velocity");
    solver.setResidual(horizon::Nodes::terminal(), functions, "target");
    Vector x0(nq + nv);
    x0 << q0, Vector::Zero(nv);
    solver.fixInitialState(x0);
    solver.setStateInitialGuess(x0.replicate(1, N + 1));
    Vector control0(nv + 12);
    control0 << Vector::Zero(nv), force0;
    solver.setInputInitialGuess(control0.replicate(1, N));

    // Solve and report convergence.
    examples::reportIterations(solver, options);
    solver.solve(options.iterations);

    // Check dynamics and contact drift at the solution.
    double residual = 0, contact_error = 0;
    std::vector<std::shared_ptr<PinocchioRobotFunction>> acceleration_checks;
    for (const auto &contact : contacts)
    {
        acceleration_checks.push_back(std::make_shared<PinocchioRobotFunction>(
            model, space, RobotQuantity::FrameAcceleration, contact.frame));
    }
    for (int k = 0; k < N; ++k)
    {
        Vector input(nq + 2 * nv + 12);
        input << solver.state(k), solver.input(k);
        Vector generalized(nv);
        inverse->evaluate(input, generalized);
        residual = std::max(residual, generalized.head(6).norm());
        Vector state_acceleration(nq + 2 * nv);
        state_acceleration << solver.state(k), solver.input(k).head(nv);
        Vector frame_acceleration(6);
        for (const auto &check : acceleration_checks)
        {
            check->evaluate(state_acceleration, frame_acceleration);
            contact_error =
                std::max(contact_error, frame_acceleration.head(3).norm());
        }
    }
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
              << "; foot drift: " << drift
              << "; contact acceleration: " << contact_error << '\n';
    return displacement > .015 && speed < 1e-3 && residual < 1e-6 &&
            drift < .001 && contact_error < 1e-6
        ? 0
        : 1;
}
