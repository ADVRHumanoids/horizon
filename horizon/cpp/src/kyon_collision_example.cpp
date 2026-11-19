// Kyon fixed-feet model; execution and reporting live in example support.
#include "kyon_collision_support.h"

using namespace kyon_example;

int main(int argc, char **argv)
{
    try
    {
        Options options;
        if (!options.parse(argc, argv))
        {
            return 0;
        }
        auto &[N, iterations, samples, threads, warmup, damping,
               collision_weight, decomposition, backend, kokkos_threads,
               collision_enabled, self_collision, urdf, collision_policy,
               scenario, output] = options;
        const double collision_scale = std::sqrt(collision_weight);
        CollisionRuntime runtime(backend, kokkos_threads);
        const auto setup_start = Clock::now();
        auto [model, space, start, geometry, sphere_count, count, fk,
              initial_points, frame_fk, initial_frames, target, obstacle,
              obstacle_radius, voxel, field, radii, self_pairs] =
            prepareScene(urdf, collision_policy, scenario, self_collision);
        CollisionBackend collision_backend(backend, collision_enabled, model,
                                           fk, geometry, sphere_count, radii,
                                           field, self_pairs, collision_weight);
        SymbolicFunctionManager manager;
        using casadi::MX;
        auto q = MX::sym("q", model->nq), u = MX::sym("u", model->nv);
        auto kinematics =
            prepareKinematics(manager, q, frame_fk, initial_frames);
        auto tool = kinematics.framePosition("dagana_1_base");
        std::vector<MX> feet;
        for (const auto *frame :
             {"contact_1", "contact_2", "contact_3", "contact_4"})
        {
            feet.push_back(
                kinematics.framePosition(frame) -
                vectorConstant(kinematics.initialFramePosition(frame)));
        }
        manager.addExpression("feet", {q, u}, MX::vertcat(feet), false);
        auto native_step = manager.addNative(
            "integrate", std::make_shared<Step>(space, 2. / N));
        manager.addExpression("dynamics", {q, u},
                              native_step(MX::vertcat({q, u})), false);
        manager.addResidual("effort", {q, u}, .05 * u);
        manager.addResidual("target", {q, u},
                            100 * (tool - vectorConstant(target)));
        if (collision_enabled && backend != "device")
        {
            collision_backend.registerResidual(manager, q, u, fk,
                                               collision_scale);
        }
        horizon::IterativeLQR solver(
            manager, "dynamics", N,
            {{"ilqr.n_threads", threads},
             {"ilqr.constr_decomp_type", std::string("svd")},
             {"ilqr.huu_reg", damping / N},
             {"ilqr.kkt_decomp_type", decomposition},
             {"ilqr.constraint_violation_threshold", 1e-7},
             {"ilqr.defect_norm_threshold", 1e-7}},
            space);
        solver.setResidual(horizon::Nodes::running(), manager, "effort");
        solver.setResidual(horizon::Nodes::terminal(), manager, "target");
        solver.setConstraint(horizon::Nodes::all(), manager, "feet");
        if (collision_enabled && backend != "device")
        {
            solver.setResidual(horizon::Nodes::all(), manager, "collision");
        }
        collision_backend.install(solver, N);
        solver.setStateBounds(model->lowerPositionLimit,
                              model->upperPositionLimit);
        solver.fixInitialState(start);
        solver.setInputBounds(-1., 1.);
        const double setup_ms = std::chrono::duration<double, std::milli>(
                                    Clock::now() - setup_start)
                                    .count();
        return Benchmark{solver,
                         collision_backend,
                         model,
                         space,
                         start,
                         fk,
                         initial_points,
                         frame_fk,
                         initial_frames,
                         target,
                         obstacle,
                         field,
                         radii,
                         self_pairs,
                         output,
                         urdf,
                         collision_policy,
                         scenario,
                         backend,
                         decomposition,
                         N,
                         samples,
                         warmup,
                         iterations,
                         threads,
                         kokkos_threads,
                         sphere_count,
                         count,
                         collision_enabled,
                         collision_weight,
                         obstacle_radius,
                         voxel,
                         damping,
                         setup_ms}
            .run();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
