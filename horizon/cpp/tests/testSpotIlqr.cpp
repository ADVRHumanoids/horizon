#include "spot_problem.h"
#include <gtest/gtest.h>
#include <iostream>

using namespace spot_test;

TEST(SpotIlqr, NativeAnalyticalAndImplicitDerivatives)
{
    ProblemOptions options;
    options.project_contacts = false;
    options.scale_regularization = false;
    for (auto form : {Formulation::AccelerationForce, Formulation::Torque})
    {
        SCOPED_TRACE(name(form));
        Problem problem(form, 12, 0, options);
        EXPECT_LT(problem.spaceDerivativeError(), 1e-7);
        EXPECT_LT(problem.derivativeError(true), 1e-6);
        EXPECT_LT(problem.derivativeError(false), 1e-6);
    }
}

TEST(SpotIlqr, MeshIndependentConvergenceThrough256)
{
    for (int N : {16, 32, 64, 128, 256})
    {
        for (int threads : {0, 2})
        {
            double objective = 0;
            for (auto form :
                 {Formulation::AccelerationForce, Formulation::Torque})
            {
                SCOPED_TRACE(std::string(name(form)) +
                             " N=" + std::to_string(N) +
                             " threads=" + std::to_string(threads));
                Problem problem(form, N, threads);
                auto m = problem.solve(20);
                ASSERT_TRUE(m.converged);
                EXPECT_LE(m.iterations, 8);
                EXPECT_GT(m.depth, .05);
                EXPECT_LT(m.pose_position, .003);
                EXPECT_LT(m.pose_rotation, .006);
                EXPECT_LT(m.foot_drift, 1e-8);
                EXPECT_LT(m.foot_velocity, 1e-6);
                EXPECT_LT(m.endpoint_velocity, 1e-8); // all eighteen components
                EXPECT_LT(m.dynamics_gap, 1e-6);
                EXPECT_LT(m.contact_acceleration, 1e-5);
                EXPECT_LT(m.newton_euler, .005);
                EXPECT_LT(m.quaternion_error, 1e-12);
                EXPECT_GT(m.min_normal, 1);
                EXPECT_GT(m.min_friction_margin, 0);
                EXPECT_GT(m.joint_margin, 0);
                EXPECT_LT(m.peak_torque, 1000);
                if (form == Formulation::AccelerationForce)
                {
                    objective = m.objective;
                }
                else
                {
                    EXPECT_NEAR(m.objective, objective, 1e-8);
                }
            }
        }
    }
}

TEST(SpotIlqr, NativeContactProjectionDerivatives)
{
    ProblemOptions options;
    options.project_contacts = true;
    for (auto form : {Formulation::AccelerationForce, Formulation::Torque})
    {
        SCOPED_TRACE(name(form));
        Problem problem(form, 16, 0, options);
        EXPECT_LT(problem.derivativeError(true), 1e-6);
    }
}

TEST(SpotIlqr, SquatWithFourContactsAndFullBasePose)
{
    Eigen::MatrixXd inverse_x, inverse_a, inverse_f, inverse_tau;
    double inverse_objective = 0;
    for (auto form : {Formulation::AccelerationForce, Formulation::Torque})
    {
        for (int threads : {0, 2})
        {
            SCOPED_TRACE(std::string(name(form)) +
                         " threads=" + std::to_string(threads));
            Problem problem(form, 16, threads);
            auto m = problem.solve();
            std::cout << name(form) << " threads=" << threads
                      << " converged=" << m.converged
                      << " iterations=" << m.iterations
                      << " solve_ms=" << m.solve_ms
                      << " objective=" << m.objective
                      << " initial=" << m.initial_objective
                      << " pose=" << m.pose_position << "," << m.pose_rotation
                      << " feet=" << m.foot_drift << "," << m.foot_velocity
                      << " NE=" << m.newton_euler
                      << " contact_acc=" << m.contact_acceleration
                      << " gap=" << m.dynamics_gap << "\n";
            EXPECT_TRUE(m.converged);
            EXPECT_LT(m.objective, .02 * m.initial_objective);
            EXPECT_GT(m.depth, .05);
            EXPECT_LT(m.pose_position, .003);
            EXPECT_LT(m.pose_rotation, .006);
            EXPECT_LT(m.foot_drift, .001);
            EXPECT_LT(m.foot_velocity, .003);
            EXPECT_LT(m.newton_euler, .005);
            EXPECT_LT(m.contact_acceleration, 1e-4);
            EXPECT_LT(m.dynamics_gap, 1e-5);
            EXPECT_LT(m.quaternion_error, 1e-12);
            EXPECT_LT(m.endpoint_velocity, 1e-5);
            EXPECT_GT(m.min_normal, 1);
            EXPECT_GT(m.min_friction_margin, 0);
            EXPECT_GT(m.joint_margin, 0);
            EXPECT_LT(m.peak_torque, 1000); // bundled URDF effort limit
#ifdef HORIZON_PROFILING
            auto profile = problem.phaseTimings();
            for (const auto &phase : {"linearize_quadratize", "backward_pass",
                                      "line_search", "solve"})
            {
                ASSERT_TRUE(profile.count(phase));
                EXPECT_FALSE(profile.at(phase).empty());
                for (double us : profile.at(phase))
                {
                    EXPECT_TRUE(std::isfinite(us) && us > 0);
                }
            }
#endif
            if (form == Formulation::AccelerationForce && threads == 0)
            {
                inverse_x = problem.states();
                inverse_a = problem.accelerations();
                inverse_f = problem.forces();
                inverse_tau = problem.torques();
                inverse_objective = m.objective;
            }
            else
            {
                EXPECT_LT((problem.states() - inverse_x).cwiseAbs().maxCoeff(),
                          .01);
                EXPECT_LT(
                    (problem.accelerations() - inverse_a).cwiseAbs().maxCoeff(),
                    .1);
                EXPECT_LT((problem.forces() - inverse_f).cwiseAbs().maxCoeff(),
                          2);
                EXPECT_LT(
                    (problem.torques() - inverse_tau).cwiseAbs().maxCoeff(), 1);
                EXPECT_NEAR(m.objective, inverse_objective,
                            .01 * (1 + inverse_objective));
            }
        }
    }
}

TEST(SpotIlqr, ProfilingSamplesAreFinite)
{
    for (auto form : {Formulation::AccelerationForce, Formulation::Torque})
    {
        Problem problem(form, 12, 0);
        auto samples = problem.profileFunctions(12);
        ASSERT_EQ(samples.dynamics_us.size(), 12);
        ASSERT_EQ(samples.stage_us.size(), 12);
        for (const auto *series : {&samples.dynamics_us, &samples.stage_us})
        {
            for (double us : *series)
            {
                EXPECT_TRUE(std::isfinite(us) && us > 0);
            }
        }
    }
}
