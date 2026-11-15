#pragma once
#include "ilqr/ilqr.h"
#include "spot_formulation.h"
#include <string>

namespace spot_test
{
struct Metrics
{
    double pose_position=0,pose_rotation=0,foot_drift=0,foot_velocity=0;
    double dynamics_gap=0,newton_euler=0,contact_acceleration=0;
    double min_normal=1e100,min_friction_margin=1e100,joint_margin=1e100;
    double peak_torque=0,quaternion_error=0,depth=0,endpoint_velocity=0;
    double objective=0,initial_objective=0,solve_ms=0,setup_ms=0;
    int iterations=0;
    bool converged=false;
};
struct Samples {std::vector<double> dynamics_us,stage_us;};
// Controlled ablations for convergence diagnostics. Defaults enforce fixed
// feet and a full-robot terminal rest condition using independent coordinates.
struct ProblemOptions
{
    enum TerminalVelocity {Full,Base,None};
    TerminalVelocity terminal_velocity=Full;
    double rank_threshold=1e-10;
    std::string kkt_decomposition="lu";
    bool project_contacts=true;
    bool scale_regularization=true;
};
class Problem
{
public:
    Problem(Formulation,int nodes,int threads,ProblemOptions options={});
    ~Problem();
    Metrics solve(int max_iterations=300);
    Samples profileFunctions(int samples);
    std::map<std::string,std::vector<double>> phaseTimings() const;
    const Eigen::MatrixXd& states() const;
    const Eigen::MatrixXd& accelerations() const;
    const Eigen::MatrixXd& forces() const;
    const Eigen::MatrixXd& torques() const;
    double derivativeError(bool dynamics);
    double spaceDerivativeError();
    void diagnoseLqStep(std::ostream& out,int iterations=0);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
