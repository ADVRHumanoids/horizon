#ifndef HORIZON_PINOCCHIO_ROBOT_H
#define HORIZON_PINOCCHIO_ROBOT_H

#include "horizon/pinocchio_kinematics.h"

namespace horizon::kinematics
{
/** Native robot computations. Matrix outputs are flattened column-major;
 * spatial vectors are [linear; angular]. Every Jacobian is analytical. */
enum class RobotQuantity
{
    QDot,
    Difference,
    FrameJacobian,
    FrameVelocity,
    FrameAcceleration,
    FrameSpatialAcceleration,
    FrameTransform,
    InverseDynamics,
    ForwardDynamics,
    MassMatrix,
    Gravity,
    NonlinearEffects,
    KineticEnergy,
    PotentialEnergy,
    CenterOfMass,
    ComPosition,
    ComVelocity,
    ComAcceleration,
    CentroidalMomentum,
    CentroidalDynamics,
    CentroidalMap
};

class PinocchioRobotFunction final : public ilqr::TangentFunction
{
  public:
    PinocchioRobotFunction(
        std::shared_ptr<const pinocchio::Model> model,
        std::shared_ptr<const PinocchioConfiguration> configuration,
        RobotQuantity quantity, std::string frame = {},
        pinocchio::ReferenceFrame reference = pinocchio::LOCAL_WORLD_ALIGNED,
        bool symbolic_gravity = false,
        std::shared_ptr<PinocchioModelWorkspace> workspace = {});
    ~PinocchioRobotFunction() override;
    std::unique_ptr<ilqr::Function> clone() const override;
      void setEvaluationContext(
        const std::shared_ptr<ilqr::EvaluationContext> &context) override;

  private:
    void evaluateImpl(ilqr::Input z, ilqr::VectorOutput out) override;
    void tangentJacobian(ilqr::Input z, ilqr::MatrixOutput jac) override;
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Fixed bilateral contact at a named frame origin. Dimension 3 constrains
 * classical linear acceleration; dimension 6 additionally constrains angular
 * acceleration. Forces/wrenches are world-aligned, at the contact origin. */
struct ContactSpec
{
    std::string frame;
    int dimension = 3;
    ContactSpec() = default;
    ContactSpec(std::string frame, int dimension = 3)
        : frame(std::move(frame)), dimension(dimension)
    {
    }
};

/** Forward input [q; v; tau], output [vdot; f]. Inverse input [q; v; vdot; f],
 * output generalized tau. Forward uses M vdot + h = S tau + J.transpose() f,
 * J vdot + gamma = 0. The contact set is fixed for the evaluator lifetime.
 * Requires independent contact rows; singular sets fail explicitly.
 * All derivatives are analytical implicit KKT derivatives.
 */
class PinocchioContactDynamics final : public ilqr::TangentFunction
{
  public:
    PinocchioContactDynamics(
        std::shared_ptr<const pinocchio::Model> model,
        std::shared_ptr<const PinocchioConfiguration> configuration,
        std::vector<ContactSpec> contacts, ilqr::Matrix actuation = {},
        bool inverse = false,
        std::shared_ptr<PinocchioModelWorkspace> workspace = {});
    ~PinocchioContactDynamics() override;
    std::unique_ptr<ilqr::Function> clone() const override;
      void setEvaluationContext(
        const std::shared_ptr<ilqr::EvaluationContext> &context) override;

  private:
    void evaluateImpl(ilqr::Input z, ilqr::VectorOutput out) override;
    void tangentJacobian(ilqr::Input z, ilqr::MatrixOutput jac) override;
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace horizon::kinematics
#endif
