#ifndef HORIZON_PINOCCHIO_COLLISION_H
#define HORIZON_PINOCCHIO_COLLISION_H

#ifndef HORIZON_WITH_PINOCCHIO
#error "Enable HORIZON_WITH_PINOCCHIO to use Pinocchio collision adapters"
#endif

#include "horizon/collision.h"
#include "horizon/device_collision.h"
#include "horizon/pinocchio_kinematics.h"
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace horizon::collision
{
// Supported compatibility aliases; general robot kinematics lives
// independently.
using kinematics::PinocchioConfiguration;
using kinematics::PinocchioModelWorkspace;
using kinematics::PinocchioStep;

/** Setup-only export; unsupported joint types fail instead of falling back. */
DeviceRobotModel exportDeviceRobot(const pinocchio::Model &model,
                                   const SphereModel &spheres);
/** Computes all sphere world centers in SoA order and their tangent-space
 * Jacobian in one Pinocchio kinematics/Jacobian pass. Each clone owns
 * independent Data.
 */
class PinocchioSphereCenters final : public ilqr::TangentFunction
{
  public:
    PinocchioSphereCenters(
        std::shared_ptr<const pinocchio::Model> model,
        std::shared_ptr<const ilqr::StateSpace> state_space,
        const SphereModel &spheres,
        std::shared_ptr<PinocchioModelWorkspace> workspace = {});
    std::unique_ptr<ilqr::Function> clone() const override;
    void setEvaluationContext(
        const std::shared_ptr<ilqr::EvaluationContext> &context) override;

  protected:
    void evaluateImpl(ilqr::Input q, ilqr::VectorOutput centers) override;
    void tangentJacobian(ilqr::Input q, ilqr::MatrixOutput jacobian) override;
    void valueAndTangentJacobian(ilqr::Input q, ilqr::VectorOutput centers,
                                 ilqr::MatrixOutput jacobian) override;

  private:
    void fillCenters(ilqr::VectorOutput centers) const;
    void fillJacobian(ilqr::MatrixOutput jacobian);
    std::shared_ptr<const pinocchio::Model> _model;
    std::shared_ptr<const ilqr::StateSpace> _state_space;
    std::shared_ptr<PinocchioModelWorkspace> _workspace_prototype, _workspace;
    std::vector<LinkSphere> _spheres;
    std::vector<pinocchio::FrameIndex> _frames;
    std::map<pinocchio::FrameIndex, std::vector<std::size_t>> _frame_spheres;
    Eigen::Matrix<double, 6, Eigen::Dynamic> _frame_jacobian;
};
} // namespace horizon::collision
namespace horizon::collision {
class PinocchioSphereClearance final : public ilqr::TangentFunction
{
public:
    PinocchioSphereClearance(std::shared_ptr<const pinocchio::Model> model,
        std::shared_ptr<const ilqr::StateSpace> space, SphereModel spheres,
        std::vector<std::shared_ptr<const DistanceField>> environment,
        std::vector<std::pair<int,int>> pairs,
        std::shared_ptr<PinocchioModelWorkspace> workspace = {});
    std::unique_ptr<ilqr::Function> clone() const override;
    void setEvaluationContext(const std::shared_ptr<ilqr::EvaluationContext>& context) override;
private:
    void evaluateImpl(ilqr::Input q, ilqr::VectorOutput value) override;
    void tangentJacobian(ilqr::Input q, ilqr::MatrixOutput jac) override;
    void valueAndTangentJacobian(ilqr::Input q, ilqr::VectorOutput value, ilqr::MatrixOutput jac) override;
    std::shared_ptr<const pinocchio::Model> _model;
    std::shared_ptr<const ilqr::StateSpace> _space;
    SphereModel _spheres;
    std::vector<std::shared_ptr<const DistanceField>> _environment;
    std::vector<std::pair<int,int>> _pairs;
    std::shared_ptr<PinocchioModelWorkspace> _workspace;
    PinocchioSphereCenters _centers;
    SphereClearanceFunction _clearance;
    ilqr::Vector _points, _value;
    ilqr::Matrix _jacobian;
};
}
#endif
