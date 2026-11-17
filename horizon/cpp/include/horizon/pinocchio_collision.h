#ifndef HORIZON_PINOCCHIO_COLLISION_H
#define HORIZON_PINOCCHIO_COLLISION_H

#ifndef HORIZON_WITH_PINOCCHIO
#error "Enable HORIZON_WITH_PINOCCHIO to use Pinocchio collision adapters"
#endif

#include "horizon/collision.h"
#include "horizon/device_collision.h"
#include "ilqr/state_space.h"
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

namespace horizon::collision
{
/** Configuration manifold and native generalized-velocity integration. */
class PinocchioConfiguration final : public ilqr::StateSpace
{
public:
    explicit PinocchioConfiguration(std::shared_ptr<const pinocchio::Model> model);
    int stateSize() const override;
    int tangentSize() const override;
    void neutral(ilqr::VectorOutput q) const override;
    void integrate(ilqr::Input q, ilqr::Input v, ilqr::VectorOutput out) const override;
    void difference(ilqr::Input a, ilqr::Input b, ilqr::VectorOutput out) const override;
    void retractionJacobian(ilqr::Input q, ilqr::MatrixOutput j) const override;
    void tangentProjection(ilqr::Input q, ilqr::MatrixOutput projection) const override;
    void differenceJacobians(ilqr::Input a, ilqr::Input b, ilqr::MatrixOutput A, ilqr::MatrixOutput B) const override;
    std::shared_ptr<const pinocchio::Model> model;
};

class PinocchioStep final : public ilqr::TangentFunction
{
public:
    PinocchioStep(std::shared_ptr<const PinocchioConfiguration> space, double dt);
    std::unique_ptr<ilqr::Function> clone() const override;
private:
    void evaluateImpl(ilqr::Input z, ilqr::VectorOutput out) override;
    void tangentJacobian(ilqr::Input z, ilqr::MatrixOutput j) override;
    std::shared_ptr<const PinocchioConfiguration> _space;
    double _dt;
    ilqr::Matrix _A, _B, _R;
    ilqr::Vector _prediction;
};

/** Setup-only export; unsupported joint types fail instead of falling back. */
DeviceRobotModel exportDeviceRobot(const pinocchio::Model& model,const SphereModel& spheres);
/** Node-local cached Pinocchio Data. Pass the same prototype to native
 * functions that need the same robot state; their EvaluationContext clones
 * and shares one Data workspace per shooting node.
 */
class PinocchioModelWorkspace
{
public:
    explicit PinocchioModelWorkspace(std::shared_ptr<const pinocchio::Model> model);
    std::shared_ptr<PinocchioModelWorkspace> cloneEmpty() const;
    std::shared_ptr<PinocchioModelWorkspace> forContext(
        const std::shared_ptr<ilqr::EvaluationContext>& context) const;
    const void* identity() const { return _identity.get(); }
    const pinocchio::Model& model() const { return *_model; }
    pinocchio::Data& data() { return _data; }
    void ensurePositions(ilqr::Input q);
    void ensureVelocity(ilqr::Input q, ilqr::Input v);
    void ensureAcceleration(ilqr::Input q, ilqr::Input v, ilqr::Input a);
    void ensureFramePlacements(ilqr::Input q);
    void ensureJointJacobians(ilqr::Input q);
private:
    PinocchioModelWorkspace(std::shared_ptr<const pinocchio::Model> model,
                            std::shared_ptr<const int> identity);
    bool same(const Eigen::VectorXd& cached, ilqr::Input value, bool valid) const;
    void resetForConfiguration(ilqr::Input q);
    std::shared_ptr<const pinocchio::Model> _model;
    std::shared_ptr<const int> _identity;
    pinocchio::Data _data;
    Eigen::VectorXd _q, _v, _a;
    bool _has_q = false, _positions = false, _velocity = false,
         _acceleration = false, _frames = false, _jacobians = false;
};

/** Computes all sphere world centers in SoA order and their tangent-space Jacobian in one
 * Pinocchio kinematics/Jacobian pass. Each clone owns independent Data.
 */
class PinocchioSphereCenters final : public ilqr::TangentFunction
{
public:
    PinocchioSphereCenters(std::shared_ptr<const pinocchio::Model> model,
                           std::shared_ptr<const ilqr::StateSpace> state_space,
                           const SphereModel& spheres,
                           std::shared_ptr<PinocchioModelWorkspace> workspace = {});
    std::unique_ptr<ilqr::Function> clone() const override;
    void setEvaluationContext(const std::shared_ptr<ilqr::EvaluationContext>& context) override;
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
}
#endif
