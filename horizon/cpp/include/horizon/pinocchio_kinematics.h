#ifndef HORIZON_PINOCCHIO_KINEMATICS_H
#define HORIZON_PINOCCHIO_KINEMATICS_H
#ifndef HORIZON_WITH_PINOCCHIO
#error "Enable HORIZON_WITH_PINOCCHIO to use native Pinocchio kinematics"
#endif
#include "ilqr/state_space.h"
#include <map>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <string>
#include <vector>

namespace horizon::kinematics
{
/** Configuration manifold and native generalized-velocity integration. */
class PinocchioConfiguration final : public ilqr::StateSpace
{
  public:
    explicit PinocchioConfiguration(
        std::shared_ptr<const pinocchio::Model> model);
    int stateSize() const override;
    int tangentSize() const override;
    void neutral(ilqr::VectorOutput q) const override;
    void integrate(ilqr::Input q, ilqr::Input v,
                   ilqr::VectorOutput out) const override;
    void difference(ilqr::Input a, ilqr::Input b,
                    ilqr::VectorOutput out) const override;
    void retractionJacobian(ilqr::Input q, ilqr::MatrixOutput j) const override;
    void tangentProjection(ilqr::Input q,
                           ilqr::MatrixOutput projection) const override;
    void differenceJacobians(ilqr::Input a, ilqr::Input b, ilqr::MatrixOutput A,
                             ilqr::MatrixOutput B) const override;
    void defectBlocks(std::vector<ilqr::DefectBlock> &blocks) const override;
    std::shared_ptr<const pinocchio::Model> model;
};

class PinocchioStep final : public ilqr::TangentDynamics
{
  public:
    PinocchioStep(std::shared_ptr<const PinocchioConfiguration> space,
                  double dt);
    std::unique_ptr<ilqr::Function> clone() const override;

  private:
    void evaluateImpl(ilqr::Input z, ilqr::VectorOutput out) override;
    void tangentOutputLinearization(ilqr::Input z, ilqr::VectorOutput value,
                                    ilqr::MatrixOutput j) override;
    std::shared_ptr<const PinocchioConfiguration> _space;
    double _dt;
    ilqr::Matrix _A, _B;
    ilqr::Vector _configuration;
};

/** Node-local cached Pinocchio Data. Pass the same prototype to native
 * functions that need the same robot state; their EvaluationContext clones
 * and shares one Data workspace per shooting node. Configuration quaternions
 * are normalized before evaluation, including at off-manifold NLP trials.
 */
class PinocchioModelWorkspace
{
  public:
    explicit PinocchioModelWorkspace(
        std::shared_ptr<const pinocchio::Model> model);
    std::shared_ptr<PinocchioModelWorkspace> cloneEmpty() const;
    std::shared_ptr<PinocchioModelWorkspace>
    forContext(const std::shared_ptr<ilqr::EvaluationContext> &context) const;
    const void *identity() const
    {
        return _identity.get();
    }
    const pinocchio::Model &model() const
    {
        return *_model;
    }
    pinocchio::Data &data()
    {
        return _data;
    }
    void ensurePositions(ilqr::Input q);
    void ensureVelocity(ilqr::Input q, ilqr::Input v);
    void ensureAcceleration(ilqr::Input q, ilqr::Input v, ilqr::Input a);
    void ensureFramePlacements(ilqr::Input q);
    void ensureJointJacobians(ilqr::Input q);
    void ensureKinematicsDerivatives(ilqr::Input q, ilqr::Input v,
                     ilqr::Input a);
    void invalidate();

  private:
    PinocchioModelWorkspace(std::shared_ptr<const pinocchio::Model> model,
                            std::shared_ptr<const int> identity);
    bool same(const Eigen::VectorXd &cached, ilqr::Input value,
              bool valid) const;
    void resetForConfiguration(ilqr::Input q);
    std::shared_ptr<const pinocchio::Model> _model;
    std::shared_ptr<const int> _identity;
    pinocchio::Data _data;
    Eigen::VectorXd _q, _input_q, _candidate_q, _v, _a;
    bool _has_q = false, _positions = false, _velocity = false,
          _acceleration = false, _frames = false, _jacobians = false,
          _derivatives = false;
};

/** Shared native frame evaluator infrastructure, independent of collision
 * geometry. */
class PinocchioFrameFunction : public ilqr::TangentFunction
{
  public:
    void setEvaluationContext(
        const std::shared_ptr<ilqr::EvaluationContext> &context) override;

  protected:
    PinocchioFrameFunction(
        std::shared_ptr<const pinocchio::Model> model,
        std::shared_ptr<const ilqr::StateSpace> configuration,
        std::shared_ptr<const ilqr::StateSpace> input_space,
        std::vector<std::string> frames, int output_size,
        std::shared_ptr<PinocchioModelWorkspace> workspace);
    void prepare(ilqr::Input q, bool derivatives);
    void frameJacobian(pinocchio::FrameIndex frame);
    std::shared_ptr<const pinocchio::Model> _model;
    std::shared_ptr<const ilqr::StateSpace> _configuration;
    std::shared_ptr<PinocchioModelWorkspace> _workspace_prototype, _workspace;
    std::vector<std::string> _names;
    std::vector<pinocchio::FrameIndex> _frames;
    ilqr::Matrix _frame_jacobian;
};

/** World frame origins in SoA order [all x; all y; all z]. */
class PinocchioFramePositions final : public PinocchioFrameFunction
{
  public:
    PinocchioFramePositions(
        std::shared_ptr<const pinocchio::Model> model,
        std::shared_ptr<const ilqr::StateSpace> configuration,
        std::vector<std::string> frames,
        std::shared_ptr<PinocchioModelWorkspace> workspace = {});
    std::unique_ptr<ilqr::Function> clone() const override;

  private:
    void evaluateImpl(ilqr::Input q, ilqr::VectorOutput out) override;
    void tangentJacobian(ilqr::Input q, ilqr::MatrixOutput jac) override;
};

/** World frame pose [xyz; quaternion xyzw]. */
class PinocchioFramePose final : public PinocchioFrameFunction
{
  public:
    PinocchioFramePose(std::shared_ptr<const pinocchio::Model> model,
                       std::shared_ptr<const ilqr::StateSpace> configuration,
                       std::string frame,
                       std::shared_ptr<PinocchioModelWorkspace> workspace = {});
    std::unique_ptr<ilqr::Function> clone() const override;

  private:
    void evaluateImpl(ilqr::Input q, ilqr::VectorOutput out) override;
    void tangentJacobian(ilqr::Input q, ilqr::MatrixOutput jac) override;
};

/** Input [q; target xyz; target quaternion xyzw], output
 * [world position error; log(R_target.transpose() * R_frame)].
 * Target quaternions are normalized internally; zero/nonfinite targets fail.
 * Analytical derivatives include all seven target coordinates.
 */
class PinocchioFramePoseError final : public PinocchioFrameFunction
{
  public:
    PinocchioFramePoseError(
        std::shared_ptr<const pinocchio::Model> model,
        std::shared_ptr<const ilqr::StateSpace> configuration,
        std::string frame,
        std::shared_ptr<PinocchioModelWorkspace> workspace = {});
    std::unique_ptr<ilqr::Function> clone() const override;

  private:
    void evaluateImpl(ilqr::Input z, ilqr::VectorOutput out) override;
    void tangentJacobian(ilqr::Input z, ilqr::MatrixOutput jac) override;
};
} // namespace horizon::kinematics
#endif
