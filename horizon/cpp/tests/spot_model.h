#pragma once

#include "ilqr/state_space.h"
#include "spot_formulation.h"
#include <Eigen/LU>
#include <array>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace spot_test
{
using namespace horizon::ilqr;
namespace pin = pinocchio;

class ConfigurationSpace final : public StateSpace
{
  public:
    explicit ConfigurationSpace(std::shared_ptr<const pin::Model> model)
        : model(std::move(model))
    {
    }
    int stateSize() const override
    {
        return nq;
    }
    int tangentSize() const override
    {
        return nv;
    }
    void neutral(VectorOutput x) const override;
    void integrate(Input x, Input dx, VectorOutput out) const override;
    void difference(Input from, Input to, VectorOutput out) const override;
    void retractionJacobian(Input x, MatrixOutput jac) const override;
    void tangentProjection(Input x, MatrixOutput projection) const override;
    void projectDirections(Input x, MatrixInput storage,
                           MatrixOutput tangent) const override;
    void retractionHessian(Input x, Input weights,
                           MatrixOutput hess) const override;
    void differenceJacobians(Input from, Input to, MatrixOutput a,
                             MatrixOutput b) const override;
    std::shared_ptr<const pin::Model> model;
};

struct Robot
{
    std::shared_ptr<const pin::Model> model;
    std::shared_ptr<const ConfigurationSpace> configuration;
    std::shared_ptr<const StateSpace> state;
    std::array<pin::FrameIndex, 4> feet;
    Vector standing, anchors, support;
    double weight;
    explicit Robot(const std::string &urdf);
};

/** Direct Pinocchio evaluation. Terms are [tau(18), contact acceleration(12),
 * positions(12), velocities(12)]. Jacobian columns are
 * [dq(18),v(18),a(18),f(12)]. Contact forces use LOCAL_WORLD_ALIGNED, applied
 * at the four foot origins.
 */
class Engine
{
  public:
    explicit Engine(std::shared_ptr<const Robot> robot);
    void evaluate(Input q, Input v, Input a, Input f, VectorOutput terms);
    void linearize(Input q, Input v, Input a, Input f, MatrixOutput jac);
    void forward(Input q, Input v, Input tau, VectorOutput acceleration_force);
    void forwardJacobian(Input q, Input v, Input tau, VectorOutput af,
                         MatrixOutput jac, MatrixOutput terms_jac);
    const Vector &lastLinearizedTerms() const
    {
        return terms;
    }
    std::shared_ptr<const Robot> robot;

  private:
    pin::Data dynamics, kinematics;
    pin::container::aligned_vector<pin::Force> external_forces;
    Matrix J, Vq, Vv, Aq, Av, Aa, M, K, residualJac;
    Vector zero, terms;
    Eigen::PartialPivLU<Matrix> lu;
};

std::shared_ptr<const Function> makeDynamics(std::shared_ptr<const Robot>,
                                             Formulation, double dt,
                                             bool project_contacts = false);
std::shared_ptr<const Function> makeStage(std::shared_ptr<const Robot>,
                                          Formulation);
} // namespace spot_test
