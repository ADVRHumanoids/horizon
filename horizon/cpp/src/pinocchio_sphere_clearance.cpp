#include "horizon/pinocchio_collision.h"
namespace horizon::collision
{
PinocchioSphereClearance::PinocchioSphereClearance(
    std::shared_ptr<const pinocchio::Model> model,
    std::shared_ptr<const ilqr::StateSpace> space, SphereModel spheres,
    std::vector<std::shared_ptr<const DistanceField>> environment,
    std::vector<std::pair<int, int>> pairs,
    std::shared_ptr<PinocchioModelWorkspace> workspace)
    : TangentFunction(space,
                      pairs.size() +
                          (environment.empty() ? 0 : spheres.spheres().size())),
      _model(std::move(model)), _space(std::move(space)),
      _spheres(std::move(spheres)), _environment(std::move(environment)),
      _pairs(std::move(pairs)),
      _workspace(workspace ? std::move(workspace)
                           : std::make_shared<PinocchioModelWorkspace>(_model)),
      _centers(_model, _space, _spheres, _workspace),
      _clearance(
          [&]
          {
              std::vector<double> r;
              for (const auto &s : _spheres.spheres())
              {
                  r.push_back(s.radius);
              }
              return r;
          }(),
          _environment, _pairs),
      _points(3 * _spheres.spheres().size()), _value(outputSize()),
      _jacobian(_points.size(), _space->tangentSize())
{
}
std::unique_ptr<ilqr::Function> PinocchioSphereClearance::clone() const
{
    return std::make_unique<PinocchioSphereClearance>(_model, _space, _spheres,
                                                      _environment, _pairs,
                                                      _workspace->cloneEmpty());
}
void PinocchioSphereClearance::setEvaluationContext(
    const std::shared_ptr<ilqr::EvaluationContext> &context)
{
    _centers.setEvaluationContext(context);
}
void PinocchioSphereClearance::evaluateImpl(ilqr::Input q,
                                            ilqr::VectorOutput value)
{
    _centers.evaluate(q, _points);
    _clearance.evaluate(_points, value);
}
void PinocchioSphereClearance::tangentJacobian(ilqr::Input q,
                                               ilqr::MatrixOutput jac)
{
    valueAndTangentJacobian(q, _value, jac);
}
void PinocchioSphereClearance::valueAndTangentJacobian(ilqr::Input q,
                                                       ilqr::VectorOutput value,
                                                       ilqr::MatrixOutput jac)
{
    _centers.valueAndTangentLinearization(q, _points, _jacobian);
    _clearance.valueAndJacobianProduct(_points, _jacobian, value, jac);
}
} // namespace horizon::collision
