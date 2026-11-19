#include "horizon/collision.h"
#include <cmath>
#include <limits>

namespace horizon::collision
{
SphereClearanceFunction::SphereClearanceFunction(
    std::vector<double> radii,
    std::vector<std::shared_ptr<const DistanceField>> environment,
    std::vector<std::pair<int, int>> pairs)
    : Function(3 * radii.size(),
               pairs.size() + (environment.empty() ? 0 : radii.size())),
      _radii(std::move(radii)), _environment(std::move(environment)),
      _pairs(std::move(pairs))
{
    for (double radius : _radii)
    {
        if (!std::isfinite(radius) || radius <= 0)
        {
            throw std::invalid_argument("invalid collision radius");
        }
    }
    for (const auto &field : _environment)
    {
        if (!field)
        {
            throw std::invalid_argument("null distance field");
        }
    }
    for (auto [i, j] : _pairs)
    {
        if (i < 0 || j < 0 || i >= _radii.size() || j >= _radii.size() ||
            i == j)
        {
            throw std::invalid_argument("invalid sphere collision pair");
        }
    }
}
std::unique_ptr<ilqr::Function> SphereClearanceFunction::clone() const
{
    return std::make_unique<SphereClearanceFunction>(_radii, _environment,
                                                     _pairs);
}
void SphereClearanceFunction::compute(ilqr::Input x, ilqr::VectorOutput value,
                                      const ilqr::MatrixInput *directions,
                                      ilqr::MatrixOutput *product)
{
    const int n = _radii.size();
    auto point = [&](int i)
    {
        return Eigen::Vector3d(x[i], x[n + i], x[2 * n + i]);
    };
    auto add = [&](int row, int i, const Eigen::Vector3d &gradient, double sign)
    {
        if (!directions)
        {
            return;
        }
        for (int a = 0; a < 3; ++a)
        {
            product->row(row).noalias() +=
                sign * gradient[a] * directions->row(a * n + i);
        }
    };
    if (product)
    {
        product->setZero();
    }
    int row = 0;
    if (!_environment.empty())
    {
        for (int i = 0; i < n; ++i)
        {
            DistanceSample closest{std::numeric_limits<double>::infinity(),
                                   Eigen::Vector3d::Zero()};
            for (const auto &field : _environment)
            {
                auto sample = field->query(point(i));
                if (sample.distance < closest.distance)
                {
                    closest = sample;
                }
            }
            value[row] = closest.distance - _radii[i];
            add(row, i, closest.gradient, 1.);
            ++row;
        }
    }
    for (auto [i, j] : _pairs)
    {
        const Eigen::Vector3d delta = point(i) - point(j);
        const double length = delta.norm();
        const Eigen::Vector3d gradient = length > 1e-12
            ? Eigen::Vector3d(delta / length)
            : Eigen::Vector3d::Zero();
        value[row] = length - _radii[i] - _radii[j];
        add(row, i, gradient, 1.);
        add(row, j, gradient, -1.);
        ++row;
    }
}
void SphereClearanceFunction::evaluateImpl(ilqr::Input x,
                                           ilqr::VectorOutput value)
{
    compute(x, value, nullptr, nullptr);
}
void SphereClearanceFunction::jacobianImpl(ilqr::Input x,
                                           ilqr::MatrixOutput jac)
{
    ilqr::Vector value(outputSize());
    ilqr::Matrix identity = ilqr::Matrix::Identity(inputSize(), inputSize());
    const ilqr::MatrixInput directions(identity);
    compute(x, value, &directions, &jac);
}
void SphereClearanceFunction::valueAndJacobianProductImpl(
    ilqr::Input x, ilqr::MatrixInput directions, ilqr::VectorOutput value,
    ilqr::MatrixOutput product)
{
    compute(x, value, &directions, &product);
}
} // namespace horizon::collision
