#include "horizon/collision.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace horizon::collision
{
SphereModel::SphereModel(std::vector<LinkSphere> spheres) : _spheres(std::move(spheres))
{
    for(const auto& sphere : _spheres)
        if(sphere.link.empty() || !sphere.center.allFinite() ||
           !std::isfinite(sphere.radius) || sphere.radius <= 0.)
            throw std::invalid_argument("sphere model contains invalid link, center, or radius");
}

std::vector<Eigen::Vector3d> SphereModel::worldCenters(
    const std::vector<Eigen::Isometry3d>& poses,
    const std::vector<std::string>& names) const
{
    if(poses.size() != names.size()) throw std::invalid_argument("link pose/name count mismatch");
    std::unordered_map<std::string, std::size_t> index;
    for(std::size_t i = 0; i < names.size(); ++i)
        if(!index.emplace(names[i], i).second) throw std::invalid_argument("duplicate link name");
    std::vector<Eigen::Vector3d> result;
    result.reserve(_spheres.size());
    for(const auto& sphere : _spheres)
    {
        auto it = index.find(sphere.link);
        if(it == index.end()) throw std::invalid_argument("missing link pose for sphere model: " + sphere.link);
        result.push_back(poses[it->second] * sphere.center);
    }
    return result;
}

SdfGrid::SdfGrid(Eigen::Vector3i dims, Eigen::Vector3d origin,
                 double voxel, std::vector<float> values)
    : _dimensions(dims), _origin(std::move(origin)), _voxel_size(voxel), _values(std::move(values))
{
    const auto count = static_cast<std::size_t>(dims.x()) * dims.y() * dims.z();
    if((dims.array() < 2).any() || !_origin.allFinite() || !std::isfinite(voxel) || voxel <= 0. ||
       _values.size() != count)
        throw std::invalid_argument("invalid SDF grid dimensions, origin, voxel size, or value count");
    for(float v : _values) if(!std::isfinite(v)) throw std::invalid_argument("SDF contains nonfinite values");
}

std::size_t SdfGrid::index(int x, int y, int z) const
{
    return (static_cast<std::size_t>(z) * _dimensions.y() + y) * _dimensions.x() + x;
}

DistanceSample SdfGrid::query(const Eigen::Vector3d& point) const
{
    if(!point.allFinite()) throw ilqr::EvaluationError("SDF query point is not finite");
    const Eigen::Vector3d p = (point - _origin) / _voxel_size;
    if((p.array() < 0.).any() || (p.array() > (_dimensions.array() - 1).cast<double>()).any())
        throw ilqr::EvaluationError("SDF query lies outside the grid");
    Eigen::Vector3i lo;
    Eigen::Vector3d f;
    for(int a = 0; a < 3; ++a)
    {
        lo[a] = std::min(static_cast<int>(std::floor(p[a])), _dimensions[a] - 2);
        f[a] = p[a] - lo[a];
    }
    double distance = 0.;
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
    for(int z = 0; z < 2; ++z) for(int y = 0; y < 2; ++y) for(int x = 0; x < 2; ++x)
    {
        const double wx = x ? f.x() : 1. - f.x();
        const double wy = y ? f.y() : 1. - f.y();
        const double wz = z ? f.z() : 1. - f.z();
        const double v = _values[index(lo.x() + x, lo.y() + y, lo.z() + z)];
        distance += wx * wy * wz * v;
        gradient.x() += (x ? 1. : -1.) * wy * wz * v / _voxel_size;
        gradient.y() += (y ? 1. : -1.) * wx * wz * v / _voxel_size;
        gradient.z() += (z ? 1. : -1.) * wx * wy * v / _voxel_size;
    }
    return {distance, gradient};
}

SphereField::SphereField(Eigen::Vector3d center, double radius)
    : _center(std::move(center)), _radius(radius)
{
    if(!_center.allFinite() || !std::isfinite(radius) || radius <= 0.)
        throw std::invalid_argument("invalid sphere obstacle");
}
DistanceSample SphereField::query(const Eigen::Vector3d& point) const
{
    const Eigen::Vector3d d = point - _center;
    const double length = std::sqrt(d.squaredNorm() + 1e-24);
    return {length - _radius, d / length};
}

PlaneField::PlaneField(Eigen::Vector3d point, Eigen::Vector3d normal)
    : _point(std::move(point)), _normal(std::move(normal))
{
    if(!_point.allFinite() || !normal.allFinite() || normal.norm() < 1e-12)
        throw std::invalid_argument("invalid plane obstacle");
    _normal.normalize();
}
DistanceSample PlaneField::query(const Eigen::Vector3d& point) const
{
    return {_normal.dot(point - _point), _normal};
}

namespace
{
double softplus(double x)
{
    if(x > 40.) return x;
    if(x < -40.) return std::exp(x);
    return std::log1p(std::exp(x));
}
double sigmoid(double x)
{
    if(x >= 0.) return 1. / (1. + std::exp(-std::min(x, 700.)));
    const double e = std::exp(std::max(x, -700.));
    return e / (1. + e);
}
}

SphereDistanceFunction::SphereDistanceFunction(
    std::vector<double> radii,
    std::vector<std::shared_ptr<const DistanceField>> environment,
    std::vector<Pair> pairs, double margin, double smoothing)
    : Function(static_cast<int>(3 * radii.size()),
               static_cast<int>(radii.size() * (environment.empty() ? 0 : 1) + pairs.size())),
      _radii(std::move(radii)), _environment(std::move(environment)),
      _self_pairs(std::move(pairs)), _margin(margin), _smoothing(smoothing)
{
    if(_radii.empty() || (_environment.empty() && _self_pairs.empty()))
        throw std::invalid_argument("collision function needs robot spheres and at least one distance query");
    if(!std::isfinite(margin) || !std::isfinite(smoothing) || smoothing <= 0.)
        throw std::invalid_argument("collision margin and smoothing must be finite; smoothing must be positive");
    for(double r : _radii) if(!std::isfinite(r) || r <= 0.) throw std::invalid_argument("invalid robot sphere radius");
    for(const auto& f : _environment) if(!f) throw std::invalid_argument("null distance field");
    for(const auto& p : _self_pairs)
        if(p.first < 0 || p.second < 0 || p.first >= static_cast<int>(_radii.size()) ||
           p.second >= static_cast<int>(_radii.size()) || p.first == p.second)
            throw std::invalid_argument("invalid self-collision pair");
}

std::unique_ptr<ilqr::Function> SphereDistanceFunction::clone() const
{
    return std::make_unique<SphereDistanceFunction>(_radii, _environment, _self_pairs, _margin, _smoothing);
}
void SphereDistanceFunction::evaluateImpl(ilqr::Input x, ilqr::VectorOutput value)
{ evaluateAndLinearize(x, value, nullptr); }
void SphereDistanceFunction::jacobianImpl(ilqr::Input x, ilqr::MatrixOutput jac)
{
    ilqr::Vector value(outputSize());
    evaluateAndLinearize(x, value, &jac);
}
void SphereDistanceFunction::valueAndJacobianImpl(ilqr::Input x, ilqr::VectorOutput value, ilqr::MatrixOutput jac)
{ evaluateAndLinearize(x, value, &jac); }
void SphereDistanceFunction::valueAndJacobianProductImpl(
    ilqr::Input x, ilqr::MatrixInput directions, ilqr::VectorOutput value,
    ilqr::MatrixOutput product)
{ evaluateAndLinearize(x, value, nullptr, &directions, &product); }

void SphereDistanceFunction::evaluateAndLinearize(ilqr::Input x, ilqr::VectorOutput value,
                                                  ilqr::MatrixOutput* jac,
                                                  const ilqr::MatrixInput* directions,
                                                  ilqr::MatrixOutput* product)
{
    if(jac) jac->setZero();
    if(product) product->setZero();
    int row = 0;
    const int count = static_cast<int>(_radii.size());
    auto addGradient = [&](int output, int sphere, const Eigen::Vector3d& gradient)
    {
        if(jac)
        {
            (*jac)(output, sphere) = gradient.x();
            (*jac)(output, count + sphere) = gradient.y();
            (*jac)(output, 2 * count + sphere) = gradient.z();
        }
        if(product)
        {
            product->row(output).noalias() += gradient.x() * directions->row(sphere)
                + gradient.y() * directions->row(count + sphere)
                + gradient.z() * directions->row(2 * count + sphere);
        }
    };
    for(int i = 0; i < static_cast<int>(_radii.size()); ++i)
    {
        const Eigen::Vector3d c(x[i], x[count + i], x[2 * count + i]);
        if(!_environment.empty())
        {
            DistanceSample closest{std::numeric_limits<double>::infinity(), Eigen::Vector3d::Zero()};
            for(const auto& field : _environment)
            {
                auto sample = field->query(c);
                if(sample.distance < closest.distance) closest = sample;
            }
            const double clear = closest.distance - _radii[i];
            const double z = (_margin - clear) / _smoothing;
            value[row] = _smoothing * softplus(z);
            if(jac || product) addGradient(row, i, -sigmoid(z) * closest.gradient);
            ++row;
        }
    }
    for(const auto& pair : _self_pairs)
    {
        const int i = pair.first, j = pair.second;
        const Eigen::Vector3d d(x[i] - x[j], x[count + i] - x[count + j],
                                x[2 * count + i] - x[2 * count + j]);
        const double length = std::sqrt(d.squaredNorm() + 1e-24);
        const Eigen::Vector3d direction = d / length;
        const double clear = length - _radii[i] - _radii[j];
        const double z = (_margin - clear) / _smoothing;
        value[row] = _smoothing * softplus(z);
        if(jac || product)
        {
            const double scale = sigmoid(z);
            addGradient(row, i, -scale * direction);
            addGradient(row, j, scale * direction);
        }
        ++row;
    }
}
}
