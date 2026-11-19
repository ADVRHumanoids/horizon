#ifndef HORIZON_COLLISION_H
#define HORIZON_COLLISION_H

#include "ilqr/function.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace horizon::collision
{
struct LinkSphere
{
    std::string link;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    double radius = 0.;
};

/** Sphere model in link-local coordinates. Link transforms are applied by the
 * robot kinematics function before feeding centers to SphereDistanceFunction.
 */
class SphereModel
{
public:
    explicit SphereModel(std::vector<LinkSphere> spheres);
    const std::vector<LinkSphere>& spheres() const { return _spheres; }
    std::vector<Eigen::Vector3d> worldCenters(
        const std::vector<Eigen::Isometry3d>& link_poses,
        const std::vector<std::string>& link_names) const;
private:
    std::vector<LinkSphere> _spheres;
};

/** Read link-local sphere collision geometry from a URDF at runtime. */
SphereModel loadSphereModelFromUrdf(const std::string& path);

/** Link pairs for which collision checking is disabled, with MoveIt reasons. */
struct DisabledCollisionPair
{
    std::string link1, link2, reason;
};
std::vector<DisabledCollisionPair> loadDisabledCollisionPairs(const std::string& path);

/** Expand enabled link pairs into sphere pairs. Spheres on the same link never
 * collide with one another. All other link pairs are enabled unless explicitly
 * disabled; a missing or malformed policy must not silently disable checks. */
std::vector<std::pair<int, int>> makeSelfCollisionPairs(
    const SphereModel& model, const std::vector<DisabledCollisionPair>& disabled);

struct DistanceSample
{
    double distance;
    Eigen::Vector3d gradient;
};

class DistanceField
{
public:
    virtual ~DistanceField() = default;
    virtual DistanceSample query(const Eigen::Vector3d& point) const = 0;
};

/** Immutable, x-fastest dense signed-distance grid with trilinear queries. */
class SdfGrid final : public DistanceField
{
public:
    SdfGrid(Eigen::Vector3i dimensions, Eigen::Vector3d origin,
            double voxel_size, std::vector<float> values);
    DistanceSample query(const Eigen::Vector3d& point) const override;
    const Eigen::Vector3i& dimensions() const { return _dimensions; }
    const Eigen::Vector3d& origin() const { return _origin; }
    double voxelSize() const { return _voxel_size; }
    const std::vector<float>& values() const { return _values; }
private:
    std::size_t index(int x, int y, int z) const;
    Eigen::Vector3i _dimensions;
    Eigen::Vector3d _origin;
    double _voxel_size;
    std::vector<float> _values;
};

class SphereField final : public DistanceField
{
public:
    SphereField(Eigen::Vector3d center, double radius);
    DistanceSample query(const Eigen::Vector3d& point) const override;
private:
    Eigen::Vector3d _center;
    double _radius;
};

class PlaneField final : public DistanceField
{
public:
    PlaneField(Eigen::Vector3d point, Eigen::Vector3d normal);
    DistanceSample query(const Eigen::Vector3d& point) const override;
private:
    Eigen::Vector3d _point, _normal;
};

/** Residuals for each robot sphere against the closest environment field,
 * followed by explicitly selected self-collision pairs. Input is SoA world
 * centers [x0,x1,...,y0,y1,...,z0,z1,...]. Residuals use a smooth softplus
 * hinge on clearance below the requested margin.
 */
class SphereDistanceFunction final : public ilqr::Function
{
public:
    using Pair = std::pair<int, int>;
    SphereDistanceFunction(std::vector<double> radii,
                           std::vector<std::shared_ptr<const DistanceField>> environment,
                           std::vector<Pair> self_pairs = {},
                           double clearance_margin = 0.02,
                           double smoothing = 0.005);
    std::unique_ptr<ilqr::Function> clone() const override;
    bool hasJacobianProduct() const override { return true; }
protected:
    void evaluateImpl(ilqr::Input x, ilqr::VectorOutput value) override;
    void jacobianImpl(ilqr::Input x, ilqr::MatrixOutput jac) override;
    void valueAndJacobianImpl(ilqr::Input x, ilqr::VectorOutput value,
                              ilqr::MatrixOutput jac) override;
    void valueAndJacobianProductImpl(ilqr::Input x, ilqr::MatrixInput directions,
                                    ilqr::VectorOutput value,
                                    ilqr::MatrixOutput product) override;
private:
    void evaluateAndLinearize(ilqr::Input x, ilqr::VectorOutput value,
                              ilqr::MatrixOutput* jac,
                              const ilqr::MatrixInput* directions = nullptr,
                              ilqr::MatrixOutput* product = nullptr);
    std::vector<double> _radii;
    std::vector<std::shared_ptr<const DistanceField>> _environment;
    std::vector<Pair> _self_pairs;
    double _margin, _smoothing;
};
/** Signed clearances, with no penalty or margin baked into the function.
 * Environment clearances precede explicitly selected self-pair clearances.
 */
class SphereClearanceFunction final : public ilqr::Function
{
public:
    SphereClearanceFunction(std::vector<double> radii,
        std::vector<std::shared_ptr<const DistanceField>> environment = {},
        std::vector<std::pair<int,int>> pairs = {});
    std::unique_ptr<ilqr::Function> clone() const override;
    bool hasJacobianProduct() const override { return true; }
private:
    void evaluateImpl(ilqr::Input x, ilqr::VectorOutput value) override;
    void jacobianImpl(ilqr::Input x, ilqr::MatrixOutput jac) override;
    void valueAndJacobianProductImpl(ilqr::Input x, ilqr::MatrixInput directions,
        ilqr::VectorOutput value, ilqr::MatrixOutput product) override;
    void compute(ilqr::Input x, ilqr::VectorOutput value, const ilqr::MatrixInput* directions,
        ilqr::MatrixOutput* product);
    std::vector<double> _radii;
    std::vector<std::shared_ptr<const DistanceField>> _environment;
    std::vector<std::pair<int,int>> _pairs;
};

}
#endif
