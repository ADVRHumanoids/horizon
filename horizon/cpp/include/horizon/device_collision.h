#pragma once
#include "horizon/collision.h"
#include <cstdint>

namespace horizon::collision
{
/** Host-side immutable export. Transforms store row-major rotation then xyz.
 * Scalar joint axes and floating-base velocities use joint-local coordinates.
 */
struct DeviceRobotModel
{
    enum JointType
    {
        Fixed,
        Revolute,
        Prismatic,
        FreeFlyer,
        Continuous
    };
    struct Joint
    {
        int parent = 0, type = Fixed, q = 0, v = 0;
        double placement[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
        double axis[3] = {0, 0, 1};
    };
    struct Sphere
    {
        int joint = 0;
        double center[3] = {0, 0, 0};
        double radius = 0;
    };
    int nq = 0, nv = 0;
    std::vector<Joint> joints;
    std::vector<Sphere> spheres;
};

struct DeviceCollisionStatistics
{
    std::uint64_t value_batches = 0, derivative_batches = 0, nodes = 0,
                  sdf_queries = 0, self_queries = 0;
    std::uint64_t upload_bytes = 0, download_bytes = 0, kernel_launches = 0,
                  device_bytes = 0;
    double prepare_ms = 0, upload_ms = 0, kinematics_ms = 0, queries_ms = 0,
           accumulation_ms = 0, download_ms = 0;
};

/** Entire collision pipeline in Kokkos::DefaultExecutionSpace. Only q is
 * uploaded; only cost / tangent gradient / packed GN Hessian are downloaded.
 * Application owns Kokkos lifetime. Grid/model/working buffers are resident.
 */
class DeviceCollisionCost final : public ilqr::BatchedCost
{
  public:
    DeviceCollisionCost(DeviceRobotModel model,
                        std::shared_ptr<const SdfGrid> grid,
                        std::vector<SphereDistanceFunction::Pair> pairs = {},
                        double weight = 1., double margin = .02,
                        double smoothing = .005, int state_size = 0,
                        int configuration_offset = 0);
    ~DeviceCollisionCost();
    std::unique_ptr<ilqr::BatchedCost> clone() const override;
    int stateSize() const override;
    int tangentSize() const override;
    void prepare(ilqr::MatrixInput states, bool derivatives) override;
    double value(int node) const override;
    ilqr::Input gradient(int node) const override;
    ilqr::Input packedHessian(int node) const override;
    DeviceCollisionStatistics statistics() const;
    void resetStatistics();
    static const char *executionSpace();

  private:
    struct Configuration;
    struct Impl;
    explicit DeviceCollisionCost(std::shared_ptr<const Configuration>);
    std::shared_ptr<const Configuration> _configuration;
    std::unique_ptr<Impl> _impl;
};
} // namespace horizon::collision
