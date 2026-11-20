#ifndef HORIZON_OPENVDB_COLLISION_H
#define HORIZON_OPENVDB_COLLISION_H
#include "horizon/collision.h"
#ifndef HORIZON_WITH_OPENVDB
#error "Enable HORIZON_WITH_OPENVDB to use OpenVDB collision fields"
#endif
namespace horizon::collision
{
class OpenVdbDistanceField final : public DistanceField
{
  public:
    explicit OpenVdbDistanceField(const std::string &path);
    DistanceSample query(const Eigen::Vector3d &point) const override;

  private:
    struct Impl;
    std::shared_ptr<const Impl> _impl;
};
} // namespace horizon::collision
#endif
