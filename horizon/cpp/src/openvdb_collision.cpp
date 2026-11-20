#include "horizon/openvdb_collision.h"
#include <cmath>
#include <limits>
#include <mutex>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <stdexcept>

namespace horizon::collision
{
struct OpenVdbDistanceField::Impl
{
    openvdb::FloatGrid::ConstPtr grid;
    Eigen::Matrix3d index_to_world;
};

namespace
{
std::once_flag init_flag;
}

OpenVdbDistanceField::OpenVdbDistanceField(const std::string &path)
{
    std::call_once(init_flag,
                   []
                   {
                       openvdb::initialize();
                   });
    auto impl = std::make_shared<Impl>();
    openvdb::io::File file(path);
    file.open();
    auto base = file.readGrid("sdf");
    file.close();
    impl->grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
    if (!impl->grid)
    {
        throw std::invalid_argument(
            "OpenVDB file must contain FloatGrid named 'sdf'");
    }
    if (impl->grid->activeVoxelCount() == 0)
    {
        throw std::invalid_argument("OpenVDB SDF has no active voxels");
    }
    const auto &transform = impl->grid->transform();
    if (!transform.isLinear())
    {
        throw std::invalid_argument(
            "OpenVDB collision fields require an affine transform");
    }
    const openvdb::Vec3d zero = transform.indexToWorld(openvdb::Vec3d(0, 0, 0));
    for (int c = 0; c < 3; ++c)
    {
        openvdb::Vec3d e(0, 0, 0);
        e[c] = 1.;
        const auto axis = transform.indexToWorld(e) - zero;
        for (int r = 0; r < 3; ++r)
        {
            impl->index_to_world(r, c) = axis[r];
        }
    }
    if (std::abs(impl->index_to_world.determinant()) < 1e-12)
    {
        throw std::invalid_argument("OpenVDB transform is singular");
    }
    _impl = std::move(impl);
}

DistanceSample OpenVdbDistanceField::query(const Eigen::Vector3d &point) const
{
    if (!point.allFinite())
    {
        throw ilqr::EvaluationError("OpenVDB query point is not finite");
    }
    const auto ijk = _impl->grid->transform().worldToIndex(
        openvdb::Vec3d(point.x(), point.y(), point.z()));
    const double index_limit =
        static_cast<double>(std::numeric_limits<int>::max() - 2);
    if (!std::isfinite(ijk.x()) || !std::isfinite(ijk.y()) ||
        !std::isfinite(ijk.z()) || std::abs(ijk.x()) > index_limit ||
        std::abs(ijk.y()) > index_limit || std::abs(ijk.z()) > index_limit)
    {
        throw ilqr::EvaluationError(
            "OpenVDB query index is outside the supported range");
    }
    // OpenVDB level sets are sparse and define a valid positive background
    // distance outside the active narrow band. Keep those points queryable so
    // collision-free initial states need not lie near an obstacle.
    const double coords[3] = {ijk.x(), ijk.y(), ijk.z()};
    int lo[3];
    double f[3];
    for (int a = 0; a < 3; ++a)
    {
        lo[a] = static_cast<int>(std::floor(coords[a]));
        f[a] = coords[a] - lo[a];
    }
    const auto accessor = _impl->grid->getConstAccessor();
    double distance = 0.;
    Eigen::Vector3d grad_index = Eigen::Vector3d::Zero();
    for (int z = 0; z < 2; ++z)
    {
        for (int y = 0; y < 2; ++y)
        {
            for (int x = 0; x < 2; ++x)
            {
                const int bit[3] = {x, y, z};
                const double w[3] = {x ? f[0] : 1. - f[0], y ? f[1] : 1. - f[1],
                                     z ? f[2] : 1. - f[2]};
                const double v = accessor.getValue(
                    openvdb::Coord(lo[0] + x, lo[1] + y, lo[2] + z));
                distance += w[0] * w[1] * w[2] * v;
                for (int a = 0; a < 3; ++a)
                {
                    double dw = bit[a] ? 1. : -1.;
                    for (int b = 0; b < 3; ++b)
                    {
                        if (a != b)
                        {
                            dw *= w[b];
                        }
                    }
                    grad_index[a] += dw * v;
                }
            }
        }
    }
    return {distance, _impl->index_to_world.inverse().transpose() * grad_index};
}
} // namespace horizon::collision
