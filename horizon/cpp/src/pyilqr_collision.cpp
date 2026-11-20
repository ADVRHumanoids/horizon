#include "horizon/collision.h"
#ifdef HORIZON_WITH_OPENVDB
#include "horizon/openvdb_collision.h"
#endif
#if defined(HORIZON_WITH_KOKKOS) && defined(HORIZON_WITH_PINOCCHIO)
#include "horizon/device_collision.h"
#include "horizon/kokkos_runtime.h"
#endif
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace horizon::collision;
void bindIlqrCollision(py::module_ &m)
{
#if defined(HORIZON_WITH_KOKKOS) && defined(HORIZON_WITH_PINOCCHIO)
    py::class_<KokkosRuntime>(m, "KokkosRuntime").def(py::init<>());
    py::class_<DeviceCollisionCost, std::shared_ptr<DeviceCollisionCost>>(
        m, "DeviceCollisionCost")
        .def_static("execution_space", &DeviceCollisionCost::executionSpace)
        .def("batch_counts",
             [](const DeviceCollisionCost &cost)
             {
                 auto stats = cost.statistics();
                 return py::make_tuple(stats.value_batches,
                                       stats.derivative_batches);
             });
#endif
    py::class_<DistanceField, std::shared_ptr<DistanceField>>(m,
                                                              "DistanceField")
        .def("query",
             [](const DistanceField &field, const Eigen::Vector3d &p)
             {
                 auto s = field.query(p);
                 return py::make_tuple(s.distance, s.gradient);
             });
    py::class_<SdfGrid, DistanceField, std::shared_ptr<SdfGrid>>(m, "SdfGrid")
        .def(
            py::init(
                [](Eigen::Vector3i dims, Eigen::Vector3d origin, double voxel,
                   py::array_t<float, py::array::c_style | py::array::forcecast>
                       values)
                {
                    return std::make_shared<SdfGrid>(
                        dims, origin, voxel,
                        std::vector<float>(values.data(),
                                           values.data() + values.size()));
                }),
            py::arg("dimensions"), py::arg("origin"), py::arg("voxel_size"),
            py::arg("values"))
        .def_property_readonly("dimensions", &SdfGrid::dimensions)
        .def_property_readonly("origin", &SdfGrid::origin)
        .def_property_readonly("voxel_size", &SdfGrid::voxelSize);
#ifdef HORIZON_WITH_OPENVDB
    py::class_<OpenVdbDistanceField, DistanceField,
               std::shared_ptr<OpenVdbDistanceField>>(m, "OpenVdbDistanceField")
        .def(py::init<const std::string &>());
#endif
    py::class_<SphereField, DistanceField, std::shared_ptr<SphereField>>(
        m, "SphereField")
        .def(py::init<Eigen::Vector3d, double>());
    py::class_<PlaneField, DistanceField, std::shared_ptr<PlaneField>>(
        m, "PlaneField")
        .def(py::init<Eigen::Vector3d, Eigen::Vector3d>());
}
