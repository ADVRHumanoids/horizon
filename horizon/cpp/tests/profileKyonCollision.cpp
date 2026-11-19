
#include "horizon/pinocchio_collision.h"
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace pin = pinocchio;
using namespace horizon::ilqr;
using namespace horizon::collision;

namespace
{
class KyonSpace final : public StateSpace
{
public:
    explicit KyonSpace(std::shared_ptr<const pin::Model> m) : model(std::move(m)) {}
    int stateSize() const override { return model->nq; }
    int tangentSize() const override { return model->nv; }
    void neutral(VectorOutput q) const override { q = pin::neutral(*model); }
    void integrate(Input q, Input dq, VectorOutput out) const override { out = pin::integrate(*model, q, dq); }
    void difference(Input a, Input b, VectorOutput out) const override { out = pin::difference(*model, a, b); }
    std::shared_ptr<const pin::Model> model;
};

template<class F>
void measure(const char* name, int iterations, F&& fn)
{
    double checksum = 0.;
    const auto begin = std::chrono::steady_clock::now();
    for(int i = 0; i < iterations; ++i) checksum += fn(i);
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - begin).count();
    std::cout << name << ": " << seconds * 1e6 / iterations << " us/op, "
              << iterations / seconds << " op/s, checksum=" << checksum << '\n';
}
}

int main(int argc, char** argv)
{
    const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 2000;
    if (argc < 4) throw std::invalid_argument("usage: profileKyonCollision ITERATIONS SPHERE_URDF COLLISION_JSON");
    auto model = std::make_shared<pin::Model>();
    pin::urdf::buildModel(argv[2], *model);
    auto sphere_model = std::make_shared<SphereModel>(loadSphereModelFromUrdf(argv[2]));
    const auto pairs = makeSelfCollisionPairs(*sphere_model, loadDisabledCollisionPairs(argv[3]));
    auto space = std::make_shared<KyonSpace>(model);
    auto workspace = std::make_shared<PinocchioModelWorkspace>(model);
    PinocchioSphereCenters centers_function(model, space, *sphere_model, workspace);
    Vector q(model->nq), center_values(3 * sphere_model->spheres().size());
    space->neutral(q);

    std::vector<double> radii;
    for(const auto& sphere : sphere_model->spheres()) radii.push_back(sphere.radius);
    auto plane = std::make_shared<PlaneField>(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());
    SphereDistanceFunction plane_function(radii, {plane}, pairs);
    Vector plane_values(plane_function.outputSize());
    Matrix plane_jac(plane_function.outputSize(), plane_function.inputSize());
    Matrix directions = Matrix::Zero(plane_function.inputSize(), model->nv);
    for(int row = 0; row < directions.rows(); ++row)
        for(int col = 0; col < directions.cols(); ++col)
            directions(row, col) = std::sin(0.013 * (row + 1) * (col + 1));
    Matrix plane_product(plane_function.outputSize(), model->nv);

    const Eigen::Vector3i dims(101, 101, 101);
    const Eigen::Vector3d origin(-2.5, -2.5, -2.5);
    constexpr double voxel = 0.05;
    std::vector<float> grid_values(static_cast<std::size_t>(dims.prod()));
    for(int z = 0; z < dims.z(); ++z)
        for(int y = 0; y < dims.y(); ++y)
            for(int x = 0; x < dims.x(); ++x)
                grid_values[(static_cast<std::size_t>(z) * dims.y() + y) * dims.x() + x] =
                    static_cast<float>(origin.z() + z * voxel);
    auto grid = std::make_shared<SdfGrid>(dims, origin, voxel, std::move(grid_values));
    SphereDistanceFunction grid_function(radii, {grid}, pairs);
    Vector grid_values_out(grid_function.outputSize());
    Matrix grid_jac(grid_function.outputSize(), grid_function.inputSize());
    Matrix center_jac(center_values.size(), model->nv);

    auto vary = [&](int i) { q[model->nq - 1] = 1e-5 * std::sin(0.01 * i); };
    // Warm caches and code pages before collecting timings.
    for(int i = 0; i < 100; ++i)
    {
        vary(i);
        centers_function.evaluate(q, center_values);
        centers_function.valueAndTangentLinearization(q, center_values, center_jac);
        plane_function.valueAndJacobian(center_values, plane_values, plane_jac);
        grid_function.valueAndJacobian(center_values, grid_values_out, grid_jac);
    }
    std::cout << "Kyon nq=" << model->nq << " nv=" << model->nv
              << " spheres=" << sphere_model->spheres().size()
              << " self_pairs=" << pairs.size()
              << " iterations=" << iterations << '\n';

    measure("Pinocchio sphere centers/value", iterations, [&](int i)
    {
        vary(i); centers_function.evaluate(q, center_values); return center_values.sum();
    });
    measure("Pinocchio sphere centers/value+Jacobian", iterations, [&](int i)
    {
        vary(i); centers_function.valueAndTangentLinearization(q, center_values, center_jac);
        return center_values.sum() + center_jac.sum();
    });
    measure("plane collision/value", iterations, [&](int)
    {
        plane_function.evaluate(center_values, plane_values); return plane_values.sum();
    });
    measure("plane collision/value+Jacobian", iterations, [&](int)
    {
        plane_function.valueAndJacobian(center_values, plane_values, plane_jac);
        return plane_values.sum() + plane_jac.sum();
    });
    measure("dense SDF collision/value", iterations, [&](int)
    {
        grid_function.evaluate(center_values, grid_values_out); return grid_values_out.sum();
    });
    measure("dense SDF collision/value+Jacobian", iterations, [&](int)
    {
        grid_function.valueAndJacobian(center_values, grid_values_out, grid_jac);
        return grid_values_out.sum() + grid_jac.sum();
    });
    measure("dense SDF collision/value+JacobianProduct", iterations, [&](int)
    {
        grid_function.valueAndJacobianProduct(center_values, directions, grid_values_out, plane_product);
        return grid_values_out.sum() + plane_product.sum();
    });
    measure("combined centers+SDF/value+Jacobians", iterations, [&](int i)
    {
        vary(i);
        centers_function.valueAndTangentLinearization(q, center_values, center_jac);
        grid_function.valueAndJacobian(center_values, grid_values_out, grid_jac);
        return center_jac.sum() + grid_jac.sum() + grid_values_out.sum();
    });
}
