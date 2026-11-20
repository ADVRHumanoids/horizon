#include "horizon/kokkos_collision.h"
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>

namespace horizon::collision
{
namespace
{
using namespace ilqr;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

struct NodeResult
{
    Vector input, value;
    Matrix jacobian;
    bool ready = false, derivatives = false;
};

class PreparedCollision final : public Function
{
  public:
    PreparedCollision(int nin, int nout, std::shared_ptr<const int> key)
        : Function(nin, nout), key(std::move(key))
    {
    }
    std::unique_ptr<Function> clone() const override
    {
        return std::make_unique<PreparedCollision>(inputSize(), outputSize(),
                                                   key);
    }
    bool hasJacobianProduct() const override
    {
        return true;
    }
    void setEvaluationContext(
        const std::shared_ptr<EvaluationContext> &context) override
    {
        result = context ? context->shareObject<NodeResult>(
                               key.get(),
                               []
                               {
                                   return std::make_shared<NodeResult>();
                               })
                         : nullptr;
    }

  private:
    void check(Input x, bool derivatives) const
    {
        if (!result || !result->ready ||
            (derivatives && !result->derivatives) ||
            result->input.size() != x.size() ||
            std::memcmp(result->input.data(), x.data(),
                        x.size() * sizeof(double)))
        {
            throw std::logic_error(
                "Kokkos collision was not prepared for this node/input/pass");
        }
    }
    void evaluateImpl(Input x, VectorOutput v) override
    {
        check(x, false);
        v = result->value;
    }
    void jacobianImpl(Input x, MatrixOutput j) override
    {
        check(x, true);
        j = result->jacobian;
    }
    void valueAndJacobianImpl(Input x, VectorOutput v, MatrixOutput j) override
    {
        check(x, true);
        v = result->value;
        j = result->jacobian;
    }
    void valueAndJacobianProductImpl(Input x, MatrixInput d, VectorOutput v,
                                     MatrixOutput p) override
    {
        check(x, true);
        v = result->value;
        multiplyJacobian(result->jacobian, d, p);
    }
    std::shared_ptr<const int> key;
    std::shared_ptr<NodeResult> result;
};
struct Statistics
{
    mutable std::mutex mutex;
    CollisionBatchStatistics value;
};
} // namespace

struct KokkosCollisionBatch::Configuration
{
    std::shared_ptr<const ilqr::Function> centers;
    std::vector<double> radii;
    std::shared_ptr<const SdfGrid> grid;
    std::vector<SphereDistanceFunction::Pair> pairs;
    double margin, smoothing;
    std::shared_ptr<const int> key = std::make_shared<const int>(0);
    std::shared_ptr<Statistics> statistics = std::make_shared<Statistics>();
    int outputs() const
    {
        return (grid ? int(radii.size()) : 0) + int(pairs.size());
    }
};

struct KokkosCollisionBatch::Impl
{
    using Space = Kokkos::DefaultExecutionSpace;
    using Memory = Space::memory_space;
    using V1 = Kokkos::View<double *, Memory>;
    using V2 = Kokkos::View<double **, Kokkos::LayoutRight, Memory>;
    using V3 = Kokkos::View<double ***, Kokkos::LayoutRight, Memory>;
    using V4 = Kokkos::View<double ****, Kokkos::LayoutRight, Memory>;
    Kokkos::View<float *, Memory> grid;
    V1 radii;
    Kokkos::View<int **, Kokkos::LayoutRight, Memory> pairs;
    V3 centers, gradients, jacobian;
    V4 center_jacobian;
    V2 residual;
    V3::HostMirror host_centers, host_jacobian;
    V4::HostMirror host_center_jacobian;
    V2::HostMirror host_residual;
    Kokkos::View<int, Memory> error;
    int nodes = 0;
    explicit Impl(const Configuration &c)
    {
        if (!Kokkos::is_initialized())
        {
            throw std::logic_error(
                "initialize Kokkos before creating a collision batch");
        }
        radii = V1("collision radii", c.radii.size());
        auto hr = Kokkos::create_mirror_view(radii);
        for (std::size_t i = 0; i < c.radii.size(); ++i)
        {
            hr(i) = c.radii[i];
        }
        Kokkos::deep_copy(radii, hr);
        pairs = decltype(pairs)("self collision pairs", c.pairs.size(), 2);
        auto hp = Kokkos::create_mirror_view(pairs);
        for (std::size_t i = 0; i < c.pairs.size(); ++i)
        {
            hp(i, 0) = c.pairs[i].first;
            hp(i, 1) = c.pairs[i].second;
        }
        Kokkos::deep_copy(pairs, hp);
        if (c.grid)
        {
            grid = decltype(grid)("environment sdf", c.grid->values().size());
            auto hg = Kokkos::create_mirror_view(grid);
            for (std::size_t i = 0; i < c.grid->values().size(); ++i)
            {
                hg(i) = c.grid->values()[i];
            }
            Kokkos::deep_copy(grid, hg);
        }
        error = decltype(error)("collision error");
    }
    void resize(int n, const Configuration &c, bool derivatives)
    {
        const int spheres = c.radii.size(), nout = c.outputs(),
                  nin = c.centers->inputSize();
        if (nodes != n)
        {
            nodes = n;
            centers = V3("world sphere centers", n, 3, spheres);
            residual = V2("collision residuals", n, nout);
            gradients = V3("collision spatial gradients", n, nout, 3);
            host_centers = Kokkos::create_mirror_view(centers);
            host_residual = Kokkos::create_mirror_view(residual);
            center_jacobian = V4();
            jacobian = V3();
        }
        if (derivatives && center_jacobian.extent(0) != std::size_t(n))
        {
            center_jacobian =
                V4("sphere storage jacobians", n, 3, spheres, nin);
            jacobian = V3("contracted collision jacobians", n, nout, nin);
            host_center_jacobian = Kokkos::create_mirror_view(center_jacobian);
            host_jacobian = Kokkos::create_mirror_view(jacobian);
        }
    }
};

KokkosCollisionBatch::KokkosCollisionBatch(
    std::shared_ptr<const ilqr::Function> centers, std::vector<double> radii,
    std::shared_ptr<const SdfGrid> grid,
    std::vector<SphereDistanceFunction::Pair> pairs, double margin,
    double smoothing)
{
    if (!centers || centers->outputSize() % 3 || radii.empty() ||
        centers->outputSize() / 3 < int(radii.size()))
    {
        throw std::invalid_argument(
            "invalid Kokkos sphere-center function or sphere count");
    }
    // Reuse the scalar backend's geometry and numeric option validation.
    std::vector<std::shared_ptr<const DistanceField>> fields;
    if (grid)
    {
        fields.push_back(grid);
    }
    SphereDistanceFunction validate(radii, fields, pairs, margin, smoothing);
    auto c = std::make_shared<Configuration>();
    c->centers = std::move(centers);
    c->radii = std::move(radii);
    c->grid = std::move(grid);
    c->pairs = std::move(pairs);
    c->margin = margin;
    c->smoothing = smoothing;
    _configuration = c;
    _impl = std::make_unique<Impl>(*c);
}
KokkosCollisionBatch::KokkosCollisionBatch(
    std::shared_ptr<const Configuration> c)
    : _configuration(std::move(c)),
      _impl(std::make_unique<Impl>(*_configuration))
{
}
KokkosCollisionBatch::~KokkosCollisionBatch() = default;
std::unique_ptr<ilqr::EvaluationBatch> KokkosCollisionBatch::clone() const
{
    return std::unique_ptr<ilqr::EvaluationBatch>(
        new KokkosCollisionBatch(_configuration));
}
std::shared_ptr<const ilqr::Function> KokkosCollisionBatch::function() const
{
    return std::make_shared<PreparedCollision>(
        _configuration->centers->inputSize(), _configuration->outputs(),
        _configuration->key);
}
const char *KokkosCollisionBatch::executionSpace()
{
    return Impl::Space::name();
}
CollisionBatchStatistics KokkosCollisionBatch::statistics() const
{
    std::lock_guard lock(_configuration->statistics->mutex);
    return _configuration->statistics->value;
}
void KokkosCollisionBatch::resetStatistics()
{
    std::lock_guard lock(_configuration->statistics->mutex);
    _configuration->statistics->value = {};
}

void KokkosCollisionBatch::prepare(
    ilqr::MatrixInput states, ilqr::MatrixInput controls,
    const std::vector<std::shared_ptr<ilqr::EvaluationContext>> &contexts,
    bool derivatives)
{
    const auto start = Clock::now();
    const auto &c = *_configuration;
    auto &w = *_impl;
    if (states.cols() > std::numeric_limits<int>::max() / c.outputs())
    {
        throw std::invalid_argument(
            "Kokkos collision batch exceeds the supported index range");
    }
    const int n = states.cols(), spheres = c.radii.size(), nout = c.outputs(),
              nin = c.centers->inputSize();
    if (n < 1 || states.rows() != nin || controls.cols() != n - 1 ||
        contexts.size() != std::size_t(n) || !states.allFinite())
    {
        throw std::invalid_argument("invalid trajectory dimensions or values "
                                    "for Kokkos collision batch");
    }
    std::vector<std::shared_ptr<NodeResult>> results;
    results.reserve(n);
    std::set<const ilqr::EvaluationContext *> unique_contexts;
    for (const auto &context : contexts)
    {
        if (!context)
        {
            throw std::invalid_argument("null node evaluation context");
        }
        if (!unique_contexts.insert(context.get()).second)
        {
            throw std::invalid_argument(
                "Kokkos collision batch requires distinct node contexts");
        }
        results.push_back(context->shareObject<NodeResult>(
            c.key.get(),
            []
            {
                return std::make_shared<NodeResult>();
            }));
        results.back()->ready = false;
    }
    w.resize(n, c, derivatives);
    const auto kinematics_start = Clock::now();
    ilqr::Vector points(c.centers->outputSize());
    ilqr::Matrix point_jac(points.size(), derivatives ? nin : 0);
    const int stride = points.size() / 3;
    for (int node = 0; node < n; ++node)
    {
        auto f = contexts[node]->share(c.centers);
        f->setEvaluationContext(contexts[node]);
        if (derivatives)
        {
            f->valueAndJacobian(states.col(node), points, point_jac);
        }
        else
        {
            f->evaluate(states.col(node), points);
        }
        for (int a = 0; a < 3; ++a)
        {
            for (int i = 0; i < spheres; ++i)
            {
                w.host_centers(node, a, i) = points[a * stride + i];
                if (derivatives)
                {
                    for (int j = 0; j < nin; ++j)
                    {
                        w.host_center_jacobian(node, a, i, j) =
                            point_jac(a * stride + i, j);
                    }
                }
            }
        }
    }
    const double kinematics_ms = elapsed(kinematics_start);
    const auto backend_start = Clock::now();
    Kokkos::deep_copy(w.centers, w.host_centers);
    if (derivatives)
    {
        Kokkos::deep_copy(w.center_jacobian, w.host_center_jacobian);
    }
    Kokkos::deep_copy(w.error, 0);
    const auto centers = w.centers;
    const auto residual = w.residual;
    const auto gradients = w.gradients;
    const auto grid = w.grid;
    const auto radii = w.radii;
    const auto pairs = w.pairs;
    const auto error = w.error;
    const bool environment = bool(c.grid);
    const int nx = environment ? c.grid->dimensions().x() : 0,
              ny = environment ? c.grid->dimensions().y() : 0,
              nz = environment ? c.grid->dimensions().z() : 0;
    const double ox = environment ? c.grid->origin().x() : 0.,
                 oy = environment ? c.grid->origin().y() : 0.,
                 oz = environment ? c.grid->origin().z() : 0.;
    const double voxel = environment ? c.grid->voxelSize() : 1.,
                 margin = c.margin, smoothing = c.smoothing;
    Kokkos::parallel_for(
        "collision distances over nodes",
        Kokkos::RangePolicy<Impl::Space>(0, n * nout),
        KOKKOS_LAMBDA(int index) {
            const int node = index / nout, row = index % nout;
            double clear = 0., g[3] = {0., 0., 0.};
            if (environment && row < spheres)
            {
                const double p[3] = {(centers(node, 0, row) - ox) / voxel,
                                     (centers(node, 1, row) - oy) / voxel,
                                     (centers(node, 2, row) - oz) / voxel};
                if (p[0] < 0 || p[1] < 0 || p[2] < 0 || p[0] > nx - 1 ||
                    p[1] > ny - 1 || p[2] > nz - 1)
                {
                    Kokkos::atomic_exchange(&error(), 1);
                    residual(node, row) = 0.;
                    for (int a = 0; a < 3; ++a)
                    {
                        gradients(node, row, a) = 0.;
                    }
                    return;
                }
                const int dims[3] = {nx, ny, nz};
                int lo[3];
                double f[3];
                for (int a = 0; a < 3; ++a)
                {
                    lo[a] = int(Kokkos::floor(p[a]));
                    if (lo[a] > dims[a] - 2)
                    {
                        lo[a] = dims[a] - 2;
                    }
                    f[a] = p[a] - lo[a];
                }
                for (int z = 0; z < 2; ++z)
                {
                    for (int y = 0; y < 2; ++y)
                    {
                        for (int x = 0; x < 2; ++x)
                        {
                            const double wx = x ? f[0] : 1 - f[0],
                                         wy = y ? f[1] : 1 - f[1],
                                         wz = z ? f[2] : 1 - f[2];
                            const double v = grid(
                                (std::size_t(lo[2] + z) * ny + lo[1] + y) * nx +
                                lo[0] + x);
                            clear += wx * wy * wz * v;
                            g[0] += (x ? 1. : -1.) * wy * wz * v / voxel;
                            g[1] += (y ? 1. : -1.) * wx * wz * v / voxel;
                            g[2] += (z ? 1. : -1.) * wx * wy * v / voxel;
                        }
                    }
                }
                clear -= radii(row);
            }
            else
            {
                const int pair = row - (environment ? spheres : 0),
                          i = pairs(pair, 0), j = pairs(pair, 1);
                double norm2 = 1e-24;
                for (int a = 0; a < 3; ++a)
                {
                    g[a] = centers(node, a, i) - centers(node, a, j);
                    norm2 += g[a] * g[a];
                }
                const double length = Kokkos::sqrt(norm2);
                clear = length - radii(i) - radii(j);
                for (int a = 0; a < 3; ++a)
                {
                    g[a] /= length;
                }
            }
            const double z = (margin - clear) / smoothing;
            residual(node, row) = smoothing *
                (z > 40 ? z
                        : (z < -40 ? Kokkos::exp(z)
                                   : Kokkos::log1p(Kokkos::exp(z))));
            const double exponential = Kokkos::exp(
                z >= 0 ? -Kokkos::fmin(z, 700.) : Kokkos::fmax(z, -700.));
            const double scale = z >= 0 ? -1. / (1. + exponential)
                                        : -exponential / (1. + exponential);
            for (int a = 0; a < 3; ++a)
            {
                gradients(node, row, a) = scale * g[a];
            }
        });
    if (derivatives)
    {
        const auto jc = w.center_jacobian;
        const auto j = w.jacobian;
        Kokkos::parallel_for(
            "collision derivative contraction over nodes",
            Kokkos::MDRangePolicy<Impl::Space,
                                  Kokkos::Rank<3, Kokkos::Iterate::Right,
                                               Kokkos::Iterate::Right>>(
                {0, 0, 0}, {n, nout, nin}),
            KOKKOS_LAMBDA(int node, int row, int col) {
                const bool env = environment && row < spheres;
                const int pair = row - (environment ? spheres : 0),
                          i = env ? row : pairs(pair, 0),
                          other = env ? 0 : pairs(pair, 1);
                double result = 0.;
                for (int a = 0; a < 3; ++a)
                {
                    result += gradients(node, row, a) *
                        (jc(node, a, i, col) -
                         (env ? 0. : jc(node, a, other, col)));
                }
                j(node, row, col) = result;
            });
        Kokkos::deep_copy(w.host_jacobian, w.jacobian);
    }
    Kokkos::deep_copy(w.host_residual, w.residual);
    int failed = 0;
    Kokkos::deep_copy(failed, w.error);
    Impl::Space().fence();
    if (failed)
    {
        throw ilqr::EvaluationError("Kokkos SDF query lies outside the grid");
    }
    const double backend_ms = elapsed(backend_start);
    for (int node = 0; node < n; ++node)
    {
        auto &r = *results[node];
        r.input = states.col(node);
        r.value.resize(nout);
        r.derivatives = derivatives;
        if (derivatives)
        {
            r.jacobian.resize(nout, nin);
        }
        for (int row = 0; row < nout; ++row)
        {
            r.value[row] = w.host_residual(node, row);
            if (derivatives)
            {
                for (int col = 0; col < nin; ++col)
                {
                    r.jacobian(row, col) = w.host_jacobian(node, row, col);
                }
            }
        }
        if (!r.value.allFinite() || (derivatives && !r.jacobian.allFinite()))
        {
            throw ilqr::EvaluationError(
                "Kokkos collision produced nonfinite output");
        }
    }
    // Publish only after every node succeeds. Readers run after prepare
    // returns.
    for (auto &r : results)
    {
        r->ready = true;
    }
    std::lock_guard lock(c.statistics->mutex);
    auto &stat = c.statistics->value;
    if (derivatives)
    {
        ++stat.derivative_batches;
    }
    else
    {
        ++stat.value_batches;
    }
    stat.nodes += n;
    stat.sdf_queries += environment ? std::uint64_t(n) * spheres : 0;
    stat.self_queries += std::uint64_t(n) * c.pairs.size();
    stat.prepare_ms += elapsed(start);
    stat.kinematics_ms += kinematics_ms;
    stat.backend_ms += backend_ms;
}
} // namespace horizon::collision
