#pragma once
#include "horizon/collision.h"
#include <cstdint>

namespace horizon::collision
{
struct CollisionBatchStatistics
{
    std::uint64_t value_batches = 0, derivative_batches = 0, nodes = 0,
                  sdf_queries = 0, self_queries = 0;
    double prepare_ms = 0., kinematics_ms = 0., backend_ms = 0.;
};

/** All-node collision evaluator. The application owns Kokkos initialization
 * and finalization. The grid is uploaded once per cloned batch. Its Kokkos
 * DefaultExecutionSpace determines the device; host/device buffers are reused.
 *
 * Centers receives one complete stored state, and returns SoA world points.
 * The first radii.size() points are collision spheres; additional points may
 * belong to feet/tasks. Its supplied derivatives remain in storage coordinates.
 */
class KokkosCollisionBatch final : public ilqr::EvaluationBatch
{
  public:
    KokkosCollisionBatch(
        std::shared_ptr<const ilqr::Function> centers,
        std::vector<double> radii, std::shared_ptr<const SdfGrid> grid,
        std::vector<SphereDistanceFunction::Pair> self_pairs = {},
        double margin = .02, double smoothing = .005);
    ~KokkosCollisionBatch();
    KokkosCollisionBatch(const KokkosCollisionBatch &) = delete;
    KokkosCollisionBatch &operator=(const KokkosCollisionBatch &) = delete;
    std::unique_ptr<ilqr::EvaluationBatch> clone() const override;
    std::shared_ptr<const ilqr::Function> function() const;
    void prepare(
        ilqr::MatrixInput states, ilqr::MatrixInput controls,
        const std::vector<std::shared_ptr<ilqr::EvaluationContext>> &contexts,
        bool derivatives) override;
    /** Statistics are shared by solver clones of this prototype; read/reset
     * only when no solve is in progress. */
    CollisionBatchStatistics statistics() const;
    void resetStatistics();
    static const char *executionSpace();

  private:
    struct Configuration;
    struct Impl;
    explicit KokkosCollisionBatch(
        std::shared_ptr<const Configuration> configuration);
    std::shared_ptr<const Configuration> _configuration;
    std::unique_ptr<Impl> _impl;
};
} // namespace horizon::collision
