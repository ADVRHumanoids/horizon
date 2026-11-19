#pragma once
// Kyon fixed-feet iLQR example and reproducible collision-backend benchmark.

#include "horizon/pinocchio_collision.h"
#include "ilqr/casadi_function.h"
#include "ilqr/ilqr.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <sstream>
#ifdef HORIZON_WITH_KOKKOS
#include "horizon/kokkos_collision.h"
#include <Kokkos_Core.hpp>
#endif

#include <tuple>
using namespace horizon::ilqr;
using namespace horizon::collision;
using namespace horizon::kinematics;
namespace pin = pinocchio;
using Clock = std::chrono::steady_clock;
namespace kyon_example
{
using Configuration = PinocchioConfiguration;
using Step = PinocchioStep;

struct Counters
{
    std::atomic<unsigned long long> values{0}, products{0}, nanos{0};
};
// Time complete collision calls, never individual sphere queries.
class TimedCollision final : public Function
{
  public:
    TimedCollision(std::shared_ptr<const Function> f,
                   std::shared_ptr<Counters> c)
        : Function(f->inputSize(), f->outputSize()), prototype(f),
          inner(f->clone()), counts(c)
    {
    }
    std::unique_ptr<Function> clone() const override
    {
        return std::make_unique<TimedCollision>(prototype, counts);
    }
    bool hasJacobianProduct() const override
    {
        return prototype->hasJacobianProduct();
    }

  private:
    void evaluateImpl(Input x, VectorOutput y) override
    {
        const auto t = Clock::now();
        inner->evaluate(x, y);
        record(t, false);
    }
    void jacobianImpl(Input x, MatrixOutput j) override
    {
        const auto t = Clock::now();
        inner->jacobian(x, j);
        record(t, true);
    }
    void valueAndJacobianImpl(Input x, VectorOutput y, MatrixOutput j) override
    {
        const auto t = Clock::now();
        inner->valueAndJacobian(x, y, j);
        record(t, true);
    }
    void valueAndJacobianProductImpl(Input x, MatrixInput d, VectorOutput y,
                                     MatrixOutput p) override
    {
        const auto t = Clock::now();
        inner->valueAndJacobianProduct(x, d, y, p);
        record(t, true);
    }
    void record(Clock::time_point t, bool derivative)
    {
        counts->nanos.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                                 t)
                .count(),
            std::memory_order_relaxed);
        (derivative ? counts->products : counts->values)
            .fetch_add(1, std::memory_order_relaxed);
    }
    std::shared_ptr<const Function> prototype;
    std::unique_ptr<Function> inner;
    std::shared_ptr<Counters> counts;
};

casadi::MX vectorConstant(const Vector &v)
{
    return casadi::MX(
        casadi::DM(std::vector<double>(v.data(), v.data() + v.size())));
}
std::string jsonString(const std::string &value)
{
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value)
    {
        if (c == '"' || c == '\\')
        {
            out << '\\' << c;
        }
        else if (c < 32)
        {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                << int(c) << std::dec;
        }
        else
        {
            out << c;
        }
    }
    out << '"';
    return out.str();
}

struct Options
{
    int N = 30, iterations = 40, samples = 3, threads = 0, warmup = 1;
    double damping = .001;
    double collision_weight = -1.;
    std::string decomposition = "lu";
    std::string backend = "scalar";
    int kokkos_threads = 1;
    bool collision_enabled = true;
    bool self_collision = true;
    std::string urdf, collision_policy, scenario = "benchmark";
    std::string output = "/tmp/kyon_trajectory";
    bool parse(int argc, char **argv)
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string key = argv[i];
            if (key == "--help")
            {
                std::cout << "Kyon fixed-feet iLQR: --urdf SPHERE_URDF "
                             "--collision-policy JSON --nodes N "
                             "--scenario benchmark|avoidance\n"
                          << "  --collision 0|1 --self-collision 0|1 --backend "
                             "scalar|kokkos|device\n"
                          << "  --samples K --warmup K --iterations K "
                             "--threads K --kokkos-threads K\n"
                          << "  --damping VALUE --collision-weight VALUE "
                             "--decomposition lu|ldlt --output PREFIX\n";
                return false;
            }
            if (i + 1 >= argc)
            {
                throw std::invalid_argument("missing option value");
            }
            const std::string value = argv[++i];
            if (key == "--nodes")
            {
                N = std::stoi(value);
            }
            else if (key == "--iterations")
            {
                iterations = std::stoi(value);
            }
            else if (key == "--samples")
            {
                samples = std::stoi(value);
            }
            else if (key == "--warmup")
            {
                warmup = std::stoi(value);
            }
            else if (key == "--urdf")
            {
                urdf = value;
            }
            else if (key == "--collision-policy")
            {
                collision_policy = value;
            }
            else if (key == "--scenario")
            {
                scenario = value;
            }
            else if (key == "--threads")
            {
                threads = std::stoi(value);
            }
            else if (key == "--damping")
            {
                damping = std::stod(value);
            }
            else if (key == "--collision-weight")
            {
                collision_weight = std::stod(value);
            }
            else if (key == "--decomposition")
            {
                decomposition = value;
            }
            else if (key == "--backend")
            {
                backend = value;
            }
            else if (key == "--kokkos-threads")
            {
                kokkos_threads = std::stoi(value);
            }
            else if (key == "--collision")
            {
                collision_enabled = std::stoi(value) != 0;
            }
            else if (key == "--self-collision")
            {
                self_collision = std::stoi(value) != 0;
            }
            else if (key == "--output")
            {
                output = value;
            }
            else
            {
                throw std::invalid_argument("unknown option: " + key);
            }
        }
        if (N < 4 || iterations < 1 || samples < 1 || threads < 0 ||
            warmup < 0 || !std::isfinite(damping) || damping < 0.)
        {
            throw std::invalid_argument(
                "invalid benchmark dimensions or damping");
        }
        if (collision_policy.empty() ||
            !std::filesystem::is_regular_file(collision_policy))
        {
            throw std::invalid_argument(
                "provide --collision-policy /path/to/disabled-collisions.json");
        }
        if (urdf.empty() || !std::filesystem::is_regular_file(urdf))
        {
            throw std::invalid_argument("provide --urdf /path/to/kyon.urdf");
        }
        if (scenario != "benchmark" && scenario != "avoidance")
        {
            throw std::invalid_argument(
                "scenario must be benchmark or avoidance");
        }
        if (collision_weight == -1.)
        {
            // Thousands of enabled sphere pairs now contribute to this sum.
            collision_weight = 100.;
        }
        if (!std::isfinite(collision_weight) || collision_weight <= 0)
        {
            throw std::invalid_argument(
                "collision weight must be finite and positive");
        }
        if ((backend != "scalar" && backend != "kokkos" &&
             backend != "device") ||
            kokkos_threads < 1)
        {
            throw std::invalid_argument(
                "backend must be scalar, kokkos, or device; Kokkos thread "
                "count must be positive");
        }
        return true;
    }
};

auto prepareScene(const std::string &urdf, const std::string &collision_policy,
                  const std::string &scenario, bool self_collision)
{
    auto model = std::make_shared<pin::Model>();
    pin::urdf::buildModel(urdf, *model);
    auto space = std::make_shared<Configuration>(model);
    Vector start = pin::neutral(*model);
    auto setJoint = [&](const std::string &name, double value)
    {
        if (!model->existJointName(name))
        {
            throw std::runtime_error("missing joint: " + name);
        }
        const auto &j = model->joints[model->getJointId(name)];
        if (j.nq() != 1)
        {
            throw std::runtime_error("expected scalar joint: " + name);
        }
        start[j.idx_q()] = value;
    };
    for (int leg = 1; leg <= 4; ++leg)
    {
        const double sign = leg % 2 ? 1. : -1.;
        setJoint("hip_pitch_" + std::to_string(leg), -.7 * sign);
        setJoint("knee_pitch_" + std::to_string(leg), 1.4 * sign);
        setJoint("hip_roll_" + std::to_string(leg),
                 (leg == 1 || leg == 4) ? .15 : -.15);
    }
    // Collision-free start for the complete xacro-derived sphere model.
    setJoint("shoulder_yaw_1", .8);
    setJoint("shoulder_pitch_1", 1.);
    setJoint("elbow_pitch_1", 2.);
    setJoint("wrist_pitch_1", 1.2);
    setJoint("wrist_yaw_1", .5);
    setJoint("shoulder_yaw_2", 1.);
    setJoint("shoulder_pitch_2", -1.15);
    setJoint("elbow_pitch_2", -1.35);
    setJoint("wrist_pitch_2", -.65);
    setJoint("wrist_yaw_2", -.75);
    const auto sphere_model = loadSphereModelFromUrdf(urdf);
    const auto disabled = loadDisabledCollisionPairs(collision_policy);
    auto geometry = sphere_model.spheres();
    const int sphere_count = geometry.size();
    const int count = sphere_count;
    auto workspace = std::make_shared<PinocchioModelWorkspace>(model);
    auto fk = std::make_shared<PinocchioSphereCenters>(model, space,
                                                       sphere_model, workspace);
    auto frame_fk = std::make_shared<PinocchioFramePositions>(
        model, space,
        std::vector<std::string>{"contact_1", "contact_2", "contact_3",
                                 "contact_4", "dagana_1_base"},
        workspace);
    Vector initial_frames(15);
    frame_fk->evaluate(start, initial_frames);
    Vector initial_points(3 * count);
    fk->evaluate(start, initial_points);
    auto point = [count](const Vector &p, int i)
    {
        return Eigen::Vector3d(p[i], p[count + i], p[2 * count + i]);
    };
    const Eigen::Vector3d initial_tool(initial_frames[4], initial_frames[9],
                                       initial_frames[14]);
    const Eigen::Vector3d target_offset = scenario == "avoidance"
        ? Eigen::Vector3d(.18, .04, .02)
        : Eigen::Vector3d(.06, .04, .02);
    const Eigen::Vector3d target = initial_tool + target_offset;

    // A sampled sphere SDF gives reproducible trilinear-query work without
    // requiring OpenVDB or mesh conversion in benchmark setup.
    Eigen::Vector3d obstacle = initial_tool + Eigen::Vector3d(.20, 0, 0);
    const double obstacle_radius = scenario == "avoidance" ? .025 : .06;
    constexpr double voxel = .04;
    if (scenario == "avoidance")
    {
        const auto tool_sphere =
            std::find_if(geometry.begin(), geometry.begin() + sphere_count,
                         [](const LinkSphere &sphere)
                         {
                             return sphere.link == "dagana_1_base";
                         });
        if (tool_sphere == geometry.begin() + sphere_count)
        {
            throw std::invalid_argument(
                "sphere URDF has no dagana_1_base collision sphere");
        }
        const int tool_sphere_index =
            static_cast<int>(tool_sphere - geometry.begin());
        obstacle = point(initial_points, tool_sphere_index) +
            .5 * target_offset + Eigen::Vector3d(0, 0, .04);
    }
    auto minClearance = [&](const Vector &p, const Eigen::Vector3d &o)
    {
        double result = 1e9;
        for (int i = 0; i < sphere_count; ++i)
        {
            result = std::min(result,
                              (point(p, i) - o).norm() - obstacle_radius -
                                  geometry[i].radius);
        }
        return result;
    };
    for (int attempt = 0; minClearance(initial_points, obstacle) <
         (scenario == "avoidance" ? .015 : .035);
         ++attempt)
    {
        if (attempt == 100)
        {
            throw std::runtime_error("cannot place obstacle");
        }
        if (scenario == "avoidance")
        {
            obstacle.z() += .005;
        }
        else
        {
            obstacle.x() += .02;
        }
    }
    const Eigen::Vector3d origin(-3, -3, -3);
    const Eigen::Vector3i dims(151, 151, 151);
    std::vector<float> grid(static_cast<std::size_t>(dims.prod()));
    for (int z = 0; z < dims.z(); ++z)
    {
        for (int y = 0; y < dims.y(); ++y)
        {
            for (int x = 0; x < dims.x(); ++x)
            {
                grid[(static_cast<std::size_t>(z) * dims.y() + y) * dims.x() +
                     x] = (origin + voxel * Eigen::Vector3d(x, y, z) - obstacle)
                              .norm() -
                    obstacle_radius;
            }
        }
    }
    auto field =
        std::make_shared<SdfGrid>(dims, origin, voxel, std::move(grid));
    std::vector<double> radii;
    for (int i = 0; i < sphere_count; ++i)
    {
        radii.push_back(geometry[i].radius);
    }
    std::vector<SphereDistanceFunction::Pair> self_pairs;
    if (self_collision)
    {
        self_pairs = makeSelfCollisionPairs(sphere_model, disabled);
    }
    return std::make_tuple(model, space, start, geometry, sphere_count, count,
                           fk, initial_points, frame_fk, initial_frames, target,
                           obstacle, obstacle_radius, voxel, field, radii,
                           self_pairs);
}

// Name frame expressions independently of collision geometry.
struct RobotKinematics
{
    casadi::MX packed_points;
    std::map<std::string, casadi::MX> frame_positions;
    std::map<std::string, Eigen::Vector3d> initial_frame_positions;

    const casadi::MX &framePosition(const std::string &name) const
    {
        return frame_positions.at(name);
    }

    const Eigen::Vector3d &initialFramePosition(const std::string &name) const
    {
        return initial_frame_positions.at(name);
    }
};

RobotKinematics
prepareKinematics(SymbolicFunctionManager &manager, const casadi::MX &q,
                  const std::shared_ptr<PinocchioFramePositions> &frame_fk,
                  const Vector &initial_frames)
{
    using casadi::MX;
    const std::vector<std::string> frames = {
        "contact_1", "contact_2", "contact_3", "contact_4", "dagana_1_base"};
    auto native_fk = manager.addNative("frame_positions", frame_fk);
    RobotKinematics result;
    result.packed_points = native_fk(q);
    const int count = frames.size();
    for (int i = 0; i < count; ++i)
    {
        const auto &p = result.packed_points;
        result.frame_positions.emplace(
            frames[i], MX::vertcat({p(i), p(count + i), p(2 * count + i)}));
        result.initial_frame_positions.emplace(
            frames[i],
            Eigen::Vector3d(initial_frames[i], initial_frames[count + i],
                            initial_frames[2 * count + i]));
    }
    return result;
}

class CollisionRuntime
{
#ifdef HORIZON_WITH_KOKKOS
    std::unique_ptr<Kokkos::ScopeGuard> kokkos_runtime;
#endif
  public:
    CollisionRuntime(const std::string &backend, int kokkos_threads)
    {
#ifdef HORIZON_WITH_KOKKOS
        if (backend != "scalar")
        {
            kokkos_runtime = std::make_unique<Kokkos::ScopeGuard>(
                Kokkos::InitializationSettings().set_num_threads(
                    kokkos_threads));
        }
#else
        if (backend != "scalar")
        {
            throw std::invalid_argument("rebuild with HORIZON_WITH_KOKKOS=ON");
        }
#endif
    }
};

class CollisionBackend
{
  public:
    std::shared_ptr<Counters> counters = std::make_shared<Counters>();
    std::shared_ptr<TimedCollision> collision;
#ifdef HORIZON_WITH_KOKKOS
    std::unique_ptr<KokkosCollisionBatch> batch;
    std::unique_ptr<DeviceCollisionCost> device;
#endif
    CollisionBackend(
        const std::string &backend, bool collision_enabled,
        const std::shared_ptr<pin::Model> &model,
        const std::shared_ptr<PinocchioSphereCenters> &fk,
        const std::vector<LinkSphere> &geometry, int sphere_count,
        const std::vector<double> &radii, const std::shared_ptr<SdfGrid> &field,
        const std::vector<SphereDistanceFunction::Pair> &self_pairs,
        double collision_weight)
    {
        collision = std::make_shared<TimedCollision>(
            std::make_shared<SphereDistanceFunction>(
                radii, std::vector<std::shared_ptr<const DistanceField>>{field},
                self_pairs),
            counters);
#ifdef HORIZON_WITH_KOKKOS
        if (collision_enabled && backend == "kokkos")
        {
            batch = std::make_unique<KokkosCollisionBatch>(fk, radii, field,
                                                           self_pairs);
        }
        if (collision_enabled && backend == "device")
        {
            device = std::make_unique<DeviceCollisionCost>(
                exportDeviceRobot(
                    *model,
                    SphereModel(std::vector<LinkSphere>(
                        geometry.begin(), geometry.begin() + sphere_count))),
                field, self_pairs, collision_weight);
        }
#endif
    }
    void
    registerResidual(SymbolicFunctionManager &manager, const casadi::MX &q,
                     const casadi::MX &u,
                     const std::shared_ptr<PinocchioSphereCenters> &centers,
                     double collision_scale)
    {
#ifdef HORIZON_WITH_KOKKOS
        if (batch)
        {
            auto native_collision =
                manager.addNative("sdf_batch", batch->function());
            manager.addResidual("collision", {q, u},
                                collision_scale * native_collision(q));
        }
        else
#endif
        {
            auto native_centers =
                manager.addNative("collision_centers", centers);
            auto native_collision = manager.addNative("sdf", collision);
            manager.addResidual("collision", {q, u},
                                collision_scale *
                                    native_collision(native_centers(q)));
        }
    }
    void install(horizon::IterativeLQR &solver, int N)
    {
#ifdef HORIZON_WITH_KOKKOS
        if (batch)
        {
            solver.addEvaluationBatch(*batch);
        }
        if (device)
        {
            solver.setBatchedCost(horizon::Nodes::all(), "collision", *device);
        }
#endif
    }
    void resetStatistics()
    {
        counters->values = 0;
        counters->products = 0;
        counters->nanos = 0;
#ifdef HORIZON_WITH_KOKKOS
        if (batch)
        {
            batch->resetStatistics();
        }
        if (device)
        {
            device->resetStatistics();
        }
#endif
    }
    struct Statistics
    {
        unsigned long long value_calls, product_calls, collision_ns;
#ifdef HORIZON_WITH_KOKKOS
        CollisionBatchStatistics batch_stats;
        DeviceCollisionStatistics device_stats;
#endif
    };
    Statistics statistics(int N) const
    {
        auto value_calls = counters->values.load(),
             product_calls = counters->products.load(),
             collision_ns = counters->nanos.load();
#ifdef HORIZON_WITH_KOKKOS
        const auto batch_stats =
            batch ? batch->statistics() : CollisionBatchStatistics{};
        const auto device_stats =
            device ? device->statistics() : DeviceCollisionStatistics{};
        if (batch)
        {
            value_calls = batch_stats.value_batches * (N + 1);
            product_calls = batch_stats.derivative_batches * (N + 1);
            collision_ns =
                static_cast<unsigned long long>(batch_stats.backend_ms * 1e6);
        }
        if (device)
        {
            value_calls = device_stats.value_batches * (N + 1);
            product_calls = device_stats.derivative_batches * (N + 1);
            collision_ns = 0;
        }
#endif
        return {value_calls, product_calls, collision_ns
#ifdef HORIZON_WITH_KOKKOS
                ,
                batch_stats, device_stats
#endif
        };
    }
    void writeStatistics(std::ostream &report, const Statistics &stats) const
    {
#ifdef HORIZON_WITH_KOKKOS
        const auto &batch_stats = stats.batch_stats;
        const auto &device_stats = stats.device_stats;
#endif
#ifdef HORIZON_WITH_KOKKOS
        if (batch)
        {
            report << ",\"kokkos_execution_space\":\""
                   << KokkosCollisionBatch::executionSpace()
                   << "\",\"batch_value_passes\":" << batch_stats.value_batches
                   << ",\"batch_derivative_passes\":"
                   << batch_stats.derivative_batches
                   << ",\"batch_kernel_launches\":"
                   << batch_stats.value_batches +
                    2 * batch_stats.derivative_batches
                   << ",\"batch_node_evaluations\":" << batch_stats.nodes
                   << ",\"batch_prepare_ms\":" << batch_stats.prepare_ms
                   << ",\"batch_kinematics_ms\":" << batch_stats.kinematics_ms
                   << ",\"batch_backend_ms\":" << batch_stats.backend_ms;
        }
        if (device)
        {
            report << ",\"device_execution_space\":\""
                   << DeviceCollisionCost::executionSpace()
                   << "\",\"device_prepare_ms\":" << device_stats.prepare_ms
                   << ",\"device_upload_ms\":" << device_stats.upload_ms
                   << ",\"device_kinematics_ms\":" << device_stats.kinematics_ms
                   << ",\"device_queries_ms\":" << device_stats.queries_ms
                   << ",\"device_accumulation_ms\":"
                   << device_stats.accumulation_ms
                   << ",\"device_download_ms\":" << device_stats.download_ms
                   << ",\"device_upload_bytes\":" << device_stats.upload_bytes
                   << ",\"device_download_bytes\":"
                   << device_stats.download_bytes
                   << ",\"device_peak_bytes\":" << device_stats.device_bytes
                   << ",\"device_kernel_launches\":"
                   << device_stats.kernel_launches
                   << ",\"device_value_batches\":" << device_stats.value_batches
                   << ",\"device_derivative_batches\":"
                   << device_stats.derivative_batches;
        }
#endif
    }
};

struct Benchmark
{
    horizon::IterativeLQR &solver;
    CollisionBackend &collision_backend;
    const std::shared_ptr<pin::Model> &model;
    const std::shared_ptr<Configuration> &space;
    const Vector &start;
    const std::shared_ptr<PinocchioSphereCenters> &fk;
    const Vector &initial_points;
    const std::shared_ptr<PinocchioFramePositions> &frame_fk;
    const Vector &initial_frames;
    const Eigen::Vector3d &target;
    const Eigen::Vector3d &obstacle;
    const std::shared_ptr<SdfGrid> &field;
    const std::vector<double> &radii;
    const std::vector<SphereDistanceFunction::Pair> &self_pairs;
    const std::string &output, &urdf, &collision_policy, &scenario, &backend,
        &decomposition;
    int N, samples, warmup, iterations, threads, kokkos_threads, sphere_count,
        count;
    bool collision_enabled;
    double collision_weight, obstacle_radius, voxel, damping, setup_ms;
    int run()
    {
        auto point = [this](const Vector &p, int i)
        {
            return Eigen::Vector3d(p[i], p[count + i], p[2 * count + i]);
        };
        std::ofstream report(output + ".json");
        if (!report)
        {
            throw std::runtime_error("cannot open report");
        }
        report << std::setprecision(12);
        report << "{\"model\":\"kinematic_configuration_velocity\",\"nodes\":"
               << N << ",\"nq\":" << model->nq << ",\"nv\":" << model->nv
               << ",\"compiler\":" << jsonString(__VERSION__)
               << ",\"collision_margin\":0.02,\"collision_smoothing\":0.005,"
                  "\"collision_weight\":"
               << collision_weight << ",\"urdf\":"
               << jsonString(std::filesystem::absolute(urdf).string())
               << ",\"collision_policy\":"
               << jsonString(
                      std::filesystem::absolute(collision_policy).string())
               << ",\"scenario\":" << jsonString(scenario)
               << ",\"warmup_solves\":" << warmup
#ifdef HORIZON_PROFILING
               << ",\"profiling_enabled\":true"
#else
               << ",\"profiling_enabled\":false"
#endif
               << ",\"backend\":\"" << backend
               << "\",\"kokkos_threads\":" << kokkos_threads
               << ",\"threads\":" << threads << ",\"damping\":" << damping
               << ",\"decomposition\":\"" << decomposition
               << "\",\"spheres\":" << sphere_count
               << ",\"voxel_size\":" << voxel << ",\"collision_enabled\":"
               << (collision_enabled ? "true" : "false")
               << ",\"setup_ms\":" << setup_ms
               << ",\"self_pairs\":" << self_pairs.size()
               << ",\"duration_s\":2,\"sdf_dimensions\":[151,151,151],\"sdf_"
                  "origin\":[-3,-3,-3]"
               << ",\"obstacle_radius\":" << obstacle_radius
               << ",\"obstacle_center\":[" << obstacle.x() << ','
               << obstacle.y() << ',' << obstacle.z() << "],\"target\":["
               << target.x() << ',' << target.y() << ',' << target.z()
               << "],\"samples\":[";
        bool valid = true;
        // Negative sample indices are warmup solves; every solve starts from
        // the same guess.
        for (int sample = -warmup; sample < samples; ++sample)
        {
            solver.setStateInitialGuess(start);
            solver.setInputInitialGuess(Vector::Zero(model->nv));
            std::ofstream trace;
            Clock::time_point begin, iteration_begin;
            std::map<std::string, std::size_t> iteration_offsets;
            for (const auto *name :
                 {"linearize_quadratize", "backward_pass", "line_search"})
            {
                const auto &timers = solver.getProfilingInfo().timings;
                auto it = timers.find(name);
                iteration_offsets[name] =
                    it == timers.end() ? 0 : it->second.size();
            }
            if (sample >= 0)
            {
                trace.open(output + "_iterations_" + std::to_string(sample) +
                           ".csv");
                trace << std::setprecision(12);
                trace
                    << "iteration,alpha,accepted,hxx_reg,cost,merit_derivative,"
                       "step_length,defect,constraint,elapsed_ms,callback_"
                       "interval_ms,linearize_quadratize_completed_since_"
                       "callback_ms,backward_pass_completed_since_callback_ms,"
                       "line_search_completed_since_callback_ms\n";
            }
            solver.setIterationCallback(
                [&](const auto &r)
                {
                    const auto now = Clock::now();
                    if (trace.is_open())
                    {
                        trace << r.iter << ',' << r.alpha << ',' << r.accepted
                              << ',' << r.hxx_reg << ',' << r.cost << ','
                              << r.merit_der << ',' << r.step_length << ','
                              << r.defect_norm << ',' << r.constraint_violation
                              << ','
                              << std::chrono::duration<double, std::milli>(
                                     now - begin)
                                     .count()
                              << ','
                              << std::chrono::duration<double, std::milli>(
                                     now - iteration_begin)
                                     .count();
                        for (const auto *name :
                             {"linearize_quadratize", "backward_pass",
                              "line_search"})
                        {
                            const auto &timers =
                                solver.getProfilingInfo().timings;
                            auto it = timers.find(name);
                            double us = 0.;
                            if (it != timers.end())
                            {
                                us = std::accumulate(
                                    it->second.begin() +
                                        std::min(iteration_offsets[name],
                                                 it->second.size()),
                                    it->second.end(), 0.);
                                iteration_offsets[name] = it->second.size();
                            }
                            trace << ',' << us / 1000.;
                        }
                        trace << '\n';
                    }
                    iteration_begin = now;
                    return true;
                });
            collision_backend.resetStatistics();
            const auto before = solver.getProfilingInfo().timings;
            const auto history_before = solver.getIterationHistory().size();
            begin = iteration_begin = Clock::now();
            const bool converged = solver.solve(iterations);
            const double ms =
                std::chrono::duration<double, std::milli>(Clock::now() - begin)
                    .count();
            const auto stats = collision_backend.statistics(N);
            const auto value_calls = stats.value_calls,
                       product_calls = stats.product_calls,
                       collision_ns = stats.collision_ns;
            double drift = 0, gap = 0, min_clear = 1e9, min_self = 1e9;
            Vector actual(3 * count), actual_frames(15), prediction(model->nq),
                defect(model->nv);
            auto framePoint = [](const Vector &p, int i)
            {
                return Eigen::Vector3d(p[i], p[5 + i], p[10 + i]);
            };
            for (int k = 0; k <= N; ++k)
            {
                fk->evaluate(solver.state(k), actual);
                frame_fk->evaluate(solver.state(k), actual_frames);
                for (int f = 0; f < 4; ++f)
                {
                    drift = std::max(drift,
                                     (framePoint(actual_frames, f) -
                                      framePoint(initial_frames, f))
                                         .norm());
                }
                for (int i = 0; i < sphere_count; ++i)
                {
                    min_clear = std::min(
                        min_clear,
                        field->query(point(actual, i)).distance - radii[i]);
                }
                for (auto [i, j] : self_pairs)
                {
                    min_self =
                        std::min(min_self,
                                 (point(actual, i) - point(actual, j)).norm() -
                                     radii[i] - radii[j]);
                }
                if (k < N)
                {
                    space->integrate(solver.state(k),
                                     (2. / N) * solver.input(k), prediction);
                    space->difference(prediction, solver.state(k + 1), defect);
                    gap = std::max(gap, defect.norm());
                }
            }
            const double error = (framePoint(actual_frames, 4) - target).norm();
            valid = valid && error < .005 && drift < 1e-4 && gap < 1e-4 &&
                (!collision_enabled || (min_clear > 0 && min_self > 0));
            if (sample >= 0)
            {
                valid = valid && converged;
            }
            const auto &history = solver.getIterationHistory();
            const bool objective_available = history.size() > history_before;
            std::cout << "sample=" << sample << " solve_ms=" << ms
                      << " target_error=" << error << " foot_drift=" << drift
                      << " gap=" << gap << " sdf_clearance=" << min_clear
                      << "\n";
            if (sample < 0)
            {
                continue;
            }
            if (sample)
            {
                report << ',';
            }
            report << "{\"solve_ms\":" << ms
                   << ",\"solver_converged\":" << (converged ? "true" : "false")
                   << ",\"iterations\":"
                   << solver.getIterationHistory().size() - history_before
                   << ",\"target_error\":" << error << ",\"objective\":";
            if (objective_available)
            {
                report << history.back().cost;
            }
            else
            {
                report << "null";
            }
            report << ",\"foot_drift\":" << drift << ",\"dynamics_gap\":" << gap
                   << ",\"min_sdf_clearance\":" << min_clear
                   << ",\"min_self_clearance\":" << min_self
                   << ",\"collision_value_calls\":" << value_calls
                   << ",\"collision_derivative_calls\":" << product_calls
                   << ",\"sdf_queries\":"
                   << (value_calls + product_calls) * sphere_count
                   << ",\"self_queries\":"
                   << (value_calls + product_calls) * self_pairs.size()
                   << ",\"collision_cpu_ms\":" << collision_ns * 1e-6;
            collision_backend.writeStatistics(report, stats);
            report << ",\"last_evaluated_cost_components\":{";
            bool cost_first = true;
            for (const auto &[name, values] : solver.getCostsValues())
            {
                double total = 0.;
                bool present = false;
                for (int k = 0; k < values.size(); ++k)
                {
                    if (std::isfinite(values[k]))
                    {
                        total += values[k] * (k == N ? 1. : 1. / N);
                        present = true;
                    }
                }
                if (!present)
                {
                    continue;
                }
                if (!cost_first)
                {
                    report << ',';
                }
                cost_first = false;
                report << jsonString(name) << ':' << total;
            }
            report << "},\"phase_us\":{";
            bool first = true;
            for (const auto &[name, timings] :
                 solver.getProfilingInfo().timings)
            {
                const auto it = before.find(name);
                const std::size_t offset =
                    it == before.end() ? 0 : it->second.size();
                if (!first)
                {
                    report << ',';
                }
                first = false;
                report << '"' << name << "\":"
                       << std::accumulate(timings.begin() +
                                              std::min(offset, timings.size()),
                                          timings.end(), 0.);
            }
            report << "},\"phase_statistics\":{";
            first = true;
            for (const auto &[name, timings] :
                 solver.getProfilingInfo().timings)
            {
                const auto it = before.find(name);
                const std::size_t offset =
                    it == before.end() ? 0 : it->second.size();
                std::vector<double> values(timings.begin() +
                                               std::min(offset, timings.size()),
                                           timings.end());
                if (values.empty())
                {
                    continue;
                }
                std::sort(values.begin(), values.end());
                const double total =
                    std::accumulate(values.begin(), values.end(), 0.);
                if (!first)
                {
                    report << ',';
                }
                first = false;
                report << jsonString(name) << ":{\"calls\":" << values.size()
                       << ",\"total_us\":" << total
                       << ",\"mean_us\":" << total / values.size()
                       << ",\"min_us\":" << values.front() << ",\"p50_us\":"
                       << .5 *
                        (values[(values.size() - 1) / 2] +
                         values[values.size() / 2])
                       << ",\"p95_us\":"
                       << values[std::min(
                              values.size() - 1,
                              std::size_t(std::ceil(.95 * values.size()) - 1))]
                       << ",\"max_us\":" << values.back() << '}';
            }
            report << "}}";
        }
        report << "],\"valid\":" << (valid ? "true" : "false") << "}\n";
        std::ofstream trajectory(output + ".csv");
        if (!trajectory)
        {
            throw std::runtime_error("cannot open trajectory output");
        }
        trajectory << std::setprecision(12);
        trajectory << "time";
        for (int i = 0; i < model->nq; ++i)
        {
            trajectory << ",q" << i;
        }
        for (int i = 0; i < model->nv; ++i)
        {
            trajectory << ",u" << i;
        }
        trajectory << '\n';
        for (int k = 0; k <= N; ++k)
        {
            trajectory << 2. * k / N;
            for (int i = 0; i < model->nq; ++i)
            {
                trajectory << ',' << solver.state(k)[i];
            }
            for (int i = 0; i < model->nv; ++i)
            {
                trajectory << ',' << (k < N ? solver.input(k)[i] : 0.);
            }
            trajectory << '\n';
        }
        return valid ? 0 : 1;
    }
};
} // namespace kyon_example
