#include "spot_problem.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>
#include <yaml-cpp/yaml.h>

using namespace spot_test;
namespace
{
struct Statistics
{
    double min = 0, median = 0, p95 = 0, max = 0, mean = 0, total = 0;
    explicit Statistics(std::vector<double> values)
    {
        if (values.empty())
        {
            throw std::runtime_error("empty timing series");
        }
        for (double v : values)
        {
            if (!std::isfinite(v) || v < 0)
            {
                throw std::runtime_error("invalid timing sample");
            }
        }
        std::sort(values.begin(), values.end());
        min = values.front();
        max = values.back();
        median = values.size() % 2
            ? values[values.size() / 2]
            : .5 * (values[values.size() / 2 - 1] + values[values.size() / 2]);
        p95 = values[size_t(std::ceil(.95 * values.size())) - 1];
        total = std::accumulate(values.begin(), values.end(), 0.0);
        mean = total / values.size();
    }
};
void write(std::ostream &out, const Statistics &s)
{
    out << "{\"min\":" << s.min << ",\"median\":" << s.median
        << ",\"p95\":" << s.p95 << ",\"max\":" << s.max
        << ",\"mean\":" << s.mean << ",\"total\":" << s.total << "}";
}
std::string quote(const std::string &s)
{
    std::ostringstream out;
    out << '"';
    for (unsigned char c : s)
    {
        switch (c)
        {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (c < 0x20)
            {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << int(c) << std::dec;
            }
            else
            {
                out << char(c);
            }
        }
    }
    out << '"';
    return out.str();
}
std::string cpuModel()
{
    std::ifstream cpu("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpu, line))
    {
        if (line.find("model name") == 0)
        {
            return line.substr(line.find(':') + 2);
        }
    }
    return "unknown";
}
std::vector<int> integers(const std::string &text, int minimum)
{
    std::vector<int> result;
    std::istringstream in(text);
    std::string item;
    while (std::getline(in, item, ','))
    {
        size_t used = 0;
        int n = std::stoi(item, &used);
        if (used != item.size() || n < minimum)
        {
            throw std::invalid_argument("invalid profiling configuration");
        }
        result.push_back(n);
    }
    if (result.empty())
    {
        throw std::invalid_argument("empty profiling configuration");
    }
    return result;
}
bool valid(const Metrics &m)
{
    return m.converged && std::isfinite(m.objective) &&
        m.objective < .02 * m.initial_objective && m.depth > .05 &&
        m.pose_position < .003 && m.pose_rotation < .006 &&
        m.foot_drift < .001 && m.foot_velocity < .003 &&
        m.newton_euler < .005 && m.contact_acceleration < 1e-4 &&
        m.dynamics_gap < 1e-5 && m.quaternion_error < 1e-12 &&
        m.endpoint_velocity < 1e-5 && m.min_normal > 1 &&
        m.min_friction_margin > 0 && m.joint_margin > 0 && m.peak_torque < 1000;
}
void quality(std::ostream &out, const Metrics &m)
{
    out << "{\"converged\":" << (m.converged ? "true" : "false")
        << ",\"objective\":" << m.objective
        << ",\"initial_objective\":" << m.initial_objective
        << ",\"position_error_m\":" << m.pose_position
        << ",\"rotation_error_rad\":" << m.pose_rotation
        << ",\"foot_drift_m\":" << m.foot_drift
        << ",\"foot_velocity_m_s\":" << m.foot_velocity
        << ",\"newton_euler_norm\":" << m.newton_euler
        << ",\"stabilized_contact_acceleration\":" << m.contact_acceleration
        << ",\"dynamics_gap\":" << m.dynamics_gap
        << ",\"min_normal_force_N\":" << m.min_normal
        << ",\"min_friction_margin_N\":" << m.min_friction_margin
        << ",\"joint_limit_margin_rad\":" << m.joint_margin
        << ",\"peak_torque_Nm\":" << m.peak_torque
        << ",\"quaternion_error\":" << m.quaternion_error
        << ",\"squat_depth_m\":" << m.depth
        << ",\"endpoint_velocity\":" << m.endpoint_velocity << "}";
}
void trajectory(const std::string &path, const Problem &problem, int nodes)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("cannot write trajectory CSV");
    }
    out << std::setprecision(16) << "time_s";
    for (auto [prefix, count] : std::vector<std::pair<std::string, int>>{
             {"q", nq}, {"v", nv}, {"a", nv}, {"f", nf}, {"tau", 12}})
    {
        for (int j = 0; j < count; ++j)
        {
            out << "," << prefix << j;
        }
    }
    out << "\n";
    for (int k = 0; k <= nodes; ++k)
    {
        out << 2.0 * k / nodes;
        for (int j = 0; j < nx; ++j)
        {
            out << "," << problem.states()(j, k);
        }
        for (const auto *controls :
             {&problem.accelerations(), &problem.forces(), &problem.torques()})
        {
            for (int j = 0; j < controls->rows(); ++j)
            {
                out << ",";
                if (k < nodes)
                {
                    out << (*controls)(j, k);
                }
            }
        }
        out << "\n";
    }
    if (!out)
    {
        throw std::runtime_error("failed to finish trajectory CSV");
    }
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        int samples = 3, warmup = 1, function_samples = 100;
        std::vector<int> nodes{16, 32}, threads{0, 2};
        std::string output = "spot_profile.json", baseline, selection = "all";
        std::vector<Formulation> forms;
        ProblemOptions options;
        double max_regression = 1.25;
        for (int i = 1; i < argc; ++i)
        {
            std::string key = argv[i];
            if (key == "--help")
            {
                std::cout << "profileSpotIlqr [--samples N] [--warmup N] "
                             "[--function-samples N] [--nodes 16,32] "
                             "[--threads 0,2] [--formulation "
                             "all|acceleration_force|torque] "
                             "[--integration projected|unprojected] "
                             "[--regularization normalized|fixed] "
                             "[--output report.json] [--baseline report.json "
                             "--max-regression 1.25]\n";
                return 0;
            }
            if (i + 1 == argc)
            {
                throw std::invalid_argument("missing profiling argument");
            }
            std::string value = argv[++i];
            if (key == "--samples")
            {
                samples = std::stoi(value);
            }
            else if (key == "--warmup")
            {
                warmup = std::stoi(value);
            }
            else if (key == "--function-samples")
            {
                function_samples = std::stoi(value);
            }
            else if (key == "--nodes")
            {
                nodes = integers(value, 12);
            }
            else if (key == "--threads")
            {
                threads = integers(value, 0);
            }
            else if (key == "--output")
            {
                output = value;
            }
            else if (key == "--baseline")
            {
                baseline = value;
            }
            else if (key == "--formulation")
            {
                selection = value;
            }
            else if (key == "--integration")
            {
                if (value != "projected" && value != "unprojected")
                {
                    throw std::invalid_argument("unknown integration mode");
                }
                options.project_contacts = value == "projected";
            }
            else if (key == "--regularization")
            {
                if (value != "normalized" && value != "fixed")
                {
                    throw std::invalid_argument("unknown regularization mode");
                }
                options.scale_regularization = value == "normalized";
            }
            else if (key == "--max-regression")
            {
                max_regression = std::stod(value);
            }
            else
            {
                throw std::invalid_argument("unknown profiling argument: " +
                                            key);
            }
        }
        if (selection == "all")
        {
            forms = {Formulation::AccelerationForce, Formulation::Torque};
        }
        else if (selection == "acceleration_force")
        {
            forms = {Formulation::AccelerationForce};
        }
        else if (selection == "torque")
        {
            forms = {Formulation::Torque};
        }
        else
        {
            throw std::invalid_argument("unknown formulation: " + selection);
        }
        if (samples < 1 || warmup < 0 || function_samples < 1 ||
            !std::isfinite(max_regression) || max_regression <= 1)
        {
            throw std::invalid_argument(
                "invalid profiling sample count or regression ratio");
        }
#ifdef HORIZON_PROFILING
        constexpr bool profiling = true;
#else
        constexpr bool profiling = false;
#endif
        const auto cpu = cpuModel();
        YAML::Node previous;
        const std::string integration =
            options.project_contacts ? "projected" : "unprojected";
        const std::string regularization =
            options.scale_regularization ? "normalized" : "fixed";
        if (!baseline.empty())
        {
            previous = YAML::LoadFile(baseline);
            if (previous["schema_version"].as<int>() != 1 ||
                previous["cpu"].as<std::string>() != cpu ||
                previous["compiler"].as<std::string>() != __VERSION__ ||
                previous["build_type"].as<std::string>() !=
                    HORIZON_SPOT_BUILD_TYPE ||
                previous["profiling_enabled"].as<bool>() != profiling ||
                previous["integration"].as<std::string>("unprojected") !=
                    integration ||
                previous["regularization"].as<std::string>("fixed") !=
                    regularization)
            {
                throw std::invalid_argument(
                    "baseline must match schema, CPU, compiler, build type, "
                    "profiling mode, integration and regularization");
            }
        }
        std::ofstream report(output);
        if (!report)
        {
            throw std::runtime_error("cannot write profiling report");
        }
        report << std::setprecision(12)
               << "{\"schema_version\":1,\"cpu\":" << quote(cpu)
               << ",\"compiler\":" << quote(__VERSION__)
               << ",\"build_type\":" << quote(HORIZON_SPOT_BUILD_TYPE)
               << ",\"hardware_threads\":"
               << std::thread::hardware_concurrency()
               << ",\"profiling_enabled\":" << (profiling ? "true" : "false")
               << ",\"samples\":" << samples << ",\"warmup\":" << warmup
               << ",\"function_samples\":" << function_samples
               << ",\"function_warmup\":10"
               << ",\"phase_units\":\"inclusive microseconds; worker times "
                  "overlap\",\"duration_s\":2"
               << ",\"integration\":" << quote(integration)
               << ",\"regularization\":" << quote(regularization)
               << ",\"terminal_velocity_condition\":"
               << quote(options.project_contacts
                            ? "all 18 velocities zero via base-6 equality and "
                              "exact Jv=0"
                            : "all 18 velocity equalities")
               << ",\"solver\":{\"constraint_decomposition\":\"svd\",\"rank_"
                  "threshold\":1e-10,\"max_iterations\":300,"
                  "\"merit_derivative_tolerance\":1e-8,\"constraint_"
                  "tolerance\":1e-7,\"defect_tolerance\":1e-7,"
                  "\"regularization_base\":1e-8}"
               << ",\"contact\":{\"kp\":25,\"kd\":10,\"friction_validation\":0."
                  "7},\"cases\":[\n";
        bool first = true, success = true;
        for (int N : nodes)
        {
            for (int pool : threads)
            {
                Eigen::MatrixXd inverse_x, inverse_a, inverse_f, inverse_tau;
                double inverse_cost = 0;
                for (auto form : forms)
                {
                    const auto case_start = std::chrono::steady_clock::now();
                    std::string phase = "setup";
                    std::vector<double> wall, iterations;
                    std::vector<Metrics> quality_samples;
                    std::ostringstream entry;
                    entry << std::setprecision(12);
                    try
                    {
                        Problem problem(form, N, pool, options);
                        for (int i = 0; i < warmup; ++i)
                        {
                            phase = "warmup " + std::to_string(i + 1);
                            auto m = problem.solve();
                            if (!valid(m))
                            {
                                success = false;
                            }
                        }
                        auto before = problem.phaseTimings();
                        Metrics last;
                        for (int i = 0; i < samples; ++i)
                        {
                            phase = "sample " + std::to_string(i + 1);
                            last = problem.solve();
                            success = success && valid(last);
                            quality_samples.push_back(last);
                            wall.push_back(last.solve_ms);
                            iterations.push_back(last.iterations);
                        }
                        phase = "function profiling";
                        auto micro = problem.profileFunctions(function_samples);
                        auto after = problem.phaseTimings();
                        Statistics solve(wall);
                        double baseline_ratio = 0;
                        if (!baseline.empty())
                        {
                            bool found = false;
                            for (auto entry : previous["cases"])
                            {
                                if (entry["nodes"].as<int>() == N &&
                                    entry["threads"].as<int>() == pool &&
                                    entry["formulation"].as<std::string>() ==
                                        name(form))
                                {
                                    found = true;
                                    double median = entry["solve_ms"]["median"]
                                                        .as<double>();
                                    if (median <= 0 || !std::isfinite(median))
                                    {
                                        throw std::invalid_argument(
                                            "invalid baseline timing");
                                    }
                                    baseline_ratio = solve.median / median;
                                    if (baseline_ratio > max_regression)
                                    {
                                        success = false;
                                    }
                                    break;
                                }
                            }
                            if (!found)
                            {
                                throw std::invalid_argument(
                                    "baseline is missing a requested "
                                    "configuration");
                            }
                        }
                        std::string csv = output + "." + name(form) + "_N" +
                            std::to_string(N) + "_T" + std::to_string(pool) +
                            ".csv";
                        trajectory(csv, problem, N);
                        entry << "{\"status\":\"completed\",\"formulation\":"
                              << quote(name(form)) << ",\"nodes\":" << N
                              << ",\"threads\":" << pool
                              << ",\"controls\":" << controlSize(form)
                              << ",\"stored_state\":" << nx
                              << ",\"tangent_state\":" << 2 * nv
                              << ",\"huu_reg\":"
                              << 1e-8 / (options.scale_regularization ? N : 1)
                              << ",\"hxx_reg_base\":"
                              << 1e-8 / (options.scale_regularization ? N : 1)
                              << ",\"trajectory_csv\":" << quote(csv)
                              << ",\"setup_ms\":" << last.setup_ms
                              << ",\"solve_ms\":";
                        write(entry, solve);
                        entry << ",\"iterations\":";
                        write(entry, Statistics(iterations));
                        entry << ",\"solve_samples_ms\":[";
                        for (size_t i = 0; i < wall.size(); ++i)
                        {
                            if (i)
                            {
                                entry << ",";
                            }
                            entry << wall[i];
                        }
                        entry << "]";
                        entry << ",\"dynamics_linearization_us\":";
                        write(entry, Statistics(micro.dynamics_us));
                        entry << ",\"stage_linearization_us\":";
                        write(entry, Statistics(micro.stage_us));
                        entry << ",\"quality_samples\":[";
                        for (size_t i = 0; i < quality_samples.size(); ++i)
                        {
                            if (i)
                            {
                                entry << ",";
                            }
                            quality(entry, quality_samples[i]);
                        }
                        entry << "],\"phase_us\":{";
                        bool first_phase = true;
                        for (const auto &[phase, values] : after)
                        {
                            size_t skip = before.count(phase)
                                ? before.at(phase).size()
                                : 0;
                            if (skip == values.size())
                            {
                                continue;
                            }
                            if (!first_phase)
                            {
                                entry << ",";
                            }
                            first_phase = false;
                            entry << quote(phase) << ":";
                            write(entry,
                                  Statistics(std::vector<double>(
                                      values.begin() + skip, values.end())));
                        }
                        entry << "}";
                        if (baseline_ratio)
                        {
                            entry << ",\"baseline_median_ratio\":"
                                  << baseline_ratio;
                        }
                        if (form == Formulation::AccelerationForce)
                        {
                            inverse_x = problem.states();
                            inverse_a = problem.accelerations();
                            inverse_f = problem.forces();
                            inverse_tau = problem.torques();
                            inverse_cost = last.objective;
                        }
                        else if (inverse_x.size())
                        {
                            double dx = (problem.states() - inverse_x)
                                            .cwiseAbs()
                                            .maxCoeff();
                            double da = (problem.accelerations() - inverse_a)
                                            .cwiseAbs()
                                            .maxCoeff();
                            double df = (problem.forces() - inverse_f)
                                            .cwiseAbs()
                                            .maxCoeff();
                            double dtau = (problem.torques() - inverse_tau)
                                              .cwiseAbs()
                                              .maxCoeff();
                            double dc = std::abs(last.objective - inverse_cost);
                            entry << ",\"versus_acceleration_force\":{\"state_"
                                     "max_abs\":"
                                  << dx << ",\"acceleration_max_abs\":" << da
                                  << ",\"force_max_abs_N\":" << df
                                  << ",\"torque_max_abs_Nm\":" << dtau
                                  << ",\"objective_difference\":" << dc << "}";
                            // Nonlinear optimization need not find identical
                            // traces at every discretization. Report these
                            // differences; testSpotIlqr gates agreement for its
                            // fixed N=16 case.
                            std::cout << "comparison N=" << N
                                      << " threads=" << pool
                                      << " state_delta=" << dx
                                      << " acceleration_delta=" << da
                                      << " force_delta=" << df
                                      << " torque_delta=" << dtau
                                      << " objective_delta=" << dc << "\n";
                        }
                        entry << "}";
                        std::cout
                            << name(form) << " N=" << N << " threads=" << pool
                            << " median=" << solve.median
                            << " ms p95=" << solve.p95 << " ms iterations="
                            << Statistics(iterations).median
                            << " valid=" << valid(last) << "\n";
                        if (!first)
                        {
                            report << ",\n";
                        }
                        first = false;
                        report << entry.str();
                        report.flush();
                    }
                    catch (const std::exception &error)
                    {
                        success = false;
                        if (form == Formulation::AccelerationForce)
                        {
                            inverse_x.resize(0, 0);
                        }
                        const double elapsed =
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - case_start)
                                .count();
                        if (!first)
                        {
                            report << ",\n";
                        }
                        first = false;
                        report << "{\"status\":\"error\",\"formulation\":"
                               << quote(name(form)) << ",\"nodes\":" << N
                               << ",\"threads\":" << pool
                               << ",\"phase\":" << quote(phase)
                               << ",\"error\":" << quote(error.what())
                               << ",\"attempt_wall_ms\":" << elapsed
                               << ",\"completed_samples\":"
                               << quality_samples.size()
                               << ",\"partial_solve_samples_ms\":[";
                        for (size_t i = 0; i < wall.size(); ++i)
                        {
                            if (i)
                            {
                                report << ",";
                            }
                            report << wall[i];
                        }
                        report << "],\"partial_iteration_counts\":[";
                        for (size_t i = 0; i < iterations.size(); ++i)
                        {
                            if (i)
                            {
                                report << ",";
                            }
                            report << iterations[i];
                        }
                        report << "],\"partial_quality_samples\":[";
                        for (size_t i = 0; i < quality_samples.size(); ++i)
                        {
                            if (i)
                            {
                                report << ",";
                            }
                            quality(report, quality_samples[i]);
                        }
                        report << "]}";
                        report.flush();
                        std::cerr << name(form) << " N=" << N
                                  << " threads=" << pool << " failed during "
                                  << phase << ": " << error.what() << "\n";
                    }
                }
            }
        }
        report << "\n],\"passed\":" << (success ? "true" : "false") << "}\n";
        report.close();
        // Confirm that the emitted report can be consumed as structured data.
        YAML::LoadFile(output);
        std::cout << "Report: " << output << "\n";
        return success ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << "\n";
        return 2;
    }
}
