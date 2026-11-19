#pragma once
#include "example_paths.h"
#include "ilqr/ilqr.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace examples
{
struct Options
{
    int nodes, threads = 1, iterations;
    double dt;
    bool quiet = false;
    std::string urdf = HORIZON_EXAMPLE_KYON;
    std::string policy = HORIZON_EXAMPLE_POLICY;
    std::string sdf = HORIZON_EXAMPLE_SDF;
};

// The walkthrough shares one CLI; dt=0 chooses a two-second whole-body horizon.
inline Options parseOptions(int argc, char **argv, int nodes, int iterations,
                            double dt, bool arm = false)
{
    try
    {
        Options opt{nodes, 1, iterations, dt};
        if (arm)
        {
            opt.urdf = HORIZON_EXAMPLE_ARM;
        }
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h")
            {
                std::cout
                    << "Options:\n"
                    << "  -N, --nodes N          shooting intervals (default "
                    << nodes << ")\n"
                    << "  --threads N            iLQR worker threads (default "
                       "1)\n"
                    << "  --max-iterations N     solver iteration limit "
                       "(default "
                    << iterations << ")\n"
                    << "  --dt SECONDS           interval duration (default "
                    << (dt == 0 ? std::string("2 / nodes") : std::to_string(dt))
                    << ")\n"
                    << "  --urdf PATH            robot model override\n"
                    << "  --policy PATH          disabled self-collision pairs "
                       "override\n"
                    << "  --sdf PATH             OpenVDB field override\n"
                    << "  --quiet                suppress iteration reports\n";
                std::exit(0);
            }
            if (arg == "--quiet")
            {
                opt.quiet = true;
                continue;
            }
            if (i + 1 == argc)
            {
                throw std::invalid_argument("missing value for " + arg);
            }
            std::string value = argv[++i];
            if (arg == "--nodes" || arg == "-N")
            {
                opt.nodes = std::stoi(value);
            }
            else if (arg == "--threads")
            {
                opt.threads = std::stoi(value);
            }
            else if (arg == "--max-iterations")
            {
                opt.iterations = std::stoi(value);
            }
            else if (arg == "--dt")
            {
                opt.dt = std::stod(value);
                if (opt.dt <= 0)
                {
                    throw std::invalid_argument("dt must be positive");
                }
            }
            else if (arg == "--urdf")
            {
                opt.urdf = value;
            }
            else if (arg == "--policy")
            {
                opt.policy = value;
            }
            else if (arg == "--sdf")
            {
                opt.sdf = value;
            }
            else
            {
                throw std::invalid_argument("unknown option: " + arg);
            }
        }
        if (opt.nodes < 2 || opt.threads < 0 || opt.iterations < 1 ||
            !std::isfinite(opt.dt) || opt.dt < 0)
        {
            throw std::invalid_argument("nodes >= 2, threads >= 0, iterations "
                                        ">= 1 and dt > 0 are required");
        }
        if (opt.dt == 0)
        {
            opt.dt = 2. / opt.nodes;
        }
        return opt;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Invalid arguments: " << error.what() << " (use --help)\n";
        std::exit(2);
    }
}

inline void reportIterations(horizon::IterativeLQR &solver, const Options &opt)
{
    if (!opt.quiet)
    {
        solver.setIterationCallback(
            [n = opt.nodes](const auto &result)
            {
                result.print(n);
                return true;
            });
    }
}
} // namespace examples
