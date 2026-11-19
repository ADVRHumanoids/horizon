# Full-body Kyon collision measurements

Measured on 2026-11-17 using the runtime xacro-derived URDF and its 75,000-trial MoveIt JSON policy. [Machine-readable results](device_collision_results.json) include model hashes, sample profiles, solver phase timers, CUDA stages, and parity checks.

The workload contains 103 spheres on all 31 collision-bearing source links, 3,799 enabled self-collision pairs, and a 151³ environment SDF. Every backend uses the same initial posture, task, collision weight (100), bounds, and tolerances. The original supplied sphere URDF omits arm collision geometry and is retained as a separate test fixture.

Release builds with profiling enabled, Core i7-12700 and RTX 3060 12 GB (driver 580.178.04), Kokkos 4.7.1, two CPU node workers. Each case uses one warmup and three measured solves. Runs were sequential after compilation and generation finished. The desktop GPU was shared; p95 with only three samples is the largest sample and should not be treated as a stable tail estimate.

| Intervals | Iterations | Scalar median ms | Kokkos residual batch median ms | Complete CUDA device median ms | Separate CPU median ms |
|---:|---:|---:|---:|---:|---:|
| 12 | 34 | 945.3 | 985.6 | 182.1 | 1090.1 |
| 30 | 29 | 2208.4 | 2181.5 | 317.9 | 2219.8 |
| 64 | 25 | 3873.6 | 3811.3 | 522.7 | 3864.9 |

All 36 measured solves converged. Minimum self clearance was at least 11.27 mm and environment clearance at least 26.77 mm at shooting nodes. The largest trajectory difference across backends was 1.1e-11; objective differences are retained in the JSON. Collision costs are soft penalties, and this is a kinematic shooting benchmark without continuous-time collision certification.

The `kokkos` path still performs robot kinematics and graph accumulation on the CPU. The `device` path performs collision kinematics, all enabled queries, tangent derivatives, and Gauss–Newton accumulation on CUDA, returning compact costs, gradients, and Hessians.

## Reproduction

```bash
python3 tools/benchmark_device_collision.py \
  --binary /path/to/cuda-build/tests/benchmarkKyonTrajectory \
  --cpu-binary /path/to/cpu-build/tests/benchmarkKyonTrajectory \
  --nodes 12 30 64 --samples 3 --threads 2 --iterations 100 \
  --output /tmp/horizon-fullbody-benchmarks
```

The script defaults to the vendored full-body URDF and JSON policy. Raw profiles, trajectories, iteration traces and logs from this run are in `/tmp/horizon-fullbody-benchmarks`. Earlier 94-sphere/100-pair measurements and CUDA-library probes describe a different partial workload and are not directly comparable to these results.

The [full-body CPU micro-profile](full_body_collision_profile.txt) uses 1,000 iterations and the same 103-sphere, 3,799-pair model. The [asset-generation validation](collision_asset_generation_validation.json) records the successful Kyon xacro → FOAM → MoveIt JSON run.
