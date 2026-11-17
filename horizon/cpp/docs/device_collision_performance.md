# Complete device collision costs: measurements and library decisions

Measured on 2026-11-16. [Machine-readable results](device_collision_results.json) preserve all 12 cases, parity checks, stage medians, library comparisons, and copy-trace events.

**Follow-up:** CUDA 13.0.2 is now installed. Native cuRoboV2 and nvblox builds, integration probes, and successful GPU memory checks are recorded in the [CUDA 13 integration report](cuda13_library_integrations.md). Toolkit and library feasibility limitations below describe the original CUDA 12 setup; its recorded performance results are unchanged.

## Full trajectory results

Release builds with profiling enabled; i7-12700, RTX 3060 12 GB, driver 580.178.04, CUDA 12.0.140, Kokkos 4.7.1, eight CPU node workers. Each case uses one warmup and 20 measured solves. Cases ran sequentially after compilation and regression tests completed. The GPU was shared with the desktop; no exclusive clocks or GPU reservation were used. Before/after utilization snapshots are in the JSON. These are profiling-build measurements, not a claim about uninstrumented deployment latency.

The problem retains configuration/generalized-velocity kinematic shooting, four fixed foot positions, the terminal `dagana_1_base` position target, 94 imported Kyon spheres, a 151³ sampled environment SDF, and 100 explicit inter-hand sphere pairs. All backends use the same objective, damping, initialization, normalization, and stopping conditions. The pair list is a benchmark mask, not a complete whole-robot exclusion policy.

| N | Iterations | Fastest measured CPU median, ms | Previous partial GPU median, ms | Complete device median, ms | Complete device p95, ms | Reduction versus CPU |
|---|---:|---:|---:|---:|---:|---:|
| 30 | 16 | 124.9 | 160.5 | 91.2 | 95.1 | 27.0% |
| 256 | 20 | 1173.4 | 1513.3 | 887.0 | 915.6 | 24.4% |
| 1024 | 12 | 2705.5 | 3536.7 | 1865.5 | 1924.9 | 31.0% |

The main CPU column is the faster of the scalar evaluator in the CUDA build and a separate GCC 13.3 CPU build. The CUDA configuration reports GCC 11.5 for NVCC's host compiler; solver sources use the wrapper's host-only compilation path. Both individual CPU cases and p95 values are retained in the JSON.

All 240 measured solves converged. Within each horizon, every backend used the same iteration count and agreed on objective at report precision. Exported CSV trajectories differ by at most approximately `1e-11`. Foot drift, dynamics defects, target error, and both sampled collision clearances passed the benchmark's existing acceptance conditions. These checks are at shooting nodes; the model is kinematic and uses soft collision costs.

For N=256 the complete device path spends approximately:

- Linearization/quadratization: 243.0 ms per solve.
- CPU backward pass: 512.4 ms per solve.
- Line search: 124.2 ms per solve.

Collision preparation is included in those solver phases, not additional to them. Median collision preparation over the entire solve is 55.3 ms, comprising these CUDA-event stage totals:

| Stage | Median ms per solve |
|---|---:|
| Upload/initialization | 3.04 |
| Device FK and sphere placement | 14.87 |
| Distances and tangent residual derivatives | 15.83 |
| Gauss–Newton accumulation | 5.87 |
| Compact download/status | 4.60 |

Host staging, result unpacking, and launch/synchronization overhead account for the difference between event totals and preparation wall time. The CPU backward pass is now the largest remaining solver phase. Further optimizing only collision matrix accumulation has a small full-solve ceiling; asynchronous overlap and additional fusion were therefore not added.

## Transfer and allocation evidence

A separate Kokkos Tools trace at N=256 verifies the evaluation data flow:

- Configuration upload: **63,736 bytes** per pass.
- Derivative download: **1,019,776 bytes** for 257 costs, 30-component gradients, and 465-entry Hessian triangles, plus four status bytes.
- Value-only download: **2,056 bytes** of costs plus four status bytes.
- No residual, world-center, spatial-gradient, center-Jacobian, or residual-Jacobian downloads.
- No working-buffer allocations after the first derivative preparation at the fixed horizon; the value-only result buffer grows once and is then reused.

Geometry uploads occur at setup for the prototype and solver clone. The trace also reports scalar initialization/fill operations as `copy` events; these are distinct from array transfers. Maximum per-workspace device payload is approximately 27.13 MiB, excluding allocator padding, CUDA context memory, and the separate prototype's resident geometry. The residual/Jacobian tile budget is 64 MiB, excluding other buffers.

Build the optional diagnostic plugin and trace separately from measurements:

```bash
c++ -shared -fPIC -I /path/to/kokkos/include tools/kokkos_copy_trace.cpp \
  -o /tmp/horizon_kokkos_copy_trace.so
KOKKOS_TOOLS_LIBS=/tmp/horizon_kokkos_copy_trace.so \
  benchmarkKyonTrajectory --backend device --nodes 256 --samples 1 \
  --threads 8 --self-collision 1 --output /tmp/kyon_copy_trace
```

## Library assessment

**Kokkos Kernels:** measured version `4.7.00`, commit `f030e6cb97cece46e14f7471108d891adc425935`, against Kokkos 4.7.1. The isolated benchmark uses 257 nodes and 30 tangent coordinates, FP64, 20 samples, and ten repeated calls per sample. It compares the production-style packed reduction with TeamGemm followed by packing, testing team sizes 32/64/128 and blocked/unblocked algorithms. Outputs agree to the benchmark tolerance.

| Residuals/node | Direct packed median, ms | Fastest tested TeamGemm + packing median, ms |
|---|---:|---:|
| 94 | 0.142 | 0.281 |
| 194 | 0.283 | 0.592 |
| 1024 | 1.491 | 3.324 |

The library computes a full Gram matrix whereas our output needs only the symmetric triangle, gradient, and scalar cost. None of the tested variants improves this workload, and accumulation is already under 1% of full N=256 solve time. Keep the direct kernel. The optional `HORIZON_BENCH_KOKKOS_KERNELS` target preserves the comparison without adding a production dependency. These results do not rule out benefits for other matrix sizes or library algorithms.

**cuRoboV2:** evaluated [commit `71ce172`](https://github.com/NVlabs/curobo/tree/71ce1726925adfdcc7d0d39dabec5f2d6bb71acd). This includes its native C++ runtime, not the legacy v1 interface. Actual CMake configuration fails because [the runtime requires CUDA 12.3](https://github.com/NVlabs/curobo/blob/71ce1726925adfdcc7d0d39dabec5f2d6bb71acd/cpp_runtime/robot/CMakeLists.txt) while the installed toolkit is 12.0.140. A dependency-free Python probe of its [joint enum](https://github.com/NVlabs/curobo/blob/71ce1726925adfdcc7d0d39dabec5f2d6bb71acd/curobo/_src/robot/types/joint_types.py) finds scalar revolute/prismatic types but no native free-flyer; the actual Kyon URDF contains the floating joint `reference`. Its [native numeric interface is float32](https://github.com/NVlabs/curobo/blob/71ce1726925adfdcc7d0d39dabec5f2d6bb71acd/cpp_runtime/robot/include/curobo_robot_runtime/api.h), and its collision-cost reductions must not replace our softplus least-squares objective.

The feasibility gates therefore stop before an integrated Kyon runtime prototype: doing that would require a newer isolated toolkit plus floating-base/tangent and precision adapters, and possibly custom cost accumulation. **No cuRoboV2 runtime speed comparison was obtained.** Its broader robot/geometry functionality could save development work for a different model or scene, but that benefit is not demonstrated for this FP64 Kyon pipeline. No system toolkit upgrade or production dependency was introduced.

**Warp:** useful geometry and differentiation components, but a second kernel/code-generation runtime does not remove a needed subsystem here. **nvblox:** useful for live TSDF/ESDF mapping, not required for this immutable dense grid. Both remain deferred rather than integrated for an unmeasured gain.

## Validation

- All 12 CTest entries pass in CPU and CUDA builds, including eight device-cost cases and Kyon trajectory smoke tests.
- CPU residual/Pinocchio references match cost, tangent gradient, and `JᵀJ`; manifold finite differences validate gradients.
- Tests cover floating bases, arbitrary axes, continuous joints, fixed frame offsets, coincident centers, environment-only/self-only/combined costs, selected/reordered nodes, clone isolation, tiled accumulation, and out-of-grid line-search recovery.
- Actual Kyon manifold descent improves an activated self-distance margin; an iLQR test separates initially overlapping spheres and matches residual-based optimization with node workers.
- Valgrind passes seven non-stress cases with zero memory errors and zero definitely lost bytes. It reports 368 possibly lost bytes in OpenMP thread-local storage allocation.
- The original CUDA 12 Compute Sanitizer failed before the first instrumented API call, including a retry with `--target-processes all`. The subsequent CUDA 13 installation successfully checked all eight collision tests with zero errors; see the follow-up report.
- CUDA/CPU Kokkos execution is tested. Other Kokkos GPU backends remain unvalidated.

Reproduce full comparisons with `tools/benchmark_device_collision.py`; raw timing/iteration/trajectory files from this run are under `/tmp/horizon-device-collision-final`. Raw copy trace, Valgrind output, cuRobo CMake probe, and Compute Sanitizer logs are under `/tmp/horizon-device-*` and `/tmp/horizon-curobo-probe.log`. Backend selection remains explicit and the scalar default is unchanged.
