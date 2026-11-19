# CUDA 13: cuRoboV2 and nvblox integration probes

Historical library feasibility measurements below used the previous 94-sphere, 100-pair workload. See [current full-body measurements](device_collision_performance.md) for the runtime URDF/JSON model.

The newer toolkit is installed and both native libraries build and run. For the current Kyon/iLQR workload, keep the existing collision implementation. **nvblox is useful as an optional map producer; replacing dense queries with its sparse hash is not a speed improvement here. cuRoboV2 has very fast native FK/Jacobians, but the tested adapter/cost composition does not improve the current collision pass.**

[Machine-readable results](cuda13_library_results.json) contain revisions, numerical errors, timings, dependencies, and validation scope. These experiments are isolated tools; the production FP64 backend and solver objective are unchanged.

## Installed toolkit and environment

CUDA **13.0.2**, compiler **13.0.88**, is installed at `/home/alaurenzi/.local/opt/cuda-13.0.2`. NVIDIA's official redistributable archives were verified against their manifest SHA-256 hashes. Compiler, runtime, NVRTC, CCCL, math libraries (including cuBLAS/cuSOLVER/cuSPARSE/cuFFT), NPP, CUPTI, Compute Sanitizer, and inspection tools are available. The driver remains 580.178.04; system `/usr/bin/nvcc` remains 12.0.

Select the new toolkit for the current shell:

```bash
source tools/cuda13_env.sh
nvcc --version
```

`CUDA13_ROOT` can override the prefix. The installer is reproducible with `python3 tools/install_cuda_toolkit.py --prefix /desired/prefix`; it installs no driver and changes no shell startup files.

cuRoboV2 uses an isolated `/tmp/horizon-curobo-v2-venv`, with read-only access to the existing PyTorch 2.9.0+cu128 and Pinocchio wheels. CUDA Python packages are pinned to driver-compatible versions: cuda-core 0.7.0, bindings 13.0.3, runtime 13.0.96, NVRTC 13.0.88, Warp 1.19.0. Unconstrained current pip extras selected CUDA 13.5 components; those were replaced inside this isolated environment. The existing ROS/Python environment was not modified.

## cuRoboV2: feasible floating-base adapter, no demonstrated backend win

Tested native runtime [revision `71ce172`](https://github.com/NVlabs/curobo/tree/71ce1726925adfdcc7d0d39dabec5f2d6bb71acd), built with CUDA 13.0.2 for SM 8.6. Exported Kyon relative to `pelvis`, with 24 articulated joints and all 21 collision frames. The floating joint is handled by a GPU quaternion transform and body-local tangent Jacobian, preserving Kyon's existing configuration convention. No Euler-joint replacement or URDF modification was required.

The native ABI returns FP32 link poses/Jacobians. The adapter applies sphere offsets, rotates articulated Jacobians into the world frame, and adds the six free-base tangent columns in FP64. It explicitly maps exported joint order to Pinocchio order. A small fused CUDA adapter is compared with a straightforward tensor implementation; their results agree to approximately `3.4e-16` absolute.

All 94 spheres and 100 explicit inter-hand pairs are evaluated against the same sampled dense SDF. The scene deliberately activates environment penalties. The prototype preserves the softplus least-squares formula and computes its gradient and Gauss–Newton matrix in FP64 using PyTorch, rather than substituting cuRobo's planner objective. Pinocchio is the independent FK/Jacobian reference.

### Timing and accuracy

Median device times in milliseconds, 20 samples. cuRobo measurements use CUDA graph replay. The Horizon reference reads the **identical configurations, float SDF samples, and FP64 oracle outputs** and reports the sum of its FK, query, and Gauss–Newton CUDA-event times. Its full preparation wall times, including compact transfers, are also in the JSON. Graph capture and execution structure differ, so these are controlled component/pass comparisons, not a full-solver speed comparison or a ceiling on optimized cuRobo integrations.

| States (N+1) | cuRobo native FK/J only | Native FK + fused sphere/tangent adapter | Prototype full collision compute | Existing Horizon FP64 collision compute |
|---:|---:|---:|---:|---:|
| 31 | 0.0188 | 0.0699 | 0.500 | 0.250 |
| 257 | 0.0257 | 0.6227 | 2.852 | 0.981 |
| 1025 | 0.0622 | 2.5421 | 11.162 | 2.911 |

The unoptimized tensor adapter is much slower (2.26 ms at 257 states); fusion reduces its FK/adapter composition to 0.62 ms. Even with fusion, expanding all sphere Jacobians into FP64 and composing the custom cost is expensive. The existing implementation projects distance derivatives directly and avoids that intermediate representation.

Across tested states, maximum sphere-position and tangent-Jacobian errors versus Pinocchio are about `1.6e-7`. Cost/gradient/Gauss–Newton relative norm errors are approximately `0.8–1.0e-6`. For 257 states, maximum absolute gradient and Hessian errors are 0.00345 and 0.0918 respectively, on the deliberately strongly weighted active cost. The existing FP64 pass agrees with the exported FP64 oracle to a maximum scaled-entry error below `1.2e-11` across all batches.

The native FK timing is encouraging, but compares FP32 library arithmetic with the existing FP64 implementation; its speed cannot be attributed to library algorithms alone. The current prototype does **not** satisfy the existing tight FP64 parity target and has no demonstrated full-solve speedup. A production candidate would need FP32 accuracy acceptance, direct projection into collision derivatives rather than materializing sphere Jacobians, and a measured benefit that justifies the dependency. The existing native FK API computes Jacobians even during value-only evaluations. These findings do not establish the performance of cuRobo's own planner/collision costs, which were not substituted for Horizon's objective.

For this workload, measured collision preparation is only about 55 ms of the previous 887 ms N=256 solve. Eliminating that preparation entirely would save about 6%, so further collision-only integration cannot yield a large full-solve improvement without moving additional work.

## nvblox: useful upstream mapping and GPU snapshots

Tested [revision `5bd2d73`](https://github.com/nvidia-isaac/nvblox/tree/5bd2d73da901ce0ba279c997a4c1f9f632b9309c), built with CUDA 13.0.2, without the renderer or PyTorch wrapper.

The probe first imports the same signed-distance samples into nvblox voxel blocks and compares dense and sparse trilinear queries. Both paths use FP64 interpolation and the exact derivative of the interpolated field. Distances and gradients agree exactly in the measured cases. Missing/unobserved corners and outside queries are rejected, rather than interpreted as free space.

| Distance/gradient queries | Dense median ms | Sparse hash median ms |
|---:|---:|---:|
| 2,914 | 0.0278 | 0.0743 |
| 24,158 | 0.1065 | 0.2626 |
| 96,350 | 0.3677 | 0.7500 |
| 1,000,000 | 3.4606 | 6.6005 |

This grid is fully observed: 6,859 live blocks, 7,000 preallocated blocks. Dense sample storage is 13.13 MiB; live sparse TSDF voxel storage is 26.79 MiB, excluding hash/pool overhead. Sparse storage becomes useful when only a small part of a large domain is observed; it is not a memory or query-speed advantage for this fully populated fixed box.

The second probe actually generates an ESDF from the synthetic observed TSDF:

- Initial full ESDF build: **13.63 ms**.
- Genuine changed-block ESDF update: **1.95 ms**. The test inserts a local obstacle; a query changes from 0.52 m clearance to 0.00 m.
- Full re-update with unchanged input: 2.10 ms; this is reported separately and is not claimed as changed-scene timing.
- Export the complete 151³ ESDF into GPU dense buffers: **2.22 ms**, including FP32 output for Horizon-style sampling and FP16 conversion/reordering into a real cuRoboV2 scene buffer.

The snapshot producer writes directly to the [stable CUDA address owned by cuRoboV2's scene](https://github.com/NVlabs/curobo/blob/71ce1726925adfdcc7d0d39dabec5f2d6bb71acd/cpp_runtime/robot/include/curobo_robot_runtime/scene.h). The probe validates the complete buffer, x-fastest versus z-fastest order, positive-outside sign, and stable address. No map-sized host round trip occurs in the update/export path; host readbacks are only validation and status checks. A cuRobo motion-planning solve consuming the map was not run.

At 4 cm voxels, the generated ESDF differs from the analytic test sphere by up to 3.71 cm; this quantization is distinct from backend numerical error. The interpolant's gradient matches finite differences within 9.1e-11. FP16 scene conversion introduces up to 1.95 mm additional error over the entire grid. Horizon's FP32 dense snapshot avoids this extra FP16 conversion. RGB-D/LiDAR ingestion performance was not tested.

## Recommended integration boundary

1. Keep Horizon's current FK/collision/Gauss–Newton path for this static-grid iLQR problem.
2. If live mapping is required, add nvblox as an optional **environment producer**, exporting a bounded FP32 dense GPU snapshot into the existing distance evaluator. Keep the custom collision objective and tangent derivatives.
3. Track the snapshot's origin, voxel-center convention, dimensions, validity mask, ownership and generation. Keep one immutable generation throughout quadratization and its line-search trials. Order producer/consumer streams with CUDA events; publish only after validity checks. Do not silently accept unknown cells.
4. Use the demonstrated nvblox-to-cuRobo FP16 scene path when adopting cuRobo's broader planning/scene stack and its precision is acceptable. A fast FK component alone does not justify that switch for this solver.

This keeps the library-specific mapping code outside the portable collision core and avoids a second production collision implementation for an unproven gain.

## Validation and reproduction

CUDA 13 Compute Sanitizer now works. All eight Horizon collision tests, the 31-state cuRobo native/fused-adapter check, and the nvblox query/update/snapshot check report **zero memory errors**. cuRobo's native CTest suite reports 10 passed and one calibration test skipped because its artifact was not supplied.

Sources live in `tools/probe_curobo_v2.py`, `tools/curobo_adapter.cu`, and the standalone `tools/nvblox_probe` CMake project. `benchmarkCollisionPass` is built with the existing Kyon/Kokkos test options. It reads the `--reference-output` artifacts from the Python probe and checks the FP64 oracle before timing. These probes are outside the production library dependency graph.

The local builds/artifacts are `/tmp/horizon-curobo-cuda13-build`, `/tmp/horizon-curobo-kyon-artifact`, `/tmp/horizon-nvblox-cuda13-build`, and `/tmp/horizon-nvblox-probe-build`. Select the toolkit explicitly when configuring native projects:

```bash
source tools/cuda13_env.sh
cmake -S /path/to/curobo/cpp_runtime/robot -B /tmp/curobo-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" -DCMAKE_CUDA_ARCHITECTURES=86
cmake -S /path/to/nvblox -B /tmp/nvblox-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBUILD_PYTORCH_WRAPPER=OFF -DBUILD_RENDERER=OFF -DBUILD_TESTING=OFF \
  -DBUILD_BENCHMARKS=OFF -DUSE_SYSTEM_EIGEN=ON
```

Use the pinned source revisions in the JSON. Export Kyon with `export_kinematics.py --urdf ... --base-link pelvis --tool-frame ...` for every occupied collision link. Compile the small fused adapter with the command at the top of its source, then pass it to `probe_curobo_v2.py --adapter-library ...`. The nvblox probe accepts `NVBLOX_SOURCE`, `NVBLOX_BUILD`, and optional `CUROBO_SOURCE`/`CUROBO_BUILD` CMake paths. Both probes have `--check-only` modes for sanitizer runs; those runs are not used as timing results.

The GPU was shared with existing applications. Timing boundaries and precision are recorded explicitly; no new full-trajectory optimization speedup is claimed from these library probes.
