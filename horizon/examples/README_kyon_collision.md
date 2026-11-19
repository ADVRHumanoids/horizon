# Kyon collision-aware iLQR example

Optimize a two-second motion of `dagana_1_base` with four fixed foot contacts,
then compare the same problem with collision costs enabled and disabled. The
default avoidance scene places a spherical obstacle in the direct path. It
uses a sampled environment SDF, Kyon's full-body 103-sphere URDF, and all 3,799
enabled self-collision sphere pairs from its disabled-link policy. `N` is the number of shooting intervals: the
trajectory contains `N+1` configurations and `N` controls.

The native C++ executable performs optimization and profiling. Python creates
an offline HTML report and optionally replays the result in Viser or ROS2/RViz.
Visualization and diagnostic FK run after optimization, outside solve timers.
For graph construction directly in Horizon Python, see the
[native Pinocchio Python example](README_ilqr_native_pinocchio.md).

## Modeling and example support

The [C++ example](../cpp/src/kyon_collision_example.cpp) keeps symbolic dynamics,
feet constraints, effort, tool target, collision-term selection, bounds, and
solver settings together. Horizon-relative node selections remove manual index
vectors:

```cpp
solver.setResidual(horizon::Nodes::running(), manager, "effort");
solver.setResidual(horizon::Nodes::terminal(), manager, "target");
solver.setConstraint(horizon::Nodes::all(), manager, "feet");
solver.setStateBounds(model->lowerPositionLimit, model->upperPositionLimit);
solver.fixInitialState(start);
solver.setInputBounds(-1., 1.);
solver.setStateInitialGuess(start);
solver.setInputInitialGuess(Vector::Zero(model->nv));
```

Contact definitions use named frame positions:

```cpp
feet.push_back(kinematics.framePosition("contact_1") -
               vectorConstant(kinematics.initialFramePosition("contact_1")));
```

Contact and tool FK uses `horizon::kinematics::PinocchioFramePositions`,
independently of collision geometry. Example support maps the frame batch to
names; collision sphere centers are evaluated separately. Both native functions
share a node-local Pinocchio workspace so robot FK can be reused.

For a full-pose target or rigid contact, use
`horizon::kinematics::PinocchioFramePoseError`. Its input is
`[q; target xyz; target quaternion xyzw]`; its output is
`[p(q) - p_target; log(R_target.T * R(q))]`. Register that output as a residual
for a soft target or as a constraint for a rigid contact. Position and rotation
components can have separate weights. See the
[Python frame and pose API](README_ilqr_native_pinocchio.md#frame-positions-and-full-pose-targets).

Vectors broadcast across the trajectory; existing explicit node-vector and
matrix APIs remain supported. `setInitialState()` sets only the node-zero
initial guess. `fixInitialState()` explicitly sets equal bounds at node zero
and initializes its guess.

[kyon_collision_support.h](../cpp/src/kyon_collision_support.h) owns scene setup,
backend installation and statistics, runtime lifetime, sampling, validation,
and serialization. Scalar, Kokkos, and device paths share that support
interface; runtime teardown follows solver and backend destruction.
[kyon_collision_support.py](kyon_collision_support.py) holds CLI, process,
replay export, and dashboard utilities. The Python entrypoint orchestrates
configuration, cases, reports, and optional replay. Existing flags, filenames,
report schemas, and exit statuses are preserved.

## Build and run

Run from the repository root. The C++ dependencies are Eigen, CasADi,
yaml-cpp, TinyXML2, and Pinocchio. Point CMake at their installed prefixes as necessary:

```bash
cmake -S horizon/cpp -B build/kyon \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/path/to/casadi;/path/to/pinocchio" \
  -DHORIZON_BUILD_PYTHON=OFF -DBUILD_TESTING=OFF \
  -DHORIZON_BUILD_EXAMPLES=ON -DHORIZON_WITH_PINOCCHIO=ON \
  -DHORIZON_PROFILING=ON
cmake --build build/kyon --target kyon_collision_example -j2
```

The reporting environment needs NumPy, Matplotlib, and the Python bindings of
Pinocchio (`pin` on PyPI). Use a Python environment compatible with the native
dependencies. The example currently requires this repository checkout for its
sphere-model data. The default runtime model and collision policy are vendored in `horizon/cpp/tests/data`:

```bash
python -m horizon.examples.kyon_collision \
  --build-dir build/kyon --N 30 --collision both \
  --samples 3 --warmup 1 --output /tmp/kyon-comparison
```

Open `/tmp/kyon-comparison/report.html`. Use `--urdf PATH --collision-policy PATH` to select a different sphere URDF
and its matching policy. No ROS installation is needed for a
headless run. The compiled executable also works directly; use
`build/kyon/kyon_collision_example --help` for its options.

```bash
# Increase the horizon; keep duration and averaged running costs fixed.
python -m horizon.examples.kyon_collision --build-dir build/kyon \
  --N 256 --collision both --output /tmp/kyon-256

# Isolate a case or omit the self-collision cost.
python -m horizon.examples.kyon_collision --build-dir build/kyon \
  --collision on --no-self-collision --output /tmp/kyon-environment-only
```

The default `avoidance` scene has a 120-iteration budget. Both scenes use
collision weight 100 with the full collision-pair set. `--scenario benchmark`
selects the small-motion timing scene.
`--collision-weight W`, `--iterations`, `--damping`, `--threads` (0 is serial),
`--samples`, and `--warmup` are configurable. The native executable defaults to
the legacy benchmark scene; the Python example explicitly selects avoidance.

The on/off problems have different objectives and can need different numbers
of iterations. Their solve-time difference measures the effect on this complete
optimization problem, rather than the isolated cost of a collision kernel.

## Profiling and reproducibility

Each invocation records exact native commands, compiler, robot/scene settings,
host platform, and an available NVIDIA GPU snapshot. Warmups and measured solves
all restart from identical initial guesses. A nonzero exit status indicates a
native/profiler error, failed convergence, or failed motion validation; a report
is still generated when the native solver exports a failed result.

Outputs include:

| File | Contents |
| --- | --- |
| `report.html`, `profile.png`, `profile.svg` | On/off comparison, accepted cost history, clearance, and all timer tables |
| `results.json` | Experiment manifest and complete profiles for every case |
| `on/profile.json`, `off/profile.json` | Setup time, raw solve samples, convergence, feasibility, collision counters, all timer distributions, and available device stages/transfer bytes |
| `*/trajectory.csv` | Final measured configuration/control trajectory |
| `*/trajectory_iterations_K.csv` | Every solver callback in measured solve K, including initial state and rejected line-search trials |
| `*/replay.npz` | Post-solve FK and sampled-SDF/self-clearance diagnostics |
| `*/process_counters.txt`, `*/run.log` | GNU time process counters (where installed) and native output |

Timer distributions contain calls, total, mean, minimum, p50, p95, and maximum
microseconds for each measured solve. Native phase totals exclude post-solve
validation/export. Nested timers overlap, and worker timers sum CPU work across
threads; they must not be added to estimate elapsed time. The dashboard stacks
only the top-level linearize/quadratize, backward, and line-search phases.
`process_counters.txt` includes native setup, warmups, validation, and export.

The trial CSV's `callback_interval_ms` measures time since the previous
callback. The `*_completed_since_callback_ms` fields collect timers that have
finished since that callback. A line-search timer finishes after its trial
callback, so its duration can appear in the next row. These are event traces,
not independent per-iteration phase profiles. The JSON's legacy `iterations`
field counts accepted steps, not failed line-search attempts.
`last_evaluated_cost_components` describes the solver's last evaluated trial;
the accepted objective is reported separately as `objective`.

For meaningful latency measurements, first run without instrumentation. Then
use a separate output directory for expensive diagnostic profiling:

```bash
python -m horizon.examples.kyon_collision --build-dir build/kyon \
  --scenario benchmark --N 12 --collision on --threads 0 \
  --samples 1 --warmup 0 --profiler callgrind --output /tmp/kyon-callgrind
callgrind_annotate /tmp/kyon-callgrind/on/callgrind.out.*

python -m horizon.examples.kyon_collision --build-dir build/kyon \
  --scenario benchmark --N 12 --collision on --threads 0 \
  --samples 1 --warmup 0 --profiler memcheck --output /tmp/kyon-memcheck
```

Valgrind adds substantial overhead; Callgrind profiles the whole native process.
Memcheck and CUDA Compute Sanitizer use exit code 99 when they detect errors.
Keep CPU threading, build flags, scene, and machine load consistent when
comparing timings. Reusing an output directory replaces its matching trajectory
traces; use a new directory when changing profiler or configuration.
The wrapper defaults `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`MKL_NUM_THREADS` to 1, preserving explicitly supplied values and recording them
in each case's manifest. `--threads` controls the solver's node workers.
Keeping inner matrix operations serial avoids nested parallelism and excessive
OpenMP spinning under Valgrind.

## GPU backend

`--backend scalar` is the native CPU reference. `--backend kokkos` batches
collision geometry but retains host work; `--backend device` keeps collision
FK, SDF queries, self-collision, and derivative accumulation in Kokkos's compiled
execution space. Only the device backend is the complete on-device collision
path. A Kokkos OpenMP build still executes on CPU; JSON records the actual
execution space.

Build with `-DHORIZON_WITH_KOKKOS=ON` and a CUDA-enabled Kokkos installation for
GPU execution. See [the Kokkos build instructions](../cpp/README.md#kokkos-residual-batches-across-horizon-nodes),
[the device backend measurements](../cpp/docs/device_collision_performance.md), and
[CUDA toolkit/library integration notes](../cpp/docs/cuda13_library_integrations.md).
An existing `tests/benchmarkKyonTrajectory` binary is also accepted by
`--binary` or `--build-dir`.

```bash
python -m horizon.examples.kyon_collision --build-dir /tmp/horizon-gpu-build \
  --backend device --N 256 --output /tmp/kyon-gpu-256

python -m horizon.examples.kyon_collision --build-dir /tmp/horizon-gpu-build \
  --backend device --scenario benchmark --N 12 --collision on \
  --samples 1 --warmup 0 --profiler cuda-memcheck \
  --sanitizer /path/to/cuda/bin/compute-sanitizer --output /tmp/kyon-cuda-check
```

Compute Sanitizer checks correctness; the native stage timers provide the GPU
timing breakdown. It is not a substitute for a kernel hardware-counter profiler.

## Optional visualization

Replay any completed result without rerunning optimization. Both viewers show
the robot meshes, obstacle, target, foot anchors, and tool paths for the enabled
and disabled trajectories. Base motion and scalar joints are replayed together;
configuration interpolation uses Pinocchio's manifold operations. Sphere
overlays use the same geometry as optimization.

For a browser viewer, install `viser` and `yourdfpy` (with their mesh dependencies):

```bash
python -m horizon.examples.kyon_collision --replay /tmp/kyon-comparison \
  --visualizer viser --port 8080 --show-spheres
```

Open `http://127.0.0.1:8080`. Playback, scrubbing, speed, trajectory selection,
and sphere visibility have controls. The server listens only on localhost.

For ROS2, source the ROS environment and provide `rclpy`, `tf2_ros`, standard
messages, `robot_state_publisher`, and optionally `rviz2`:

```bash
source /opt/ros/jazzy/setup.bash
python -m horizon.examples.kyon_collision --replay /tmp/kyon-comparison \
  --visualizer ros2 --rviz --show-spheres
```

The example starts one robot-state publisher per case, publishes joint states
in `/kyon_collision_on` and `/kyon_collision_off`, and publishes corresponding
`on/` and `off/` TF frames under `world`. Markers use
`/kyon_collision/markers`. It writes `kyon_collision.rviz` and publisher parameter
files beside the report. RViz uses the generated configuration. Ctrl-C stops
replay and its child processes. `--speed` changes playback speed;
`--visualization-seconds 5` gives a bounded smoke test.

Core Kyon mesh assets must be available. Missing optional external sensor meshes
(for example `zed_description`) are omitted with a warning while their frames
are preserved. Both viewers resolve Kyon's package mesh paths from the URDF
checkout or the sourced ROS package index.

## Interpretation and validation

This is kinematic configuration-space shooting, controlled by generalized
velocity, with Pinocchio integration and fixed foot equalities. It is not a
rigid-body dynamics or actuator-feasibility example. Running costs are averaged
over N; terminal target costs are not. Collision residuals use a 20 mm margin,
5 mm smoothing, and Gauss–Newton curvature.

Validation checks convergence, target error below 5 mm, maximum foot drift and
dynamics gap below 0.1 mm (the gap uses tangent-coordinate norm), and positive
environment/self clearance when collision costs are enabled. Disabled runs
still report clearance but may penetrate the obstacle intentionally. The
self-pair mask covers the two hands, not all possible robot link pairs.
Soft collision costs are checked at shooting knots; continuous swept collision
avoidance is not guaranteed by this example.

Profiling is optional for solving, reports, and Viser/ROS2 replay. A build with
`HORIZON_PROFILING=OFF` still reports solve time and trajectory diagnostics;
internal solver phase timings are marked unavailable. Enable
`HORIZON_PROFILING=ON` only when you want those detailed timer tables.

Run the headless integration regression against either build:

```bash
HORIZON_KYON_EXAMPLE_BINARY="$PWD/build/kyon/kyon_collision_example" \
  python -m unittest horizon.tests.test_kyon_collision_example -v
```
