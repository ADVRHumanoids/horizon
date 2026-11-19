# Native Pinocchio from Horizon Python

For the complete native kinematics/dynamics API, analytic Jacobian table, and
torque-driven contact dynamics, see the [native robot guide](README_native_robot.md).

Python can configure the new function manager and compose CasADi MX expressions
around native C++ Pinocchio calls. Graph construction happens in Python;
values and analytical first derivatives run in C++, including parallel iLQR
node evaluation. There are no Python FK callbacks in the solve loop.

The existing `Problem` / `Solver.make_solver('ilqr', problem, opts)` interface
and low-level CasADi setters remain supported. Native dynamics and robot costs
are optional additions.

## Build

Use the Python interpreter/environment that provides Horizon's dependencies:

```bash
cmake -S horizon/cpp -B build/python-native \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/path/to/casadi;/path/to/pinocchio;/path/to/pybind11" \
  -DPYTHON_EXECUTABLE="$(command -v python)" \
  -DHORIZON_BUILD_PYTHON=ON -DHORIZON_WITH_PINOCCHIO=ON \
  -DHORIZON_PROFILING=ON -DBUILD_TESTING=ON
cmake --build build/python-native --target pyilqr -j2
ctest --test-dir build/python-native -R pythonIlqrInterop --output-on-failure
```

The existing build rule copies the extension into `horizon/solvers`. CTest
explicitly loads the extension from its own build directory, so an older
package copy cannot mask a regression. The Kyon test additionally needs
`KYON_URDF`;
otherwise that portion is skipped. Python Pinocchio is used for setup and an
independent reference check, and is required by the Kyon example/test.

Without `HORIZON_WITH_PINOCCHIO`, the managers, CasADi-native adapters, and
legacy Python API remain available; `PinocchioModel` is omitted.

## Model through Problem

The [native robot guide](README_native_robot.md) shows direct MX expressions,
full pose targets, underactuated contact dynamics, force bounds and collision
helpers. Costs, constraints and dynamics are defined on `Problem`; call
`Solver.make_solver` to compile for a backend. There is no public export step
or native registration layer to configure.

The [kinematic example](ilqr_native_pinocchio.py) defines a velocity-controlled
motion with fixed feet and a tool position objective. It preserves its
historical objective through explicit residual weights. General Problem
residuals are squared norms summed over active nodes for either backend.
The legacy `NativeModel` adapter and low-level managers remain supported.

```bash
python -m horizon.examples.ilqr_native_pinocchio --urdf /path/to/kyon.urdf --N 30
python -m horizon.examples.ilqr_native_pinocchio --urdf /path/to/kyon.urdf --N 30 --solver ipopt
```

Robot presets, solver options and validation live in
[native_pinocchio_support.py](native_pinocchio_support.py). The
[torque-controlled example](native_contact_motion.py) returns a Problem with
contacts, a full tool pose residual, positive fZ, self collision and environment
collision constraints. Use IPOPT for its hard function inequalities.

Native first derivatives run in C++. IPOPT uses ambient configuration
coordinates; iLQR infers the robot manifold from native calls, with an optional
explicit `state_space` in `Problem.setDiscreteDynamics`. Native residuals use
Gauss–Newton curvature, and native scalar costs require IPOPT limited-memory
Hessians. Keep graph topology fixed after constructing a solver, and build a
fresh Problem for each backend. Parameters, bounds and active nodes can be
updated before repeated solves.

The profiling results below describe the earlier low-level implementation of
the same kinematic example and objective. They are retained as historical
measurements, not measurements of every modeling frontend.

### IPOPT bottleneck investigation and fixes

The slow initial result was dominated by the **constraint Jacobian**, not the
Gauss--Newton Hessian. At N=30 the baseline solve took 0.502 s, of which
0.319 s was `nlp_jac_g`; `nlp_hess_l` took only 0.0077 s. See the
[baseline phase timings](../cpp/docs/ipopt_profile_before.json).

Three costs were identified and addressed:

- Horizon's default ten-thread CasADi maps create threads during numerical
  evaluations while each native adapter protects one mutable workspace with a
  mutex. For this workload the resulting overhead outweighs the benefit.
  `Problem(..., thread_map_num=1)` now allows explicit serial maps; the native
  example defaults to one map thread. `--map-threads` overrides it. The general
  Problem default remains ten for compatibility. iLQR worker count is separate.
- IPOPT's monotone barrier strategy required 44 and 27 iterations at N=12 and
  N=30. The example now uses `mu_strategy=adaptive`, which takes 15 and 11.
  `mu_min=1e-13` keeps the existing trajectory agreement checks satisfied;
  the original `tol=1e-11`, objective, bounds, and constraints are preserved.
- Native CasADi callbacks used allocating DM input/output conversions, and
  ambient Pinocchio derivatives factored a full robot-sized tangent projection
  matrix. Callbacks now use CasADi's direct buffer interface. The projection
  solves independent joint blocks; a regression compares it against the dense
  reference, including nonunit quaternion NLP trial points.

A controlled profile with the final native kernels isolates map and barrier
choices (three solves per case; medians):

| Intervals | Map threads | Barrier | Iterations | Solve |
|---:|---:|---|---:|---:|
| 12 | 10 | monotone | 44 | 0.5428 s |
| 12 | 1 | monotone | 44 | 0.2227 s |
| 12 | 1 | adaptive | 15 | 0.0826 s |
| 30 | 10 | monotone | 27 | 0.4326 s |
| 30 | 1 | monotone | 27 | 0.3313 s |
| 30 | 1 | adaptive | 11 | 0.1524 s |

The final isolated iLQR/IPOPT comparison gives:

| Intervals | iLQR | IPOPT | IPOPT before fixes | IPOPT objective |
|---:|---:|---:|---:|---:|
| 12 | 0.0228 s | 0.0798 s | 0.6251 s | 5.308068e-6 |
| 30 | 0.0580 s | 0.1538 s | 0.5222 s | 5.308151e-6 |

All runs pass the original feasibility and objective checks. The existing
trajectory tolerances are unchanged. [Final comparison data](../cpp/docs/ipopt_native_results_optimized.json)
and [per-kernel profiles](../cpp/docs/ipopt_profile_after.json) include raw runs.
Reproduce without running competing benchmarks concurrently:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m tools.profile_native_ipopt \
  --urdf /path/to/kyon.urdf --nodes 12 30 --samples 3 --output /tmp/profile.json
```

The remaining bottleneck is still constraint differentiation: roughly 88 ms
of a 152 ms N=30 solve, versus 0.7 ms for the GN Hessian. Native callbacks
currently declare dense Jacobians; CasADi differentiates the horizon maps with
large directional sweeps (the N=30 dynamics uses 64- and 28-direction batches).
Propagating proven structural native sparsity would reduce this work and the
KKT structure. Numerical sampling must not be used to infer structural zeros.

A separate [IPOPT detailed timer](../cpp/docs/ipopt_linear_timing.txt) reports
about 38 ms in the primal-dual system solver and 81 ms evaluating equality
Jacobians. Timer categories are nested and must not be summed independently.
Linear-system work is material but is not the original dominant cost. These
figures describe this example and machine, not a general solver ranking.
