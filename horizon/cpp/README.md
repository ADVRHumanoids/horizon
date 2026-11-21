# C++ iLQR functions and state spaces

For the complete native kinematics/dynamics API, analytic Jacobian table, and
torque-driven contact dynamics, see the [native robot guide](../examples/README_native_robot.md).

For a complete Kyon motion example with collision on/off comparisons, profiling,
and optional Viser or ROS2/RViz replay, see the
[Kyon example guide](../examples/README_kyon_collision.md).

The solver accepts pure CasADi models, native C++ models with supplied
derivatives, and expressions mixing both. Existing `IterativeLQR(casadi::Function,
N, options)` and CasADi cost/constraint setters remain available. The new C++
interfaces live in `horizon::ilqr`; Python bindings retain their existing API.

Python can also configure native function graphs and call C++ Pinocchio directly.
See the [Python manager guide](../examples/README_ilqr_native_pinocchio.md) and
its fixed-feet Kyon example, including integration with Horizon's `Problem`.

## State and control bounds

`setStateBounds(lb, ub)` accepts `stateSize()` by `N+1` matrices;
`setInputBounds(lb, ub)` accepts `nu` by `N` matrices. Use `-infinity` and
`+infinity` for missing sides. Bounds with `lb == ub` remain hard equality
constraints in the Riccati or baseline KKT solve. `setInitialState()` sets an
initial guess; fix the initial state using equal bounds at node zero.

Enable augmented Lagrangian handling of the remaining component bounds:

```cpp
horizon::IterativeLQR solver(dynamics, N, {
    {"ilqr.enable_auglag", 1},
    {"ilqr.rho_base", 1.0},
    {"ilqr.rho_max", 1e8}
});
solver.setStateBounds(state_lb, state_ub);
solver.setInputBounds(control_lb, control_ub);
bool converged = solver.solve(200);
```

The same options pass through the Python solver. State inequalities apply to
stored state coordinates, including the terminal node; their derivatives are
converted to tangent coordinates for custom state spaces. There is no terminal
control bound. General nonlinear inequality functions are not supported here.

For each component, the method eliminates a slack constrained to `[lb, ub]`.
With signed multiplier `lambda`, it uses
`z = lambda + rho * (v - projection(v + lambda/rho, [lb, ub]))` and adds
`(z*z - lambda*lambda)/(2*rho)` to the cost. Its gradient is `z` and its
curvature is `rho` in the shifted active region, zero elsewhere. Multiplier
updates set `lambda = z`, so lower-bound multipliers are negative, upper-bound
multipliers positive, and inactive multipliers can return to zero. Running
penalties have the same `1/N` scaling as running costs.

The solver updates multipliers after an inner subproblem becomes stationary,
using its current penalty. It increases the next penalty only if the maximum
bound violation fails to decrease by `ilqr.auglag_progress_ratio` (default
`0.5`), with growth factor `ilqr.rho_growth_factor` (default `10`) and cap
`ilqr.rho_max`. `ilqr.auglag_inner_tolerance` (default `1e-10`) controls the
relative merit derivative for inner convergence; the control-increment and
equality/defect tolerances also apply. `solve(max_iter)` counts all inner
iterations and returns false if that budget expires before convergence.

`ForwardPassResult.bound_violation` reports the maximum component violation.
Stopping with AL enabled requires bound feasibility and convergence of the
projected multiplier update, as well as the usual stationarity and equality
checks. Duals and the penalty warm-start across solves. Changing a bound clears
its multiplier; passing identical bounds preserves it. Line-search filter
entries reset when the AL objective changes.

AL is disabled by default. When disabled, `rho_base` defaults to zero; explicitly
setting a positive `rho_base` retains a quadratic bound penalty without dual
updates or hard inequality feasibility guarantees. Enabling AL defaults it to
`1.0` and requires a positive finite starting penalty. Invalid or NaN bounds,
reversed bounds, and invalid penalty options throw `std::invalid_argument`.

## C++ formatting and linting

From the repository root, check all C++ sources and headers with clang-format
15 or newer and clang-tidy:

```bash
python3 tools/lint_cpp.py --build-dir /path/to/build
```

The build directory must contain `compile_commands.json`; configure your CMake
build with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` to generate it. Enable the optional
backends you intend to check: sources absent from the build database may fail
clang-tidy because their include paths or compile definitions are unavailable.
Headers are formatted directly and checked by clang-tidy through their includes.

To apply fixes, or to format without a configured build:

```bash
python3 tools/lint_cpp.py --build-dir /path/to/build --fix
python3 tools/lint_cpp.py --format-only --fix
```

Use `--tidy-only` to skip formatting, or `--clang-format clang-format-18`
and `--clang-tidy clang-tidy-18` to select versioned executables. The script
includes tracked and new C++ files throughout the repository, excludes ignored
build artifacts, and returns a nonzero status if either tool reports a failure.
Fix mode runs clang-tidy before clang-format so inserted braces get formatted.
The `.clang-format` configuration preserves manually inserted blank lines before
control statements. Clang-format cannot automatically insert those blank lines.

## Symbolic expressions around native functions

`SymbolicFunctionManager` is an external registry and expression builder:

```cpp
using namespace horizon::ilqr;
using casadi::MX;

SymbolicFunctionManager functions;
auto fk = functions.addNative("tool_position", native_fk);
auto q = MX::sym("q", nq);
auto u = MX::sym("u", nu);
auto ref = MX::sym("p_ref", 3);

// cartesianError is an ordinary CasADi Function.
auto residual = cartesianError({fk(q) - ref}).at(0);
functions.addResidual("tool_tracking", {q, u, ref}, residual);
functions.addExpression("dynamics", {q, u}, discrete_dynamics, false);

horizon::IterativeLQR solver(functions, "dynamics", N, options, robot_space);
solver.setResidual({N}, functions, "tool_tracking");
solver.setParameterValue("p_ref", references); // 3 by N+1
```

The first two input symbols are the stored state and control. Further symbols
are named parameters; their values have one column per node, including node N.
Use the entire robot state as the first symbol when it includes more than q;
native calls can take slices of it. Multiple calls, shared results, nested
calls, symbolic argument transforms, and nonlinear symbolic operations are
supported. For poses, choose a consistent representation and a group-based
pose error, such as your `log(T_ref.inverse() * T)` operation.

`fk(q)` creates a placeholder owned by this builder. Finalize expressions with
the **same manager** using `addExpression`, `addResidual`, or `addCost`;
constructing a plain CasADi Function from an unfinalized placeholder leaves a
free symbol. The manager lowers expressions to a numerical graph. CasADi AD
handles pure symbolic segments; native segments call C++ directly. Shared native
call results are evaluated once per forward sweep. The compiled function owns
its graph, so the manager can be destroyed after solver construction/setup.

`addResidual` builds first derivatives only. `solver.setResidual` defines
`0.5 * residual.squaredNorm()` with a tangent-space Gauss–Newton Hessian.
`addCost` builds exact second derivatives; every reachable native call must
provide a weighted Hessian. `addExpression(..., false)` is suitable for dynamics
and equality constraints, which need only first derivatives.

See [ilqr_mixed_example.cpp](src/ilqr_mixed_example.cpp) for a complete executable
with native Cartesian FK, symbolic Cartesian error, and an iLQR solve.

## Implementing native functions

Derive from `Function` and provide:

- `evaluateImpl(x, value)`;
- `jacobianImpl(x, J)`, with output components in rows;
- `clone()`, returning an evaluator with independent mutable workspace;
- optionally `hasHessian()` and `weightedHessianImpl(x, weights, H)`, computing
  `sum_i weights[i] * Hessian(f_i)`.

All output buffers are pre-sized Eigen references. Implementations overwrite
the entire output. `CallbackFunction` is a convenience for stateless/reentrant
callbacks. For Pinocchio, a subclass can share an immutable Model and create a
fresh Data in each clone. Do not capture a shared mutable Data in callbacks.
Throw `EvaluationError` for a recoverable numerical/domain failure. Nonfinite
native outputs raise this exception automatically. Line search retries a
shorter step; errors at the accepted iterate and programming errors propagate.

For native FK whose evaluation takes nq stored configuration coordinates but
whose Jacobian has nv tangent columns, derive from **`TangentFunction`** instead.
Pass the configuration `StateSpace` to its constructor, implement
`evaluateImpl`, `tangentJacobian`, and `clone`. For exact costs, also implement
`tangentWeightedHessian` and return true from `hasHessian`. The adapter converts
the supplied derivatives internally and preserves the local first/second
derivatives when the solver pulls them back through the state retraction.
For a scalar stage whose native input space is the product of the solver's
state-space instance and Euclidean control/parameter coordinates, the solver
uses the native tangent gradient and exact Hessian directly. Mixed exact-cost
graphs also use those derivatives for compatible native leaves; transformed
arguments retain the ambient compatibility path.
Do not pad an nv-column Jacobian with zero columns to make it nq columns.

`FunctionManager::add(name, function, parameters)` also supports direct
registration without symbolic expressions. A stage map receives the concatenated
vector `[state; control; parameters...]`. Register CasADi functions with
`CasadiFunction(f, exact_hessian)`; it concatenates vector inputs in declaration
order and generates the selected derivatives at setup. `ComposedFunction`
offers explicit `outer(inner(x))` composition, including the full Hessian chain
rule when both components supply second derivatives.

## User-defined state operations

The solver defaults to `EuclideanSpace`. Supply any `StateSpace` as the fourth
argument after options for a CasADi dynamics constructor, or after options for
the manager constructor. The operations define the geometry; no robot group
is hard-coded into the solver:

```cpp
auto robot_space = std::make_shared<CallbackStateSpace>(
    neutral_robot_state, nv,
    [](Input x, Input dx, VectorOutput out) { robot_sum(x, dx, out); },
    [](Input from, Input to, VectorOutput out) { robot_difference(from, to, out); });
```

Bind `robot_sum` and `robot_difference` to the operations for your supplied
Pinocchio model or your own oplus/ominus. `difference(from,to)` returns the
tangent at `from` that takes it to `to`. The state and tangent sizes may differ.
Provide a valid neutral state; guesses, dynamics results, and fixed state
targets must belong to the same state space.

`CallbackStateSpace` accepts optional analytical callbacks for:

- the storage Jacobian of `integrate(x,dx)` at dx=0;
- its scalar weighted Hessian at dx=0;
- the two tangent Jacobians of
  `difference(integrate(from,a), integrate(to,b))` at a=b=0.

Without those callbacks the base class uses central finite differences. For
performance-sensitive robot problems, supply analytical derivatives or implement
`StateSpace` directly. State-space operations are const and must be reentrant;
they can run on different nodes concurrently. `QuaternionSpace` is an optional
SO(3) implementation with `[qx,qy,qz,qw]` storage and body increments.
`ProductSpace` combines components, e.g. configuration, Euclidean velocity,
control, and parameters. The quaternion logarithm uses the shortest rotation
and is nonsmooth at the pi branch cut.

The backward recursion and KKT system use tangent dimensions. Line search
retracts increments through the supplied sum operation. Dynamics gaps use the
supplied difference, and linearization transports derivatives between the
predicted and next shooting states, including infeasible shooting trajectories.
Exact cost Hessians include retraction curvature; residual Hessians use
Gauss–Newton directly in tangent coordinates.

Bounds refer to stored components. A fully fixed state uses the supplied
difference and tangent-sized equality equations, avoiding redundant quaternion
component constraints. Partially fixed components and inequality penalties use
storage coordinates and are pulled back through the retraction. `setInitialState`
sets the first guess; fix its bounds as well to enforce an initial condition.
Terminal functions must be independent of control: the final node has no
control decision variable (the existing API passes the last control for evaluation).

## Performance, ownership, and validation

Each active horizon node owns cloned function workspaces, including legacy
CasADi wrappers. Native evaluation never crosses a Python boundary or a CasADi
callback. Derivative construction and graph lowering occur at setup; numerical
graph buffers are reused. Graphs allocate the quadratic tape only when an ambient exact Hessian is
requested. The joint `valueAndJacobian` path runs a residual
graph in one sweep and evaluates a shared native call once; native subclasses
can override `valueAndJacobianImpl` to share their own evaluation work.
Tangent-native subclasses can override `valueAndTangentJacobian` instead.
The solver uses `valueAndJacobianProduct(x, directions, value, product)` with
directions `[Rx, Iu, 0p]`, so native derivatives stay in tangent coordinates and
Gauss–Newton terms are accumulated directly there. Existing ambient Jacobian
and exact-Hessian interfaces remain supported. CasADi segments propagate their
CSC nonzeros directly; selection/concatenation seeds avoid dense products.
`StateSpace::tangentProjection` and `projectDirections` have generic fallbacks;
product, Euclidean and quaternion spaces use blockwise implementations. Custom
state operations continue to work without implementing these optimizations.

Each node has a separate `EvaluationContext`. Native calls registered through
`addNative` share one cloned evaluator and memoized value/Jacobian across that
node's dynamics, residual and constraint graphs. Keys compare the complete
native input bitwise; caches are invalidated for every linearization and trial
cost pass, and failures are not memoized. No native scratch or cache is shared
across nodes or solvers. Outside a solver, bind the same context explicitly with
`setEvaluationContext` and call `invalidate()` at evaluation-pass boundaries.
Cloning a graph creates independent numerical workspaces again.

The Spot dynamics graph reuses its native robot stage and passes the recovered
acceleration to a native integration/projection call. Native buffers and known
identity/zero blocks are reused, without changing the objective or dynamics.
`ilqr.n_threads` enables independent node evaluation;
worker exceptions reach the caller and workers are joined at destruction.
Pure CasADi models retain the existing `ilqr.codegen_enabled` option. Pure
symbolic managed stages use perturbation-form derivatives for supported state
spaces and honor this option; mixed graphs evaluate their symbolic segments
through CasADi's numerical evaluator. Small supported symbolic stages use a
bounded scalar expansion; larger stages and stages with embedded calls retain
the original graph representation.

Intermediate costs are averaged over N and the final cost is added once.
Gradients and Hessians now use this same normalization. This corrects the old
mismatch between cost evaluation and quadratization and can change trajectories
that previously relied on that mismatch. Merit weights also account for
normalized feasibility measures and fixed-state multipliers.

The interface is independent of a robotics library or device backend. The
optional Kokkos collision backend prepares horizon-wide batches before node
evaluation; symbolic graphs, Pinocchio kinematics and backward recursion remain
on the CPU. See the Kokkos section below for supported fields and validation.

Build and test against your installed CasADi prefix:

```bash
cmake -S horizon/cpp -B /tmp/horizon-ilqr-build \
  -DCMAKE_PREFIX_PATH=/path/to/dependencies \
  -DHORIZON_BUILD_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/horizon-ilqr-build -j2
ctest --test-dir /tmp/horizon-ilqr-build --output-on-failure
/tmp/horizon-ilqr-build/ilqr_mixed_example
```

The regression suites check chain-rule derivatives against finite differences,
symbolic graphs with shared/nested calls, tangent-native derivatives, arbitrary
custom sum/difference, SO(3) derivatives and shooting, parameters and node targets,
serial/parallel agreement, both solver paths, and worker failure propagation.

`TangentDynamics` exposes a prediction in storage coordinates and a Jacobian
in the prediction's tangent coordinates. The direct solver path requires the
same state-space instance; other functions retain the ambient compatibility
path. With Pinocchio enabled, a small nonzero-defect direct/fallback comparison
can be run with:

```bash
build/cuda/tests/testFrameKinematics --gtest_also_run_disabled_tests \
  --gtest_filter=FrameKinematics.DISABLED_ProfileTangentOutputDefectLinearization
```

Use a Release build with `HORIZON_PROFILING=OFF` for wall-time measurements.
This microbenchmark measures one linearization, not complete solver sweeps.
For software instruction counts, set `HORIZON_BENCH_DYNAMICS_PATH` to `direct`
or `fallback` and run each mode under Callgrind. With profiling enabled on the
same Release build, Callgrind counted 364 million versus 987 million
instructions for the complete process, including startup and 10,000 calls.
These counts are not hardware counters or end-to-end solver measurements.
The disabled `IlqrFunctions.DISABLED_ProfileGraphTangentPropagation` and
`IlqrFunctions.DISABLED_ProfileCasadiLocalDerivatives` tests compare matching
direct and compatibility derivatives on small identical inputs. In the same
profiling-enabled Release build, the graph test took 1.12–1.15 versus
1.31–1.33 microseconds per call, and the expanded CasADi test took 0.51–0.52
versus 1.76–2.25 microseconds. Callgrind process totals for 10,000 calls were
188 versus 217 million instructions for the graph and 91 versus 313 million
for CasADi. These are isolated microbenchmarks, not solver-level speedups.
`ManagedIlqr.DISABLED_ProfileLocalDerivativeSweeps` compares complete dynamics,
cost and constraint derivative sweeps for the same 16-node quaternion problem,
including nonzero defects. In the profiling-enabled build, a sweep took 97
versus 242 microseconds in serial and 88 versus 161 microseconds with two
threads. An earlier Callgrind run counted 1.72 versus 4.12 billion instructions
for the complete serial benchmark process, including setup and 1,000 sweeps,
before the compatibility adapter was changed to explicitly disable tangent
interfaces.
These measurements do not include a complete solve.
With profiling disabled, three repeats of that sweep measured 77.2–78.9
versus 220.4–224.8 microseconds in serial and 48.4–56.2 versus 135.4–148.8
microseconds with two threads. The profiling-disabled direct/fallback
microbenchmarks measured 1.99–2.04/5.71–5.74 microseconds for dynamics,
1.06–1.07/1.26–1.27 for graph propagation, and 0.38–0.39/1.80–1.82 for
CasADi. Structured/dense defect transport measured 1.16–1.19/1.14–1.15,
4.64–4.65/5.14–5.18, and 9.55–9.77/27.18–27.38 microseconds at tangent
sizes 7, 18, and 36 respectively; the size-7 structured API uses the dense
solve. The disabled `ManagedIlqr.DISABLED_ProfileSymbolicShooting` measures
complete solves of a separate five-node quaternion shooting problem, with
identical initial guesses and solver settings. Across nine profiling-disabled
samples, serial direct/fallback solves took 230–287/325–352 microseconds;
two-thread timings varied substantially (298–587/348–550 microseconds).
Solver construction is excluded, and each sample checks solve and trajectory
parity. These small-problem timings depend on the machine and do not establish
a general end-to-end speedup.

For a real Pinocchio model, enable `-DHORIZON_TEST_PINOCCHIO=ON` and include its
installation prefix in `CMAKE_PREFIX_PATH`. The optional
[testPinocchioIlqr.cpp](tests/testPinocchioIlqr.cpp) provides complete adapters
for robot configuration operations, frame-position FK with an analytical
nv-column Jacobian, and native integrated dynamics using `dIntegrate`. It checks
the mixed symbolic residual derivatives and solves a free-flyer robot shooting
problem with nq different from nv. Pinocchio is an optional test dependency;
the core library does not depend on it.

## Sphere collision residuals

`horizon/collision.h` contains link-local sphere models, dense SDF grid queries,
analytic sphere/plane fields, and `SphereDistanceFunction`. The distance
function accepts SoA world-space centers `[x0,x1,...,y0,y1,...,z0,z1,...]`, subtracts
robot radii, and returns a smooth softplus residual below the requested
clearance. It supplies analytical Jacobians, so it can be registered with
`FunctionManager` and used with `IterativeLQR::setResidual`. Self-collision
pairs are explicit; adjacent or otherwise exempt pairs should be omitted by
the caller. Grid values are x-fastest signed distances in world units, and
queries outside a finite dense grid throw `EvaluationError`. OpenVDB narrow-band
grids use their positive background distance for samples outside the active
band.

When Pinocchio support is enabled with `-DHORIZON_WITH_PINOCCHIO=ON`,
`horizon/pinocchio_collision.h` provides `PinocchioSphereCenters`, which
computes all Kyon sphere world centers and tangent Jacobians using one
joint-kinematics/Jacobian pass per evaluation. `PinocchioModelWorkspace`
provides node-local `Data` shared through the solver's `EvaluationContext`;
functions given the same workspace prototype can reuse cached q/v/a forward
kinematics, frame placements, and joint Jacobians. Each shooting node and
solver still receives independent mutable `Data`. Custom native dynamics that
share the same prototype can bind their per-node workspace with
`prototype->forContext(context)` and call `ensureVelocity` or
`ensureAcceleration` before their Pinocchio algorithms. Register the aggregate center function through
`SymbolicFunctionManager::addNative` and feed its output to a
`SphereDistanceFunction` in the same expression graph; shared native results
are evaluated once per node and pass.

```cpp
auto model_spheres = std::make_shared<collision::SphereModel>(collision::loadSphereModelFromUrdf(sphere_urdf));
auto disabled = collision::loadDisabledCollisionPairs(collision_policy_json);
auto self_pairs = collision::makeSelfCollisionPairs(*model_spheres, disabled);
auto robot_workspace = std::make_shared<collision::PinocchioModelWorkspace>(pin_model);
auto centers = functions.addNative("kyon_centers",
    std::make_shared<collision::PinocchioSphereCenters>(pin_model, robot_space,
                                                        *model_spheres, robot_workspace));
std::vector<double> radii;
for(const auto& sphere : model_spheres->spheres()) radii.push_back(sphere.radius);
auto distances = functions.addNative("collision_distances",
    std::make_shared<collision::SphereDistanceFunction>(radii, environment_fields, self_pairs));
auto residual = distances(centers(q));
functions.addResidual("collision", {q, u}, residual);
solver.setResidual(all_nodes, functions, "collision");
```

The runtime loader reads every collision sphere directly from a URDF; a JSON
policy supplies disabled link pairs with MoveIt reasons. `makeSelfCollisionPairs`
enables all distinct-link sphere pairs except those disabled in that policy.
Same-link pairs are excluded. Unknown policy links and malformed input fail
explicitly. No robot-specific sphere data is compiled into Horizon.

The original vendored Kyon fixture is `tests/data/kyon_spheres.urdf`, with
`kyon_disabled_collisions.json`. It has no collision geometry for the arm
links. Examples and benchmarks instead use the xacro-derived
`kyon_full_spheres.urdf` and `kyon_full_disabled_collisions.json`: 103 spheres
on all 31 links with source collision geometry, with 3,799 enabled sphere pairs.
Both policies were generated with 75,000 trials. `Adjacent` and `Default` are
MoveIt classifications; `Never` is based on random samples and is not a proof
that a pair can never collide. Review and adjust the disabled-pair policy for
the intended operating range and sphere approximation.

Generate a new sphere URDF and JSON policy from a xacro command with FOAM and
the JSON-capable `moveit_compute_default_collisions` installed:

```bash
python3 tools/generate_collision_assets.py \
  --sphere-urdf /tmp/robot_spheres.urdf \
  --collision-json /tmp/robot_disabled_collisions.json \
  --trials 75000 -- xacro /path/to/robot.urdf.xacro
```

Source the robot's ROS environment first so xacro and `package://` mesh
resources resolve. The script preserves sphere primitives and applies mesh
scale and pose before FOAM fitting. FOAM fitting is an approximation.

OpenVDB support is optional (`-DHORIZON_WITH_OPENVDB=ON`). It enables
`OpenVdbDistanceField` for grids named `sdf` and the STL mesh converter:

```bash
horizon_mesh_to_sdf environment.stl environment.vdb 0.02 5
```

The converter accepts binary or ASCII STL, writes a signed level-set grid, and
takes voxel size in world units and the narrow-band width in voxels. Its mesh
input must describe a closed surface. Pinocchio collision kinematics are independently optional via
`-DHORIZON_WITH_PINOCCHIO=ON`.

The Kyon integration test reads the vendored fixture at runtime:

```bash
cmake -S horizon/cpp -B /tmp/horizon-kyon-build \
  -DHORIZON_BUILD_PYTHON=OFF -DHORIZON_WITH_PINOCCHIO=ON \
  -DHORIZON_TEST_KYON_COLLISION=ON
cmake --build /tmp/horizon-kyon-build -j2
ctest --test-dir /tmp/horizon-kyon-build -R testKyonCollision --output-on-failure
```

`profileKyonCollision` (built with the Kyon test option) accepts an iteration
count and runtime paths, e.g. `profileKyonCollision 1000 horizon/cpp/tests/data/kyon_full_spheres.urdf horizon/cpp/tests/data/kyon_full_disabled_collisions.json`. Use a Release build for timings.
Collision Jacobian products contract the three nonzero center derivatives per
sphere directly with directions. Sphere kinematics computes one frame Jacobian
per occupied link and applies each sphere offset without allocating a matrix
per sphere. The integration tests cover Kyon self-distance derivatives and
configuration descent, plus an iLQR solve separating overlapping spheres.

The imported sphere URDF is an approximation supplied by the Kyon package;
the converter does not fit spheres to meshes or certify whole-robot coverage.
Self-collision exclusions remain explicit. Exactly coincident sphere centers
have zero separation gradient and require a noncoincident initial guess.
OpenVDB level sets saturate outside their narrow band: choose a band wider than
the largest robot radius plus clearance margin, and initialize away from deep
penetrations with saturated gradients. OpenVDB support still needs a build and
integration run on a machine with that dependency installed.

## Complete device collision costs

`DeviceCollisionCost` in `horizon/device_collision.h` runs robot forward
kinematics, sphere placement, dense-SDF/self queries, tangent derivatives, and
Gauss–Newton accumulation in `Kokkos::DefaultExecutionSpace`. Enable
`HORIZON_WITH_KOKKOS`; the setup-only Pinocchio exporter additionally needs
`HORIZON_WITH_PINOCCHIO`. The same implementation runs with CPU or CUDA Kokkos
builds. Other Kokkos GPU backends have not been validated here.

```cpp
Kokkos::ScopeGuard runtime; // outlives every cost and solver
auto robot = collision::exportDeviceRobot(*pin_model, sphere_model);
collision::DeviceCollisionCost cost(std::move(robot), sdf_grid, self_pairs,
                                     10000.); // squared residual weight
solver.setBatchedCost(all_nodes, "collision", cost);
```

This replaces the collision residual registration; do not register both paths
for the same objective. The direct `ilqr::BatchedCost` interface is state-only,
with unscaled cost, gradient, and a row-wise packed upper triangle of the
Gauss–Newton Hessian. The solver applies its existing running `1/N` and terminal
normalization once and exposes the contribution through ordinary named cost
reporting. It clones the prototype into a solver-owned workspace. Cost objects
at different nodes read that workspace after preparation completes. Selected
nodes are gathered in registration order, so inactive nodes are not queried.

For a state containing additional components, pass its stored dimension and
configuration offset to the cost constructor, and the matching tangent block
offset to `setBatchedCost`. The caller must use a state space whose configuration
block has Pinocchio's tangent convention; matching dimensions alone cannot
establish that convention. Control and mixed derivatives are zero.

The model exporter supports scalar revolute/prismatic joints (including
arbitrary axes), continuous revolute joints, and quaternion free-flyers. Fixed
frame placements are folded into link-local sphere coordinates. Unsupported
joints, including mimic joints, are rejected. Floating-base configurations use
`xyz, qx, qy, qz, qw` with body-local linear/angular tangent velocities; unit
quaternions and continuous-joint coordinates must be normalized.

Geometry, grids, masks, and working buffers remain resident. A derivative pass
uploads configurations and returns only scalar cost, tangent gradient, and
packed symmetric Hessian plus status. A value-only pass returns costs and
status, reusing existing allocations without downloading derivative buffers.
The residual/Jacobian workspace is tiled at 64 MiB; this limit excludes resident
geometry, transforms, configurations, and compact outputs. Exactly coincident
spheres retain the scalar backend's zero separation gradient. Failed prepares
invalidate results, and out-of-grid trial costs use normal line-search retries.
`DeviceCollisionStatistics` reports successful batch counts, logical transfer
bytes (excluding setup geometry), peak live device-buffer bytes, custom kernel
launches, total host preparation time, and per-stage timings. CUDA stage timings
use CUDA events; CPU stage timings use fenced wall time. Kokkos copy/fill
implementation kernels are not included in the custom launch count.
Statistics aggregate a prototype and its solver clones; reset them only while
those solvers are idle. Mutable evaluation buffers remain private to each clone.

Run the complete Kyon benchmark with fixed feet, the `dagana_1_base` target,
environment SDF, and every enabled self-collision pair:

```bash
benchmarkKyonTrajectory --urdf horizon/cpp/tests/data/kyon_full_spheres.urdf \
  --collision-policy horizon/cpp/tests/data/kyon_full_disabled_collisions.json \
  --backend device --nodes 256 --samples 20 --threads 8 \
  --self-collision 1 --output /tmp/kyon_device
python3 tools/benchmark_device_collision.py \
  --binary /path/to/cuda-build/tests/benchmarkKyonTrajectory \
  --cpu-binary /path/to/cpu-build/tests/benchmarkKyonTrajectory \
  --samples 20 --output /tmp/kyon_device_comparison
```

The comparison script runs backends sequentially at N=30, 256, and 1024. It
preserves raw timings, iteration traces, trajectories, median/p95 summaries,
and checks convergence, objective, and trajectory parity. The default runtime
model has 103 spheres and 3,799 enabled self-collision pairs.
The device benchmark's CPU feet/target functions compute only five frame
origins; collision sphere kinematics is entirely on the device. Post-solve CPU
validation is outside the measured solve.

An isolated Kokkos Kernels comparison is available through
`HORIZON_BENCH_KOKKOS_KERNELS=ON` and `KokkosKernels_DIR`. The optional
`benchmarkCollisionAccumulation` target compares the packed reduction with
TeamGemm plus packing. Production code does not depend on Kokkos Kernels.
Measured results and the cuRoboV2 feasibility assessment are recorded in
[`docs/device_collision_performance.md`](docs/device_collision_performance.md).
The follow-up [`CUDA 13 library integration report`](docs/cuda13_library_integrations.md)
contains native cuRoboV2/free-base adapter tests, nvblox mapping and direct GPU
scene exchange, and an identical-input FP64 reference comparison. The recent
toolkit is isolated under the user-local prefix; `source tools/cuda13_env.sh`
selects it for the current shell. These standalone probes add no production
dependency on either library.

To inspect copy/allocation events separately from timing runs, compile
`tools/kokkos_copy_trace.cpp` as a shared library against the same Kokkos
installation and set `KOKKOS_TOOLS_LIBS` to that library. This diagnostic plugin
does not participate in the production build or timed comparisons.

## Kokkos residual batches across horizon nodes

Configure `HORIZON_WITH_KOKKOS=ON` and provide an installed Kokkos 4+ prefix
through `CMAKE_PREFIX_PATH` (or `Kokkos_DIR`). This dependency is optional.
`KokkosCollisionBatch` in `horizon/kokkos_collision.h` supports an immutable
dense SDF and explicit self-collision pairs, using `Kokkos::DefaultExecutionSpace`.
Kokkos's build selects the execution backend; changing solver node-thread count
does not change that execution space.

```cpp
Kokkos::ScopeGuard runtime; // before constructing batches/solvers
// centers maps a complete stored state to SoA sphere centers. Extra task/foot
// points may follow the collision spheres in each coordinate block.
collision::KokkosCollisionBatch batch(centers, radii, sdf_grid, self_pairs);
auto native = functions.addNative("collision_batch", batch.function());
functions.addResidual("collision", {x, u}, 100 * native(x));
// Construct solver and install the residual as usual, then register preparation.
solver.setResidual(all_nodes, functions, "collision");
solver.addEvaluationBatch(batch);
```

The application owns Kokkos initialization/finalization, and all solver/batch
objects must be destroyed before the runtime guard. The solver clones each
registered batch, including its mutable buffers. Grid, radii, and pair data are
uploaded once per clone; buffers are reused for unchanged horizon dimensions.
Node results live in independent solver evaluation contexts. The function is
intentionally unavailable without preparation: standalone users must explicitly
call `prepare(states, controls, contexts, derivatives)` and bind those contexts.

Before each linearization and every trial cost pass, the solver invalidates
native caches, prepares the entire N+1-node trajectory, then evaluates nodes.
CPU kinematics gathers world centers and, when requested, center Jacobians.
One kernel covers all node/sphere and node/self-pair distances; a second kernel
contracts spatial derivatives with center Jacobians. Values and contracted
Jacobians return to the CPU for existing graph propagation and Gauss–Newton
accumulation. Prepared kinematics can also be reused by feet and target graphs.
This batches collision work across nodes, not Pinocchio or the backward pass.

Out-of-grid queries raise `EvaluationError` after synchronization, allowing
line-search retries. Failed batches publish no node results. Bitwise input
checks and pass flags reject stale or unavailable derivatives. The first
implementation uses one dense SDF; analytic primitive and OpenVDB field objects
cannot be passed directly. A null grid enables self-collision-only batches.

For the Kyon benchmark, build and run:

```bash
cmake -S horizon/cpp -B /tmp/horizon-kokkos-build \
  -DCMAKE_BUILD_TYPE=Release -DHORIZON_BUILD_PYTHON=OFF \
  -DHORIZON_WITH_PINOCCHIO=ON -DHORIZON_TEST_KYON_COLLISION=ON \
  -DHORIZON_WITH_KOKKOS=ON -DHORIZON_PROFILING=ON \
  -DCMAKE_PREFIX_PATH="/path/to/kokkos;/path/to/pinocchio;/path/to/casadi"
cmake --build /tmp/horizon-kokkos-build -j2
ctest --test-dir /tmp/horizon-kokkos-build --output-on-failure
/tmp/horizon-kokkos-build/tests/benchmarkKyonTrajectory \
  --urdf horizon/cpp/tests/data/kyon_full_spheres.urdf \
  --collision-policy horizon/cpp/tests/data/kyon_full_disabled_collisions.json \
  --nodes 256 --samples 3 --iterations 40 --backend kokkos \
  --kokkos-threads 2 --output /tmp/kyon_kokkos
```

`--backend scalar` retains the reference implementation. `--kokkos-threads`
configures Kokkos CPU threads independently of `--threads` for solver node
evaluation. JSON includes execution space, successful batch/pass counts,
kernel launches, node evaluations, and preparation/kinematics/backend times.
Backend time includes host/device transfers, kernels and synchronization, but not
host kinematics. Preparation time additionally includes host kinematics and
publishing node results; these nested timings must not be added together.

Historical partial-model measurements below used Kokkos 4.7.1 OpenMP/Serial on CPU and CUDA 12.0 on an RTX 3060
(Ampere 8.6). At N=256, 20 iterations produced
38 value batches and 20 derivative batches, each over 257 nodes: 78 kernel
launches for 1,401,164 SDF queries. With serial solver evaluation, scalar solves
took 2.72 s and Kokkos with two CPU threads took 2.63–2.64 s on the development
host. Both retained the same target, contacts, clearance and convergence.
The modest end-to-end improvement reflects host kinematics, transfer/staging
and solver costs.

For CUDA, build Kokkos with `Kokkos_ENABLE_CUDA=ON` and the correct architecture
(`Kokkos_ARCH_AMPERE86=ON` for the RTX 3060), then configure Horizon in a separate
build directory with `CMAKE_CXX_COMPILER=/path/to/kokkos/bin/nvcc_wrapper` and
`Kokkos_DIR` pointing to that CUDA-enabled installation. CPU solver translation
units use the wrapper's `--host-only` mode to avoid expensive device analysis of
Eigen/Pinocchio/CasADi templates. Build `testKokkosCollision` and
`benchmarkKyonTrajectory`, and run the same `--backend kokkos` command; the JSON
execution space must read `Cuda`.

Three N=256 samples from the same CUDA-enabled build averaged 2,764 ms for the
scalar path and 2,667 ms for CUDA. Transfer-inclusive backend time averaged
71 ms versus 152 ms for scalar collision evaluation. Both took 20 iterations
and preserved the same geometry. This is about a 3.5% end-to-end improvement,
not a large GPU speedup. The five CUDA parity/lifecycle tests passed, including
self collisions, node workers and out-of-grid retries. Compute Sanitizer 12.0
could not instrument the application on this host (terminated before its first
instrumented API call), so no GPU memory-sanitizer pass is claimed.
Tests cover scalar parity, self-collision finite differences, terminal nodes,
node workers, clone isolation, horizon resizing, stale-result rejection and
out-of-grid line-search retries. The core also builds with Kokkos disabled.
Memcheck found no invalid accesses or definitely lost allocations; full leak
checking reported 352 bytes as possibly lost in libgomp/pthread TLS allocation
during OpenMP initialization.

## Kyon fixed-foot trajectory benchmark

`benchmarkKyonTrajectory` is built with `HORIZON_TEST_KYON_COLLISION=ON`.
Enable `HORIZON_PROFILING=ON` and use a Release build to collect solver phase
timings. It solves a two-second configuration trajectory using generalized
velocity controls and Pinocchio integration. Four `contact_*` frame positions
are equality constraints at every shooting node; `dagana_1_base` has a terminal
position target offset by (0.06, 0.04, 0.02) metres from its initial position.
Joint position and velocity bounds are included. This is a kinematic benchmark:
it does not enforce force balance, friction, actuator torques, or contact
constraints between shooting nodes.

A sphere obstacle is sampled into a 151-cubed dense SDF with 4 cm voxels.
All 103 imported robot spheres and all 3,799 enabled self pairs are queried at every node; the scene is placed
deterministically clear of the initial configuration. Collision residuals use
2 cm clearance and soft penalties. Scene parameters are included in the JSON
report so the grid can be reproduced. This benchmark measures environment and complete enabled self-collision costs.

```bash
/tmp/horizon-kyon-build/tests/benchmarkKyonTrajectory \
  --urdf horizon/cpp/tests/data/kyon_full_spheres.urdf \
  --collision-policy horizon/cpp/tests/data/kyon_full_disabled_collisions.json \
  --nodes 30 --samples 3 --iterations 40 --threads 0 \
  --output /tmp/kyon_trajectory
# Collision-disabled convergence/control comparison:
/tmp/horizon-kyon-build/tests/benchmarkKyonTrajectory \
  --urdf horizon/cpp/tests/data/kyon_full_spheres.urdf \
  --collision-policy horizon/cpp/tests/data/kyon_full_disabled_collisions.json \
  --nodes 30 --samples 3 --iterations 40 --collision 0 \
  --output /tmp/kyon_trajectory_baseline
```

One warmup solve is excluded from the report. Each measured solve resets the
state and control guesses. JSON records solve time, iteration count, collision
call time and query counts, nested solver phase times in microseconds, target
error, maximum foot drift, shooting defects and minimum sampled SDF clearance.
For multiple threads, collision time is summed across calls, not elapsed time;
nested phase times must not be added together. CSV stores the final trajectory
with configuration and control columns. A nonzero exit code indicates failed
geometric acceptance checks or a setup/evaluation error. The CTest entry
`kyonTrajectorySmoke` checks a 12-interval solve.

Historical Release measurements below used an earlier environment-only model; they do not describe the full self-collision model. See [current measurements](docs/device_collision_performance.md).

| Intervals | Threads | Iterations | Solve time | Collision call time | SDF queries |
|---|---:|---:|---:|---:|---:|
| 30 | 0 | 14 | 306–316 ms | 15.9–16.2 ms | 93,248 |
| 256 | 0 | 31 | 5,637 ms | 289 ms | 1,570,270 |
| 256 | 2 | 31 | 3,425 ms | 304 ms summed | 1,570,270 |

At 256 intervals, target error was 0.26 micrometres, maximum foot drift 15.5 nm,
and minimum SDF clearance 40.7 mm. The collision-disabled 30-interval solve took
about 36 ms / 3 iterations and penetrated the obstacle by 7.9 mm. Its timing
difference includes different convergence and graph work, not just query cost.

Subsequent profiling found that the shared native cache rebuilt a dense
collision Jacobian, bypassing its direct directional kernel. Functions can now
opt in with `hasJacobianProduct()` to preserve that kernel through the cache.
Such directional calls are recomputed for their supplied directions; ordinary
native functions retain value/Jacobian memoization. Graph lowering also removes
proven pure selections and concatenations and omits unused symbolic inputs.
These transformations happen at setup and preserve exact Hessian composition.

The trajectory benchmark now defaults to control damping `0.001 / N`, which
stabilizes the late contact-residual oscillation without changing the objective
or stopping tolerances. Set `--damping 0` to reproduce the undamped formulation;
`--decomposition qr` compares a QR KKT solve against the default LU. Excessive
damping (`0.01 / N`) slowed convergence in this problem. Accepted/rejected steps
and regularization are exported to `<output>_iterations_<sample>.csv`.

Final serial N=256 measurements over three samples were 2.75–2.77 s and 20
iterations, versus 5.55 s / 31 iterations before these changes. Collision-cost
quadratization dropped from 2.35 s to 0.79 s, and total linearization and
quadratization from 4.51 s to 2.06 s. The two-thread solve took 1.77 s. Target
error remained below 0.44 micrometres, maximum foot drift below 8 nm, and SDF
clearance remained 40.7 mm. LU and QR produced the same convergence and geometry;
no square-root backward pass was added for this benchmark.

Regularization escalation now starts at `1e-9` when the current value is zero,
rather than jumping directly to `1000` with the default growth factor. It fails
explicitly above a finite limit instead of retrying indefinitely. This fix is
covered by a regression test; it did not account for the benchmark speedup,
since the recorded solves did not require escalation.

### Historical CPU node parallelism versus CUDA batching

For the earlier incomplete collision model, with the same CUDA-enabled Release build, N=256, damping 0.001 and two measured
samples after warmup, the following mean solve times were recorded on the
Core i7-12700 / RTX 3060 development host:

| CPU node workers (`--threads`) | Scalar collision | CUDA collision batch |
|---:|---:|---:|
| 0 | 2.770 s | 2.719 s |
| 2 | 1.777 s | 1.921 s |
| 4 | 1.292 s | 1.550 s |
| 8 | 1.066 s | 1.409 s |

All used 20 iterations and reached the same objective and trajectory. The
worker count denotes persistent node workers; the caller handles the terminal
node and any remainder. GPU gathering/preparation is currently serial on the
CPU, which limits scaling when combined with node workers. At eight workers,
the scalar solve spent about 372 ms in linearization/quadratization, 498 ms in
the backward pass and 191 ms in line search. CPU node parallelism is the
preferred configuration for this workload:

```bash
benchmarkKyonTrajectory --urdf horizon/cpp/tests/data/kyon_full_spheres.urdf \
  --collision-policy horizon/cpp/tests/data/kyon_full_disabled_collisions.json \
  --nodes 256 --samples 3 --iterations 40 \
  --backend scalar --threads 8 --output /tmp/kyon_cpu_parallel
```

A small damping sweep was not adopted as a new default: `--damping 0.00025`
converged in 13 iterations at N=256 but reached a higher-cost local solution;
`0.0005` failed to converge within 60 iterations at both tested horizons.
JSON now includes the final objective, and measured samples that do not report
solver convergence force `valid=false` and a nonzero benchmark exit status even
when their geometric errors fall below the looser acceptance checks. The
objective and stopping tolerances were not changed for the worker comparison.

## Spot squat and formulation profiling

Enable `HORIZON_TEST_SPOT=ON` to build `testSpotIlqr` and `profileSpotIlqr`.
They load the bundled `horizon/examples/urdf/spot.urdf` without meshes or ROS:
19 configuration coordinates (free-flyer plus twelve joints), 18 velocities,
and four point contacts at `lf_foot`, `rf_foot`, `lh_foot`, and `rh_foot`.
The state is `(q,v)`: 37 stored coordinates and 36 tangent coordinates.

The two formulations use the same two-second, 6 cm down-and-up squat reference,
with small nonzero x/y and roll/pitch/yaw references to exercise all six base
pose components. They have the same physical running/terminal objective,
initial standing guess, contact anchors, and rest conditions. Controls differ:

| Formulation | Controls | Acceleration, force, and torque relationship |
|---|---|---|
| `acceleration_force` | `(vdot, f)`, 30 variables | The first six rows of `rnea(q,v,vdot) - J.transpose()*f` are zero Newton–Euler constraints. Joint torques are the remaining twelve rows. Contact acceleration is an equality constraint. |
| `torque` | Actuated joint torques, 12 variables | A native contact KKT solve recovers `vdot` and `f` from `[M, -J.transpose(); J, 0]`, with the six base generalized forces fixed to zero. |

**Robot computations stay in native Pinocchio.** RNEA, CRBA, frame positions,
velocities, classical accelerations, Jacobians and analytical derivative
algorithms run directly in C++. The torque-control Jacobian uses implicit
differentiation of the contact equations. The derivative of `J.transpose()*f`
is contracted through Pinocchio's external-wrench RNEA derivatives in one sweep,
with the correction for forces fixed in world coordinates. This replaces one
Jacobian-time-variation sweep per configuration tangent coordinate. CasADi only combines native
outputs into weighted residuals and constraint expressions. No symbolic robot
model or CasADiKinDyn binary is used.

Both use the same discrete group integration and stabilized contact condition
`J*vdot + Jdot*v + 25*(p-p_anchor) + 10*J*v = 0`. After the group update, native
IK fixes the twelve joint positions while preserving the base pose, and
`J_joint*v_joint = -J_base*v_base` fixes compatible joint velocities. The
projection Jacobian comes from these implicit equations and Pinocchio's native
frame/velocity derivatives; it uses no finite differences or symbolic robot
model. Robot state retractions and differences remain Pinocchio's operations.
This is a contact-consistent discrete dynamics model, rather than relying on
an acceleration-level constraint to preserve positions at a finite time step.
The foot anchoring residual remains in the objective with the same weight.

The terminal rest equality uses six independent base velocity coordinates.
For the nonsingular leg Jacobians on this motion, exact `J*v=0` then implies
all eighteen generalized velocities are zero. The tests independently check
the full velocity vector, foot positions, and foot velocities. Imposing all
eighteen terminal equalities with the earlier unprojected update introduced
weak constraint directions and amplified discretization drift; at larger grids
it caused long solves, nonphysical contact forces, and numerical failures.

Running costs are averaged over `N` intervals. The benchmark's `huu_reg` and
`hxx_reg_base` are therefore `1e-8/N`, so a mesh refinement does not strengthen
the regularization relative to the physical running objective. Using a fixed
`1e-8` ridge made iteration counts increase even after contact drift was fixed.
Costs retain Gauss–Newton Hessians; these corrections require only native
values and first derivatives.

The tests require less than 1 mm drift, 3 mm/s foot speed, 3 mm base position
error, and 6 mrad base rotation error. They also check Newton–Euler residuals,
dynamics gaps, quaternion norms, terminal velocity, joint limits, positive
normal force, and a friction margin with coefficient 0.7. Friction and actuator
limits are feasibility checks on the result, not extra optimization inequalities.
The URDF's generous 1000 Nm effort limit is the actuator check used here.

The regression compares both formulations and serial/two-worker execution,
including recovered forces, accelerations and joint torques. Native analytical
and implicit Jacobians are checked against tangent-space finite differences at
a perturbed robot state before relying on the motion result. Both formulations
use the same SVD constraint decomposition and convergence tolerances, with a
300-iteration cap; these settings are recorded in the profiling report.
`MeshIndependentConvergenceThrough256` additionally solves both formulations
with serial/two-worker execution at `N=16,32,64,128,256`, requiring at most eight
iterations and checking full terminal rest, contact feasibility, and agreement
of the physical objective. Wall-time scaling is measured by the profiler,
without machine-dependent time assertions in CTest.

```bash
cmake -S horizon/cpp -B /tmp/horizon-spot-build \
  -DCMAKE_PREFIX_PATH="/path/to/casadi;/path/to/pinocchio" \
  -DCMAKE_BUILD_TYPE=Release -DHORIZON_BUILD_PYTHON=OFF \
  -DHORIZON_TEST_SPOT=ON -DHORIZON_PROFILING=ON
cmake --build /tmp/horizon-spot-build --target testSpotIlqr profileSpotIlqr -j1
ctest --test-dir /tmp/horizon-spot-build -L spot --output-on-failure
/tmp/horizon-spot-build/tests/profileSpotIlqr \
  --nodes 16,32 --threads 0,2 --warmup 1 --samples 5 \
  --function-samples 200 --output /tmp/spot_profile.json
```

The JSON report records setup time separately, raw solve samples, min/median/
p95/max/mean solve and linearization timings, iteration counts, feasibility for
every measured run, and differences between formulations. Profiling-enabled
builds also report inclusive solver-phase timing statistics; worker timings
overlap and must not be summed to estimate wall time. Each timed solve resets
the trajectories to the same standing guess. Function timings include ten
warmup calls and measure joint value/Jacobian evaluation, including the native
contact solve for torque control.
Each case also writes a CSV trajectory beside the JSON report, containing
configuration, velocity, recovered acceleration, contact force and joint torque.
The terminal row leaves controls empty because there is no terminal control.
The profiler gates convergence and physical feasibility for every run, and
records inter-formulation trace differences without requiring equality at every
horizon. Different nonlinear parameterizations can yield different converged
traces. Exact agreement is tested separately at the fixed 16-interval regression
case; inspect the reported differences when comparing other discretizations.

There is no universal wall-time threshold in CTest. For regression checks on the
same machine/build, pass `--baseline previous.json --max-regression 1.25` to fail
if a configuration's median solve time increases by more than 25%. The reader
checks CPU model, compiler, build type, profiling mode, and requested configurations.
Use a quiet machine with matching compiler flags and CPU power settings when
comparing baselines. Performance reports are measurements, not real-time guarantees.

For horizon scaling, keep the two-second duration and costs fixed and run
`--nodes 16,32,64,128,256 --threads 0,2`. `--formulation torque` or
`--formulation acceleration_force` isolates one formulation; the default is
`all`. Each case records `status: completed` or `status: error`. Numerical
exceptions are recorded with their phase, message, elapsed attempt time and
any completed measured samples, and the profiler continues to the next case.
The overall exit status is nonzero when convergence/physical validation fails
or a case raises an exception. Failed-attempt times are not solve-time medians.

Defaults are `--integration projected --regularization normalized`. For
controlled ablations, `--integration unprojected` restores the earlier group
update and full eighteen-component terminal velocity equality, and
`--regularization fixed` restores the mesh-independent ridge value. Baseline
comparisons require matching integration and regularization modes. Old reports
without these fields are interpreted as unprojected/fixed.
`diagnoseSpotIlqr N torque|acceleration_force solve full|base|none rank_threshold
lu|qr|ldlt project_contacts scale_regularization` exposes endpoint/rank ablations;
the final two arguments are `0` or `1`. Replacing `solve` with an iteration
count assembles an independent dense square-root QP reference for that iterate;
this diagnostic is intended for small grids.

For before/after implementation benchmarks, preserve the original binary and
run both versions with identical `--nodes`, `--threads`, `--samples`, `--warmup`
and build settings. Then compare phases and exported trajectories:

```bash
python3 horizon/cpp/tests/compareSpotProfiles.py \
  --before before.json --after after.json --output comparison.json
```

The comparison checks matching protocols/configurations, physical feasibility
of every measured solve, objective agreement, and all stored state/control
samples. It writes solve and linearization speedups to JSON and CSV. Use a
profiling-enabled build for phase attribution and repeat end-to-end latency
measurements without profiling when evaluating deployment performance.
Check status before reading timing fields. This matters at finer grids: the
contact model is bilateral, so solver convergence alone does not establish
positive normal force or friction feasibility.
