# C++ iLQR functions and state spaces

The solver accepts pure CasADi models, native C++ models with supplied
derivatives, and expressions mixing both. Existing `IterativeLQR(casadi::Function,
N, options)` and CasADi cost/constraint setters remain available. The new C++
interfaces live in `horizon::ilqr`; Python bindings retain their existing API.

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
graph buffers are reused. Residual graphs avoid allocating the quadratic tape
needed for exact Hessians. The joint `valueAndJacobian` path runs a residual
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
Pure CasADi models retain the existing `ilqr.codegen_enabled` option. Managed
graphs currently run their symbolic segments through CasADi's numerical evaluator.

Intermediate costs are averaged over N and the final cost is added once.
Gradients and Hessians now use this same normalization. This corrects the old
mismatch between cost evaluation and quadratization and can change trajectories
that previously relied on that mismatch. Merit weights also account for
normalized feasibility measures and fixed-state multipliers.

The interface is independent of a robotics library or device backend. GPU
batch evaluation across nodes remains future work, as in PLAN.md; this change
does not implement a GPU scheduler or a collision-distance backend.

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

For a real Pinocchio model, enable `-DHORIZON_TEST_PINOCCHIO=ON` and include its
installation prefix in `CMAKE_PREFIX_PATH`. The optional
[testPinocchioIlqr.cpp](tests/testPinocchioIlqr.cpp) provides complete adapters
for robot configuration operations, frame-position FK with an analytical
nv-column Jacobian, and native integrated dynamics using `dIntegrate`. It checks
the mixed symbolic residual derivatives and solves a free-flyer robot shooting
problem with nq different from nv. Pinocchio is an optional test dependency;
the core library does not depend on it.

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
