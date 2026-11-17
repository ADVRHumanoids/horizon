# Native Pinocchio from Horizon Python

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
`KYON_URDF`, `HORIZON_KYON_URDF`, or a sibling `iit-kyon-ros-pkg` checkout;
otherwise that portion is skipped. Python Pinocchio is used for setup and an
independent reference check, and is required by the Kyon example/test.

Without `HORIZON_WITH_PINOCCHIO`, the managers, CasADi-native adapters, and
legacy Python API remain available; `PinocchioModel` is omitted.

## Build a native graph

```python
import casadi as cs
from horizon.solvers import pyilqr
from horizon.solvers.function_manager import SymbolicFunctionManager

robot = pyilqr.PinocchioModel('/path/to/kyon.urdf')
functions = SymbolicFunctionManager()
q = cs.MX.sym('q', robot.nq)
v = cs.MX.sym('v', robot.nv)
target = cs.MX.sym('target', 3)

fk = functions.addNative('tool_fk', robot.framePositions(['dagana_1_base']))
step = functions.addNative('integrate', robot.step(0.05))
functions.addExpression('dynamics', [q, v], step(cs.vertcat(q, v)),
                        exact_hessian=False)
functions.addResidual('target', [q, v, target], 100*(fk(q) - target))

solver = pyilqr.IterativeLQR(functions, 'dynamics', 30,
                           {'ilqr.n_threads': 2}, robot.configuration_space)
solver.setResidual([30], functions, 'target')
# Set parameter values, bounds, and initial guesses, then call solve().
```

`PinocchioModel` loads the URDF into a C++ model. It respects a floating joint
already declared in the URDF and does not insert another one. Kyon has nq=31
stored configuration coordinates and nv=30 tangent coordinates.
`configuration_space` supplies Pinocchio integration/difference operations;
`step(dt)` maps `[q; generalized_velocity]` to the next configuration, with
analytical integration derivatives. `framePositions` returns frame origins in
SoA order `[x0, x1, ..., y0, y1, ..., z0, z1, ...]`.
For arbitrary link-local points, `sphereCenters` accepts `pyilqr.LinkSphere`
objects. Functions from the same robot share node-local Pinocchio workspaces;
different nodes/solvers own separate mutable workspaces.

Use the **same manager** to finalize expressions containing its native calls.
The returned MX objects are placeholders until `addExpression`, `addResidual`,
or `addCost` lowers them. Do not put an unfinished placeholder into an ordinary
CasADi Function or Horizon residual. The Python/C++ serialization bridge runs
only during setup and preserves shared-symbol identity, nested native calls,
and symbolic transforms. Evaluators own their compiled graphs independently of
the Python manager's lifetime.

`addResidual` defines a half squared norm with Gauss–Newton curvature. Pinocchio
FK and integration expose analytical first derivatives; use residuals and
first-derivative dynamics/constraints for these adapters. `addCost` requests
exact second derivatives and rejects native calls without weighted Hessians.
`FunctionManager.addCasadi` also registers ordinary SX/MX CasADi functions.
No Python-callback trampoline is exposed for native node functions.

## Use Horizon's Problem interface

The [complete Kyon example](ilqr_native_pinocchio.py) combines ordinary Horizon
variables, bounds, effort costs, parameter updates, and solution dictionaries
with native dynamics, fixed foot equalities, and a tool target:

```bash
python -m horizon.examples.ilqr_native_pinocchio \
  --urdf /path/to/kyon.urdf --N 30 --threads 2
```

It uses the optional factory arguments:

```python
solver = Solver.make_solver(
    'ilqr', problem, opts,
    function_manager=functions, dynamics='dynamics',
    state_space=robot.configuration_space,
)
solver.ilqr.setConstraint(all_nodes, functions, 'feet')
solver.ilqr.setResidual([N], functions, 'target')
solver.solve()
q_solution = solver.getSolutionDict()['q']
```

In this mode the native manager supplies dynamics, so `Problem.setDynamics`
is not required. Ordinary Horizon costs and constraints still use the legacy
CasADi path. Register native terms through `solver.ilqr`; their active nodes
can be updated with `setIndices`/`updateIndices`. Horizon parameters whose names
match manager inputs are refreshed on every `solver.solve()`. If the native
dynamics uses `Problem.setDt`, declare a scalar `dt` input as the example does.
Fixed integration steps can instead be used with the low-level solver.

The example controls generalized velocity, rather than physical joint torque.
It checks convergence, target error, and foot drift. Native graphs cannot be
exported as a CasADi-only saved problem; `solver.save()` raises an explicit
`NotImplementedError` in manager mode. Legacy problem saving remains supported.

The regression covers SX/MX legacy graphs against an independent dense QP,
GN/exact costs, serial/two-worker execution, changing parameters and active
nodes, warm starts, callbacks, and exports. Kyon checks native FK values and
Jacobian directions, unit quaternions, fixed feet, and repeated solves with a
changed target. It patches Python Pinocchio FK to fail during optimization,
checking that the native solve does not enter that Python function.

## CasADi/IPOPT with the same native graphs

`FunctionManager.asCasadi(name, nx, nu)` exports a registered graph as a CasADi
MX Function. Values and analytical Jacobians execute in a C++ CasADi Callback;
there is no Python numerical callback or finite differencing. Each exported
adapter clones the graph and protects its workspace during threaded CasADi
maps. Calls to the same adapter are serialized; different adapters own separate
workspaces. Cross-function/node Pinocchio caching remains specific to iLQR.
The extension must use the same CasADi library/ABI as Python: the binding
transfers a Function into an allocated SWIG wrapper at setup time.

Use exported functions in ordinary Horizon constraints and residuals:

```python
fk_residual = functions.asCasadi('target', robot.nq, robot.nv)
problem.createFinalResidual(
    'tool_target', fk_residual(q, cs.MX.zeros(robot.nv), target)/np.sqrt(2))
solver = Solver.make_solver(
    'ipopt', problem, {'ipopt.tol': 1e-11},
    function_manager=functions, dynamics='dynamics', gauss_newton=True,
)
```

The solver retains the manager and adds multiple-shooting equalities for the
named **discrete** dynamics. Inputs after state/control match Horizon parameter
names; `dt` uses `Problem.setDt`. Do not also transcribe the dynamics through
`Transcriptor`. Native residuals/constraints use `asCasadi` before constructing
the solver. If using exported functions without passing `function_manager`,
keep the manager alive yourself for the entire lifetime of those functions.
Native callback graphs cannot be serialized or code generated.

`gauss_newton=True` supplies IPOPT's `hess_lag(x, p, sigma, lambda)` with the
upper triangle of `sigma * (2 J_r.T J_r + H_scalar_costs)`. Constraint multiplier
curvature and residual second derivatives are omitted. The Jacobian and GN
matrix are evaluated at the current NLP variables and parameters on each
Hessian request. Scalar Horizon costs keep exact curvature, so express native
least-squares terms with `createResidual`, rather than `createCost(sumsqr(...))`.
Custom `hess_lag` and limited-memory Hessian options conflict with this mode
and are rejected. Legacy IPOPT defaults are unchanged.

Horizon IPOPT minimizes `sum(sumsqr(r))`; iLQR's GN residual setters minimize
`0.5*sumsqr(r)`, with running costs additionally divided by N. To compare the
same objective, scale IPOPT running residuals by `1/sqrt(2*N)` and terminal
residuals by `1/sqrt(2)`. The existing example handles these scales:

```bash
python -m horizon.examples.ilqr_native_pinocchio --urdf /path/to/kyon.urdf --N 30 --solver ipopt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m tools.benchmark_native_ipopt \
  --urdf /path/to/kyon.urdf --nodes 12 30 --samples 3 --output /tmp/comparison.json
```

The floating-base example uses ambient configuration coordinates for IPOPT and
Pinocchio integration for its discrete step. Fixed initial configuration and
step equalities preserve the unit quaternion. iLQR uses its tangent state
space. Initial foot constraints are redundant with the fixed initial state and
are omitted from IPOPT. Neither solver changes the robot model.

Initial measurements before the profiling fixes below (CasADi 3.8.2, two iLQR workers, three fresh solves;
medians exclude setup and post-solve diagnostics):

| Intervals | iLQR solve | IPOPT solve | iLQR objective | IPOPT objective |
|---:|---:|---:|---:|---:|
| 12 | 0.0247 s | 0.6251 s | 5.307923e-6 | 5.308150e-6 |
| 30 | 0.0548 s | 0.5222 s | 5.307925e-6 | 5.308330e-6 |

These are example-specific timings, not a general speed ranking. Both reach
about 2.34e-8 m target error. The benchmark checks feet, discrete dynamics,
quaternion norms, velocity bounds, and matched objective values on every run.
[Raw measurements](../cpp/docs/ipopt_native_results.json) include setup times
and all numerical diagnostics. The interoperability regression also compares
a nonlinear native residual problem, verifies SX/MX GN scaling with nonzero
residuals, repeats solves after parameter changes, and runs the existing legacy
QP and native Pinocchio tests.

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
