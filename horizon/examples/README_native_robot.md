# Native robot expressions in Problem

Use `NativeRobot` for robot computations and `Problem` for the optimization
model. There is no public export or registration step. Build with
`HORIZON_WITH_PINOCCHIO=ON`; see the [build guide](README_ilqr_native_pinocchio.md).

```python
import casadi as cs
import numpy as np
from horizon.problem import Problem
from horizon.robot import NativeRobot
from horizon.solvers import Solver

robot = NativeRobot('/path/to/robot.urdf')
problem = Problem(30, abstract_casadi_type=cs.MX)
q = problem.createStateVariable('q', robot.nq)
v = problem.createStateVariable('v', robot.nv)
tau = problem.createInputVariable('tau', robot.na)
problem.setDt(2./30)
q0 = robot.configuration_space.neutral()  # replace with a feasible stance
problem.setInitialState(np.r_[q0, np.zeros(robot.nv)])
q.setInitialGuess(q0)
v.setInitialGuess(np.zeros(robot.nv))
tau.setInitialGuess(np.zeros(robot.na))

contacts = ['left_foot', 'right_foot']  # point contacts at frame origins
a, forces = robot.contactDynamics(q, v, tau, contacts)
dt = problem.getDt()
problem.setDiscreteDynamics(cs.vertcat(
    robot.integrate(q, dt*v + .5*dt**2*a), v + dt*a))
problem.createIntermediateResidual('effort', .01*tau)
target = problem.createParameter('tool_target', 7)
target.assign(robot.framePose('tool', q0))  # xyz + quaternion xyzw
error = robot.framePoseError('tool', q, target)
problem.createFinalResidual('tool', cs.vertcat(100*error[:3], 30*error[3:]))
problem.createConstraint('positive_fz', forces[2::3], nodes=range(30),
    bounds={'lb': np.ones(len(contacts)), 'ub': np.full(len(contacts), np.inf)})
solver = Solver.make_solver('ipopt', problem, {'ipopt.max_iter': 200})
solver.solve()
```

Set robot joint, velocity and torque limits as appropriate; the complete
[contact motion example](native_contact_motion.py) includes those bounds and
collision constraints. Neutral configuration is not necessarily a feasible
contact stance.

`framePoseError` returns `[p(q)-p_target; log(R_target.T @ R(q))]`. Translation
uses world coordinates; orientation uses target-frame axes. Quaternions are
normalized internally, with equivalent signs. The rotation log has a branch
cut at pi. Frames define feet, tools and contact origins independently of
collision geometry. `framePositions(frames, q)` returns a 3-by-number-of-frames
matrix; `jacobian` returns 6-by-nv, and `massMatrix` returns nv-by-nv.

Native expressions accept numeric arrays or MX variables and compose with
ordinary CasADi algebra and functions. They can be evaluated before solver
creation. Robot values and analytic first derivatives run in C++; CasADi
differentiates the surrounding algebra. At solver creation Horizon discovers
the native calls, builds the internal iLQR registry or IPOPT transcription,
and retains their owners. Keep the robot alive while building standalone MX
graphs; once registered as Problem dynamics, costs or constraints, Problem
retains the required native objects. SX robot expressions are unsupported.

A Problem residual means its squared norm, summed over its active nodes, for
both solvers. Native residual models use Gauss–Newton curvature. Native scalar
costs use IPOPT limited-memory Hessians; native iLQR costs must be residuals.
iLQR supports equality function constraints and its state/control bound
machinery. Use IPOPT for hard bounded force or collision function constraints.
Build a fresh Problem per backend; reuse the solver for parameter, bound and
active-node updates without changing the graph topology.

## Actuation and contacts

`NativeRobot` defaults to joint actuation when the leading joint is a six-DOF
free flyer, and full actuation otherwise. `robot.na` is the torque input size;
`robot.actuation_matrix` is S. Supply `actuation_matrix=S` to the constructor
for other layouts, or to `contactDynamics`/`forwardDynamics` for a particular
operation. `forwardDynamics(q,v,tau)` applies S; `aba(q,v,generalized_tau)`
accepts full generalized torque.

Contact dynamics solves
`M*a + h = S*tau + J.T*f`, `J*a + gamma = 0`.
Strings denote 3D point contacts; `pyilqr.Contact(frame, 6)` denotes rigid
contacts. Output forces act on the robot in world-aligned axes at the frame
origin: xyz for point contacts, `[force xyz; torque xyz]` for rigid contacts,
concatenated in contact order. Thus `forces[2::3]` selects fZ only for an
all-point-contact list. `inverseDynamics(q,v,a,f,contacts)` returns generalized
torques, including unactuated base components.

Contacts are fixed bilateral acceleration constraints. Initialize consistent
positions and velocities and check integration drift. Positive fZ, friction,
contact switching and impacts require explicit modeling. Rank-deficient KKT
systems fail explicitly. The sensitivity solve analytically differentiates
both accelerations and forces.

## Collision helpers

```python
from horizon.collision import (Box, Sphere, Mesh, build_sdf, CollisionSpheres,
                               self_collision, environment_collision)
sdf = build_sdf([
    Box(center=[1., 0., .5], size=[.3, .4, 1.]),
    Sphere(center=[.5, .8, .5], radius=.2),
    Mesh('/path/to/watertight_obstacle.stl'),
], bounds=([-2., -2., -1.], [2., 2., 3.]), voxel_size=.02)
geometry = CollisionSpheres.from_urdf('/path/to/sphere_geometry.urdf')
self_distance = self_collision(robot, q, geometry,
                               disabled_pairs=[('base', 'upper_leg')])
world_distance = environment_collision(robot, q, geometry, sdf)
for name, distance in [('self', self_distance), ('world', world_distance)]:
    if distance.numel():
        problem.createConstraint(name, distance,
            bounds={'lb': np.full(distance.numel(), .02),
                    'ub': np.full(distance.numel(), np.inf)})
```

Available scene primitives are Sphere, Box, Capsule, Cylinder and Plane; Mesh
accepts a rigid pose and positive scale. Mesh sampling requires `trimesh` and
`rtree`, a watertight mesh and consistent winding. `build_sdf` samples the
union once during setup. Native C++ interpolation and clearance derivatives
run during optimization. The SDF bounds must contain queried sphere centers;
out-of-domain evaluations fail rather than extrapolate.

Self collision checks all cross-link sphere pairs except explicitly disabled
link pairs (a list or a JSON policy path); same-link pairs are omitted.
Distances are signed surface clearances. Robot spheres approximate geometry,
and the sampled SDF approximates environment distances. Check resolution,
coverage and clearance between optimization nodes for the application.
Spheres serve collision checking only; foot/tool kinematics needs no spheres.

To assemble the complete torque-controlled example with this scene:

```python
from horizon.examples.native_contact_motion import build_problem
problem = build_problem('/path/to/kyon.urdf', q0, target_pose, geometry, sdf,
                        disabled_pairs='/path/to/collision_policy.json')
solver = Solver.make_solver('ipopt', problem, {'ipopt.max_iter': 300})
solver.solve()
```

Here `q0` is a contact-consistent Kyon configuration and `target_pose` is the
seven-component tool target. Set contact/tool frame names explicitly when
using another robot. Scene geometry and collision policy are user inputs.

## Metadata and compatibility

`robot.nq`, `nv`, `mass`, `joint_names`, `lower_position_limit`,
`upper_position_limit`, `velocity_limits`, and `effort_limits` expose metadata.
`joint_iq`, `joint_nq`, `joint_iv`, and `joint_nv` locate joint blocks. Native
clones own independent mutable workspaces for solver node evaluation.

`robot.raw` exposes the existing `pyilqr.PinocchioModel` factories below.
Those factories default to full generalized actuation. `NativeModel` remains
a compatibility adapter with its original half-squared, horizon-averaged
objective convention. New models should use Problem directly. The option
`ilqr.legacy_problem_cost_scaling=True` retains historical Python iLQR cost
normalization; low-level C++ setter conventions remain unchanged.

## Low-level factories and derivatives

Here `q` has `nq` stored coordinates, while `v`, `a`, and generalized `tau`
have `nv` tangent coordinates. Inputs are concatenated in the order shown.
Every output is a column vector. Matrices are flattened **column-major**:

```python
M = robot.massMatrix().evaluate(q0).reshape(robot.nv, robot.nv, order='F')
```

Spatial vectors use `[linear; angular]`. Poses use world-frame translation
and quaternion coefficients in **xyzw** order. The `reference` argument accepts
`pyilqr.LOCAL`, `pyilqr.WORLD`, or `pyilqr.LOCAL_WORLD_ALIGNED`; the default is
`LOCAL_WORLD_ALIGNED`. They are also available as `robot.LOCAL`, etc.
`LOCAL` uses the frame origin and frame axes; `LOCAL_WORLD_ALIGNED` uses the
frame origin and world axes; `WORLD` applies the spatial-vector transform to
the world origin and world axes. Centroidal quantities use world axes and
angular momentum about the center of mass.

The table describes the Jacobian of the returned function with respect to
**all its inputs**, including parameters such as target poses and gravity.
There are no CasADi-based or finite-difference robot Jacobians in this API.

| Python factory | Packed input | Packed output | Value computation | Jacobian source |
| --- | --- | --- | --- | --- |
| `step(dt)` | `[q; v]` | `integrate(q, dt*v)` | Pinocchio integration | Analytic `dIntegrate` and configuration coefficient Jacobians |
| `integrate()` | `[q; displacement]` | next `q` | Pinocchio integration, unit scale | Analytic, same as `step(1)` |
| `difference()` | `[q_from; q_to]` | tangent displacement (`nv`) | Pinocchio difference | Analytic `dDifference` |
| `qdot()` | `[q; v]` | configuration-coordinate derivative (`nq`) | Native configuration retraction map | Analytic joint-coordinate rules, including quaternion, planar, continuous, and composite joints |
| `framePositions(frames)` | `q` | frame origins `[all x; all y; all z]` | Pinocchio frame placements | Analytic frame Jacobians |
| `framePose(frame)` | `q` | `[xyz; quaternion xyzw]` | Pinocchio frame placement | Analytic frame Jacobian and quaternion derivative |
| `fk(frame)` | `q` | `[xyz; vec(rotation_matrix)]` (`12`) | Pinocchio frame placement | Analytic frame Jacobian and rotation-matrix derivative |
| `framePoseError(frame)` | `[q; target xyz; target quaternion xyzw]` | `[position error; rotation-log error]` (`6`) | Frame placement and SO(3) log | Analytic frame Jacobian, `Jlog3`, quaternion normalization derivatives |
| `jacobian(frame, reference=...)` | `q` | `vec(J_frame)` (`6*nv`) | Pinocchio frame Jacobian | Analytic frame-velocity derivatives, one basis direction per matrix column |
| `frameVelocity(frame, reference=...)` | `[q; v]` | frame velocity (`6`) | Pinocchio frame velocity | Analytic kinematics derivatives and coordinate transforms |
| `frameAcceleration(frame, reference=...)` | `[q; v; a]` | `J*a + Jdot*v` (`6`) | Classical acceleration for LWA; spatial acceleration for LOCAL/WORLD | Analytic kinematics derivatives and velocity cross-product terms |
| `frameSpatialAcceleration(frame, reference=...)` | `[q; v; a]` | spatial acceleration (`6`) | Pinocchio spatial acceleration | Analytic kinematics derivatives and coordinate transforms |
| `rnea(sym_g=False)` | `[q; v; a]`, additionally `g_xyz` if `sym_g=True` | generalized torque (`nv`) | Pinocchio RNEA | Analytic `computeRNEADerivatives`; gravity-input derivatives from the CoM Jacobian |
| `inverseDynamics(contacts=[])` | `[q; v; a; contact_forces]` | generalized torque (`nv`) | RNEA minus contact generalized forces | Analytic RNEA and contracted frame-Jacobian derivatives |
| `aba()` / `forwardDynamics()` | `[q; v; tau]` | `vdot` (`nv`) | Pinocchio ABA | Analytic `computeABADerivatives` |
| `crba()` / `massMatrix()` | `q` | `vec(M)` (`nv*nv`) | Pinocchio CRBA, full symmetric matrix | Analytic RNEA derivatives, one basis acceleration per matrix column |
| `gravity()` | `q` | gravity torque (`nv`) | Pinocchio generalized gravity | Analytic RNEA derivatives at zero velocity/acceleration |
| `nonlinearEffects()` | `[q; v]` | Coriolis/centrifugal and gravity torque (`nv`) | Pinocchio nonlinear effects | Analytic RNEA derivatives at zero acceleration |
| `kineticEnergy()` | `[q; v]` | scalar kinetic energy | Pinocchio kinetic energy | Analytic contracted mass-matrix derivatives and `M*v` |
| `potentialEnergy()` | `q` | scalar potential energy | Pinocchio potential energy | Analytic generalized gravity |
| `centerOfMass()` | `[q; v; a]` | `[com; vcom; acom]` (`9`) | Pinocchio center of mass | Analytic CoM/centroidal derivatives |
| `comPosition()` | `q` | CoM position (`3`) | Pinocchio center of mass | Analytic CoM Jacobian |
| `comVelocity()` | `[q; v]` | CoM velocity (`3`) | Pinocchio center of mass | Analytic centroidal derivatives |
| `comAcceleration()` | `[q; v; a]` | CoM acceleration (`3`) | Pinocchio center of mass | Analytic centroidal derivatives |
| `centroidalMomentum()` | `[q; v]` | momentum `[linear; angular]` (`6`) | Pinocchio centroidal momentum | Analytic centroidal derivatives |
| `computeCentroidalDynamics()` / `centroidalDynamics()` | `[q; v; a]` | `[momentum; momentum_rate]` (`12`) | Pinocchio centroidal momentum/time variation | Analytic `computeCentroidalDynamicsDerivatives` |
| `ccrba()` / `centroidalMap()` | `q` | `vec(A_g)` (`6*nv`) | Pinocchio CCRBA | Analytic momentum derivatives, one basis velocity per matrix column |
| `contactDynamics(contacts, actuation_matrix=...)` | `[q; v; tau]` | `[vdot; contact_forces]` | Native constrained dynamics KKT solve | Analytic implicit differentiation of the KKT equations; Pinocchio RNEA and frame derivatives |

The Jacobians of matrix-valued quantities are analytical but can require
`nv` native derivative passes. Prefer vector-valued computations such as
`rnea`, `frameVelocity`, or `centroidalMomentum` when those are what the model
actually needs.

`function.jacobian(z)` returns an **ambient** Jacobian with `inputSize()`
columns. In C++, `TangentFunction::valueAndTangentLinearization` provides the
corresponding manifold Jacobian. Quaternion normalization is included in the
ambient extension, so IPOPT trial points need not have exactly unit
quaternions. Invalid zero/nonfinite quaternion inputs fail explicitly.

The native functions expose first derivatives, suitable for Gauss–Newton
objectives and constraints; they do not advertise exact Hessians. CasADi may
still differentiate **outer model algebra** such as weights, sums, or symbolic
integration compositions. It does not differentiate robot graphs for the
Jacobians listed above.

