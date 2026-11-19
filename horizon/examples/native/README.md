# Native API walkthrough

Each numbered Python example has a C++ counterpart in `horizon/cpp/examples`.
Run them in order:

1. `01_symbolic`: a CasADi differential-drive model and a terminal objective.
2. `02_parameters`: supply the target as a parameter rather than baking it into the expression.
3. `03_robot_kinematics`: load a URDF and use native frame kinematics and configuration integration.
4. `04_self_collision`: introduce a soft sphere-pair self-clearance cost on the simple arm.
5. `05_environment_primitive`: steer the tool around a sphere on its direct route.
6. `06_environment_mesh`: replace that sphere with a mesh-derived OpenVDB field at the same mid-route location.
7. `07_whole_body_ik`: hold four feet fixed while moving a floating-base robot's tool.
8. `08_whole_body_self_collision`: add the bundled robot's full sphere model and self-collision exclusions.
9. `09_contact_forward`: solve a small arm motion with joint torques as controls and constrained forward dynamics.
10. `10_contact_inverse`: use generalized acceleration and contact forces as controls, with the first six inverse-dynamics rows and contact accelerations constrained to zero.
11. `11_accelerated_collision`: revisit lesson 8 with the Kokkos batched collision cost.

`spot_motions` is a native iLQR port of the Spot jump, wheelie, leap, wall-jump,
and twist examples. It defaults to the bundled Spot URDF and accepts `--action`,
the common solver options, and `--visualize` for browser playback in Viser:

```sh
python -m horizon.examples.native.spot_motions --action leap --visualize
```

The same action interface is available through the Spot command and the action
dispatcher:

```sh
python -m horizon.examples.native.spot --action jump_forward --visualize
python -m horizon.examples.native.spot_actions --action wheelie --visualize
```

Lessons 05 and 06 place an obstacle on the direct tool route, with the start and target clear. They sample clearance within each interval as well as at the knots. Clearance penalties are soft; sampled checks do not prove continuous collision freedom. Lesson 04 uses an exaggerated self-clearance margin on the simple arm; a multi-link self-collision example would make this lesson more realistic.

Run Python lessons from the repository root, for example `python -m horizon.examples.native.03_robot_kinematics`. For C++, configure `horizon/cpp` with `HORIZON_BUILD_EXAMPLES=ON`; lessons 3 through 11 also require `HORIZON_WITH_PINOCCHIO=ON`. Build the corresponding `native_...` target. With `BUILD_TESTING=ON`, CTest runs each built lesson; Python lessons additionally require `HORIZON_BUILD_PYTHON=ON`.

For example, with the dependencies on `CMAKE_PREFIX_PATH`:

```sh
cmake -S horizon/cpp -B build -DHORIZON_WITH_PINOCCHIO=ON -DHORIZON_BUILD_PYTHON=ON
cmake --build build --target native_03_robot_kinematics pyilqr
ctest --test-dir build -R 'native_03_robot_kinematics' --output-on-failure
```

Enable `HORIZON_WITH_OPENVDB` and/or `HORIZON_WITH_KOKKOS` to build the corresponding optional lessons. Building the default target also builds the mesh converter needed by its CTest fixture.

All lessons accept the same optional arguments: `--nodes` (`-N`), `--threads`,
`--max-iterations`, `--dt`, and `--quiet`. The default time step for the
whole-body IK lessons is `2 / nodes`; the other lessons use a fixed step.
`--urdf`, `--policy`, and `--sdf` override the bundled assets where applicable.
For example:

```sh
python -m horizon.examples.native.03_robot_kinematics --nodes 40 --threads 2
build/native_03_robot_kinematics --nodes 40 --threads 2
```

The mesh lesson requires `HORIZON_WITH_OPENVDB=ON`. It defaults to the bundled
`obstacle.vdb`, generated from the adjacent closed STL using voxel size 0.025
and a 16-voxel band. To regenerate a field or supply your own:

```sh
build/horizon_mesh_to_sdf horizon/cpp/examples/data/obstacle.stl /tmp/obstacle.vdb 0.025 16
python -m horizon.examples.native.06_environment_mesh --sdf /tmp/obstacle.vdb
```

CTest also generates its own mesh fixture. The full-body example uses the
repository's Kyon fixture and collision policy; sphere geometry and sampled
exclusion policies are approximations. The collision costs are soft and
clearance checks are at knots, not between them.

The contact lessons use fixed bilateral point contacts, initialized in static equilibrium. They check dynamics and foot drift at shooting nodes; they do not impose unilateral force or friction limits.

Lesson 11 requires `HORIZON_WITH_KOKKOS=ON` and runs on the execution backend selected by the Kokkos build (CPU or CUDA). Kokkos must remain initialized until every solver and collision cost is destroyed.

Python lesson 11 moves the tool by `(0.4, 0, -0.5)` m. It starts with
`ilqr.hxx_reg=10` and `ilqr.hxx_reg_growth_factor=10` to damp the large initial
steps; the solver reduces this damping as steps are accepted. The much smaller
control regularization alone can send this motion into a colliding local
minimum. The example requires solver convergence within the iteration limit,
as well as successful tracking, fixed feet, and positive knot clearance.


To view a Python solution, add `--visualize`. Lessons 1–2 show a matplotlib
trajectory plot; robot lessons start a Viser server at `http://127.0.0.1:8080`.
Install `viser`, `yourdfpy`, and `trimesh` for robot playback (the numerical
examples do not require them when visualization is disabled). Use
`--viewer-port` to change the port (or use 0 to select a free port), and stop Viser with Ctrl-C. Playback uses
solved knots; the time slider pauses playback when dragged. For a custom mesh
SDF, pass `--mesh path/to/mesh.stl` alongside `--sdf` to display its geometry.
For unattended checks, `--visualization-seconds 1` closes the viewer after one
second; the default keeps it open.

When the optional viewer dependencies are present at CMake configure time,
`ctest --test-dir build -L visualization --output-on-failure` runs short,
headless smoke tests for every enabled Python lesson.
