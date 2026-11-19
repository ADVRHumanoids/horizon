# Kyon runtime collision fixtures

`kyon_spheres.urdf` is an unchanged copy of the supplied sphere URDF from
`iit-kyon-ros-pkg/kyon_urdf/urdf/kyon_spheres.urdf`, source revision
`0f2208d4452556f3d5b704af1f6e721dbbb6b521`. It contains 96 spheres on 23 links,
but no collision geometry on the arms. Its matching policy is
`kyon_disabled_collisions.json`.

Examples and benchmarks use `kyon_full_spheres.urdf` and
`kyon_full_disabled_collisions.json`. The full fixture was generated from the
same checkout's `kyon.urdf.xacro`, with its default arguments, using FOAM
revision `116928f71aaa7c40356d79c84d3c9ff1f4497d90`, depth 1 and branch 3.
It covers all 31 collision-bearing links of that xacro, with 103 spheres.
Visual meshes remain package URIs; headless kinematics and sphere collision
tests do not need those mesh resources.

Both JSON policies were generated with the JSON-capable
`moveit_compute_default_collisions`, based on revision
`675ef71a773ca53a84a758d4dd5b5f134f51c111`, using 75,000 trials. The full model
has 137 disabled link pairs and 3,799 enabled sphere pairs. Same-link spheres
are always excluded from self-collision checking. Every remaining distinct-link
sphere pair is enabled unless its link pair appears in the policy.

MoveIt records `Adjacent`, `Default`, `Always`, or `Never` reasons when those
classifications occur. `Never` is based on sampled configurations, not a
proof over the full joint domain. Sphere fitting and sampled exclusions are
approximations; review the policy for the intended robot configuration range.
Sampling and sphere fitting can produce different results on regeneration.

With FOAM and the modified collision generator installed, source the ROS
environment for `kyon_urdf` and its `zed_description` dependency, then run from
the Horizon root:

```bash
python3 tools/generate_collision_assets.py \
  --sphere-urdf /tmp/kyon_full_spheres.urdf \
  --collision-json /tmp/kyon_full_disabled_collisions.json \
  --trials 75000 --threads 2 --branch 3 --depth 1 -- \
  xacro /path/to/kyon_urdf/urdf/kyon.urdf.xacro
```

The pipeline resolves package meshes, applies scale and pose in URDF order,
preserves existing spheres, checks that collision-bearing links are retained,
and invokes MoveIt on the resulting sphere URDF. The full Kyon command was
validated end to end with 2,000 sampling trials; the vendored policy was then
generated with 75,000 trials. No C++ sphere header is generated.
