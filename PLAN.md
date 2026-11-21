## Tangent-space derivatives and solver performance

### Goal

Make tangent coordinates the primary derivative representation throughout iLQR. Native functions and dynamics should supply local derivatives directly; CasADi functions should be differentiated in perturbation coordinates. Avoid converting these derivatives to ambient coordinates and back. Continue storing trajectories in their existing state representation, including quaternion coordinates, and preserve the existing mathematical problem and support for infeasible shooting trajectories.

### 1. Solver-facing tangent contracts

- Define explicit local derivative interfaces for dynamics, constraints, residuals, and scalar costs. Associate each with its state-space identity, base point, dimensions, and tangent-coordinate convention.
- Dynamics return their predicted state in storage coordinates and derivatives mapping state-tangent and control perturbations to the tangent space at that predicted state.
- Constraints and residuals return values and derivatives with respect to the state tangent and controls. Exact scalar costs return local gradients and Hessians, including retraction curvature.
- Retain the ambient `Function` interface for external users and unsupported functions. Isolate conversion in compatibility adapters that feed the same tangent-based solver core.

### 2. Direct native derivatives

- Expose the tangent derivatives native functions already compute and use them directly in iLQR, without constructing ambient Jacobians or projecting ambient seed directions back into tangent space.
- Add a tangent-output dynamics contract. For Pinocchio integration, use `dIntegrate` to assemble the local dynamics derivatives directly, including velocity and acceleration terms, without converting the predicted-state derivative to storage coordinates and back.
- Extend graph lowering so native leaves consume tangent directions directly when the space, base point, argument selection, and coordinate ordering are compatible. Preserve a correct ambient fallback for arbitrary transformed, sliced, or reordered arguments until explicit local rules are available.
- Preserve node-local workspace ownership, cache invalidation, and concurrent evaluation safety.

### 3. CasADi perturbation-form differentiation

For a CasADi function `f(x, u, p)`, construct at setup:

```text
f_local(x, u, p, dx, du) = f(integrate(x, dx), u + du, p)
```

Differentiate with respect to `dx` and `du`, then evaluate at zero to obtain local Jacobians. Compile values and derivatives together where useful, allowing CasADi to simplify and exploit sparsity without a separate runtime ambient-to-tangent projection.

- Supply symbolic retractions for supported state spaces and compose them for Cartesian products. Unsupported spaces use the compatibility path.
- For exact scalar Hessians, differentiate through the actual retraction so its curvature is included. Do not silently substitute Gauss-Newton or a linearized retraction when exact derivatives are required.
- For mixed CasADi/native graphs, implement a tangent-aware native derivative bridge. Wrapping the existing ambient native callback in a symbolic retraction alone would retain the projection internally.
- For manifold-valued dynamics, compile derivatives in the tangent at the nominal prediction, treating that nominal prediction as fixed when defining the output difference. Use native tangent-output dynamics where available.

### 4. Structured defect transport

At a multiple-shooting node, the predicted next state and the independently stored next state can differ. The dynamics derivative is expressed at the prediction, while the backward pass uses increments at the stored next state. Transport between those tangent spaces is therefore required for nonzero defects; do not assume feasible trajectories or remove this operation universally.

- Provide a state-space defect linearization and transport operation. Product spaces dispatch to components; Euclidean components use identity operations; Pinocchio configuration uses joint blocks, including the coupled free-flyer block.
- Retain the existing dense formulation for arbitrary or coupled state spaces. Preserve singularity handling and the information needed by globalization; independent block thresholds must not silently change the current invertibility decision.
- Compare the structured operation against the current dense solve before considering further analytic Lie-group specializations.

### 5. Streamline workspaces and solver paths

- Remove ambient derivative buffers and conversion products from direct tangent paths. Reuse node-local geometry and derivative workspaces where still required.
- Skip empty state-bound equality work and share compatible retraction information. Preserve full-state versus partial storage-coordinate bound semantics.
- Preserve graph selection and sparsity where it reduces direction propagation; benchmark specialized products and solves against Eigen at actual problem dimensions.
- Avoid duplicating optimization logic between direct and compatibility paths, and do not change multiple-shooting rollout or globalization semantics as a performance shortcut.

### 6. Validation and measurement

- Compare local Jacobians against manifold finite differences and the existing implementation for native, CasADi, and mixed graphs, including parameters, arbitrary seeds, coordinate conventions, and fallback paths.
- Check exact cost Hessians, retraction curvature, Gauss-Newton behavior, state bounds, cache invalidation, exceptions, and multithreaded node evaluation.
- Compare dynamics derivatives and transported `A`, `B`, and defect at zero and nonzero defects, quaternion sign changes and branch boundaries, product spaces, and near-singular or mixed-scale blocks.
- Compare solver steps, multipliers, residuals, accepted trials, and trajectories against the existing solver in serial and multithreaded runs.
- Measure native leaves, CasADi local derivatives, graph propagation, defect transport, and full derivative sweeps separately on the same problem, state, solver settings, and thread count. Report profiling-disabled wall time separately from instrumented timings and instruction counts.

Implementation order: tangent contracts, direct native derivatives, CasADi perturbation compilation, structured defect transport, then solver cleanup. Validate correctness and measure each stage before proceeding.

