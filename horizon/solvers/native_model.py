"""Problem-aware native graphs with identical iLQR and IPOPT residual semantics."""
import casadi as cs
import numpy as np

from horizon.solvers import Solver
from horizon.solvers.function_manager import SymbolicFunctionManager


class NativeModel:
    """Build one native problem for either backend.

    Residuals mean half squared norms; running terms are divided by the number
    of shooting intervals. Use a fresh Problem for each backend instance.
    The solver retains this adapter, its native graphs and CasADi callbacks.
    """

    def __init__(self, problem, state_space=None):
        if problem.default_abstract_casadi_type is not cs.MX:
            raise ValueError("NativeModel requires Problem(..., abstract_casadi_type=cs.MX)")
        self.problem = problem
        self.state_space = state_space
        self.functions = SymbolicFunctionManager()
        self.dt = cs.MX.sym("dt")
        self._terms = []
        self._dynamics = None
        self._built = False
        self._uses_control = {}

    def addNative(self, name, function):
        if self._built:
            raise RuntimeError("Register native calls before constructing the solver")
        if name in self.functions._signatures:
            raise ValueError(f"Native function name already registered: {name}")
        return self.functions.addNative(name, function)

    def _register(self, name, expression, timestep=False):
        if name in self.functions._signatures:
            raise ValueError(f"Native function name already registered: {name}")
        inputs = [self.problem.getState().getVars(), self.problem.getInput().getVars()]
        dependencies = [cs.MX(expression)]
        for argument, result, _ in reversed(self.functions._calls):
            if any(cs.depends_on(value, result) for value in dependencies):
                dependencies.append(argument)
        self._uses_control[name] = any(cs.depends_on(value, inputs[1]) for value in dependencies)
        inputs += [p for p in self.problem.getParameters().values()
                   if any(cs.depends_on(value, p) for value in dependencies)]
        timestep = timestep or any(cs.depends_on(value, self.dt) for value in dependencies)
        if timestep:
            if any(p.name() == "dt" for p in inputs[2:]):
                raise ValueError("The name dt is reserved for model.dt")
            inputs.append(self.dt)
        self.functions.addExpression(name, inputs, expression, exact_hessian=False)

    def setDynamics(self, expression):
        if self._built:
            raise RuntimeError("Set dynamics before constructing the solver")
        self._register("native_dynamics_map", expression, timestep=True)
        self._dynamics = "native_dynamics_map"

    def _term(self, kind, name, expression, nodes):
        if self._built:
            raise RuntimeError("Register terms before constructing the solver")
        nodes = [int(nodes)] if isinstance(nodes, (int, np.integer)) else list(nodes)
        if not nodes or len(set(nodes)) != len(nodes) or any(k < 0 or k >= self.problem.getNNodes() for k in nodes):
            raise ValueError("Term nodes must be unique valid problem nodes")
        self._register(name, expression)
        if self.problem.getNNodes() - 1 in nodes and self._uses_control[name]:
            raise ValueError("Terminal terms cannot depend on controls")
        self._terms.append((kind, name, nodes))
        return name

    def createConstraint(self, name, expression, nodes=None):
        """Register an equality constraint, by default at all state nodes."""
        return self._term("constraint", name, expression,
                          range(self.problem.getNNodes()) if nodes is None else nodes)

    def createIntermediateResidual(self, name, expression, nodes=None):
        """Register half squared residuals with horizon normalization."""
        nodes = range(self.problem.getNNodes() - 1) if nodes is None else nodes
        nodes = [int(nodes)] if isinstance(nodes, (int, np.integer)) else list(nodes)
        if any(k == self.problem.getNNodes() - 1 for k in nodes):
            raise ValueError("Running residuals cannot include the terminal node")
        return self._term("residual", name, expression, nodes)

    def createFinalResidual(self, name, expression):
        return self._term("residual", name, expression, [self.problem.getNNodes() - 1])

    def make_solver(self, solver_type="ilqr", options=None):
        if solver_type not in ("ilqr", "ipopt"):
            raise ValueError("solver_type must be ilqr or ipopt")
        if self._built:
            raise RuntimeError("NativeModel already built; reuse its solver for repeated solves")
        if self._dynamics is None:
            raise ValueError("Set native dynamics before constructing the solver")
        options = dict(options or {})
        options.setdefault("ilqr.legacy_problem_cost_scaling", True)
        kwargs = dict(function_manager=self.functions, dynamics=self._dynamics)
        if solver_type == "ilqr":
            options["ilqr.enable_gn"] = True
            kwargs["state_space"] = self.state_space
        else:
            kwargs["gauss_newton"] = True
            x = self.problem.getState().getVars()
            u = self.problem.getInput().getVars()
            N = self.problem.getNNodes() - 1
            for kind, name, nodes in self._terms:
                function = self.functions.asCasadi(name, x.numel(), u.numel())
                pars = [self.problem.getDt() if function.name_in(i) == "dt"
                        else self.problem.getParameters(function.name_in(i))
                        for i in range(2, function.n_in())]
                # State-only terminal functions still use the native [x; u] ABI.
                control = u if self._uses_control[name] else cs.MX.zeros(u.numel())
                if N in nodes and self._uses_control[name]:
                    raise ValueError("Terminal terms cannot depend on controls")
                value = function(x, control, *pars)
                if kind == "constraint":
                    self.problem.createConstraint(name, value, nodes=nodes)
                else:
                    scale = np.sqrt(2 if nodes == [N] else 2 * N)
                    self.problem.createResidual(name, value / scale, nodes=nodes)
        solver = Solver.make_solver(solver_type, self.problem, options, **kwargs)
        if solver_type == "ilqr":
            for kind, name, nodes in self._terms:
                setter = solver.ilqr.setConstraint if kind == "constraint" else solver.ilqr.setResidual
                setter(nodes, self.functions, name)
        solver.native_model = self
        self._built = True
        return solver
