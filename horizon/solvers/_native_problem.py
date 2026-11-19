"""Internal lowering of Problem MX expressions; never a modeling interface."""
import casadi as cs
import numpy as np

from horizon.native import _calls, _expanded
from horizon.functions import Residual, RecedingResidual

_SCALE = "horizon_problem_cost_scale"


class NativeCompilation:
    def __init__(self, problem, backend, options, kwargs):
        from horizon.solvers import pyilqr
        from horizon.solvers.function_manager import SymbolicFunctionManager
        self.problem = problem
        self.manager = SymbolicFunctionManager()
        self.records = {}
        self.terms = []
        self.has_cost = False
        self.scale = cs.MX.sym(_SCALE)
        self.x = problem.getState().getVars()
        self.u = problem.getInput().getVars()
        self.dt = cs.MX.sym("dt")
        dynamics = getattr(problem, "_discrete_dynamics", None)
        if dynamics is None:
            integrator = problem.getIntegrator()
            if integrator is None:
                raise ValueError("Define Problem dynamics before creating a native solver")
            dynamics = integrator(self.x, self.u, self.dt)
            if isinstance(dynamics, (tuple, list)):
                dynamics = dynamics[0]
        self.dynamics = cs.MX(dynamics)
        self.owners = list(getattr(problem, "_native_dynamics_owners", ()))
        containers = [problem.function_container.getCost(), problem.function_container.getCnstr()]
        native_terms = [fun for container in containers for fun in container.values()
                        if getattr(fun, "_native_owners", ()) and not getattr(fun, "_native_transcription", False)]
        self.owners += [owner for fun in native_terms for owner in fun._native_owners]
        self.owners += [call[2] for call in _calls(self.dynamics)]
        if backend == "ipopt":
            if not getattr(problem, "_native_transcribed", False):
                expression = self.dynamics
                if cs.depends_on(expression, self.dt):
                    expression = cs.substitute(expression, self.dt, cs.MX(problem.getDt()))
                constraint = problem.createIntermediateConstraint(
                    "horizon_discrete_dynamics", expression-problem.getState().getVarOffset(1).getVars())
                constraint._native_transcription = True
                problem._native_transcribed = True
            native_scalar = any(not isinstance(fun, (Residual, RecedingResidual))
                                for fun in native_terms if fun in problem.function_container.getCost().values())
            if native_scalar or options.get("ipopt.hessian_approximation") == "limited-memory" or kwargs.get("gauss_newton") is False:
                options.setdefault("ipopt.hessian_approximation", "limited-memory")
                kwargs.setdefault("gauss_newton", False)
            else:
                kwargs.setdefault("gauss_newton", True)
            return
        if backend != "ilqr":
            raise ValueError("Native Problem expressions currently support ilqr and ipopt")
        if "function_manager" in kwargs or "dynamics" in kwargs:
            raise ValueError("Problem-native expressions manage their own internal function registry")
        self._register("horizon_discrete_dynamics", self.dynamics, timestep=True)
        kwargs.update(function_manager=self.manager, dynamics="horizon_discrete_dynamics")
        kwargs.setdefault("state_space", getattr(problem, "_discrete_state_space", None) or self._state_space(pyilqr))
        options["ilqr.enable_gn"] = True
        for fun in native_terms:
            name = fun.getName()
            is_constraint = name in problem.function_container.getCnstr()
            expression = fun._f
            if not is_constraint:
                if not isinstance(fun, (Residual, RecedingResidual)):
                    raise ValueError("iLQR native scalar costs require exact Hessians; use residuals or IPOPT")
                expression = np.sqrt(2)*self.scale*expression
                self.has_cost = True
            self._register(name, expression, cost=not is_constraint)
            self.terms.append((fun, is_constraint))
            if is_constraint:
                self._targets(fun)
        problem._native_compile_active = True

    def _state_space(self, pyilqr):
        spaces = []
        calls = _calls(self.dynamics)
        for fun in self.problem.function_container.getCost().values():
            calls += _calls(fun._f)
        for fun in self.problem.function_container.getCnstr().values():
            if not getattr(fun, "_native_transcription", False):
                calls += _calls(fun._f)
        for var in self.problem.getState().var_list:
            matches = [p.configuration_space for argument, _, p in calls
                       if p.configuration_space is not None and p.nq == var.numel()
                       and cs.is_equal(argument[:p.nq], var)]
            spaces.append(matches[0] if matches else pyilqr.EuclideanSpace(var.numel()))
        return spaces[0] if len(spaces) == 1 else pyilqr.ProductSpace(spaces)

    def _register(self, name, expression, timestep=False, cost=False):
        expression = _expanded(expression)
        for argument, result, primitive in _calls(expression):
            key = hash(result)
            if key not in self.records:
                self.records[key] = (argument, result, primitive,
                                     cs.MX.sym(f"horizon_value_{len(self.records)}", result.numel()))
        old = [record[1] for record in self.records.values()]
        new = [record[3] for record in self.records.values()]
        replace = lambda value: cs.graph_substitute(value, old, new) if old else value
        self.manager._calls = [(replace(a), placeholder, primitive.native)
                               for a, _, primitive, placeholder in self.records.values()]
        pars = [p for p in self.problem.getParameters().values() if cs.depends_on(expression, p)]
        inputs = [self.x, self.u, *pars]
        if timestep and cs.depends_on(expression, self.dt):
            inputs.append(self.dt)
        if cost:
            inputs.append(self.scale)
        self.manager.addExpression(name, inputs, replace(expression), exact_hessian=False)

    @staticmethod
    def _targets(fun):
        lower, upper = fun.getLowerBounds(), fun.getUpperBounds()
        nodes = fun.getNodes()
        if lower.shape[1] != len(nodes):
            feasible = np.flatnonzero(fun._getFeasNodes()).tolist()
            columns = [feasible.index(node) for node in nodes]
            lower, upper = lower[:, columns], upper[:, columns]
        if not np.array_equal(lower, upper) or not np.all(np.isfinite(lower)):
            raise ValueError(f'iLQR supports equality function constraints; "{fun.getName()}" has inequalities. Use IPOPT for hard bounded native constraints.')
        return [lower[:, k] for k in range(lower.shape[1])]

    def attach(self, solver):
        for fun, constraint in self.terms:
            if constraint:
                solver.ilqr.setConstraint(fun.getNodes(), self.manager, fun.getName(), self._targets(fun))
            else:
                solver.ilqr.setResidual(fun.getNodes(), self.manager, fun.getName())
        if self.has_cost:
            solver.ilqr.setParameterValue(_SCALE, np.r_[np.full(solver.N, np.sqrt(solver.N)), 1.][None, :])

    def refresh_constraints(self, solver):
        for fun, constraint in self.terms:
            if constraint:
                solver.ilqr.setConstraint(fun.getNodes(), self.manager, fun.getName(), self._targets(fun))


def prepare(problem, backend, options, kwargs):
    functions = [*problem.function_container.getCost().values(), *problem.function_container.getCnstr().values()]
    if not (hasattr(problem, "_discrete_dynamics") or getattr(problem, "_native_dynamics_owners", ())
            or any(getattr(fun, "_native_owners", ()) for fun in functions)):
        return None
    return NativeCompilation(problem, backend, options, kwargs)
