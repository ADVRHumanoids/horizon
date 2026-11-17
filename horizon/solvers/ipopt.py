from .nlpsol import NlpsolSolver
from horizon.problem import Problem
from typing import Dict
import casadi as cs

class IpoptSolver(NlpsolSolver):

    def __init__(self, prb: Problem, opts: Dict = None, *, gauss_newton=False,
                 function_manager=None, dynamics=None) -> None:
        self.function_manager = function_manager
        if dynamics is not None:
            if function_manager is None:
                raise ValueError('A named dynamics function requires function_manager')
            if not gauss_newton:
                raise ValueError('Native dynamics requires gauss_newton=True')
            x = prb.getState().getVars()
            u = prb.getInput().getVars()
            function = function_manager.asCasadi(dynamics, x.numel(), u.numel())
            if function.size1_out(0) != x.numel():
                raise ValueError('Dynamics output must match the state dimension')
            parameters = [prb.getDt() if function.name_in(i) == 'dt'
                          else prb.getParameters(function.name_in(i))
                          for i in range(2, function.n_in())]
            prb.createIntermediateConstraint('native_dynamics',
                function(x, u, *parameters) - prb.getState().getVarOffset(1).getVars())
        # remove ilqr options from solver options
        filtered_opts = None
        if opts is not None:
            filtered_opts = {k: opts[k] for k in opts.keys() if not k.startswith('ilqr.')}
        super().__init__(prb, opts=filtered_opts, solver_plugin='ipopt', gauss_newton=gauss_newton)
