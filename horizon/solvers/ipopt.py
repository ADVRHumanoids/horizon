from .nlpsol import NlpsolSolver
from horizon.problem import Problem
from typing import Dict
import casadi as cs

class IpoptSolver(NlpsolSolver):

    def __init__(self, prb: Problem, opts: Dict) -> None:
        # remove ilqr options from solver options
        filtered_opts = None
        if opts is not None:
            filtered_opts = {k: opts[k] for k in opts.keys() if not k.startswith('ilqr.')}
        super().__init__(prb, opts=filtered_opts, solver_plugin='ipopt')
