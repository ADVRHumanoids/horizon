import typing
from horizon.problem import Problem
from typing import Dict, Iterable
from abc import ABC, abstractmethod
import numpy as np

class Solver(ABC):

    """
    Solver is an abstract interface for a generic solver
    that aims to find a solution of an instance of
    horizon.Problem.

    The typical usage requires to (i) given an horizon.Problem,
    call the factory make_solver() to retrieve a solver concrete instance, 
    (ii) to call solve() on it, and (iii) to retrieve the solution as
    solver.x_opt, and solver.u_opt.
    """

    @classmethod
    def make_solver(cls, 
                    type: str,
                    prb: Problem, 
                    dt: float, 
                    opts: Dict = None):
        """
        Construct a solver from its type name

        Args:
            type (str): a string indicating the solver type (e.g., blocksqp, ipopt, ilqr)
            prb (Problem): the horizon's problem instance to be solved
            dt (float): the discretization step
            opts (Dict, optional): A solver-dependent Dict of options. Defaults to None.
        """

        if type == 'blocksqp':
            from . import blocksqp
            return blocksqp.BlockSqpSolver(prb, dt, opts)
        elif type == 'ipopt':
            from . import ipopt
            return ipopt.IpoptSolver(prb, dt, opts)
        elif type == 'ilqr':
            from . import ilqr
            return ilqr.SolverILQR(prb, dt, opts)
        else:
            raise KeyError(f'unsupperted solver type "{type}"')

    def __init__(self, 
                 prb: Problem, 
                 dt: float, 
                 opts: Dict = None) -> None:
        """
        Construct a solver

        Args:
            prb (Problem): the horizon's problem instance to be solved
            dt (float): the discretization step
            opts (Dict, optional): A solver-dependent Dict of options. Defaults to None.
        """
        
        # handle inputs
        if opts is None:
            opts = dict()

        self.opts = opts
        self.prb = prb
        self.dt = dt

        # save state and control
        self.x = prb.getState().getVars()
        self.u = prb.getInput().getVars()
        self.xdot = prb.getDynamics()

        # derived classes should at least provide the optimal state trajectory, 
        # and input trajectory
        self.nx = self.x.size1()
        self.nu = self.u.size1()
        self.x_opt = np.zeros(shape=(self.nx, prb.getNNodes()))
        self.u_opt = np.zeros(shape=(self.nu, prb.getNNodes()-1))

        # setup real-time iteration options
        self.rti = opts.get('realtime_iteration', False)

        if self.rti:
            self.configure_rti()
            del self.opts['realtime_iteration']
        

    @abstractmethod
    def solve(self) -> bool:
        """
        Solve the horizon's problem

        Returns:
            bool: success flag
        """
        pass 

    def configure_rti(self) -> bool:
        """
        Take appropriate actions to make the solver suitable for
        a RTI scheme

        Returns:
            bool: success flag
        """