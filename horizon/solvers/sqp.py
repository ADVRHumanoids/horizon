#try:
from .pysqp import SQPGaussNewtonSX
from horizon.solvers.pyilqr import IterativeLQR
#except ImportError:
#    print('failed to import pysqp extension; did you compile it?')
#    exit(1)

from .solver import Solver
from horizon.problem import Problem
from horizon.functions import Cost, RecedingCost, Residual, RecedingResidual
from typing import Dict
import numpy as np
import casadi as cs
import itertools


class GNSQPSolver(Solver):

    def __init__(self, prb: Problem, opts: Dict, qp_solver_plugin: str) -> None:

        filtered_opts = None 
        if opts is not None:
            filtered_opts = {k[6:]: opts[k] for k in opts.keys() if k.startswith('gnsqp.')}

        super().__init__(prb, opts=filtered_opts)

        if self.opts['qp_solver'] == 'osqp':
            if 'osqp.verbose' not in self.opts:
                self.opts['osqp.verbose'] = False

            if 'osqp.polish' not in self.opts:
                self.opts['osqp.polish'] = True

        self.prb = prb

        # generate problem to be solved
        self.var_container = self.prb.var_container
        self.fun_container = self.prb.function_container

        # generate problem to be solver
        var_list = list()
        for var in prb.var_container.getVarList(offset=False):
            var_list.append(var.getImpl())
        w = cs.veccat(*var_list)  #

        fun_list = list()
        for fun in prb.function_container.getCnstr().values():
            fun_list.append(fun.getImpl())
        g = cs.veccat(*fun_list)

        # todo: residual, recedingResidual should be the same class
        # sqp only supports residuals, warn the user otherwise
        fun_list = list()
        for fun in self.fun_container.getCost().values():
            fun_to_append = fun.getImpl()
            if fun_to_append is not None:
                if type(fun) in (Cost, RecedingCost):
                    print('warning: sqp solver does not support costs that are not residuals')
                    fun_list.append(fun_to_append[:])
                elif type(fun) in (Residual, RecedingResidual):
                    fun_list.append(fun_to_append[:])
                else:
                    raise Exception('wrong type of function found in fun_container')
        
        f = cs.veccat(*fun_list)
        
        # build parameters
        par_list = list()
        for par in self.var_container.getParList(offset=False):
            par_list.append(par.getImpl())
        p = cs.veccat(*par_list)

        # create solver from prob
        F = cs.Function('f', [w, p], [f], ['w', 'p'], ['f'])
        G = cs.Function('g', [w, p], [g], ['w', 'p'], ['g'])

        # create solver
        print(self.opts)
        self.solver = SQPGaussNewtonSX('gnsqp', qp_solver_plugin, F, G, self.opts)

        self.set_iteration_callback()


    def set_iteration_callback(self, cb=None):
        if cb is None:
            self.solver.setIterationCallback(self._iter_callback)
        else:
            self.solver.setIterationCallback(cb)

    def _iter_callback(self, fpres):
            if not fpres.accepted:
                return
            fpres.print(self.prb.getNNodes()-1)


    def solve(self) -> bool:
        
        # update lower/upper bounds of variables
        lbw = self._getVarList('lb')
        ubw = self._getVarList('ub')
        
        # update initial guess of variables
        w0 = self._getVarList('ig')
        
        # update lower/upper bounds of constraints
        lbg = self._getFunList('lb')
        ubg = self._getFunList('ub')

        # update parameters
        p = self._getParList()
        if p.size1() == 0:
            p = cs.DM([])

        # solve
        sol = self.solver.solve(x0=w0, p=p, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # self.cnstr_solution = self._createCnsrtSolDict(sol)

        # get solution dict
        self.var_solution = self._createVarSolDict(sol)
        self.var_solution['x_opt'] = self.x_opt
        self.var_solution['u_opt'] = self.u_opt

        # get solution as state/input
        self._createVarSolAsInOut(sol)

        # build dt_solution as an array
        self._createDtSol()

        return True

    def getSolutionDict(self):
        return self.var_solution

    def getConstraintSolutionDict(self):
        return self.cnstr_solution

    def getDt(self):
        return self.dt_solution

    def getHessianComputationTime(self):
        return self.solver.getHessianComputationTime()

    def getQPComputationTime(self):
        return self.solver.getQPComputationTime()

    def getLineSearchComputationTime(self):
        return self.solver.getLineSearchComputationTime()

    def getObjectiveIterations(self):
        return self.solver.getObjectiveIterations()

    def getConstraintNormIterations(self):
        return self.solver.getConstraintNormIterations()

    def setAlphaMin(self, alpha_min):
        self.solver.setAlphaMin(alpha_min)

    def getAlpha(self):
        return self.solver.getAlpha()

    def getBeta(self):
        return self.solver.getBeta()

    def setBeta(self, beta):
        self.solver.setBeta(beta)
