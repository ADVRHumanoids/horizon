from horizon.solvers import Solver
from horizon.problem import Problem
from horizon.functions import Cost, RecedingCost, Residual, RecedingResidual
from typing import Dict, List
import casadi as cs
import numpy as np
import pprint
class NlpsolSolver(Solver):

    def __init__(self, prb: Problem, opts: Dict, solver_plugin: str) -> None:

        super().__init__(prb, opts=opts)

        # generate problem to be solved
        self.var_container = self.prb.var_container
        self.fun_container = self.prb.function_container

        self.vars_impl = dict()
        self.pars_impl = dict()

        self.cond_warm_start = self.opts.get('ipopt.warm_start_init_point', 'no') == 'yes'
        self.lam_x0 = None
        self.lam_g0 = None

        # dictionary of implemented variables
        self.dict_sol = dict(x0=None, lbx=None, ubx=None, lbg=None, ubg=None, p=None)

        j, w, g, p = self.build()
        # implement the abstract state variable with the current node
        # self.prb.var_container.build()
        # implement the constraints and the cost functions with the current node
        # self.function_container.build()

        # get j, w, g
        # j = self.function_container.getCostFImplSum()
        # w = self.var_container.getVarImplList()
        # g = self.function_container.getCnstrFList()
        # p = self.var_container.getParameterList()

        self.prob_dict = {'f': j, 'x': w, 'g': g, 'p': p}

        self.iter_counter_callback = self.IterCountCallback('iter_counter_callback', w.shape[0], g.shape[0], p.shape[0])

        self.opts['iteration_callback'] =  self.iter_counter_callback

        # create solver from prob
        self.solver = cs.nlpsol('solver', solver_plugin, self.prob_dict, self.opts,)

    class IterCountCallback(cs.Callback):
        def __init__(self, name, nx, ng, np, opts={}, iter_counter = -1):
            cs.Callback.__init__(self)

            self.nx = nx
            self.ng = ng
            self.np = np

            self.iter_counter = iter_counter
            # Initialize internal objects
            self.construct(name, opts)

        def get_n_in(self): return cs.nlpsol_n_out()
        def get_n_out(self): return 1
        def get_name_in(self, i): return cs.nlpsol_out(i)
        def get_name_out(self, i): return "ret"

        def get_sparsity_in(self, i):
            n = cs.nlpsol_out(i)
            if n=='f':
                return cs.Sparsity. scalar()
            elif n in ('x', 'lam_x'):
                return cs.Sparsity.dense(self.nx)
            elif n in ('g', 'lam_g'):
                return cs.Sparsity.dense(self.ng)
            else:
                return cs.Sparsity(0,0)

        def eval(self, arg):

            self.iter_counter = self.iter_counter + 1

            return [0]

    def build(self):
        """
        fill the dictionary "state_var_impl"
            - key: nodes (nNone, n0, n1, ...) nNone contains single variables that are not projected in nodes
            - val: dict with name and value of implemented variable
        """

        # todo it seems tht i only need self.vars in var_container.
        # ORDERED AS VARIABLES
        # build variables
        var_list = list()
        for var in self.var_container.getVarList(offset=False):
            # x_2_3 --> dim 2 of node 3
            # order is: x_0_0, x_1_0, x_0_1, x_1_1 ...
            var_list.append(var.getImpl())
        w = cs.veccat(*var_list)

        # build parameters
        par_list = list()
        for par in self.var_container.getParList(offset=False):
            par_list.append(par.getImpl())
        p = cs.veccat(*par_list)

        # build constraint functions list
        fun_list = list()
        for fun in self.fun_container.getCnstr().values():
            fun_to_append = fun.getImpl()
            if fun_to_append is not None:
                fun_list.append(fun_to_append)
        g = cs.veccat(*fun_list)

        # todo: residual, recedingResidual should be the same class
        # treat differently cost and residual (residual must be quadratized)
        fun_list = list()
        for fun in self.fun_container.getCost().values():
            fun_to_append = fun.getImpl()
            if fun_to_append is not None:
                if type(fun) in (Cost, RecedingCost):
                    fun_list.append(fun_to_append[:])
                elif type(fun) in (Residual, RecedingResidual):
                    fun_list.append(cs.sumsqr(fun_to_append[:]))
                else:
                    raise Exception('wrong type of function found in fun_container')

        # if it is empty, just set j to []
        j = cs.sum1(cs.veccat(*fun_list)) if fun_list else []

        return j, w, g, p

    def solve(self) -> bool:

        self.iter_counter_callback.iter_counter = 0 # resetting iteration number at each solve 

        # update lower/upper bounds of variables
        lbw = self._getVarList('lb')
        ubw = self._getVarList('ub')
        # update initial guess of variables
        w0 = self._getVarList('ig')
        # update parameters
        p = self._getParList()
        # update lower/upper bounds of constraints
        lbg = self._getFunList('lb')
        ubg = self._getFunList('ub')

        # last guard
        if lbg.shape != self.prob_dict['g'].shape:
            raise ValueError(
                f'Constraint bounds have mismatching shape: {lbg.shape}. Allowed dimensions: {self.prob_dict["g"].shape}. '
                f'Be careful: if you added constraints or variables after loading the problem, you have to rebuild it before solving it!')

        # update solver arguments
        self.dict_sol['x0'] = w0
        self.dict_sol['lbx'] = lbw
        self.dict_sol['ubx'] = ubw
        self.dict_sol['lbg'] = lbg
        self.dict_sol['ubg'] = ubg
        self.dict_sol['p'] = p

        # solve
        sol = self.solver(**self.dict_sol)

        if self.cond_warm_start:
            self.dict_sol['lam_x0'] = sol['lam_x']
            self.dict_sol['lam_g0'] = sol['lam_g']

        self.cnstr_solution = self._createCnsrtSolDict(sol)

        #getting lagrange multipliers of constraints
        self.cnstr_solution['lam_x'] = np.array(sol['lam_x'])
        self.cnstr_solution['lam_g'] = np.array(sol['lam_g'])

        # retrieve state and input trajector

        # get solution dict
        self.var_solution = self._createVarSolDict(sol)

        self.var_solution["opt_cost"] = float(sol['f'])

        self.var_solution["n_iter2sol"] =  self.iter_counter_callback.iter_counter

        # get solution as state/input
        self._createVarSolAsInOut(sol)
        self.var_solution['x_opt'] = self.x_opt
        self.var_solution['u_opt'] = self.u_opt

        # build dt_solution as an array
        self._createDtSol()

        return self.solver.stats()['success']

    def getSolutionDict(self):
        return self.var_solution

    def getConstraintSolutionDict(self):
        return self.cnstr_solution

    def getDt(self):
        return self.dt_solution

    def getSolutionState(self):
        return self.var_solution['x_opt']

    def getSolutionInput(self):
        return self.var_solution['u_opt']

if __name__ == '__main__':

    # from matplotlib import pyplot as plt
    #
    # # create problem
    # N = 100
    # dt = 0.03
    # prb = Problem(N)
    #
    # # create variables
    # p = prb.createStateVariable('p', 2)
    # theta = prb.createStateVariable('theta', 1)
    # v = prb.createInputVariable('v', 1)
    # omega = prb.createInputVariable('omega', 1)
    #
    # p.setBounds([99, 99], [99, 99], nodes=50)
    # # define dynamics
    # x = prb.getState().getVars()
    # u = prb.getInput().getVars()
    # xdot = cs.vertcat(v * cs.cos(theta),
    #                   v * cs.sin(theta),
    #                   omega)
    # prb.setDynamics(xdot)
    #
    # # Cost function
    # x_tgt = np.array([1, 0, 0])
    # prb.createIntermediateCost("reg", 1e-6 * cs.sumsqr(u))
    # prb.createFinalConstraint("gothere", x - x_tgt)
    #
    # # initial state
    # x0 = np.array([0, 0, np.pi / 2])
    # prb.setInitialState(x0=x0)
    #
    # # TEST ILQR
    # sol = NlpsolSolver(prb, dt, {}, 'ipopt')
    # sol.solve()
    # print(sol.x_opt.shape)
    # print(sol.x_opt)
    # # print(sol.u_opt)

    # exit()

    N = 10
    dt = 0.01
    prob = Problem(10)
    x = prob.createStateVariable('x', 2)
    y = prob.createStateVariable('y', 4)
    u = prob.createInputVariable('u', 2)
    z = prob.createSingleVariable('z', 4)
    j = prob.createSingleParameter('j', 1)
    p = prob.createParameter('p', 2)

    z.setBounds([77, 77, 77, 77], [77, 77, 77, 77])
    x_next = x.getVarOffset(1)
    x_prev = x.getVarOffset(-1)
    f = prob.createSingleParameter('f', 4)
    #
    a = prob.createVariable('a', 2, nodes=range(0, 5))

    x.setInitialGuess([1, 1], nodes=0)
    x.setInitialGuess([10, 10], nodes=10)
    a.setBounds([0, 0], [5, 5])

    p.assign([7, 7], nodes=range(0, 4))
    p.assign([2, 2], nodes=4)

    j.assign([44])

    print(z.getUpperBounds(range(3, 5)))
    #
    cnsrt0 = prob.createIntermediateConstraint('cnsrt0', y[2:4] + u)
    ## =========
    cnsrt1 = prob.createIntermediateConstraint('cnsrt1', x + u)
    cnsrt1.setLowerBounds([-np.inf, -np.inf])
    ## this is new, bitches!
    print(cnsrt1.getImpl(2))  # the constraints get implemented as soon as it get created muahahah
    ## =========
    # cnsrt2 = prob.createConstraint('cnsrt2', x * y[0:2], nodes=[3, 8])
    ## =========
    # cnsrt3 = prob.createConstraint('cnsrt3', x + p)
    ## =========
    # cnsrt4 = prob.createConstraint('cnsrt4', x + f[0:2])
    ## =========
    # cnsrt5 = prob.createConstraint('cnsrt5', p + f[0:2] + z[2:4])
    ## =========
    # cnsrt6 = prob.createConstraint('cnsrt6', x + z[0:2])
    ## =========
    # this should be the same
    # cnsrt7 = prob.createIntermediateConstraint('cnsrt7', x_next - x)
    # cnsrt8 = prob.createConstraint('cnsrt8', x - x_prev, nodes=range(1, N+1))
    # cnsrt9 = prob.createConstraint('cnsrt9', y, nodes=N)
    #

    cost1 = prob.createCost('cost1', x + p)
    # =========

    # todo check if everything is allright!
    for i in range(N):
        x.setLowerBounds(np.array(range(i, i + 2)), nodes=i)

    p.assign([20, 20], nodes=4)
    # f.assign([121, 122, 120, 119])
    xdot = cs.vertcat(y, u)
    # xdot = cs.vertcat(u)
    prob.setDynamics(xdot)
    sol = NlpsolSolver(prb=prob, opts=dict(), solver_plugin='ipopt')
    sol.solve()
