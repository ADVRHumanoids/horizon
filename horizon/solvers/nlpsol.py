from horizon.solvers import Solver
from horizon.problem import Problem
from typing import Dict
import casadi as cs
import numpy as np
import pprint

class NlpsolSolver(Solver):
    
    def __init__(self, prb: Problem, dt: float, opts: Dict, solver_plugin: str) -> None:
        
        super().__init__(prb, dt, opts=opts)
        
        self.solution: Dict[str:np.array] = None 
        
        # generate problem to be solver
        self.var_container = self.prb.var_container
        self.fun_container = self.prb.function_container

        self.vars_impl = dict()
        self.pars_impl = dict()

        # dictionary of implemented variables

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

        # create solver from prob
        self.solver = cs.nlpsol('solver', solver_plugin, self.prob_dict, self.opts)

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
            var_list.append(var.getImpl())

        w = cs.vertcat(*var_list) #


        # build parameters
        par_list = list()
        for par in self.var_container.getParList():
            par_list.append(par.getImpl())

        p = cs.vertcat(*par_list)


        # print(f'w {w.shape[0]}: {w}')
        # print(f'p {p.shape[0]}: {p}')

        # this is good but the problem is that, without some tampering, i get the variables repeated
        # ORDERED AS NODES
        # w = self.var_container.getVarImplList() # ordered as nodes
        # print(w)


        # or ...
        # ===================================================================================
        # ===================================================================================
        # self.vars_impl['nNone'] = dict()
        # self.pars_impl['nNone'] = dict()
        # for node in self.prb.nodes:
        #     node_name = 'n' + str(node)
        #     # get all implemented vars at node n
        #     # get all the variable implemented at node n as a list
        #     var_list_in_node = self.var_container.getVarImpl(node)
        #     self.vars_impl[node_name] = var_list_in_node # or this
        #
        # var_impl_list = self.vars_impl.values()
        # w = cs.vertcat(*var_impl_list)
        # or ...
        # ===================================================================================
        # ===================================================================================
        # for node in self.prb.nodes:
        #     node_name = 'n' + str(node)
        #     # get all implemented vars at node n
        #     # get all the variable implemented at node n as a list
        #     for name_var, value_var in self.var_container.vars.items():
        #         var_impl = value_var.getImpl(node) #doing a list of these
        #         # not important right?
        #         # var_bound_min = self.vars[name].getLowerBounds(node)
        #         # var_bound_max = self.vars[name].getUpperBounds(node)
        #         # initial_guess = self.vars[name].getInitialGuess(node)
        #         var_dict = dict(var=var_impl) #lb=var_bound_min, ub=var_bound_max, w0=initial_guess)
        #         self.vars_impl[node_name].update({name_var: var_dict})  # or this
        #
        # var_impl_list = list()
        # for vars_in_node in self.vars_impl.values():
        #     for var_abstract in vars_in_node.keys():
        #         # get from var_impl the relative var
        #
        #         var_impl_list.append(vars_in_node[var_abstract]['var'])
        #
        # w = cs.vertcat(*var_impl_list)
        # ===================================================================================
        # ===================================================================================
        # functions

        # should I do it inside function? maybe yes
        # get function from fun_container
        # fun_list = list()
        # for fname, fval in self.fun_container.cnstr_container.items():
        #     used_vars = list()
        #     # prepare variables
        #     for var_name in fval.getVariables().keys():
        #         var_impl = self.var_container.vars[var_name].getImpl(fval.getNodes())
        #         # reshape them for all-in-one evaluation of function
        #         var_impl_matrix = cs.reshape(var_impl, (fval.getDim(), len(fval.getNodes())))
        #         # generic input --> row: dim // column: nodes
        #         # [[x_0_0, x_1_0, ... x_N_0],
        #         #  [x_0_1, x_1_1, ... x_N_1]]
        #         used_vars.append(var_impl_matrix)
        #
        #     # compute function with all used variables on all active nodes
        #     fun_eval = fval.fun(*used_vars)
        #     # reshape it as a vector for solver
        #     fun_eval_vector = cs.reshape(fun_eval, (fval.getDim() * len(fval.getNodes()), 1))
        #
        #     fun_list.append(fun_eval_vector)
        #
        # g = cs.vertcat(*fun_list)
        # print(f'g ({g.shape}: {g})')
        # ===================================================================================
        # ===================================================================================
        # or ...

        # build constraint functions list
        fun_list = list()
        for fun in self.fun_container.getCnstr().values():
            fun_list.append(fun.getImpl())
        g = cs.vertcat(*fun_list)

        # build cost functions list
        fun_list = list()
        for fun in self.fun_container.getCost().values():
            fun_list.append(fun.getImpl())
        j = cs.sum1(cs.vertcat(*fun_list))

        # print(f'g ({g.shape[0]}): {g}')
        # print(f'j ({j.shape[0]}): {j}')


        return j, w, g, p


    def solve(self) -> bool:
        # update lower bounds of variables
        lb_list = list()
        for var in self.var_container.getVarList(offset=False):
            lb_list.append(var.getLowerBounds())
        lbw = cs.vertcat(*lb_list)

        # update upper bounds of variables
        ub_list = list()
        for var in self.var_container.getVarList(offset=False):
            ub_list.append(var.getUpperBounds())
        ubw = cs.vertcat(*ub_list)

        # update initial guess of variables
        w0_list = list()
        for var in self.var_container.getVarList(offset=False):
            w0_list.append(var.getInitialGuess())
        w0 = cs.vertcat(*w0_list)
        # to transform it to matrix form ---> vals = np.reshape(vals, (self.shape[0], len(self.nodes)), order='F')

        # update parameters
        p_list = list()
        for par in self.var_container.getParList():
            p_list.append(par.getValues())
        p = cs.vertcat(*p_list)

        # update lower bounds of constraints
        lbg_list = list()
        for fun in self.fun_container.getCnstr().values():
            lbg_list.append(fun.getLowerBounds())
        lbg = cs.vertcat(*lbg_list)

        # update upper bounds of constraints
        ubg_list = list()
        for fun in self.fun_container.getCnstr().values():
            ubg_list.append(fun.getUpperBounds())
        ubg = cs.vertcat(*ubg_list)

        # print(f'lbw ({lbw.shape[0]}): {lbw}')
        # print(f'ubw ({ubw.shape[0]}): {ubw}')
        # print(f'w0 ({w0.shape[0]}): {w0}')
        # print(f'p ({p.shape[0]}): {p}')
        # print(f'lbg ({lbg.shape[0]}): {lbg}')
        # print(f'ubg ({ubg.shape[0]}): {ubg}')

        # getlowerboundList(node):  to be defined in variables.py....
        #     ublist = list()
        #     for var in self.vars.values():
        #          ublist.append(var.getLowerBounds(node))
        #     return ublist
        # todo careful about ordering!
        #   this list should be ordered as the var_impl_list. Both comes from the same vars?

        # solve
        sol = self.solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=p)

        # retrieve state and input trajector
        input_vars = [v.getName() for v in self.prb.getInput().var_list]
        state_vars = [v.getName() for v in self.prb.getState().var_list]

        # get solution dict
        pos = 0
        solution_dict = dict()
        for var in self.var_container.getVarList(offset=False):
            val_sol = sol['x'][pos: pos + var.shape[0] * len(var.getNodes())]
            # this is to divide in rows the each dim of the var
            val_sol_matrix = np.reshape(val_sol, (var.shape[0], len(var.getNodes())), order='F')
            solution_dict[var.getName()] = val_sol_matrix
            pos = pos + var.shape[0] * len(var.getNodes())

        self.solution = solution_dict

        # get solution as state/input
        pos = 0
        for name, var in self.var_container.getVar().items():
            val_sol = sol['x'][pos: pos + var.shape[0] * len(var.getNodes())]
            val_sol_matrix = np.reshape(val_sol, (var.shape[0], len(var.getNodes())), order='F')
            if name in state_vars:
                off, _ = self.prb.getState().getVarIndex(name)
                self.x_opt[off:off+var.shape[0], :] = val_sol_matrix
            elif name in input_vars:
                off, _ = self.prb.getInput().getVarIndex(name)
                self.u_opt[off:off+var.shape[0], :] = val_sol_matrix
            else:
                pass
            pos = pos + var.shape[0] * len(var.getNodes())


        # print(f'{self.x_opt.shape}:, {self.x_opt}')
        # print(f'{self.u_opt.shape}:, {self.u_opt}')

        return True

    def getSolutionDict(self):
        return self.solution

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
    # x.setInitialGuess([10, 10], nodes=10)
    # a.setBounds([0, 0], [5, 5])
    #
    # p.assign([7, 7], nodes=range(0, 4))
    # p.assign([2, 2], nodes=4)
    #
    # j.assign([44])
    #
    # print(z.getUpperBounds(range(3, 5)))

    # cnsrt0 = prob.createIntermediateConstraint('cnsrt0', y[2:4] + u)
    # =========
    # cnsrt1 = prob.createIntermediateConstraint('cnsrt1', x + u)
    # cnsrt1.setLowerBounds([-np.inf, -np.inf])
    # this is new, bitches!
    # print(cnsrt1.getImpl(2)) # the constraints get implemented as soon as it get created muahahah
    # =========
    # cnsrt2 = prob.createConstraint('cnsrt2', x * y[0:2], nodes=[3, 8])
    # =========
    # cnsrt3 = prob.createConstraint('cnsrt3', x + p)
    # =========
    # cnsrt4 = prob.createConstraint('cnsrt4', x + f[0:2])
    # =========
    # cnsrt5 = prob.createConstraint('cnsrt5', p + f[0:2] + z[2:4])
    # =========
    # cnsrt6 = prob.createConstraint('cnsrt6', x + z[0:2])
    # =========
    # this should be the same
    # cnsrt7 = prob.createIntermediateConstraint('cnsrt7', x_next - x)
    # cnsrt8 = prob.createConstraint('cnsrt8', x - x_prev, nodes=range(1, N+1))
    cnsrt9 = prob.createConstraint('cnsrt9', y, nodes=N)


    # cost1 = prob.createCostFunction('cost1', x+p)
    # =========

    # todo check if everything is allright!
    for i in range(N):
        x.setLowerBounds(np.array(range(i, i+2)), nodes=i)

    p.assign([20, 20], nodes=4)
    # f.assign([121, 122, 120, 119])
    xdot = cs.vertcat(y, u)
    # xdot = cs.vertcat(u)
    prob.setDynamics(xdot)
    sol = NlpsolSolver(prb=prob, dt=dt, opts=dict(), solver_plugin='ipopt')
    sol.solve()


