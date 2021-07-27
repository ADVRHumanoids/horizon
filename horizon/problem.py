import casadi as cs
from horizon import function as fc
from horizon import nodes as nd
from horizon import state_variables as sv
import numpy as np
import logging
import sys
import pickle
import horizon.misc_function as misc

class Problem:

    def __init__(self, N, crash_if_suboptimal=False, logging_level=logging.INFO):

        self.opts = None
        self.solver = None
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(level=logging_level)
        self.debug_mode = self.logger.isEnabledFor(logging.DEBUG)
        stdout_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(stdout_handler)

        self.crash_if_suboptimal = crash_if_suboptimal

        self.nodes = N + 1
        # state variable to optimize
        self.var_container = sv.VariablesContainer(self.nodes)
        self.function_container = fc.FunctionsContainer(self.var_container, self.nodes, self.logger)
        self.prob = None

        self.state = sv.State()

    def createStateVariable(self, name, dim):
        var = self.var_container.setStateVar(name, dim)
        self.state.addVariable(var)
        return var

    def createInputVariable(self, name, dim):
        var = self.var_container.setInputVar(name, dim)
        return var

    # def setVariable(self, name, var):

    # assert (isinstance(var, (cs.casadi.SX, cs.casadi.MX)))
    # setattr(Problem, name, var)
    # self.var_container.append(name)

    # def getStateVariable(self, name):
    #
    #     for var in self.var_container:
    #         if var.getName() == name:
    #             return var
    #     return None
    def getState(self):
        return self.state

    def getInput(self):
        input = list(self.var_container.getInputVars().values())
        return input

    def _getUsedVar(self, f):
        used_var = dict()
        for name_var, value_var in self.var_container.getVarAbstrDict().items():
            used_var[name_var] = list()
            for var in value_var:
                if cs.depends_on(f, var):
                    used_var[name_var].append(var)

        return used_var

    # @classmethod
    # def createFunction(self, fun_type, **kwargs):
    #         return self.function[fun_type](**kwargs)

    def createConstraint(self, name, g, nodes=None, bounds=None):

        # get nodes as a list
        nodes = misc.checkNodes(nodes, range(self.nodes))

        # get vars that constraint depends upon
        used_var = self._getUsedVar(g)

        if self.debug_mode:
            self.logger.debug(f'Creating Constraint Function "{name}": active in nodes: {nodes}')

        # create internal representation of a constraint
        fun = fc.Constraint(name, g, used_var, nodes, bounds)

        self.function_container.addFunction(fun)

        return fun

    def createCostFunction(self, name, j, nodes=None):

        nodes = misc.checkNodes(nodes, range(self.nodes))

        used_var = self._getUsedVar(j)

        if self.debug_mode:
            self.logger.debug(f'Creating Cost Function "{name}": active in nodes: {nodes}')

        fun = fc.CostFunction(name, j, used_var, nodes)

        self.function_container.addFunction(fun)

        return fun

    def removeCostFunction(self, name):

        # if self.debug_mode:
        #     self.logger.debug('Functions before removal: {}'.format(self.costfun_container))
        self.function_container.removeFunction(name)
        # if self.debug_mode:
        #     self.logger.debug('Function after removal: {}'.format(self.costfun_container))

    def removeConstraint(self, name):
        self.function_container.removeFunction(name)

    def setNNodes(self, n_nodes):
        self.nodes = n_nodes + 1 # todo because I decided so
        self.var_container.setNNodes(self.nodes)
        self.function_container.setNNodes(self.nodes)

    def getNNodes(self) -> int:
        return self.nodes

    def createProblem(self, opts=None):

        if opts is not None:
            self.opts = opts

        # this is to reset both the constraints and the cost functions everytime I create a problem
        self.var_container.clear()
        self.function_container.clear()

        for k in range(self.nodes):
        # implement the abstract state variable with the current node
            self.var_container.update(k)

        for k in range(self.nodes):
        # implement the constraints and the cost functions with the current node
            self.function_container.update(k)

        j = self.function_container.getCostFImplSum()
        w = self.var_container.getVarImplList()
        g = self.function_container.getCnstrFList()

        # self.logger.debug('state var unraveled:', self.state_var_container.getVarImplList())
        # self.logger.debug('constraints unraveled:', cs.vertcat(*self. ...))
        # self.logger.debug('cost functions unraveled:', cs.vertcat(*self. ...))
        # self.logger.debug('cost function summed:', self.j)
        # self.logger.debug('----------------------------------------------------')

        # if self.debug_mode:
        #     self.logger.debug('cost fun: {}'.format(j))
        #     self.logger.debug('state variables: {}'.format(w))
        #     self.logger.debug('constraints: {}'.format(g))

        self.prob = {'f': j, 'x': w, 'g': g}

        if self.opts is not None:
            if "nlpsol.ipopt" in self.opts:
                if self.opts["nlpsol.ipopt"]:
                    self.solver = cs.nlpsol('solver', 'ipopt', self.prob)

    def getProblem(self):
        return self.prob

    def setSolver(self, solver):
        self.solver = solver

    def getSolver(self):
        return self.solver

    def solveProblem(self):

        # t_start = time.time()
        if self.solver is None:
            self.logger.warning('Problem is not created. Nothing to solve!')
            return 0

        self.var_container.updateBounds()
        self.var_container.updateInitialGuess()

        w0 = self.var_container.getInitialGuessList()

        if self.debug_mode:
            self.logger.debug('Initial guess vector for variables: {}'.format(self.var_container.getInitialGuessList()))


        lbw = self.var_container.getBoundsMinList()
        ubw = self.var_container.getBoundsMaxList()

        lbg = self.function_container.getLowerBoundsList()
        ubg = self.function_container.getUpperBoundsList()


        if self.debug_mode:

            j = self.function_container.getCostFImplSum()
            w = self.var_container.getVarImplList()
            g = self.function_container.getCnstrFList()

            self.logger.debug('================')
            self.logger.debug('len w: {}'.format(w.shape))
            self.logger.debug('len lbw: {}'.format(len(lbw)))
            self.logger.debug('len ubw: {}'.format(len(ubw)))
            self.logger.debug('len w0: {}'.format(len(w0)))
            self.logger.debug('len g: {}'.format(g.shape))
            self.logger.debug('len lbg: {}'.format(len(lbg)))
            self.logger.debug('len ubg: {}'.format(len(ubg)))


            # self.logger.debug('================')
            # self.logger.debug('w: {}'.format(w))
            # self.logger.debug('lbw: {}'.format(lbw))
            # self.logger.debug('ubw: {}'.format(ubw))
            # self.logger.debug('g: {}'.format(g))
            # self.logger.debug('lbg: {}'.format(lbg))
            # self.logger.debug('ubg: {}'.format(ubg))
            # self.logger.debug('j: {}'.format(j))

        # t_to_set_up = time.time() - t_start
        # print('T to set up:', t_to_set_up)
        # t_start = time.time()

        sol = self.solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # t_to_solve = time.time() - t_start
        # print('T to solve:', t_to_solve)
        # t_start = time.time()

        if self.crash_if_suboptimal:
            if not self.solver.stats()['success']:
                raise Exception('Optimal solution NOT found.')

        w_opt = sol['x'].full().flatten()

        # split solution for each variable
        solution_dict = {name: np.zeros([var.shape[0], var.getNNodes()]) for name, var in self.var_container.getVarAbstrDict(past=False).items()}

        pos = 0
        for node, val in self.var_container.getVarImplDict().items():
            for name, var in val.items():
                dim = var['var'].shape[0]
                node_number = int(node[node.index('n') + 1:])
                solution_dict[name][:, node_number] = w_opt[pos:pos + dim]
                pos = pos + dim


        # t_to_finish = time.time() - t_start
        # print('T to finish:', t_to_finish)
        return solution_dict

    def getStateVariables(self):
        return self.var_container.getVarAbstrDict()

    def getConstraints(self):
        return self.function_container.getCnstrFDict()

    def scopeNodeVars(self, node):

        return self.var_container.getVarImplAtNode(node)

    def scopeNodeConstraints(self, node):
        return self.function_container.getCnstrFImplAtNode(node)

    def scopeNodeCostFunctions(self, node):
        return self.function_container.getCostFImplAtNode(node)

    def serialize(self):

        self.var_container.serialize()
        self.function_container.serialize()


        if self.prob:

            # self.prob.clear()
            # print('serializing f (type: {}): {}'.format(type(self.prob['f']), self.prob['f']))
            # print('serializing x (type: {}): {}'.format(type(self.prob['x']), self.prob['x']))
            # print('serializing g (type: {}): {}'.format(type(self.prob['g']), self.prob['g']))

            self.prob['f'] = self.prob['f'].serialize()
            self.prob['x'] = self.prob['x'].serialize()
            self.prob['g'] = self.prob['g'].serialize()

            # print('serialized f (type: {}): {}'.format(type(self.prob['f']), self.prob['f']))
            # print('serialized x (type: {}): {}'.format(type(self.prob['x']), self.prob['x']))
            # print('serialized g (type: {}): {}'.format(type(self.prob['g']), self.prob['g']))


        return self

    def deserialize(self):

        self.var_container.deserialize()
        self.function_container.deserialize()

        if self.prob:
            self.prob['f'] = cs.Sparsity.deserialize(self.prob['f']) if self.function_container.getNCostFun() == 0 else cs.SX.deserialize(self.prob['f'])
            self.prob['x'] = cs.SX.deserialize(self.prob['x'])
            self.prob['g'] = cs.Sparsity.deserialize(self.prob['g']) if self.function_container.getNCnstrFun() == 0 else cs.SX.deserialize(self.prob['g'])

            # print('deserializing f', self.prob['f'])
            # print('deserializing x', self.prob['x'])
            # print('deserializing g', self.prob['g'])

        return self


if __name__ == '__main__':

    import horizon.utils.transcription_methods as tm
    import horizon.utils.integrators as integ
    nodes = 10
    prb = Problem(nodes, logging_level=logging.DEBUG)
    x = prb.createStateVariable('x', 2)
    v = prb.createStateVariable('v', 2)
    u = prb.createInputVariable('u', 2)

    print(x.getNNodes())
    print(u.getNNodes())

    print(prb.var_container.getVarsDim())
    danieli = prb.createConstraint('danieli', x)
    xprev = x.getVarOffset(-1)
    xprev_copy = x.getVarOffset(-1)
    xnext = x.getVarOffset(+1)

    print(id(xprev))
    print(id(xprev_copy))
    exit()
    state = prb.getState()
    state_prev = state.getVarOffset(-1)



    dt = 0.01
    state_dot = cs.vertcat(v, u)
    # opts = dict()
    # opts['tf'] = dt
    # dae = dict()
    # dae['x'] = cs.vertcat(*prb.getState())
    # dae['p'] = cs.vertcat(*prb.getInput())
    # dae['ode'] = state_dot
    # dae['quad'] = cs.sumsqr(u)

    # integrator = integ.RK4(dae, opts, cs.SX)
    hl = tm.TranscriptionsHandler(prb, 0.01, state_dot=state_dot)
    # hl.set_integrator(integrator)
    hl.setMultipleShooting()

    prb.createProblem({'nlpsol.ipopt': 10})

    prb.solveProblem()

    exit()
    # ==================================================================================================================
    # ======================================= bounds as list but also other stuff =====================================
    # ==================================================================================================================

    nodes = 10
    prb = Problem(nodes, logging_level=logging.DEBUG)
    x = prb.createStateVariable('x', 2)
    y = prb.createStateVariable('y', 2)
    z = prb.createInputVariable('z', 2)
    danieli = prb.createConstraint('danieli', x+y, range(4, 10))

    x.setBounds([2,2], [2,2])
    danieli.setBounds([12, 12],[12, 12], list(range(4, 10)))
    prb.createProblem({"nlpsol.ipopt":True})
    sol = prb.solveProblem()

    print(sol)
    exit()


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================

    # nodes = 8
    # prb = Problem(nodes)
    # x = prb.createStateVariable('x', 2)
    # y = prb.createStateVariable('y', 2)
    # danieli = prb.createConstraint('danieli', x+y)
    #
    # danieli.setBounds([12, 12],[12, 12], 4)
    # prb.createProblem()
    # sol = prb.solveProblem()
    #
    # print(sol)
    # exit()
    #
    #
    # print('===PICKLING===')
    # prb = prb.serialize()
    # prb_serialized = pickle.dumps(prb)
    #
    # print('===DEPICKLING===')
    # prb_new = pickle.loads(prb_serialized)
    # prb_new.deserialize()
    #
    # sv_new = prb_new.getStateVariables()
    # cnstr_new = prb_new.getConstraints()
    #
    # # these two are different
    # print('x', x)
    # print('new x', sv_new['x'])
    #
    # # todo how to check if the new state variable x is used by all the constraints?
    #
    # # oebus = prb_new.createConstraint('oebus', x)  # should not work
    # oebus = prb_new.createConstraint('oebus', sv_new['x'])  # should work
    # prb_new.createProblem()
    #
    # exit()
    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    # nodes = 8
    # prb = Problem(nodes)
    # x = prb.createStateVariable('x', 2)
    # y = prb.createStateVariable('y', 2)
    # # todo something wrong here
    # danieli = prb.createConstraint('danieli', x+y)
    # sucua = prb.createCostFunction('sucua', x*y)
    #
    #
    # prb.createProblem()
    #
    # print('===PICKLING===')
    #
    # prb = prb.serialize()
    # prb_serialized = pickle.dumps(prb)
    #
    #
    # print('===DEPICKLING===')
    # prb_new = pickle.loads(prb_serialized)
    # prb_new.deserialize()
    #
    # prb_new.createProblem()
    # print(prb_new.prob)
    #
    # exit()
    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================

    nodes = 8
    prb = Problem(nodes, logging_level=logging.INFO)
    x = prb.createStateVariable('x', 2)
    y = prb.createStateVariable('y', 2)

    x.setBounds([-2, -2], [2, 2])

    # todo this is allright but I have to remember that if I update the nodes (from 3 to 6 for example) i'm not updating the constraint nodes
    # todo so if it was active on all the node before, then it will be active only on the node 1, 2, 3 (not on 4, 5, 6)


    scoping_node = nodes
    # print('var at nodes {}  BEFORE creating the problem: '.format(scoping_node), prb.scopeNodeVars(scoping_node))
    # print('number of nodes of {}: {}'.format(x, x.getNNodes()))
    # print('bounds of function {} at node {} are: {}'.format(x, scoping_node, x.getBounds(scoping_node)))

    # print('getVarImplList way before:', prb.state_var_container.getVarImplList())
    danieli = prb.createConstraint('danieli', x+y)
    sucua = prb.createCostFunction('sucua', x*y, nodes=list(range(3, 15)))
    pellico = prb.createCostFunction('pellico', x-y, nodes=[0, 4, 6])

    danieli.setBounds(lb=[-1, -1], ub=[1,1], nodes=3)

    prb.createProblem({"nlpsol.ipopt":True})

    for i in range(nodes+1):
        print(x.getBounds(i))

    # print('var at nodes {}  AFTER creating the problem: '.format(scoping_node), prb.scopeNodeVars(scoping_node))
    # print('getVarImplList before:', prb.state_var_container.getVarImplList())
    new_n_nodes = 5
    # print('================== Changing n. of nodes to {} =================='.format(new_n_nodes))
    prb.setNNodes(new_n_nodes)
    scoping_node = 8
    # print('var at nodes {} AFTER changing the n. of nodes but BEFORE rebuilding: {}'.format(scoping_node, prb.scopeNodeVars(scoping_node)))
    # print('number of nodes of {}: {}'.format(x, x.getNNodes()))
    # print('bounds of function {} at node {} are: {}'.format(x, scoping_node, x.getBounds(scoping_node)))
    # print('getVarImplList after but before create:', prb.state_var_container.getVarImplList())
    prb.createProblem()

    # todo check why this is so
    # print('after:', prb.state_var_container.getVarImplList())

    scoping_node = 8
    # print('var at nodes {} AFTER changing the n. of nodes but AFTER rebuilding: {}'.format(scoping_node, prb.scopeNodeVars(scoping_node))) # should not work
    # print('number of nodes of {}: {}'.format(x, x.getNNodes()))
    # print('bounds of function {} at node {} are: {}'.format(x, scoping_node, x.getBounds(scoping_node)))
    # x.setBounds(10)
    # danieli.setNodes([1,6])
    prb.scopeNodeVars(2)


    x.setBounds([2, 8], [2, 8], 5)

    for i in range(new_n_nodes+1):
        print(x.getBounds(i))

    # todo what do I do?
    # is it better to project the abstract variable as soon as it is created, to set the bounds and everything?
    # or is it better to wait for the buildProblem to generate the projection of the abstract value along the horizon line?






