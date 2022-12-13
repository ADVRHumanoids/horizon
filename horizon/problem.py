import time
import warnings

import casadi as cs
from numpy.core.fromnumeric import var
from horizon import functions as fc
from horizon import variables as sv
import numpy as np
import logging
import sys
import pickle
import horizon.misc_function as misc
from typing import Union, Dict, List
# from horizon.type_doc import BoundsDict
from collections.abc import Iterable
import inspect


class Problem:
    """
    Main class of Horizon, a tool for dynamic optimization using the symbolic framework CASADI. It is a useful tool
    to describe and solve parametric non-linear problems for trajectory planning. It follows the structure of a
    generic shooting method, dividing the interval of planning [ta, tb] into a given number of shooting nodes,
    where each nodes contains decision variables (state, input ..) of the problem.

    Horizon greatly simplifies the description of the problem, allowing the user to work only with abstract definitions
    of variables and functions, which are internally managed, evaluated and projected along the optimization horizon
    by the framework.
    """

    # todo probably better to set logger, not logging_level
    def __init__(self, N: int, casadi_type=cs.MX, abstract_casadi_type=cs.SX, receding=False,
                 logging_level=logging.INFO):
        """
        Initialize the optimization problem.

        Args:
            N: number of INTERMEDIATE nodes (transitions) in the optimization horizon. IMPORTANT: the final node is automatically generated. The problem will have N+1 nodes.
            logging_level: accepts the level of logging from package logging (INFO, DEBUG, ...)
        """
        self.opts = None
        self.is_receding = receding
        self.thread_map_num = 10
        self.default_casadi_type = casadi_type
        self.default_abstract_casadi_type = abstract_casadi_type

        self.default_solver = cs.nlpsol
        self.default_solver_plugin = 'ipopt'

        self.logger = logging.getLogger('logger')
        self.logger.setLevel(level=logging_level)
        self.debug_mode = self.logger.isEnabledFor(logging.DEBUG)
        stdout_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(stdout_handler)

        self.nodes = N + 1
        # state variable to optimize
        self.var_container = sv.VariablesContainer(self.is_receding, self.logger)
        self.function_container = fc.FunctionsContainer(self.is_receding, self.thread_map_num, self.logger)

        self.state_aggr = sv.StateAggregate()
        self.input_aggr = sv.InputAggregate()

        self.state_der: Union[cs.SX, cs.MX] = None
        self.f_int: cs.Function = None
        self.dt = None

    def createStateVariable(self, name: str, dim: int, casadi_type=None, abstract_casadi_type=None) -> sv.StateVariable:
        """
        Create a State Variable active on ALL the N+1 nodes of the optimization problem.
        Remember: the State of the problem contains, in order of creation, all the State Variables created.
        Args:
            name: name of the variable
            dim: dimension of the variable

        Returns:
            instance of the State Variable

        """
        casadi_type = self.default_casadi_type if casadi_type is None else casadi_type
        abstract_casadi_type = self.default_abstract_casadi_type if abstract_casadi_type is None else abstract_casadi_type

        if self.state_der is not None:
            raise RuntimeError('createStateVariable must be called *before* setDynamics')

        # binary array to select which nodes are "active" for the variable. In this case, all of them
        nodes_array = np.ones(self.nodes).astype(int)

        var = self.var_container.setStateVar(name, dim, nodes_array, casadi_type, abstract_casadi_type)
        self.state_aggr.addVariable(var)
        return var

    def createInputVariable(self, name: str, dim: int, casadi_type=None, abstract_casadi_type=None) -> sv.InputVariable:
        """
        Create an Input Variable active on all the nodes of the optimization problem except the final one. (Input is not defined on the last node)
        Remember: the Input of the problem contains, in order of creation, all the Input Variables created.
        Args:
            name: name of the variable
            dim: dimension of the variable

        Returns:
            instance of Input Variable
        """
        casadi_type = self.default_casadi_type if casadi_type is None else casadi_type
        abstract_casadi_type = self.default_abstract_casadi_type if abstract_casadi_type is None else abstract_casadi_type

        # binary array to select which nodes are "active" for the variable. In this case, all of them
        nodes_array = np.ones(self.nodes).astype(int)
        nodes_array[-1] = 0

        var = self.var_container.setInputVar(name, dim, nodes_array, casadi_type, abstract_casadi_type)
        self.input_aggr.addVariable(var)
        return var

    def createSingleVariable(self, name: str, dim: int, casadi_type=None,
                             abstract_casadi_type=None) -> sv.SingleVariable:
        """
        Create a node-independent Single Variable of the optimization problem. It is a single decision variable which is not projected over the horizon.

        Args:
            name: name of the variable
            dim: dimension of the variable

        Returns:
            instance of Single Variable
        """
        casadi_type = self.default_casadi_type if casadi_type is None else casadi_type
        abstract_casadi_type = self.default_abstract_casadi_type if abstract_casadi_type is None else abstract_casadi_type

        nodes_array = np.ones(self.nodes).astype(int)  # dummy, cause it is the same on all the nodes

        var = self.var_container.setSingleVar(name, dim, nodes_array, casadi_type, abstract_casadi_type)
        return var

    def createVariable(self, name: str, dim: int, nodes: Iterable = None, casadi_type=None,
                       abstract_casadi_type=None) -> Union[
        sv.StateVariable, sv.SingleVariable]:
        """
        Create a generic Variable of the optimization problem. Can be specified over a desired portion of the horizon nodes.

        Args:
            name: name of the variable
            dim: dimension of the variable
            nodes: nodes the variables is defined on. If not specified, the variable is created on all the nodes.

        Returns:
            instance of Variable

        """
        # todo: right now the variable is created only in the nodes specified
        #     better to create it on every nodes anyway?

        casadi_type = self.default_casadi_type if casadi_type is None else casadi_type
        abstract_casadi_type = self.default_abstract_casadi_type if abstract_casadi_type is None else abstract_casadi_type

        nodes_array = np.ones(self.nodes).astype(int) if nodes is None else \
            misc.getBinaryFromNodes(self.nodes, misc.checkNodes(nodes, np.ones(self.nodes))).astype(int)

        var = self.var_container.setVar(name, dim, nodes_array, casadi_type, abstract_casadi_type)
        return var

    def createParameter(self, name: str, dim: int, nodes: Iterable = None, casadi_type=None,
                        abstract_casadi_type=None) -> Union[
        sv.Parameter, sv.SingleParameter]:
        """
        Create a Parameter used in the optimization problem. Can be specified over a desired portion of the horizon nodes.
        Parameters are specified before building the problem and can be 'assigned' afterwards, before solving the problem.

        Args:
            name: name of the parameter
            dim: dimension of the parameter
            nodes: nodes the parameter is defined on. If not specified, the parameter is created on all the nodes.

        Returns:
            instance of Parameter

        """
        casadi_type = self.default_casadi_type if casadi_type is None else casadi_type
        abstract_casadi_type = self.default_abstract_casadi_type if abstract_casadi_type is None else abstract_casadi_type


        nodes_array = np.zeros(self.nodes).astype(int)
        nodes_array[nodes] = 1

        par = self.var_container.setParameter(name, dim, nodes_array, casadi_type, abstract_casadi_type)
        return par

    def createSingleParameter(self, name: str, dim: int, casadi_type=None,
                              abstract_casadi_type=None) -> sv.SingleParameter:
        """
        Create a node-independent Single Parameter used to solve the optimization problem. It is a single parameter which is not projected over the horizon.
        Parameters are specified before building the problem and can be 'assigned' afterwards, before solving the problem.

        Args:
            name: name of the parameter
            dim: dimension of the parameter

        Returns:
            instance of Single Parameter

        """
        casadi_type = self.default_casadi_type if casadi_type is None else casadi_type
        abstract_casadi_type = self.default_abstract_casadi_type if abstract_casadi_type is None else abstract_casadi_type

        nodes_array = np.ones(self.nodes).astype(int)
        par = self.var_container.setSingleParameter(name, dim, nodes_array, casadi_type, abstract_casadi_type)
        return par

    def setParameter(self, par):
        assert (isinstance(par, sv.AbstractVariable))
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

    def getState(self) -> sv.StateAggregate:
        """
        Getter for the aggregate State defined for the problem. The State contains, in order, all the State Variables created.
        Returns:
            instance of State
        """
        return self.state_aggr

    def getInput(self) -> sv.InputAggregate:
        """
        Getter for the aggregate Input defined for the problem. The Input contains, in order, all the Input Variables created.

        Returns:
            instance of Input
        """
        return self.input_aggr

    def setIntegrator(self, f_int: cs.Function):
        """
        Setter for the integrator function. The integrator function is used to integrate the system dynamics.
        Args:
            f_int: integrator function
        """
        self.f_int = f_int
        if f_int.size1_in(0) != self.getState().getVars().shape[0] or \
                f_int.size1_in(1) != self.getInput().getVars().shape[0] or \
                f_int.size1_in(2) != 1:
            raise ValueError(f"Integrator function {f_int} must have the following input arguments: "
                             "state, input, time (with appropriate input sizes)")

    def getIntegrator(self) -> cs.Function:
        return self.f_int

    def setDynamics(self, xdot, integrator='RK4'):
        """
        Setter of the system Dynamics used in the optimization problem.
        Remember that the variables in "xdot" are to be ordered as the variable in the state "x"

        Args:
            xdot: derivative of the State describing the dynamics of the system
        """
        nx = self.getState().getVars().shape[0]
        if xdot.shape[0] != nx:
            raise ValueError(f'state derivative dimension mismatch ({xdot.shape[0]} != {nx})')

        self.state_der = xdot

        import horizon.transcriptions.integrators as integrators

        dae = {
            'x': self.getState().getVars(),
            'p': self.getInput().getVars(),
            'ode': self.state_der,
            'quad': 0
        }

        f_int = integrators.__dict__[integrator](dae, {}, self.default_abstract_casadi_type)

        self.setIntegrator(f_int)

    def getDynamics(self) -> cs.SX:
        """
        Getter of the system Dynamics used in the optimization problem.

        Returns:
            instance of the derivative of the state

        """
        if self.state_der is None:
            raise ValueError('dynamics not defined, have you called setDynamics?')
        return self.state_der

    def resetDynamics(self):
        self.state_der = None

    def setDt(self, dt):
        """
        Setter of the system dt used in the optimization problem.

        Args:
            dt: dt of the system
        """

        # checks on dt
        # todo check that each dt in list has one dimension only
        if isinstance(dt, List):
            print('EXPERIMENTAL: you are setting a vector of dt. Be careful!')
            if len(dt) != self.getNNodes() - 1:
                raise Exception('Wrong dimension of dt vector.')
        elif isinstance(dt, (cs.SX, cs.MX, int, float)):
            pass
        else:
            raise ValueError(f'dt of type: {type(dt)} is not supported.')

        self.dt = dt

    def getDt(self):
        """
        Getter of the system dt used in the optimization problem.

        Returns:
            instance of the dt

        """
        if self.dt is None:
            raise ValueError('dt not defined, have you called setDt?')
        return self.dt

    def setInitialState(self, x0: Iterable):
        self.getState().setBounds(lb=x0, ub=x0, nodes=0)

    def getInitialState(self) -> np.array:
        lb, ub = self.getState().getBounds(node=0)
        if np.any(lb != ub):
            return None
        return lb

    def _getUsedVar(self, f: Union[cs.SX, cs.MX]) -> list:
        """
        Finds all the variable used by a given CASADI function

        Args:
            f: function to be checked

        Returns:
            list of used variable

        """
        used_var = self._getUsedSym(f, self.var_container.getVarList(offset=True))
        return used_var

    def _getUsedPar(self, f) -> list:
        """
        Finds all the parameters used by a given CASADI function

        Args:
            f: function to be checked

        Returns:
            list of used parameters

        """
        used_par = self._getUsedSym(f, self.var_container.getParList(offset=True))
        return used_par

    def _getUsedSym(self, f, sym_list):

        used_sym = list()
        # todo add guards

        if isinstance(f, cs.SX):
            for sym in sym_list:
                if cs.depends_on(f, sym):
                    used_sym.append(sym)

        # todo is this ok?
        elif isinstance(f, cs.MX):
            sym_in_f = cs.symvar(f)
            for symbol in sym_in_f:
                for sym in sym_list:
                    if str(sym) == symbol.name():
                        used_sym.append(sym)

        return used_sym

    # def _autoNodes(self, type, nodes=None):
    #
    #     active_nodes_array = np.ones(self.nodes)
    #     if type == 'generic' or type == 'intermediate':
    #         if nodes is None:
    #             if type == 'intermediate':
    #                 active_nodes_array[-1] = 0
    #         else:
    #             active_nodes_array = misc.getBinaryFromNodes(self.nodes, nodes)
    #     elif type == 'final':
    #         # FINAL
    #         active_nodes_array[:-1] = 0
    #     else:
    #         raise Exception('type in autoNodes not recognized')
    #
    #     return active_nodes_array

    def createConstraint(self, name: str,
                         g,
                         nodes: Union[int, Iterable] = None,
                         bounds=None) -> Union[fc.Constraint, fc.RecedingConstraint]:
        """
        Create a Constraint Function of the optimization problem.

        Args:
            name: name of the constraint
            g: constraint function
            nodes: nodes the constraint is active on. If not specified, the Constraint is active on ALL the nodes.
            bounds: bounds of the constraint. If not specified, the bounds are set to zero.

        Returns:
            instance of Constraint

        """
        # todo add guards
        # todo create private function to handle the conversion from nodes to binary array of nodes:
        #   createConstraint(nodes) calls _createFun(nodes) that insides calls fc.Constraint or fc.Cost with array nodes
        if nodes is None:
            # all the nodes
            active_nodes_array = np.ones(self.nodes).astype(int)
        else:
            active_nodes_array = misc.getBinaryFromNodes(self.nodes, nodes).astype(int)

            # nodes = misc.checkNodes(nodes, range(self.nodes))

        used_var = self._getUsedVar(g)  # these now are lists!
        used_par = self._getUsedPar(g)

        if self.debug_mode:
            self.logger.debug(
                f'Creating Constraint Function "{name}": active in nodes: {misc.getNodesFromBinary(active_nodes_array)} using vars {used_var}')

        # create internal representation of a constraint
        fun = self.function_container.createConstraint(name, g, used_var, used_par, active_nodes_array, bounds)

        return fun

    def createFinalConstraint(self, name: str,
                              g,
                              bounds=None):
        """
        Create a Constraint Function only active on the last node of the optimization problem.

        Args:
            name: name of the constraint
            g: constraint function
            bounds: bounds of the constraint. If not specified, the bounds are set to zero.

        Returns:
            instance of Constraint

        """
        u = self.getInput().getVars()
        if cs.depends_on(g, u):
            raise RuntimeError(f'final constraint "{name}" must not depend on the input')
        return self.createConstraint(name, g, nodes=self.nodes - 1, bounds=bounds)

    def createIntermediateConstraint(self, name: str,
                                     g,
                                     nodes: Union[int, Iterable] = None,
                                     bounds=None):
        """
        Create a Constraint Function that can be active on all the nodes except the last one

        Args:
            name: name of the constraint
            g: constraint function
            nodes: nodes the constraint is active on. If not specified, the constraint is active on all the nodes except the last one
            bounds: bounds of the constraint. If not specified, the bounds are set to zero.

        Returns:
            instance of Constraint

        """
        if nodes is None:
            nodes = range(self.nodes - 1)

        return self.createConstraint(name, g, nodes=nodes, bounds=bounds)

    def createCost(self, name: str,
                   j,
                   nodes: Union[int, Iterable] = None):
        """
        Create a Cost Function of the optimization problem.

        Args:
            name: name of the cost function
            j: cost function
            nodes: nodes the cost function is active on. If not specified, the cost function is active on ALL the nodes.

        Returns:
            instance of Cost Function

        """
        # todo add guards
        if nodes is None:
            nodes_array = np.ones(self.nodes).astype(int)
        else:
            nodes_array = misc.getBinaryFromNodes(self.nodes, nodes).astype(int)
            # nodes = misc.checkNodes(nodes, range(self.nodes))

        used_var = self._getUsedVar(j)
        used_par = self._getUsedPar(j)

        if self.debug_mode:
            self.logger.debug(
                f'Creating Cost Function "{name}": active in nodes: {misc.getNodesFromBinary(nodes_array)}')

        fun = self.function_container.createCost(name, j, used_var, used_par, nodes_array)

        # if receding, add a weight for activating/disabling the node
        if self.is_receding:
            fun._setWeightMask(self.default_casadi_type, self.default_abstract_casadi_type)
            weight_mask = fun._getWeightMask()
            self.var_container._pars[weight_mask.getName()] = weight_mask

        return fun

    def createFinalCost(self, name: str, j):
        """
        Create a Cost Function only active on the last node of the optimization problem.

        Args:
            name: name of the cost function
            j: cost function

        Returns:
            instance of Cost Function

        """
        u = self.getInput().getVars()
        if cs.depends_on(j, u):
            raise RuntimeError(f'final cost "{name}" must not depend on the input')
        return self.createCost(name, j, nodes=self.nodes - 1)

    def createIntermediateCost(self,
                               name: str,
                               j,
                               nodes: Union[int, Iterable] = None):
        """
        Create a Cost Function that can be active on all the nodes except the last one.

        Args:
            name: name of the function
            j: function
            nodes: nodes the function is active on. If not specified, the cost function is active on all the nodes except the last one

        Returns:
            instance of Function

        """
        if nodes is None:
            nodes = range(self.nodes - 1)

        return self.createCost(name, j, nodes=nodes)

    def createResidual(self, name: str,
                       j,
                       nodes: Union[int, Iterable] = None):
        """
        Create a Residual Function of the optimization problem.

        Args:
            name: name of the function
            j: function
            nodes: nodes the function is active on. If not specified, it is active on ALL the nodes.

        Returns:
            instance of Residual Function

        """
        # todo add guards
        if nodes is None:
            nodes_array = np.ones(self.nodes).astype(int)
        else:
            nodes_array = misc.getBinaryFromNodes(self.nodes, nodes).astype(int)

        used_var = self._getUsedVar(j)
        used_par = self._getUsedPar(j)

        if self.debug_mode:
            self.logger.debug(
                f'Creating Residual Function "{name}": active in nodes: {misc.getNodesFromBinary(nodes_array)}')

        fun = self.function_container.createResidual(name, j, used_var, used_par, nodes_array)

        # if receding, add a weight for activating/disabling the node
        if self.is_receding:
            fun._setWeightMask(self.default_casadi_type, self.default_abstract_casadi_type)
            weight_mask = fun._getWeightMask()
            self.var_container._pars[weight_mask.getName()] = weight_mask

        return fun

    def createFinalResidual(self, name: str, j):
        """
        Create a Residual Function only active on the last node of the optimization problem.

        Args:
            name: name of the residual function
            j: function

        Returns:
            instance of Residual Function

        """
        u = self.getInput().getVars()
        if cs.depends_on(j, u):
            raise RuntimeError(f'final residual "{name}" must not depend on the input')
        return self.createResidual(name, j, nodes=self.nodes - 1)

    def createIntermediateResidual(self,
                                   name: str,
                                   j,
                                   nodes: Union[int, Iterable] = None):
        """
        Create a Residual Function that can be active on all the nodes except the last one.

        Args:
            name: name of the function
            j: function
            nodes: nodes the function is active on. If not specified, the function is active on all the nodes except the last one

        Returns:
            instance of Function

        """
        if nodes is None:
            nodes = range(self.nodes - 1)

        return self.createResidual(name, j, nodes=nodes)

    def removeVariable(self, name: str) -> bool:
        """
        remove the desired variable.

        Args:
            name: name of the cost function to be removed

        Returns: False if function name is not found.

        """
        # todo add also remove from variable instance not only name
        return self.var_container.removeVar(name)

    def removeCostFunction(self, name: str) -> bool:
        """
        remove the desired cost function.

        Args:
            name: name of the cost function to be removed

        Returns: False if function name is not found.

        """
        # if self.debug_mode:
        #     self.logger.debug('Functions before removal: {}'.format(self.costfun_container))
        return self.function_container.removeFunction(name)
        # if self.debug_mode:
        #     self.logger.debug('Function after removal: {}'.format(self.costfun_container))

    def removeConstraint(self, name: str) -> bool:
        """
        remove the desired constraint.

        Args:
            name: name of the constraint function to be removed

        Returns: False if function name is not found.

        """
        return self.function_container.removeFunction(name)

    # def setNNodes(self, n_nodes: int):
    #     """
    #     set a desired number of nodes of the optimization problem.
    #
    #     Args:
    #         n_nodes: new number of nodes
    #
    #     """
    #     self.nodes = n_nodes + 1  # todo because I decided so
    #     self.var_container.setNNodes(self.nodes)
    #     self.function_container.setNNodes(self.nodes)

    def getNNodes(self) -> int:
        """
        Getter for the number of nodes of the optimization problem.

        Returns:
            the number of optimization nodes
        """
        return self.nodes

    def modifyNodes(self, nodes_map):
        """
        get a map of old_node: number of new_nodes
        """
        from horizon.variables import SingleVariable, InputVariable, StateVariable, Variable
        old_to_new = dict()
        # number of supplementary node inserted
        n_supplementary_nodes = 0
        for old_node in range(self.getNNodes()):
            old_to_new[old_node] = old_node + n_supplementary_nodes
            if old_node in nodes_map.keys():
                # if in map, add node + supplementary nodes
                n_supplementary_nodes += nodes_map[old_node]

        # last new node + 1 is the number of total new nodes
        self.nodes = old_to_new[self.getNNodes() - 1] + 1

        # MODIFY VARIABLES NODES
        for var in self.var_container._vars.values():
            print(f'modifying var {var.getName()}')
            print('old_nodes:', var.getNodes())

            node_array = self.__array_from_node_map(var.getNodes(), old_to_new, nodes_map)
            var._setNodes(node_array)

            if hasattr(var, 'var_offset'):
                for val in var.var_offset.values():
                    val._impl = var._impl
                    val._nodes_array = node_array

            print('new_nodes:', var.getNodes())

        # MODIFY PARAMETER NODES
        for par in self.var_container._pars.values():
            print(f'modifying par {par.getName()}')
            print('old_nodes:', par.getNodes())

            node_array = self.__array_from_node_map(par.getNodes(), old_to_new, nodes_map)
            par._setNodes(node_array)

            if hasattr(par, 'var_offset'):
                for val in par.var_offset.values():
                    val._impl = par._impl
                    val._nodes_array = node_array

            print('new_nodes:', par.getNodes())

        # MODIFY FUNCTIONS NODES
        for name, cnsrt in self.getConstraints().items():
            print(f'========================== constraint {name} =========================================')
            # old_n = self.old_cnsrt_nodes[name]
            # old_lb, old_ub = self.old_cnrst_bounds[name]

            print('old nodes:', cnsrt.getNodes())
            # todo
            #  if constraint depends on dt, what to do?
            #  if it is a variable, it is ok. Can be changed and recognized easily.
            #  What if it is a constant?
            #  I have to change that constant value to the new value (old dt to new dt).
            #  a possible thing is that i "mark" it, so that I can find it around.
            #  Otherwise it would be impossible to understand which constraint depends on dt?
            feas_node_array = self.__array_from_node_map(misc.getNodesFromBinary(cnsrt._getFeasNodes()), old_to_new, nodes_map)
            node_array = self.__array_from_node_map(cnsrt.getNodes(), old_to_new, nodes_map)

            cnsrt._setFeasNodes(feas_node_array)
            cnsrt.setNodes(misc.getNodesFromBinary(node_array))
            print('new nodes:', cnsrt.getNodes())

        for name, cost in self.getCosts().items():
            print(f'========================== cost {name} =========================================')
            # old_n = self.old_cnsrt_nodes[name]
            # old_lb, old_ub = self.old_cnrst_bounds[name]

            print('old nodes:', cost.getNodes())

            feas_node_array = self.__array_from_node_map(misc.getNodesFromBinary(cost._getFeasNodes()), old_to_new,
                                                         nodes_map)
            node_array = self.__array_from_node_map(cost.getNodes(), old_to_new, nodes_map)

            cost._setFeasNodes(feas_node_array)
            cost.setNodes(misc.getNodesFromBinary(node_array))
            print('new nodes:', cost.getNodes())

    def __array_from_node_map(self, old_nodes, old_to_new, nodes_map):

        node_array = np.zeros(self.nodes)
        for old_node in old_nodes:
            new_node = old_to_new[old_node]
            node_array[new_node] = 1

            if old_node in nodes_map.keys() and old_node + 1 in old_nodes:
                node_array[new_node + 1: new_node + nodes_map[old_node] + 1] = 1

        return node_array

    def getVariables(self, name: str = None):
        """
        Getter for a desired variable of the optimization problem.

        Args:
            name: name of the desired variable. If not specified, a dict with all the variables is returned

        Returns:
            the desired variable/s
        """
        var = self.var_container.getVar(name)

        return var

    def getParameters(self, name: str = None):
        """
        Getter for a desired parameter of the optimization problem.

        Args:
            name: name of the desired parameter. If not specified, a dict with all the parameters is returned

        Returns:
            the desired parameter/s
        """
        par = self.var_container.getPar(name)

        return par

    def getConstraints(self, name=None):
        """
        Getter for a desired constraint of the optimization problem.

        Args:
            name: name of the desired constraint. If not specified, a dict with all the constraint is returned

        Returns:
            the desired constraint/s
        """
        fun = self.function_container.getCnstr(name)

        return fun

    def getCosts(self, name=None):
        """
        Getter for a desired constraint of the optimization problem.

        Args:
            name: name of the desired constraint. If not specified, a dict with all the constraint is returned

        Returns:
            the desired constraint/s
        """
        fun = self.function_container.getCost(name)

        return fun

    def evalFun(self, fun: fc.Function, solution):
        """
        Evaluates a given function over the solution found.

        Args:
            fun: function to evaluate

        Returns:
            fun evaluated at all nodes using the solution of horizon problem
        """
        fun_to_evaluate = fun.getFunction()
        all_vars = list()
        for var in fun.getVariables():
            var_name = var.getName()
            # careful about ordering
            # todo this is very ugly, but what can I do (wanted to do it without the if)
            if True or isinstance(var, sv.SingleVariable):
                all_vars.append(solution[var_name])
            else:
                # this is required because:
                # I retrieve from the solution values only the nodes that the function is active on.
                # suppose a function is active on nodes [3, 4, 5, 6].
                # suppose a variable is defined on nodes [0, 1, 2, 3, 4, 5, 6].
                #   its solution will be [a, b, c, d, e, f, g]. I'm interested in the values at position [3, 4, 5, 6]
                # suppose a variable is defined only on nodes [3, 4, 5, 6].
                #   its solution will be [a, b, c, d]. I'm interested in the values at position [0, 1, 2, 3], which corresponds to nodes [3, 4, 5, 6]
                node_index = [i for i in range(len(var.getNodes())) if
                              var.getNodes()[i] in np.array(fun.getNodes()) + var.getOffset()]
                all_vars.append(solution[var_name][:, node_index])

        all_pars = list()
        for par in fun.getParameters():
            # careful about ordering
            # todo this is very ugly, but what can I do (wanted to do it without the if)
            if True or isinstance(par, sv.SingleParameter):
                all_pars.append(par.getValues())
            else:
                par_matrix = np.reshape(par.getValues(), (par.getDim(), len(par.getNodes())), order='F')
                all_pars.append(par_matrix[:, fun.getNodes()])

        fun_evaluated = fun_to_evaluate(*(all_vars + all_pars)).toarray()
        return fun_evaluated

    def toParameter(self, var_name):

        warnings.warn('EXPERIMENTAL FUNCTION: toParameter')
        # TODO isn't there a way to change just the variable and everything else changes accordingly?
        # check if name of variable exists
        if var_name not in self.getVariables().keys():
            raise Exception(f'variable {var_name} not recognized.')

        # substitute variable with parameter
        par = self.var_container._vars.pop(var_name).toParameter()
        self.var_container._pars[var_name] = par

        # if the variable is also the dt, set new dt
        if var_name == self.getDt().getName():
            if isinstance(self.getDt(), List):
                raise NotImplementedError('tbd')
            else:
                self.setDt(self.getParameters(var_name))

        # dt = self.getDt()
        # if isinstance(dt, List):
        #     for i in range(len(dt)):
        #         if isinstance(dt[i], (sv.Variable, sv.SingleVariable)):
        #             if dt[i].getName() == var_name:
        #                 dt[i] = par

        if var_name in [input_var.getName() for input_var in self.getInput()]:
            self.getInput().removeVariable(var_name)
        #     self.getInput().addVariable(par)
        #
        if var_name in [state_var.getName() for state_var in self.getState()]:
            self.getState().removeVariable(var_name)
        #     self.getState().addVariable(par)

        # modify constraints on their core (correct?)
        for fun in self.getConstraints().values():
            for i in range(len(fun.vars)):
                var_index_to_remove = list()
                if fun.vars[i].getName() == var_name:
                    # get the right offset of the parameter, if present
                    var_index_to_remove.append(i)
                    fun.pars.append(par.getParOffset(fun.vars[i].getOffset()))

                for index in sorted(var_index_to_remove, reverse=True):
                    del fun.vars[index]

                fun._project()

        # modify constraints on their core (correct?)
        for fun in self.getCosts().values():
            for i in range(len(fun.vars)):
                var_index_to_remove = list()
                if fun.vars[i].getName() == var_name:
                    # get the right offset, if present
                    var_index_to_remove.append(i)
                    fun.pars.append(par.getParOffset(fun.vars[i].getOffset()))

                for index in sorted(var_index_to_remove, reverse=True):
                    del fun.vars[index]
                # if fun.vars[i].getName() == var_name:
                #     fun.vars[i] = par

                fun._project()

        # finally, change integrator dimension, rebuilding the dynamics
        self.setDynamics(self.getDynamics())

        # for fun in self.getConstraints().values():
        #     for fun_var in fun.getVariables():
        #         if fun_var.getName() == var_name:
        #             print(f'found reference of {var_name} in {fun.getName()}')
        #             f_name = fun.getName()
        #             f_nodes = fun.getNodes()
        #             self.removeConstraint(fun.getName())
        #             self.createConstraint(f_name, fun._f, f_nodes)

    def scopeNodeVars(self, node: int):
        """np.where(pos_nodes
        Scope the variables active at the desired node of the optimization problem.

        Args:
            node: desired node to scope

        Returns:
            all the active variables at the desired node
        """
        raise Exception('scopeNodeVars yet to be re-implemented')
        return self.var_container.getVarImplAtNode(node)

    def scopeNodeConstraints(self, node):
        """
        Scope the constraints active at the desired node of the optimization problem.

        Args:
            node: desired node to scope

        Returns:
            all the active constraint at the desired node
        """
        raise Exception('scopeNodeConstraint yet to be re-implemented')
        return self.function_container.getCnstrFImplAtNode(node)

    def scopeNodeCostFunctions(self, node):
        """
        Scope the cost functions active at the desired node of the optimization problem.

        Args:
            node: desired node to scope

        Returns:
            all the active cost functions at the desired node
        """
        raise Exception('scopeNodeCostFunctinos yet to be re-implemented')
        return self.function_container.getCostFImplAtNode(node)

    def serialize(self):
        """
        Serialize this class. Used for saving it.

        Returns:
            instance of the serialized class "Problem"
        """
        raise Exception('serialize yet to implement')
        # self.var_container.serialize()
        self.function_container.serialize()
        if self.state_der is not None:
            self.state_der = self.state_der.serialize()

        return self

    def deserialize(self):
        """
        Deserialize this class. Used for loading it.

        Returns:
            instance of the deserialized class "Problem"
        """
        raise Exception('serialize yet to implement')
        # self.var_container.deserialize()
        self.function_container.deserialize()
        if self.state_der is not None:
            self.state_der = cs.SX.deserialize(self.state_der)

        return self

    def save(self):
        data = dict()

        data['n_nodes'] = self.getNNodes() - 1

        # save state variables
        data['state'] = list()
        for sv in self.getState():
            var_data = dict()
            var_data['name'] = sv.getName()
            var_data['size'] = sv.size1()
            var_data['lb'] = sv.getLowerBounds().flatten('F').tolist()
            var_data['ub'] = sv.getUpperBounds().flatten('F').tolist()
            var_data['initial_guess'] = sv.getInitialGuess().flatten('F').tolist()
            data['state'].append(var_data)

        # save input variables
        data['input'] = list()
        for sv in self.getInput():
            var_data = dict()
            var_data['name'] = sv.getName()
            var_data['size'] = sv.size1()
            var_data['lb'] = sv.getLowerBounds().flatten('F').tolist()
            var_data['ub'] = sv.getUpperBounds().flatten('F').tolist()
            var_data['initial_guess'] = sv.getInitialGuess().flatten('F').tolist()
            data['input'].append(var_data)

        # save parameters
        data['param'] = dict()
        for p in self.var_container.getParList():
            var_data = dict()
            var_data['name'] = p.getName()
            var_data['size'] = p.getDim()
            var_data['values'] = p.getValues().flatten('F').tolist()
            data['param'][var_data['name']] = var_data

        # save cost and constraints
        data['cost'] = dict()
        for f in self.function_container.getCost().values():
            f: fc.Function = f
            var_data = dict()
            var_data['name'] = f.getName()
            var_data['repr'] = str(f.getFunction())
            var_data['var_depends'] = [v.getName() for v in f.getVariables()]
            var_data['param_depends'] = [v.getName() for v in f.getParameters()]
            var_data['nodes'] = f.getNodes()
            var_data['function'] = f.getFunction().serialize()
            data['cost'][var_data['name']] = var_data

        data['constraint'] = dict()
        for f in self.function_container.getCnstr().values():
            f: fc.Function = f
            var_data = dict()
            var_data['name'] = f.getName()
            var_data['repr'] = str(f.getFunction())
            var_data['var_depends'] = [v.getName() for v in f.getVariables()]
            var_data['param_depends'] = [v.getName() for v in f.getParameters()]
            var_data['nodes'] = f.getNodes()
            var_data['function'] = f.getFunction().serialize()
            var_data['lb'] = f.getLowerBounds().flatten('F').tolist()
            var_data['ub'] = f.getUpperBounds().flatten('F').tolist()
            data['constraint'][var_data['name']] = var_data

        return data


def pickleable(obj):
    try:
        pickle.dumps(obj)
    except pickle.PicklingError:
        return False
    return True

if __name__ == '__main__':

    import pickle
    from transcriptions import transcriptor
    from horizon.solvers import Solver
    from horizon.utils import plotter
    import matplotlib.pyplot as plt

    N = 10
    nodes_vec = np.array(range(N + 1))  # nodes = 10
    dt = 0.01
    prb = Problem(N, receding=True, casadi_type=cs.SX)
    x = prb.createStateVariable('x', 2)
    y = prb.createInputVariable('y', 2)
    mimmo = prb.createIntermediateConstraint('cost_y', y - x)

    print(mimmo.getNodes())
    mimmo.setBounds([-np.inf, -np.inf], [np.inf, np.inf], nodes=4)
    mimmo.setBounds([0, 1], [0, 1], nodes=4)
    print(mimmo.getNodes())

    exit()
    # ============================================================
    # ============================================================
    # ============================================================

    N = 10
    nodes_vec = np.array(range(N + 1))  # nodes = 10
    dt = 0.01
    prb = Problem(N, receding=True, casadi_type=cs.SX)
    x = prb.createStateVariable('x', 2)
    y = prb.createInputVariable('y', 2)
    dan = prb.createParameter('dan', 2)
    x.setBounds([-2, -2], [2, 2])
    y.setBounds([-5, -5], [5, 5])
    prb.createCost('cost_x', x)
    mimmo = prb.createIntermediateConstraint('cost_y', y - dan)

    print(mimmo.getUpperBounds())
    print(mimmo.getLowerBounds())
    exit()

    print(dan.getValues())
    print(dan.getNodes())
    dan.assign([[2, 3, 4]], [])
    print(dan.getValues())
    exit()
    # for i in range(500):
    #     cnsrt = prb.createConstraint(f'cnsrt_{i}', x - i * y, nodes=[])
    #     print(cnsrt.getBounds())

    prb.setDynamics(x)
    prb.setDt(dt)

    opts = dict()
    opts['ipopt.linear_solver'] = 'ma27'
    # opts['ipopt.check_derivatives_for_naninf'] = 'yes'
    # opts['ipopt.jac_c_constant'] = 'yes'
    # opts['ipopt.jac_d_constant'] = 'yes'
    # opts['ipopt.hessian_constant'] = 'yes'
    solv = Solver.make_solver('ipopt', prb, opts)
    tic = time.time()
    solv.solve()
    toc = time.time() - tic
    print(toc)
    print(solv.getSolutionDict()['x'])
    exit()
    # N = 3
    # nodes_vec = np.array(range(N+1))    # nodes = 10
    # dt = 0.01
    # prb = Problem(nodes, crash_if_suboptimal=True, receding=True)
    # x = prb.createStateVariable('x', 6)
    # u = prb.createInputVariable('u', 2)
    # p = prb.createSingleParameter('p', 6)
    #
    # prb.setDynamics(x)
    # prb.setDt(dt)
    # x.setBounds([1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2])
    #
    # constr1 = prb.createIntermediateConstraint('constr', x[2:4] ** 2 + p[2:4] - u)
    #
    # constr1.setBounds([1, 1], [1, 1])
    # solver = Solver.make_solver('ipopt', prb)
    #
    # all_sol = dict()
    # for i in range(100):
    #     p.assign(6 * [2 * i])
    #     solver.solve()
    #     sol = solver.getSolutionDict()
    #     all_sol[i] = sol
    #
    # exit()
    N = 10
    nodes_vec = np.array(range(N + 1))  # nodes = 10
    dt = 0.01
    prb = Problem(N, receding=True, casadi_type=cs.SX)
    x = prb.createStateVariable('x', 2)
    y = prb.createInputVariable('y', 2)
    par = prb.createParameter('par', 2)

    cost = prb.createCost('cost', x, nodes=[3, 4, 5])

    y_prev = y.getVarOffset(-1)
    prb.setDynamics(x)
    prb.setDt(dt)

    print(par.getValues())
    print('assigning new values ...')
    par.assign([1, 2], nodes=[3, 4, 5, 6, 7, 8])
    print(par.getValues())
    print('shifting ...')
    par.shift()
    print(par.getValues())
    print('assigning new values ...')
    par.assign([3, 3], nodes=10)
    print(par.getValues())
    print('shifting ...')
    par.shift()
    print(par.getValues())
    exit()

    x.setUpperBounds([2, 3], nodes=[4, 5, 6])
    print(x.getUpperBounds())
    print('shifting ...')
    x.shift()
    print(x.getUpperBounds())
    print('setting new bounds ...')
    x.setUpperBounds([1, 2], nodes=10)
    print(x.getUpperBounds())
    print('shifting ...')
    x.shift()
    print('shifting ...')
    x.shift()
    print(x.getUpperBounds())
    exit()

    # p1 = prb.createParameter('p1', 4)
    # cnsrt = prb.createConstraint('cnsrt', x - y_prev, nodes=[1])
    # cnsrt = prb.createConstraint('cnsrt', x, nodes=[1])
    # cost1 = prb.createCost('cost_x', 1e-3 * cs.sumsqr(x))
    # cost2 = prb.createIntermediateCost('cost_y', 1e-3 * cs.sumsqr(y))
    c = prb.createConstraint('c1', x[0], nodes=[3, 4, 5, 6])

    print(c.getNodes())
    c.setLowerBounds(2)
    print(c.getLowerBounds())
    print(c.getUpperBounds())
    # c.setNodes([2, 3, 4, 5],  erasing=True)
    print('shifting ...')
    c.shift()
    print('shifting ...')
    c.shift()
    print('shifting ...')
    c.shift()
    print('adding new nodes ...')
    c.setNodes([9, 10])
    c.setLowerBounds(-5, nodes=[9])
    c.setLowerBounds(-10, nodes=[10])
    print(c.getLowerBounds())
    print(c.getUpperBounds())
    print('shifting ...')
    c.shift()
    print(c.getLowerBounds())
    print(c.getUpperBounds())
    print('changing bounds ...')
    c.setLowerBounds(-23, nodes=[8])
    c.setLowerBounds(-25, nodes=[9])
    print(c.getLowerBounds())
    print(c.getUpperBounds())
    print('shifting ...')
    c.shift()
    print(c.getLowerBounds())
    print(c.getUpperBounds())
    exit()
    x.setInitialGuess([5, 5])
    # print(x.getInitialGuess())

    x.setBounds([-10, -10], [10, 10])
    x.setBounds([7, 7], [7, 7], nodes=2)
    print(x.getLowerBounds())
    print(x.getUpperBounds())
    opts = dict()
    opts['ipopt.tol'] = 1e-12
    opts['ipopt.constr_viol_tol'] = 1e-12
    slvr = Solver.make_solver('ipopt', prb, opts=opts)
    slvr.solve()
    sol = slvr.getSolutionDict()

    print('x:\n', sol['x'])
    # print('y:\n', sol['y'])
    old_sol = sol.copy()
    print('=========== receding the horizon: ==============')

    print('old initial guess: ')
    print(x.getInitialGuess())
    new_ig_elem = [3, 2]
    x.setInitialGuess(sol['x'][:, 1:], nodes=nodes_vec[:-1])
    x.setInitialGuess(new_ig_elem, nodes=N)
    print('new initial guess: ')
    print(x.getInitialGuess())

    print('old bounds: ')
    print(x.getLowerBounds())
    print(x.getUpperBounds())

    # === shifting algorithm ===
    shifted_lb = x.getLowerBounds()[:, 1:]
    shifted_ub = x.getUpperBounds()[:, 1:]
    new_lb_elem = np.array([3, 3])
    new_ub_elem = np.array([3, 3])

    print('new bounds: ')
    x.setLowerBounds(shifted_lb, nodes=nodes_vec[:-1])
    x.setUpperBounds(shifted_ub, nodes=nodes_vec[:-1])
    x.setLowerBounds(new_lb_elem, nodes=N)
    x.setUpperBounds(new_ub_elem, nodes=N)

    print(x.getLowerBounds())
    print(x.getUpperBounds())

    print('manage constraints')
    print('old nodes:')
    nodes_cnsrt = c.getNodes()
    print(nodes_cnsrt)
    print(c.getImpl())
    c.setNodes(nodes_cnsrt - 1, erasing=True)
    print('new nodes:')
    print(c.getNodes())

    print(c.getImpl())
    print(x.setBounds([2, 2], [2, 2], 6))
    slvr.solve()
    sol = slvr.getSolutionDict()

    print('old x:\n', old_sol['x'])
    print('x:\n', sol['x'])
    # print('y:\n', sol['y'])

    # np.concatenate((shifted_lb, np.reshape(new_lb, (2, 1))))
    # x.getUpperBounds()[1:]
    # x.setBounds(, )
    exit()

    # print('before', prob.var_container._vars)
    # print('before', prob.var_container._pars)
    # print('before:', [elem.getFunction() for elem in prob.function_container._cnstr_container.values()])
    # print('before:', [elem.getFunction() for elem in prob.function_container._costfun_container.values()])

    # for fun in prob.function_container._cnstr_container.values():
    #     print(f"does {fun._f} depends on {prob.var_container._vars['y']}: {cs.depends_on(fun._f, prob.var_container._vars['y'])}")
    #
    # prob.serialize()
    # print('===PICKLING===')
    # prob_serialized = pickle.dumps(prob)
    # print('===DEPICKLING===')
    # prob_new = pickle.loads(prob_serialized)
    # prb = prob_new.deserialize()
    #
    # for fun in prb.function_container._cnstr_container.values():
    #     print(f"does {fun._f} depends on {prb.var_container._vars['y']}: {cs.depends_on(fun._f, prb.var_container._vars['y'])}")
    #
    # exit()
    #
    # transcriptor.Transcriptor.make_method('multiple_shooting', prb, dt)
    # sol = Solver.make_solver('ipopt', prb, dt)
    # sol.solve()
    # solution = sol.getSolutionDict()
    #
    #
    # exit()
    #
    # N = 10
    # dt = 0.01
    # prob = Problem(10)
    # x = prob.createStateVariable('x', 1)
    # y = prob.createVariable('y', 1, nodes=range(5, 11))
    #
    # cnsrt = prob.createConstraint('cnsrt', x+y, nodes=range(5, 11))
    #
    # xdot = cs.vertcat(x)
    # prob.setDynamics(xdot)
    #
    # sol = Solver.make_solver('ipopt', prob, dt)
    # sol.solve()
    # solution = sol.getSolutionDict()
    #
    # print(solution)
    # hplt = plotter.PlotterHorizon(prob, solution)
    # hplt.plotVariables()
    # hplt.plotFunctions()
    #
    # plt.show()
