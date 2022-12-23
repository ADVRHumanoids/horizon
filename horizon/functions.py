import casadi as cs
import numpy as np
from horizon import misc_function as misc
from horizon import variables as sv
from collections import OrderedDict
import pickle
import time
from typing import Union, Iterable
import itertools

# from horizon.type_doc import BoundsDict

default_thread_map = 10

# todo think about a good refactor for the RecedingCost:
#  it requires a parameter, weight_mask, to activate/disable the cost on the desired nodes
#     1. the parameter can be easily generated from the problem.py, but then it needs to be injected in the CostFunction
#        this is good because the problem.py has also the knowledge of the casadi type
#        RecedingCost(..., Param)
#        ._setWeightMask() if constructed AFTER the function creation, should override _f and _fun
#    2. the parameter can be generated from inside the function, and then added to the var_container used by the problem.py
#       RecedingCost(...)
#       par = fun._getWeightMask() and var_container.add(par)
#  I would like to find a way to instantiate the receding cost with everything in place already:
#

def getRanges(i):
    for a, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
        b = list(b)
        yield b[0][1], b[-1][1]

class AbstractFunction:
    """
        Function of Horizon: generic function of SX values from CASADI.

        Notes:
            This function is an abstract representation of its projection over the nodes of the optimization problem.
            An abstract function gets internally implemented at each node, using the variables at that node.
    """

    def __init__(self, name: str, f: Union[cs.SX, cs.MX], used_vars: list, used_pars: list, active_nodes_array: np.ndarray):
        """
        Initialize the Horizon Function.

        Args:
            name: name of the function
            f: SX or MX function
            used_vars: variable used in the function
            used_pars: parameters used in the function
            active_nodes_array: binary array specifying the nodes the function is active on
        """
        self._f = f
        self._name = name
        self._active_nodes_array = active_nodes_array.copy()
        self._feas_nodes_array = active_nodes_array.copy()
        self._n_feas_nodes = np.sum(self._feas_nodes_array).astype(int)
        # todo isn't there another way to get the variables from the function g?
        # todo these are copies, but it is wrong, they should be exactly the objects pointed in var_container
        self.vars = used_vars
        self.pars = used_pars

        # create function of CASADI, dependent on (in order) [all_vars, all_pars]
        all_input = self.vars + self.pars
        all_names = [f'{i.getName()}_{str(i.getOffset())}' for i in all_input]

        self._fun = cs.Function(name, self.vars + self.pars, [self._f], all_names, ['f'])

    def getName(self) -> str:
        """
        Getter for the name of the function

        Returns:
            name of the function
        """
        return self._name

    def getDim(self) -> int:
        """
        Getter for the dimension of the function

        Returns:
            dimension of the function
        """
        return self._f.shape[0]

    def getFunction(self) -> cs.Function:
        """
        Getter for the CASADI function

        Returns:
            instance of the CASADI function
        """
        return self._fun

    def getNodes(self) -> list:
        """
        Getter for the active nodes of the function.

        Returns:
            a list of the nodes where the function is active

        """
        return misc.getNodesFromBinary(self._active_nodes_array)

    def setNodes(self, nodes, erasing=False):
        """
        Setter for the active nodes of the function.

        Args:
            nodes: list of desired active nodes.
            erasing: choose if the inserted nodes overrides the previous active nodes of the function. 'False' if not specified.
        """
        pos_nodes = misc.convertNodestoPos(nodes, self._feas_nodes_array)
        # todo check for repetition in setted nodes? otherwise this does not work
        # todo this way of checking is not robust
        if len(pos_nodes) != len(nodes):
            feas_nodes = misc.getNodesFromBinary(self._feas_nodes_array)
            raise Exception(f'You are trying to set nodes of the function where it is NOT defined. Available nodes: {feas_nodes}. If you want to change the nodes freely in the horizon, use the receding mode.')

        if erasing:
            self._active_nodes_array[:] = 0

        self._active_nodes_array[nodes] = 1

    def _getFeasNodes(self):
        """
        advanced method.
        Get the FEASIBLE node array of the function.
        """
        return self._feas_nodes_array.copy()
    def _setFeasNodes(self, feas_nodes_array):
        """
        Caution, advanced method.
        Change the FEASIBLE node array of the function.
        Reset the ACTIVE nodes.
        """
        self._feas_nodes_array = feas_nodes_array.copy()
        self._active_nodes_array = feas_nodes_array.copy()

        self._n_feas_nodes = np.sum(self._feas_nodes_array).astype(int)

    def getVariables(self, offset=True) -> list:
        """
        Getter for the variables used in the function.

        Returns:
            a dictionary of all the used variable. {name: cs.SX or cs.MX}
        """
        if offset:
            ret = self.vars
        else:
            ret = list()
            for var in self.vars:
                if var.getOffset() == 0:
                    ret.append(var)

        return ret

    def getParameters(self) -> list:
        """
        Getter for the parameters used in the function.

        Returns:
            a dictionary of all the used parameters. {name: cs.SX or cs.MX}
        """
        return self.pars

    def _projectNodes(self, thread_map_num):
        """
        Implements the function at the desired node using the desired variables.

        Args:
            used_vars: the variable used at the desired node

        Returns:
            the implemented function
        """
        num_nodes = self._n_feas_nodes
        if num_nodes == 0:
            # if the function is not specified on any nodes, don't implement
            self._fun_impl = None
        else:
            # mapping the function to use more cpu threads
            self._fun_map = self._fun.map(num_nodes, 'thread', thread_map_num)
            # print('usedvar:')
            used_var_impl = self._getUsedVarImpl()
            # print('usedpar:')
            used_par_impl = self._getUsedParImpl()
            all_vars = used_var_impl + used_par_impl

            fun_eval = self._fun_map(*all_vars)
            self._fun_impl = fun_eval


    def _getUsedElemImpl(self, elem_container):
        # todo there should be an option to specify which nodes i'm querying (right now it's searching all the feaasible nodes)
        # todo throw with a meaningful error when nodes inserted are wrong
        used_elem_impl = list()
        for elem in elem_container:
            impl_nodes = misc.getNodesFromBinary(self._feas_nodes_array)
            # tic = time.time()
            elem_impl = elem.getImpl(impl_nodes)
            # print('get_used:', time.time() - tic)
            used_elem_impl.append(elem_impl)

        # exit()
        return used_elem_impl

    def _getUsedVarImpl(self):
        return self._getUsedElemImpl(self.vars)

    def _getUsedParImpl(self):
        return self._getUsedElemImpl(self.pars)

    # def __reduce__(self):
    #     """
    #     Experimental function to serialize this element.
    #
    #     Returns:
    #         instance of this element serialized
    #     """
    #     return (self.__class__, (self._name, self._f, self.vars, self.pars, self._nodes, ))

    # def serialize(self):
    #     """
    #     Serialize the Function. Used to save it.
    #
    #     Returns:
    #         serialized instance of Function
    #     """
    #
    #     self._f = self._f.serialize()
    #
    #     for i in range(len(self.vars)):
    #         self.vars[i] = self.vars[i].serialize()
    #
    #     for node, item in self._fun_impl.items():
    #         self._fun_impl[node] = item.serialize()
    #
    #     # self._fun = self._fun.serialize()

        # return self

    # def deserialize(self):
    #     """
    #     Deserialize the Function. Used to load it.
    #
    #     Returns:
    #         deserialized instance of Function
    #     """
    #
    #     self._f = cs.SX.deserialize(self._f)
    #
    #     for i in range(len(self.vars)):
    #         self.vars[i] = cs.SX.deserialize(self.vars[i])
    #
    #     for node, item in self._fun_impl.items():
    #         self._fun_impl[node] = cs.SX.deserialize(item)
    #
    #     # self._fun = cs.Function.deserialize(self._fun)
    #
    #     return self

class Function(AbstractFunction):
    """
        Function of Horizon: generic function of SX values from CASADI.

        Notes:
            This function is an abstract representation of its projection over the nodes of the optimization problem.
            An abstract function gets internally implemented at each node, using the variables at that node.
    """

    def __init__(self, name: str, f: Union[cs.SX, cs.MX], used_vars: list, used_pars: list, active_nodes_array: np.ndarray, thread_map_num=None):
        """
        Initialize the Horizon Function.

        Args:
            name: name of the function
            f: SX function
            used_vars: variable used in the function
            used_pars: parameters used in the function
            active_nodes_array: binary array specifying the nodes the function is active on
        """
        super().__init__(name, f, used_vars, used_pars, active_nodes_array)

        self._f = f

        if thread_map_num is None:
            self.thread_map_num = default_thread_map
        else:
            self.thread_map_num = thread_map_num

        self._fun_impl = None
        self._project()

    def getImpl(self, nodes=None):
        """
        Getter for the CASADI function implemented at the desired node
        Args:
            node: the desired node of the implemented function to retrieve

        Returns:
            instance of the CASADI function at the desired node
        """
        if self._fun_impl is None:
            return None

        if nodes is None:
            nodes = misc.getNodesFromBinary(self._active_nodes_array)
        else:
            nodes = misc.checkNodes(nodes, self._active_nodes_array)

        # I have to convert the input nodes to the corresponding column position:
        # function active on [5, 6, 7] means that the columns are 0, 1, 2 so i have to convert, for example, 6 --> 1
        pos_nodes = misc.convertNodestoPos(nodes, self._feas_nodes_array)

        # getting the column corresponding to the nodes requested
        fun_impl = cs.vertcat(*[self._fun_impl[:, pos_nodes]])
        return fun_impl

    def _project(self):
        """
        Implements the function at the desired node using the desired variables.

        Args:
            used_vars: the variable used at the desired node

        Returns:
            the implemented function
        """
        super()._projectNodes(thread_map_num=self.thread_map_num)

    def getNodes(self) -> list:
        """
        Getter for the active nodes of the function.

        Returns:
            a list of the nodes where the function is active

        """
        return misc.getNodesFromBinary(self._active_nodes_array)


    def setNodes(self, nodes, erasing=True):
        """
        Setter for the active nodes of the function.

        Args:
            nodes: list of desired active nodes.
            erasing: choose if the inserted nodes overrides the previous active nodes of the function. 'False' if not specified.
        """
        # todo check for repetition in setted nodes?
        super().setNodes(nodes, erasing)

        # usually the number of nodes stays the same, while the active nodes of a function may change.
        # If the number of nodes changes, also the variables change. That is when this reprojection is required.
        self._project()

    def _setFeasNodes(self, feas_nodes_array):
        """
        Caution, advanced method.
        Change the FEASIBLE node array of the function.
        """
        super()._setFeasNodes(feas_nodes_array)

class RecedingFunction(AbstractFunction):

    def __init__(self, name: str, f: Union[cs.SX, cs.MX], used_vars: list, used_pars: list, active_nodes_array: np.ndarray, thread_map_num=None):
        # TODO: probably everything breaks if there are variables only defined on specific nodes
        super().__init__(name, f, used_vars, used_pars, active_nodes_array)

        if thread_map_num is None:
            self.thread_map_num = default_thread_map
        else:
            self.thread_map_num = thread_map_num

        self.weight_mask = None

        total_nodes = np.array(range(self._active_nodes_array.size))
        self._feas_nodes_array = self._computeFeasNodes(used_vars, used_pars, total_nodes)

        self._n_feas_nodes = np.sum(self._feas_nodes_array).astype(int)

        # if the function is active (self._nodes_array) in some nodes where the variables it involves are not defined (self._var_nodes) throw an error.
        # this is true for the offset variables also: an offset variable of a variable defined on [0, 1, 2] is only valid at [1, 2].
        check_feas_nodes = self._feas_nodes_array - self._active_nodes_array
        if (check_feas_nodes < 0).any():
            raise ValueError(f'Function "{self.getName()}" cannot be active on nodes: {np.where(check_feas_nodes < 0)}')

        self._fun_impl = None
        self._project()

    def _computeFeasNodes(self, vars, pars, total_nodes):

        temp_nodes = total_nodes.copy()

        # IF RECEDING, it is important to define two concepts:
        # - function EXISTS: the function exists only on the nodes where ALL the variables and parameters of the function are defined.
        # - function is ACTIVE: the function can be activated/disabled on the nodes where it exists
        for var in vars:
            # getNodes() in a OffsetVariable returns the nodes of the base variable.
            # here I want the nodes where the variable is actually defined, so I need to consider also the offset
            # when getting the nodes:

            # #todo very bad hack to remove SingleVariables and SingleParameters
            if -1 in var.getNodes():
                continue
            var_nodes = np.array(var.getNodes()) - var.getOffset()
            temp_nodes = np.intersect1d(temp_nodes, var_nodes)
        for par in pars:
            if -1 in par.getNodes():
                continue
            par_nodes = np.array(par.getNodes()) - par.getOffset()
            temp_nodes = np.intersect1d(temp_nodes, par_nodes)

        feas_nodes_array = misc.getBinaryFromNodes(total_nodes.size, temp_nodes)

        return feas_nodes_array

    def getImpl(self, nodes=None):
        """
        Getter for the CASADI function implemented at the desired node
        Args:
            node: the desired node of the implemented function to retrieve

        Returns:
            instance of the CASADI function at the desired node
        """
        # todo return the implemented function on all nodes always??

        return cs.vertcat(*[self._fun_impl])

    def _project(self):
        """
        Implements the function at the desired node using the desired variables.

        Args:
            used_vars: the variable used at the desired node

        Returns:
            the implemented function
        """
        super()._projectNodes(thread_map_num=self.thread_map_num)

    def setNodes(self, nodes, erasing=True):
        """
        Setter for the active nodes of the function.

        Args:
            nodes: list of desired active nodes.
            erasing: choose if the inserted nodes overrides the previous active nodes of the function. 'False' if not specified.
        """
        # todo check for repetition in setted nodes?
        # super().setNodes(nodes, erasing)

        if erasing:
            self._active_nodes_array[:] = 0

        self._active_nodes_array[nodes] = 1

        # usually the number of nodes stays the same, while the active nodes of a function may change.
        # If the number of nodes changes, also the variables change. That is when this reprojection is required.
        # todo: if the function is receding, it is already defined on ALL the nodes. So, the reprojection is useless, I believe.
        # self._project()

class AbstractBounds:
    """
    Bounds helper of Horizon.
    """

    def __init__(self, fun_dim, init_bounds):
        """
        Initialize the bounds helper.

        Args:
        """

        # set_bounds exist to reset to old value the bounds of restored non-active nodes
        self.bounds = dict()
        self.set_bounds = dict()

        self.bounds['lb'] = init_bounds['lb']
        self.bounds['ub'] = init_bounds['ub']

        # default value of constraints is 0.
        self.set_bounds['lb'] = np.zeros_like(init_bounds['lb'])
        self.set_bounds['ub'] = np.zeros_like(init_bounds['ub'])

        self.fun_dim = fun_dim

    def _set_initial_bounds(self, bounds):

        # manage bounds
        if bounds is not None:
            if 'nodes' not in bounds:
                bounds['nodes'] = None

            if 'lb' in bounds:
                if 'ub' not in bounds:
                    bounds['ub'] = np.full(self.fun_dim, np.inf)

            if 'ub' in bounds:
                if 'lb' not in bounds:
                    bounds['lb'] = np.full(self.fun_dim, -np.inf)

            self.setBounds(lb=bounds['lb'], ub=bounds['ub'], nodes=bounds['nodes'])

    def _setVals(self, val_type, val, nodes=None):
        """
        Generic setter.

        Args:
            val_type: type of value
            val: desired values to set
            nodes: which nodes the values are applied on
        """
        val_checked = misc.checkValueEntry(val)
        if val_checked.shape[0] != self.fun_dim:
            raise Exception('Wrong dimension of upper bounds inserted.')

        # todo guards (here it is assumed that bounds is a row)
        val_type[:, nodes] = val_checked

    def setLowerBounds(self, bounds, nodes=None):
        """
        Setter for the lower bounds of the function.

        Args:
            bounds: desired bounds of the function
            nodes: nodes of the function the bounds are applied on. If not specified, the function is bounded along ALL the nodes.
        """
        self._setVals(self.set_bounds['lb'], bounds, nodes)
        self._setVals(self.bounds['lb'], bounds, nodes)

    def setUpperBounds(self, bounds, nodes=None):
        """
        Setter for the upper bounds of the function.

        Args:
            bounds: desired bounds of the function
            nodes: nodes of the function the bounds are applied on. If not specified, the function is bounded along ALL the nodes.
        """
        self._setVals(self.set_bounds['ub'], bounds, nodes)
        self._setVals(self.bounds['ub'], bounds, nodes)

    def setBounds(self, lb, ub, nodes=None):
        """
        Setter for the bounds of the function.

        Args:
            lb: desired lower bounds of the function
            ub: desired upper bounds of the function
            nodes: nodes of the function the bounds are applied on. If not specified, the function is bounded along ALL the nodes.
        """
        self.setLowerBounds(lb, nodes)
        self.setUpperBounds(ub, nodes)

    def _getVals(self, val_type, nodes=None):
        """
        wrapper function to get the desired argument from the constraint.

        Args:
            val_type: type of the argument to retrieve
            node: desired node at which the argument is retrieved. If not specified, this returns the desired argument at all nodes.

        Returns:
            value/s of the desired argument
        """
        pass

    def getLowerBounds(self, node: int = None):
        """
        Getter for the lower bounds of the function.

        Args:
            node: desired node at which the lower bounds are retrieved. If not specified, this returns the lower bounds at all nodes.

        Returns:
            value/s of the lower bounds

        """
        lb = self._getVals(self.bounds['lb'], node)
        return lb

    def getUpperBounds(self, node: int = None):
        """
        Getter for the upper bounds of the function.

        Args:
            node: desired node at which the upper bounds are retrieved. If not specified, this returns the upper bounds at all nodes.

        Returns:
            value/s of the upper bounds

        """
        ub = self._getVals(self.bounds['ub'], node)
        return ub

    def getBounds(self, nodes=None):
        """
        Getter for the bounds of the function.

        Args:
            node: desired node at which the bounds are retrieved. If not specified, this returns the bounds at all nodes.

        Returns:
            value/s of the upper bounds
        """
        return self.getLowerBounds(nodes), self.getUpperBounds(nodes)

    def setNodes(self, nodes, erasing=True):
        """
        Setter for the active nodes of the constraint function.

        Args:
            nodes: list of desired active nodes.
            erasing: choose if the inserted nodes overrides the previous active nodes of the function. 'False' if not specified.
        """

        # todo check for repetition in setted nodes?
        # todo think about this, it depends on how the mechanics of the receding works
        if erasing:
            # apply mask without erasing old values
            self.bounds['lb'][:] = -np.inf
            self.bounds['ub'][:] = np.inf

        # restore old bounds where set
        self.bounds['lb'][:, nodes] = self.set_bounds['lb'][:, nodes]
        self.bounds['ub'][:, nodes] = self.set_bounds['ub'][:, nodes]

        # set 0. all the remaining "new nodes", if any


class Constraint(Function, AbstractBounds):
    """
    Constraint Function of Horizon.
    """

    def __init__(self, name: str, f: Union[cs.SX, cs.MX], used_vars: list, used_pars: list,
                 active_nodes_array: np.ndarray,
                 bounds=None, thread_map_num=None):
        """
        Initialize the Constraint Function.

        Args:
            name: name of the constraint function
            f: constraint SX function
            used_vars: variable used in the function
            used_pars: parameters used in the function
            active_nodes_array: nodes the function is active on
            bounds: bounds of the constraint. If not specified, the bounds are set to zero.
        """
        Function.__init__(self, name, f, used_vars, used_pars, active_nodes_array, thread_map_num)

        # constraints are initialized to 0.: 0. <= x <= 0.
        num_nodes = int(np.sum(active_nodes_array))
        init_bounds = dict()
        init_bounds['lb'] = np.full((f.shape[0], num_nodes), 0.)
        init_bounds['ub'] = np.full((f.shape[0], num_nodes), 0.)

        AbstractBounds.__init__(self, f.shape[0], init_bounds)
        self._set_initial_bounds(bounds)

    def _setVals(self, val_type, val, nodes=None):

        # todo make a wrapper function for this Guard
        if nodes is None:
            nodes = misc.getNodesFromBinary(self._active_nodes_array)
        else:
            nodes = misc.checkNodes(nodes, self._active_nodes_array)

        pos_nodes = misc.convertNodestoPos(nodes, self._feas_nodes_array)

        super()._setVals(val_type, val, pos_nodes)

    def _getVals(self, val_type, nodes=None):

        if nodes is None:
            nodes = misc.getNodesFromBinary(self._active_nodes_array)
        else:
            nodes = misc.checkNodes(nodes, self._active_nodes_array)

        pos_nodes = misc.convertNodestoPos(nodes, self._feas_nodes_array)

        # todo what is this???
        if len(nodes) == 0:
            return np.zeros((self.getDim(), 0))

        vals = val_type[:, pos_nodes]

        return vals

    def setNodes(self, nodes, erasing=True):
        """
        Setter for the active nodes of the constraint function.

        Args:
            nodes: list of desired active nodes.
            erasing: choose if the inserted nodes overrides the previous active nodes of the function. 'False' if not specified.
        """
        # todo check for repetition in setted nodes?
        Function.setNodes(self, nodes, erasing)
        # todo am I wrong?
        pos_nodes = misc.convertNodestoPos(nodes, self._feas_nodes_array)
        AbstractBounds.setNodes(self, pos_nodes, erasing)

class RecedingConstraint(RecedingFunction, AbstractBounds):
    """
    Constraint Function of Horizon.
    """

    def __init__(self, name: str, f: Union[cs.SX, cs.MX], used_vars: list, used_pars: list, active_nodes_array: np.ndarray,
                 bounds=None, thread_map_num=None):
        """
        Initialize the Constraint Function.

        Args:
            name: name of the constraint function
            f: constraint SX function
            used_vars: variable used in the function
            used_pars: parameters used in the function
            active_nodes_array: nodes the function is active on
            bounds: bounds of the constraint. If not specified, the bounds are set to zero.
        """
        RecedingFunction.__init__(self, name, f, used_vars, used_pars, active_nodes_array, thread_map_num)

        num_nodes = int(np.sum(self._feas_nodes_array))
        temp_lb = -np.inf * np.ones([f.shape[0], num_nodes])
        temp_ub = np.inf * np.ones([f.shape[0], num_nodes])

        # this is zero only on the nodes where the function is ACTIVE (which are generally different from the nodes where the function EXISTS)
        self._active_nodes = misc.getNodesFromBinary(self._active_nodes_array)
        pos_nodes = misc.convertNodestoPos(self._active_nodes, self._feas_nodes_array)

        temp_lb[:, pos_nodes] = 0.
        temp_ub[:, pos_nodes] = 0.

        init_bounds = dict()
        init_bounds['lb'] = temp_lb
        init_bounds['ub'] = temp_ub

        AbstractBounds.__init__(self, f.shape[0], init_bounds)
        self._set_initial_bounds(bounds)

    def _checkActiveNodes(self, nodes):

        # useful to deactivate nodes if lb and ub are -inf/inf
        # pos_nodes = misc.convertNodestoPos(self._active_nodes, self._feas_nodes_array)

        for node in nodes:
            if np.isinf(self.bounds['lb'][:, node]).all() and np.isinf(self.bounds['ub'][:, node]).all():
                self._active_nodes_array[node] = 0
            else:
                self._active_nodes_array[node] = 1

        self._active_nodes = misc.getNodesFromBinary(self._active_nodes_array)


    def _setVals(self, val_type, val, nodes=None):

        if nodes is None:
            nodes = self._active_nodes
        else:
            # todo: I BELIEVE THIS SHOULD BE self._feas_nodes_array
            nodes = misc.checkNodes(nodes, self._feas_nodes_array)

        # pos_nodes = misc.convertNodestoPos(nodes, self._feas_nodes_array)

        super()._setVals(val_type, val, nodes)
        # todo put it also in normal constraint
        # todo if setting bounds DEACTIVATE a node, maybe it should be able to ACTIVATE it too
        self._checkActiveNodes(nodes)

    def _getVals(self, val_type, nodes=None):
        # todo return the bounds on all nodes always??
        # todo: this should return only active nodes

        # if nodes is None:
        #     nodes = misc.getNodesFromBinary(self._active_nodes_array)
        # else:
        #     nodes, _ = misc.checkNodes(nodes, self._active_nodes_array)
        #
        # pos_nodes = misc.convertNodestoPos(nodes, self._feas_nodes_array)
        #
        # # todo what is this???
        # if len(nodes) == 0:
        #     return np.zeros((self.getDim(), 0))
        #
        # vals = val_type[:, pos_nodes]
        #
        # return vals

        return val_type

    def setNodes(self, nodes, erasing=True):
        # todo check for repetition in setted nodes?
        RecedingFunction.setNodes(self, nodes, erasing)

        # pos_nodes = misc.convertNodestoPos(nodes, self._feas_nodes_array)
        AbstractBounds.setNodes(self, nodes, erasing)

        # print('=======================')
        # print('name:', self.getName())
        # print('nodes:', list(getRanges(misc.getNodesFromBinary(self._active_nodes_array))))
        # for dim in range(self.bounds['lb'].shape[0]):
        #     print(f'lb_{dim}:', list(getRanges(misc.getNodesFromBinary(np.isfinite(self.bounds['lb'][dim, :]).astype(int)))))
        # for dim in range(self.bounds['ub'].shape[0]):
        #     print(f'ub_{dim}:', list(getRanges(misc.getNodesFromBinary(np.isfinite(self.bounds['ub'][dim, :]).astype(int)))))

    def _setFeasNodes(self, feas_nodes_array):
        """
        Caution, advanced method.
        Change the FEASIBLE node array of the function.
        Reset the ACTIVE nodes and the bounds
        """
        super()._setFeasNodes(feas_nodes_array)

        n_nodes = int(np.sum(self._feas_nodes_array))
        self.bounds['lb'] = np.full((self.getDim(), n_nodes), 0.)
        self.bounds['ub'] = np.full((self.getDim(), n_nodes), 0.)

        self.set_bounds['lb'] = np.full((self.getDim(), n_nodes), 0.)
        self.set_bounds['ub'] = np.full((self.getDim(), n_nodes), 0.)

    def shift(self):
        shift_num = -1

        print(f'============= CONSTRAINT ================')
        print(f'NAME: {self.getName()}')
        print(f'OLD VALUES:\n {self.getLowerBounds()}')
        print(f'OLD VALUES:\n {self.getUpperBounds()}')

        active_nodes = misc.getNodesFromBinary(self._active_nodes_array)

        # if the constraint is only defined on given nodes, I have to get the right nodes:
        pos_nodes = misc.convertNodestoPos(active_nodes, self._feas_nodes_array)

        active_lb = self.getLowerBounds()[:, pos_nodes]
        active_ub = self.getUpperBounds()[:, pos_nodes]


        old_nodes = np.array(self.getNodes())
        shifted_nodes = old_nodes + shift_num
        mask_nodes = shifted_nodes >= 0

        masked_nodes = shifted_nodes[mask_nodes]
        masked_lb = active_lb[:, mask_nodes]
        masked_ub = active_ub[:, mask_nodes]

        self.setNodes(masked_nodes, erasing=True)
        self.setLowerBounds(masked_lb)
        self.setUpperBounds(masked_ub)

        print(f'NEW VALUES:\n {self.getLowerBounds()}')
        print(f'NEW VALUES:\n {self.getUpperBounds()}')

class Cost(Function):
    """
    Cost Function of Horizon.
    """

    def __init__(self, name, f, used_vars, used_pars, active_nodes_array, thread_map_num=None):
        """
        Initialize the Cost Function.

        Args:
            name: name of the function
            f: SX function
            used_vars: variable used in the function
            used_pars: parameters used in the function
            active_nodes_array: binary array specifying the nodes the function is active on
        """

        super().__init__(name, f, used_vars, used_pars, active_nodes_array, thread_map_num)

    def setNodes(self, nodes, erasing=True):
        super().setNodes(nodes, erasing)

class RecedingCost(RecedingFunction):
    """
    Cost Function of Horizon.
    """

    def __init__(self, name, f, used_vars, used_pars, active_nodes_array, thread_map_num=None):
        """
        Initialize the Cost Function.

        Args:
            name: name of the function
            f: SX function
            used_vars: variable used in the function
            used_pars: parameters used in the function
            active_nodes_array: binary array specifying the nodes the function is active on
        """

        # create weight mask to select which nodes (among the feasible) are active or not

        super().__init__(name, f, used_vars, used_pars, active_nodes_array, thread_map_num)

    def _setWeightMask(self, casadi_type, abstract_casadi_type):

        dim_weight_mask = 1
        self.weight_mask = sv.RecedingParameter(f'{self.getName()}_weight_mask',
                                                dim_weight_mask,
                                                self._feas_nodes_array,
                                                casadi_type,
                                                abstract_casadi_type)

        self.pars.append(self.weight_mask)

        # override _f and _fun
        self._f = self.weight_mask * self._f
        all_input = self.vars + self.pars
        all_names = [i.getName() for i in all_input]
        self._fun = cs.Function(self.getName(), self.vars + self.pars, [self._f], all_names, ['f'])
        self._zero_nodes_mask = np.zeros([self.weight_mask.getDim(), np.sum(self._feas_nodes_array).astype(int)])

        self.setNodes(misc.getNodesFromBinary(self._active_nodes_array), erasing=True)

    def _getWeightMask(self):
        return self.weight_mask

    def setNodes(self, nodes, erasing=True):
        super().setNodes(nodes, erasing)
        # eliminate/enable cost functions by setting their weight
        nodes_mask = self._zero_nodes_mask.copy()
        # nodes_mask = np.zeros([self.weight_mask.getDim(), np.sum(self._feas_nodes_array).astype(int)])
        nodes_mask[:, nodes] = 1
        self.weight_mask.assign(nodes_mask)

    # def shift(self):
    # pass
    # shift_num = -1

    # print(f'============= COST ================')
    # print(f'NAME: {self.getName()}')
    # print(f'NEW VALUES:\n {self.weight_mask.getValues()}')

class Residual(Cost):
    """
    Residual Function of Horizon.
    """

    def __init__(self, name, f, used_vars, used_pars, active_nodes_array, thread_map_num=None):
        """
        Initialize the Residual Function.

        Args:
            name: name of the function
            f: SX function
            used_vars: variable used in the function
            used_pars: parameters used in the function
            active_nodes_array: binary array specifying the nodes the function is active on
        """
        super().__init__(name, f, used_vars, used_pars, active_nodes_array, thread_map_num)

class RecedingResidual(RecedingCost):
    """
    Residual Function of Horizon.
    """

    def __init__(self, name, f, used_vars, used_pars, active_nodes_array, thread_map_num=None):
        """
        Initialize the Residual Function.

        Args:
            name: name of the function
            f: SX function
            used_vars: variable used in the function
            used_pars: parameters used in the function
            active_nodes_array: binary array specifying the nodes the function is active on
        """
        super().__init__(name, f, used_vars, used_pars, active_nodes_array, thread_map_num)

class FunctionsContainer:
    """
    Container of all the function of Horizon.
    It is used internally by the Problem to get the abstract and implemented function.

    Methods:
        build: builds the container with the updated functions.
    """

    def __init__(self, receding, thread_map_num, logger=None):
        """
        Initialize the Function Container.

        Args:
            receding: activate receding horizon
            thread_map_num: number of cpu threads used when implementing casadi map
            logger: a logger reference to log data
        """
        self._logger = logger
        self.is_receding = receding
        self.thread_map_num = thread_map_num

        # containers for the constraint functions
        self._cnstr_container = OrderedDict()

        # containers for the cost functions
        self._cost_container = OrderedDict()

    def createConstraint(self, name, g, used_var, used_par, nodes_array, bounds):

        if self.is_receding:
            fun_constructor = RecedingConstraint
        else:
            fun_constructor = Constraint

        fun = fun_constructor(name, g, used_var, used_par, nodes_array, bounds, thread_map_num=self.thread_map_num)
        self.addFunction(fun)

        return fun

    def createCost(self, name, j, used_var, used_par, nodes_array):

        if self.is_receding:
            fun_constructor = RecedingCost
        else:
            fun_constructor = Cost

        fun = fun_constructor(name, j, used_var, used_par, nodes_array, thread_map_num=self.thread_map_num)
        self.addFunction(fun)

        return fun

    def createResidual(self, name, j, used_var, used_par, nodes_array):

        if self.is_receding:
            fun_constructor = RecedingResidual
        else:
            fun_constructor = Residual

        fun = fun_constructor(name, j, used_var, used_par, nodes_array, thread_map_num=self.thread_map_num)
        self.addFunction(fun)

        return fun

    def addFunction(self, fun: Function):
        """
        Add a function to the Function Container.

        Args:
            fun: a Function (can be Constraint or Cost Function) o add
        """
        if isinstance(fun, (Constraint, RecedingConstraint)):
            if fun.getName() not in self._cnstr_container:
                self._cnstr_container[fun.getName()] = fun
            else:
                raise Exception(f'Function name "{fun.getName()}" already inserted.')
        elif isinstance(fun, (Cost, RecedingCost)):
            if fun.getName() not in self._cost_container:
                self._cost_container[fun.getName()] = fun
            else:
                raise Exception(f'Function name "{fun.getName()}" already inserted.')
        elif isinstance(fun, (Residual, RecedingResidual)):
            if fun.getName() not in self._cost_container:
                self._cost_container[fun.getName()] = fun
            else:
                raise Exception(f'Function name "{fun.getName()}" already inserted.')
        else:
            raise Exception('Function type not implemented')

    def removeFunction(self, fun_name: str):
        """
        Remove a function from the Function Container.

        Args:
            fun_name: name of the function to be removed
        """
        if fun_name in self._cnstr_container:
            del self._cnstr_container[fun_name]
            return True
        elif fun_name in self._cost_container:
            del self._cost_container[fun_name]
            return True
        else:
            return False

    def getFunction(self, fun_name: str):
        """
        Getter for a Function inside the Function Container.
        Args:
            fun_name: name of the function to retrieve
        """
        if fun_name in self._cnstr_container:
            return self._cnstr_container[fun_name]
        elif fun_name in self._cost_container:
            return self._cost_container[fun_name]
        else:
            return None

    def getCnstr(self, name=None) -> OrderedDict:
        """
        Getter for the dictionary of all the abstract constraint functions.

        Args:
            name of constraint. If not specified, returns all of them
        Returns:
            ordered dict of the functions {name: fun}
        """
        if name is None:
            cnsrt_dict = self._cnstr_container
        else:
            if name in self._cnstr_container:
                cnsrt_dict = self._cnstr_container[name]
            else:
                cnsrt_dict = None
        return cnsrt_dict

    def getCost(self, name=None) -> OrderedDict:
        """
        Getter for the dictionary of all the abstract cost functions.

        Args:
            name of constraint. If not specified, returns all of them

        Returns:
            ordered dict of the functions {name: fun}
        """
        if name is None:
            cost_dict = self._cost_container
        else:
            cost_dict = self._cost_container[name]
        return cost_dict

    def getCnstrDim(self) -> int:
        """
        Getter for the dimension of all the constraints
        Returns:
            the total dimension of the constraint
        """
        total_dim = 0
        for cnstr in self._cnstr_container.values():
            total_dim += cnstr.getDim() * len(cnstr.getNodes())

        return total_dim

    def setNNodes(self, n_nodes):
        """
        set a desired number of nodes to Function Container.
        Args:
            n_nodes: the desired number of nodes to be set
        """

        # todo should find a way to understand which is the transcription function and project it over all the nodes
        # this is required to update the function_container EACH time a new number of node is set
        for cnstr in self._cnstr_container.values():
            # this is required to update the nodes consistently.
            # For instance, the horizon problem is specified on [0, 1, 2, 3, 4].
            # Consider a function containing an input variable. it is active on [0, 1, 2, 3].
            # I change the nodes to [0, 1, 2, 3]. The function must be updated accordingly: [0, 1, 2]
            # I change the nodes to [0, 1, 2, 3, 4, 5]. The function must be updated accordingly: [0, 1, 2, 4]
            available_nodes = set(range(n_nodes))
            # get only the variable it depends (not all the offsetted variables)
            for var in cnstr.getVariables(offset=False):
                if not var.getNodes() == [
                    -1]:  # todo very bad hack to check if the variable is a SingleVariable (i know it returns [-1]
                    available_nodes.intersection_update(var.getNodes())

            cnstr.setNodes([i for i in cnstr.getNodes() if i in available_nodes], erasing=True)

        for cost in self._cost_container.values():
            available_nodes = set(range(n_nodes))
            for var in cost.getVariables(offset=False):
                if not var.getNodes() == [-1]:
                    available_nodes.intersection_update(var.getNodes())
            cost.setNodes([i for i in cost.getNodes() if i in range(n_nodes)], erasing=True)

    def serialize(self):
        """
        Serialize the Function Container. Used to save it.

        Returns:
            instance of serialized Function Container

        """
        raise Exception('serialize yet to implement')
        for name, item in self._cnstr_container.items():
            self._cnstr_container[name] = item.serialize()

        for name, item in self._cost_container.items():
            self._cost_container[name] = item.serialize()

    def deserialize(self):
        """
        Deerialize the Function Container. Used to load it.

        Returns:
            instance of deserialized Function Container

        """
        raise Exception('serialize yet to implement')
        for name, item in self._cnstr_container.items():
            item.deserialize()
            new_vars = item.getVariables()
            for var in new_vars:
                print(var.getName(), var.getOffset())
                print(f'{item._f} depends on {var}?', cs.depends_on(item._f, var))


        for name, item in self._cnstr_container.items():
            self._cnstr_container[name] = item.deserialize()

        # these are CASADI functions
        for name, item in self._cost_container.items():
            self._cost_container[name] = item.deserialize()


if __name__ == '__main__':
    x = cs.SX.sym('x', 2)
    y = cs.SX.sym('y', 2)
    fun = x + y
    used_var = dict(x=x, y=y)
    funimpl = Function('dan', fun, used_var, 1)

    funimpl = funimpl.serialize()
    print('===PICKLING===')
    funimpl_serialized = pickle.dumps(funimpl)
    print(funimpl_serialized)
    print('===DEPICKLING===')
    funimpl_new = pickle.loads(funimpl_serialized)
