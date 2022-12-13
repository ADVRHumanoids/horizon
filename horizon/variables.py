import copy
import time
from typing import List
import casadi as cs
from collections import OrderedDict
import logging
import numpy as np
import pickle
import horizon.misc_function as misc
import pprint
from abc import ABC, abstractmethod
import itertools

def getRanges(i):
    for a, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
        b = list(b)
        yield b[0][1], b[-1][1]
'''
now the StateVariable is only abstract at the very beginning.
Formerly
'''

# todo create function checker to check if nodes are in self.nodes and if everything is ok with the input (no dict, no letters...)

class AbstractVariable(ABC):
    """
    Abstract Variable of Horizon Problem.

    Notes:
          Horizon allows the user to work only with abstract variables. Internally, these variables are projected over the horizon nodes.
    """

    def __init__(self, tag: str, dim: int, abstract_casadi_type):
        """
        Initialize the Abstract Variable. Inherits from the symbolic CASADI variable SX or MX.

        Args:
            tag: name of the variable
            dim: dimension of the variable
        """

        # dynamically add casadi_type as base class 
        class AVClass(self.__class__, abstract_casadi_type):
            pass

        self.__class__ = AVClass

        # save the cs type
        self._abstract_casadi_type = abstract_casadi_type

        # initialize base class
        super().__init__(abstract_casadi_type.sym(tag, dim))

        # human readable name
        self._tag = tag

        # size (same as self.size1())
        self._dim = dim

        # offset of a variable is used to point to the desired previous/next implemented
        # Example:
            # offset 1 of variable x --> refers to implemented variable x at the next node
        self._offset = 0

    def getDim(self) -> int:
        """
        Getter for the dimension of the abstract variable.

        Returns:
            dimension of the variable
        """
        return self._dim

    @abstractmethod
    def getName(self):
        ...
        # return self.tag

    def getOffset(self):
        return self._offset

    def __getitem__(self, item):
        var_slice = super().__getitem__(item)
        view = AbstractVariableView(self, var_slice, item)
        return view

    # todo old stuff
    # def __getitem__(self, item):
    #     view = AbstractVariableView(self, item)
    #     return view

class AbstractVariableView:
    def __init__(self, parent: AbstractVariable, var_slice, indices):

        # dynamically add casadi_type as base class
        class AVVClass(self.__class__, parent._abstract_casadi_type):
            pass
        
        self.__class__ = AVVClass

        # initialize base class
        super().__init__(var_slice)

        # save parent and view's indices
        self._parent = parent
        self._indices = indices

        # todo debug
        if isinstance(self._indices, slice):
            self._dim = len(range(*self._indices.indices(self._parent.shape[0])))
        elif hasattr(self._indices, '__len__'):
            self._dim = len(self._indices)
        else:
            self._dim = 1

    def getName(self):
        return self._parent.getName()

    def getDim(self):
        return self._dim

    def __getitem__(self, item):
        var_slice = super().__getitem__(item)
        view = self.__class__(self._parent, var_slice, item)
        return view


class OffsetTemplate(AbstractVariable):  # note parametrize casadi type!!!
    def __init__(self, parent_name, tag, dim, offset, nodes_array, impl, casadi_type, abstract_casadi_type):
        """
        Initialize the Offset Variable.

        Args:
            tag: name of the variable
            dim: dimension of the variable
            nodes: nodes the variable is defined on
            offset: offset of the variable (which (previous/next) node it refers to
            impl: implemented variables it refers to (of base class Variable)
        """
        self._tag = tag
        self._dim = dim
        super().__init__(tag, dim, abstract_casadi_type)

        self._casadi_type = casadi_type

        self.parent_name = parent_name
        self._offset = offset
        self._impl = impl
        self._nodes_array = nodes_array

    def getImpl(self, nodes=None):
        """
        Getter for the implemented offset variable.

        Args:
            node: node at which the variable is retrieved

        Returns:
            implemented instances of the abstract offsetted variable
        """
        if nodes is None:
            nodes = misc.getNodesFromBinary(self._nodes_array)

        nodes_array = np.array(nodes)
        # offset the node of self.offset
        offset_nodes = nodes_array + self._offset
        offset_nodes = misc.checkNodes(offset_nodes, self._nodes_array)

        var_impl = self._impl['var'][:, offset_nodes]

        return var_impl

    def getName(self):
        """
        Get name of the variable. Warning: not always same as the tag

        Returns:
            name of the variable

        """
        return self.parent_name

    def getNodes(self):
        """
        Getter for the active nodes.

        Returns:
            list of active nodes
        """
        return misc.getNodesFromBinary(self._nodes_array)

    def __reduce__(self):

        for node in self._impl.keys():
            self._impl[node]['var'] = self._impl[node]['var'].serialize()

        return (self.__class__, (self.parent_name, self._tag, self._dim, self._offset, self._impl,))

# todo add also absolute position of variable
#   not only relative position (offseted to the present one)

class SingleParameter(AbstractVariable):
    """
    Single Parameter of Horizon Problem.
    It is used for parametric problems: it is a symbolic variable in the optimization problem but it is not optimized.
    Rather, it is kept parametric and can be assigned before solving the problem.
    Parameters are specified before building the problem and can be 'assigned' afterwards, before solving the problem.
    The assigned value is the same along the horizon, since this parameter is node-independent.
    The Parameter is abstract, and gets implemented automatically.
    """
    def __init__(self, tag, dim, nodes_array, casadi_type, abstract_casadi_type):
        """
        Initialize the Single Parameter: a node-independent parameter which is not projected over the horizon.

        Args:
            tag: name of the parameter
            dim: dimension of the parameter
            dummy_nodes: useless input, used to simplify the framework mechanics
        """
        super().__init__(tag, dim, abstract_casadi_type)

        self._casadi_type = casadi_type
        self._nodes_array = nodes_array
        self._impl = dict()
        self._impl['var'] = self._casadi_type.sym(self._tag + '_impl', self._dim)
        self._impl['val'] = np.zeros([self._dim, 1])

    def assign(self, val, indices=None):
        """
        Assign a value to the parameter. Can be assigned also after the problem is built, before solving the problem.
        If not assigned, its default value is zero.

        Args:
            val: value of the parameter
        """
        val = misc.checkValueEntry(val)

        if indices is None:
            indices_vec = np.array(range(self._dim)).astype(int)
        else:
            indices_vec = np.array(indices).astype(int)

        val_checked = misc.checkValueEntry(val)
        if val_checked.shape[0] != indices_vec.size:
            raise Exception('Wrong dimension of parameter values inserted.')

        self._impl['val'][indices_vec] = val_checked

    def _getVals(self, val_type, nodes):
        """
        wrapper function to get the desired argument from the variable.

        Args:
            val_type: type of the argument to retrieve
            nodes: if None, returns an array of the desired argument

        Returns:
            value/s of the desired argument
        """
        if nodes is None:
            val_impl = self._impl[val_type]
        else:
            nodes, _ = misc.checkNodes(nodes, self._nodes_array)
            num_nodes = int(np.sum(self._nodes_array[nodes]))
            val_impl = cs.repmat(self._impl[val_type], 1, num_nodes)

        if isinstance(val_impl, cs.DM):
            val_impl = val_impl.toarray()

        return val_impl

    def getImpl(self, nodes=None):
        """
        Getter for the implemented parameter. Node is useless, since this parameter is node-independent.

        Args:
            node: useless input, used to simplify the framework mechanics

        Returns:
            instance of the implemented parameter
        """
        return self._getVals('var', nodes)

    def getNodes(self):
        """
        Getter for the active nodes.

        Returns:
            -1 since this parameter is node-independent
        """
        # todo what if I return all the nodes?
        return [-1]

    def getValues(self, nodes=None):
        """
        Getter for the value assigned to the parameter. It is the same throughout all the nodes, since this parameter is node-independent.

        Args:
            dummy_node: useless input, used to simplify the framework mechanics

        Returns:
            value assigned to the parameter
        """
        return self._getVals('val', nodes)

    def getParOffset(self, node):

        """
        Getter for the offset parameter. Since the parameter is the same along the horizon, it will point to itself.

        Args:
            nodes: offset of the node (shift to n node before or after)
        """
        return self

    def getParOffsetDict(self):
        """
        Getter for the offset parameter. Useless, since this parameter is node-independent.

        Returns:
            empty dict
        """
        return dict()

    def getName(self):
        """
        getter for the name of the parameter

        Returns:
            name of the parameter
        """
        return self._tag

    def __getitem__(self, item):
        par_slice = super().__getitem__(item)
        view = SingleParameterView(self, par_slice, item)
        return view

class SingleParameterView(AbstractVariableView):
    def __init__(self, parent: SingleParameter, var_slice, indices):
        super().__init__(parent, var_slice, indices)

    def assign(self, vals):
        """
        Assign a value to the parameter. Can be assigned also after the problem is built, before solving the problem.
        If not assigned, its default value is zero.

        Args:
            vals: value of the parameter
        """
        indices = np.array(range(self._parent._dim))[self._indices]
        self._parent.assign(vals, indices=indices)

class Parameter(AbstractVariable):
    """
    Parameter of Horizon Problem.
    It is used for parametric problems: it is a symbolic variable in the optimization problem but it is not optimized. Rather, it is kept parametric and can be assigned before solving the problem.
    """
    def __init__(self, tag, dim, nodes_array, casadi_type, abstract_casadi_type):
        """
        Initialize the Parameter: an abstract parameter projected over the horizon.
        Parameters are specified before building the problem and can be 'assigned' afterwards, before solving the problem.
        The assigned value can vary along the horizon, since this parameter is node-dependent.
        The Parameter is abstract, and gets implemented automatically.

        Args:
            tag: name of the parameter
            dim: dimension of the parameter
            nodes: nodes the parameter is implemented at
        """
        super().__init__(tag, dim, abstract_casadi_type)

        self._casadi_type = casadi_type

        self._par_offset = dict()
        self._impl = OrderedDict()

        self._nodes_array = nodes_array
        self._project()

    def _project(self):
        """
        Implements the parameter along the horizon nodes.
        Generates an ordered dictionary containing the implemented parameter and its value at each node {node: {par, val}}
        """
        # todo how to re-project keeping the old variables
        #   right now it only rewrite everything
        new_par_impl = OrderedDict()

        # count how many nodes are active and create a matrix with only than number of columns (nodes)
        num_nodes = np.sum(self._nodes_array).astype(int)
        proj_dim = [self._dim, num_nodes]
        # the MX variable is created: dim x n_nodes
        par_impl = self._casadi_type.sym(self._tag, proj_dim[0], proj_dim[1])
        par_value = np.zeros([proj_dim[0], proj_dim[1]])

        new_par_impl['var'] = par_impl
        new_par_impl['val'] = par_value

        self._impl.clear()
        self._impl.update(new_par_impl)

        # par_impl = self._casadi_type.sym(self._tag, dim)
        # for n in self._nodes:
        #     if 'n' + str(n) in self._impl:
        #         new_par_impl['n' + str(n)] = self._impl['n' + str(n)]
        #     else:
        #         par_impl = self._casadi_type.sym(self._tag + '_' + str(n), self._dim)
        #         new_par_impl['n' + str(n)] = dict()
        #         new_par_impl['n' + str(n)]['par'] = par_impl
        #         new_par_impl['n' + str(n)]['val'] = np.zeros(self._dim)

    def getNodes(self):
        """
        Getter for the nodes of the parameters.

        Returns:
            list of nodes the parameter is active on.
        """
        return misc.getNodesFromBinary(self._nodes_array)

    def _setNNodes(self, n_nodes):
        """
        set a desired number of nodes to the parameter.

        Args:
            n_nodes: the desired number of nodes to be set
        """
        self._nodes = list(n_nodes)
        self._project()

    def _setNodes(self, node_array):

        self._nodes_array = node_array
        self._project()

    def assign(self, val, nodes=None, indices=None):
        """
       Assign a value to the parameter at a desired node. Can be assigned also after the problem is built, before solving the problem.
       If not assigned, its default value is zero.

       Args:
           val: value of the parameter
           nodes: nodes at which the parameter is assigned
       """
        # nodes
        if nodes is None:
            nodes = misc.getNodesFromBinary(self._nodes_array)
        else:
            nodes = misc.checkNodes(nodes, self._nodes_array)

        pos_nodes = misc.convertNodestoPos(nodes, self._nodes_array)

        val = misc.checkValueEntry(val)

        # indices
        if indices is None:
            indices_vec = np.array(range(self._dim)).astype(int)
        else:
            indices_vec = np.array(indices).astype(int)

        val_checked = misc.checkValueEntry(val)

        if val_checked.shape[0] != indices_vec.size:
            raise Exception(f'Wrong dimension of parameter values inserted: ({val_checked.shape[0]} != {indices_vec.size})')

        # if a matrix of values is being provided, check cols match len(nodes)
        multiple_vals = val_checked.ndim == 2 and val_checked.shape[1] != 1

        if multiple_vals and val_checked.shape[1] != len(nodes):
            raise Exception(f'Wrong dimension of parameter inserted.')
        # todo this is because what I receive as val is 1-dimensional array which cannot be assigned to a matrix
        self._impl['val'][np.ix_(indices_vec, pos_nodes)] = val_checked

    def getImpl(self, nodes=None):
        """
        Getter for the implemented parameter.

        Args:
            node: node at which the parameter is retrieved. If not specified, this function returns an SX array with all the implemented parameters along the nodes.

        Returns:
            implemented instances of the abstract parameter
        """
        return self._getVals('var', nodes)

    def getValues(self, nodes=None):
        """
        Getter for the value of the parameter.

        Args:
            node: node at which the value of the parameter is retrieved. If not specified, this function returns a matrix with all the values of the parameter along the nodes.

        Returns:
            value/s of the parameter
        """
        return self._getVals('val', nodes)

    def _getVals(self, val_type, nodes):
        """
        wrapper function to get the desired argument from the variable.

        Args:
            val_type: type of the argument to retrieve
            nodes: if None, returns an array of the desired argument

        Returns:
            value/s of the desired argument
        """
        if nodes is None:
            nodes = misc.getNodesFromBinary(self._nodes_array)
        else:
            nodes = misc.checkNodes(nodes, self._nodes_array)

        pos_nodes = misc.convertNodestoPos(nodes, self._nodes_array)

        par_impl = self._impl[val_type][:, pos_nodes]

        return par_impl

    def getName(self):
        """
        getter for the name of the parameter

        Returns:
            name of the parameter
        """
        return self._tag

    def getParOffset(self, node):

        """
        Getter for the offset parameter. An offset parameter is used to point to the desired implemented instance of the abstract parameter.

        Examples:
            Abstract parameter "p". Horizon nodes are N.

            Implemented parameter "p" --> p_0, p_1, ... p_N-1, p_N

            Offset parameter "p-1" points FOR EACH NODE at variable "p" implemented at the PREVIOUS NODE.

        Args:
            nodes: offset of the node (shift to n node before or after)
        """

        # if node == 0:
        #     return self

        if node > 0:
            node = f'+{node}'

        if node in self._par_offset:
            return self._par_offset[node]
        else:

            createTag = lambda name, node: name + str(node) if node is not None else name

            new_tag = createTag(self._tag, node)
            par = OffsetTemplate(self._tag, new_tag, self._dim, int(node), self._nodes_array, self._impl, self._casadi_type, self._abstract_casadi_type)

            self._par_offset[node] = par
        return par

    def getParOffsetDict(self):
        """
        Getter for the offset variables.

        Returns:
            dict with all the offset variables referring to this abstract variable
        """
        return self._par_offset

    def __getitem__(self, item):
        par_slice = super().__getitem__(item)
        view = ParameterView(self, par_slice, item)
        return view

    def __reduce__(self):
        """
        Experimental function to serialize this element.

        Returns:
            instance of this element serialized
        """
        return (self.__class__, (self._tag, self._dim, self._nodes,))

class ParameterView(AbstractVariableView):
    def __init__(self, parent: SingleParameter, var_slice, indices):
        super().__init__(parent, var_slice, indices)

    def assign(self, vals, nodes=None):
        """
       Assign a value to the parameter at a desired node. Can be assigned also after the problem is built, before solving the problem.
       If not assigned, its default value is zero.

       Args:
           vals: value of the parameter
           nodes: nodes at which the parameter is assigned
       """
        indices = np.array(range(self._parent._dim))[self._indices]
        self._parent.assign(vals, nodes=nodes, indices=indices)

    def getValues(self, nodes=None):
        """
                Getter for the value of the parameter.

                Args:
                    node: node at which the value of the parameter is retrieved. If not specified, this function returns a matrix with all the values of the parameter along the nodes.

                Returns:
                    value/s of the parameter
                """
        par_impl = self._parent.getValues(nodes)[self._indices, :]

        return par_impl

class SingleVariable(AbstractVariable):
    """
    Single Variable of Horizon Problem: generic variable of the optimization problem.
    The single variable is the same along the horizon, since it is node-independent.
    The Variable is abstract, and gets implemented automatically.
    """
    def __init__(self, tag, dim, nodes_array, casadi_type, abstract_casadi_type):
        """
        Initialize the Single Variable: a node-independent variable which is not projected over the horizon.
        The bounds of the variable are initialized to -inf/inf.

        Args:
            tag: name of the variable
            dim: dimension of the variable
            nodes_array: binary array specifying which node is active
        """
        super().__init__(tag, dim, abstract_casadi_type)

        self._casadi_type = casadi_type
        self._nodes_array = nodes_array
        self._impl = dict()
        # todo do i create another var or do I use the SX var inside SingleVariable?
        self._impl['var'] = self._casadi_type.sym(self._tag + '_impl', self._dim)
        self._impl['lb'] = np.full([self._dim, 1], -np.inf)
        self._impl['ub'] = np.full([self._dim, 1], np.inf)
        self._impl['w0'] = np.zeros([self._dim, 1])

    def _setVals(self, val_type, val, indices=None):
        """
        Generic setter.

        Args:
            val_type: desired type of values to set
            val: values
            indices: select the indices to set
        """
        if indices is None:
            indices_vec = np.array(range(self._dim)).astype(int)
        else:
            indices_vec = np.array(indices).astype(int)

        val_checked = misc.checkValueEntry(val)
        if val_checked.shape[0] != indices_vec.size:
            raise Exception('Wrong dimension of parameter values inserted.')

        self._impl[val_type][indices_vec] = val_checked

    def setLowerBounds(self, bounds):
        """
        Setter for the lower bounds of the variable.

        Args:
            bounds: value of the lower bounds
        """
        self._setVals('lb', bounds)

    def setUpperBounds(self, bounds):
        """
        Setter for the upper bounds of the variable.

        Args:
            bounds: value of the upper bounds
        """
        self._setVals('ub', bounds)

    def setBounds(self, lb, ub):
        """
        Setter for the bounds of the variable.

        Args:
            lb: value of the lower bounds
            ub: value of the upper bounds
        """
        self.setLowerBounds(lb)
        self.setUpperBounds(ub)

    def setInitialGuess(self, val):
        """
        Setter for the initial guess of the variable.

        Args:
            val: value of the initial guess
        """
        self._setVals('w0', val)

    def _getVals(self, val_type, nodes):
        """
        wrapper function to get the desired argument from the variable.

        Args:
            val_type: type of the argument to retrieve
            dummy_node: if None, returns an array of the desired argument

        Returns:
            value/s of the desired argument
        """
        if nodes is None:
            val_impl = self._impl[val_type]
        else:
            nodes, _ = misc.checkNodes(nodes, self._nodes_array)
            num_nodes = int(np.sum(self._nodes_array[nodes]))
            val_impl = cs.repmat(self._impl[val_type], 1, num_nodes)

        if isinstance(val_impl, cs.DM):
            val_impl = val_impl.toarray()

        return val_impl

    def getImpl(self, dummy_node=None):
        """
        Getter for the implemented variable. Node is useless, since this variable is node-independent.

        Args:
            dummy_node: useless input, used to simplify the framework mechanics

        Returns:
            implemented instances of the abstract variable
        """
        return self._getVals('var', dummy_node)

    def getLowerBounds(self, dummy_node=None):
        """
        Getter for the lower bounds of the variable.

        Args:
            node: useless input, used to simplify the framework mechanics

        Returns:
            values of the lower bounds

        """
        return self._getVals('lb', dummy_node)

    def getUpperBounds(self, dummy_node=None):
        """
        Getter for the upper bounds of the variable.

        Args:
            node: useless input, used to simplify the framework mechanics

        Returns:
            values of the upper bounds

        """
        return self._getVals('ub', dummy_node)

    def getBounds(self, dummy_node=None):
        """
        Getter for the bounds of the variable.

        Args:
            node: useless input, used to simplify the framework mechanics

        Returns:
            values of the bounds

        """
        return self.getLowerBounds(dummy_node), self.getUpperBounds(dummy_node)

    def getInitialGuess(self, dummy_node=None):
        """
        Getter for the initial guess of the variable.

        Args:
            node: useless input, used to simplify the framework mechanics

        Returns:
            values of the initial guess
        """
        return self._getVals('w0', dummy_node)

    def getNodes(self):
        """
        Getter for the active nodes of the variable.

        Returns:
            -1 since this parameter is node-independent
        """
        # todo what if I return all the nodes?
        return [-1]

    def getVarOffset(self, node):

        """
        Getter for the offset variable. Since the variable is the same along the horizon, it will point to itself.

        Args:
            nodes: offset of the node (shift to n node before or after)
        """
        return self

    def getVarOffsetDict(self):
        """
        Getter for the offset variables. Useless, since this variable is node-independent.

        Returns:
            empty dict
        """
        return dict()

    def getImplDim(self):
        """
        Getter for the dimension of the implemented variables.

        Returns:
            dimension of the variable
        """
        return self.shape[0]

    def getName(self):
        """
        getter for the name of the variable

        Returns:
            name of the variable
        """
        return self._tag

    def toParameter(self):
        # self, tag, dim, nodes_array, casadi_type, abstract_casadi_type
        return SingleParameter(self._tag, self.getDim(), self._nodes_array, self._casadi_type, self._abstract_casadi_type)

    def __getitem__(self, item):
        var_slice = super().__getitem__(item)
        view = SingleVariableView(self, var_slice, item)
        return view

class SingleVariableView(AbstractVariableView):
    def __init__(self, parent: SingleVariable, var_slice, indices):
        super().__init__(parent, var_slice, indices)

    def _setVals(self, val_type, input_val):
        """
        Generic setter.

        Args:
            val_type: desired type of values to set
            input_val: values
        """
        indices = np.array(range(self._parent._dim))[self._indices]
        self._parent._setVals(val_type, input_val, indices=indices)

    def setLowerBounds(self, bounds):
        """
        Setter for the lower bounds of the variable.

        Args:
            bounds: value of the lower bounds
        """
        self._setVals('lb', bounds)

    def setUpperBounds(self, bounds):
        """
        Setter for the upper bounds of the variable.

        Args:
            bounds: value of the upper bounds
        """
        self._setVals('ub', bounds)

    def setBounds(self, lb, ub):
        """
        Setter for the bounds of the variable.

        Args:
            lb: value of the lower bounds
            ub: value of the upper bounds
        """
        self.setLowerBounds(lb)
        self.setUpperBounds(ub)

    def setInitialGuess(self, val):
        """
        Setter for the initial guess of the variable.

        Args:
            val: value of the initial guess
        """
        self._setVals('w0', val)

class Variable(AbstractVariable):
    """
    Variable of Horizon Problem: generic variable of the optimization problem.
    The Variable is abstract, and gets implemented automatically over the horizon nodes.

    Examples:
        Abstract variable "x". Horizon nodes are N.

        Implemented variable "x" --> x_0, x_1, ... x_N-1, x_N
    """
    def __init__(self, tag, dim, nodes_array, casadi_type, abstract_casadi_type):
        """
        Initialize the Variable.
        The bounds of the variable are initialized to -inf/inf.

        Args:
            tag: name of the variable
            dim: dimension of the variable
            nodes_array: binary array specifying the variable is defined on
        """
        super().__init__(tag, dim, abstract_casadi_type)

        self._abstract_casadi_type = abstract_casadi_type
        self._casadi_type = casadi_type

        self._nodes_array = np.array(nodes_array)

        self.var_offset = dict()
        self._impl = dict()

        # project the variable over the optimization nodes
        self._project()

    def _setVals(self, val_type, val, nodes=None, indices=None):
        """
        Generic setter.

        Args:
            val_type: desired values to set
            val: values
            nodes: which nodes the values are applied on
            indices: which indices the values are applied on
        """
        # nodes
        if nodes is None:
            nodes = misc.getNodesFromBinary(self._nodes_array)
        else:
            nodes = misc.checkNodes(nodes, self._nodes_array)

        pos_nodes = misc.convertNodestoPos(nodes, self._nodes_array)

        # indices
        if indices is None:
            indices_vec = np.array(range(self._dim)).astype(int)
        else:
            indices_vec = np.array(indices).astype(int)

        val_checked = misc.checkValueEntry(val)
        if val_checked.shape[0] != indices_vec.size:
            raise Exception(f'Wrong dimension of variable values inserted: {val_checked.shape[0]} instead of {indices_vec.size}')

        self._impl[val_type][np.ix_(np.atleast_1d(indices_vec), pos_nodes)] = val_checked

    def setLowerBounds(self, bounds, nodes=None):
        """
        Setter for the lower bounds of the variable.

        Args:
            bounds: desired bounds of the variable
            nodes: which nodes the bounds are applied on. If not specified, the variable is bounded along ALL the nodes
        """
        self._setVals('lb', bounds, nodes)

    def setUpperBounds(self, bounds, nodes=None):
        """
        Setter for the upper bounds of the variable.

        Args:
            bounds: desired bounds of the variable
            nodes: which nodes the bounds are applied on. If not specified, the variable is bounded along ALL the nodes
        """
        self._setVals('ub', bounds, nodes)

    def setBounds(self, lb, ub, nodes=None):
        """
        Setter for the bounds of the variable.

        Args:
            lb: desired lower bounds of the variable
            ub: desired upper bounds of the variable
            nodes: which nodes the bounds are applied on. If not specified, the variable is bounded along ALL the nodes
        """
        self.setLowerBounds(lb, nodes)
        self.setUpperBounds(ub, nodes)

    def setInitialGuess(self, val, nodes=None):
        """
        Setter for the initial guess of the variable.

        Args:
            val: desired initial guess of the variable
            nodes: which nodes the bounds are applied on. If not specified, the variable is bounded along ALL the nodes
        """
        self._setVals('w0', val, nodes)

    def getVarOffset(self, node):

        """
        Getter for the offset variable. An offset variable is used to point to the desired implemented instance of the abstract variable.

        Examples:
            Abstract variable "x". Horizon nodes are N.

            Implemented variable "x" --> x_0, x_1, ... x_N-1, x_N

            Offset variable "x-1" points FOR EACH NODE at variable "x" implemented at the PREVIOUS NODE.

        Args:
            nodes: offset of the node (shift to n node before or after)
        """

        # todo call it .getOffset()

        # if node == 0:
        #     return self

        if node > 0:
            node = f'+{node}'

        if node in self.var_offset:
            return self.var_offset[node]
        else:

            createTag = lambda name, node: name + str(node) if node is not None else name

            new_tag = createTag(self._tag, node)
            var = OffsetTemplate(self._tag, new_tag, self._dim, int(node), self._nodes_array, self._impl, self._casadi_type, self._abstract_casadi_type)

            self.var_offset[node] = var
        return var

    def getVarOffsetDict(self):
        """
        Getter for the offset variables.

        Returns:
            dict with all the offset variables referring to this abstract variable
        """
        return self.var_offset

    def _setNNodes(self, n_nodes):
        """
        set a desired number of nodes to the variable.

        Args:
            n_nodes: the desired number of nodes to be set
        """
        self._nodes_array = list(n_nodes)
        self._project()

    def _setNodes(self, node_array):

        self._nodes_array = node_array
        self._project()
    # def _project(self):
    #     """
    #     Implements the variable along the horizon nodes.
    #     Generates an ordered dictionary containing the implemented variables and its value at each node {node: {var, lb, ub, w0}}
    #     """
    #     new_var_impl = OrderedDict()
    #
    #     for n in self._nodes_array:
    #         if 'n' + str(n) in self._impl:
    #             # when reprojecting, if the implemented variable is present already, use it. Do not create a new one.
    #             new_var_impl['n' + str(n)] = self._impl['n' + str(n)]
    #         else:
    #             var_impl = self._casadi_type.sym(self._tag + '_' + str(n), self._dim)
    #             new_var_impl['n' + str(n)] = dict()
    #             new_var_impl['n' + str(n)]['var'] = var_impl
    #             new_var_impl['n' + str(n)]['lb'] = np.full(self._dim, -np.inf)
    #             new_var_impl['n' + str(n)]['ub'] = np.full(self._dim, np.inf)
    #             new_var_impl['n' + str(n)]['w0'] = np.zeros(self._dim)
    #
    #     # this is to keep the instance at the same memory position (since it is shared by the OffsetVariable)
    #     self._impl.clear()
    #     self._impl.update(new_var_impl)

    def _project(self):
        """
        Implements the variable along the horizon nodes.
        Generates an ordered dictionary containing the implemented variables and its value at each node {node: {var, lb, ub, w0}}
        """
        # todo how to re-project keeping the old variables
        #   right now it only rewrite everything

        new_var_impl = OrderedDict()
        # self._nodes contains the actual nodes on which the variable is defined
        num_nodes = np.sum(self._nodes_array).astype(int)
        proj_dim = [self._dim, num_nodes]
        # the MX variable is created: dim x n_nodes
        var_impl = self._casadi_type.sym(self._tag, proj_dim[0], proj_dim[1])
        var_lb = np.full((proj_dim[0], proj_dim[1]), -np.inf)
        var_ub = np.full((proj_dim[0], proj_dim[1]), np.inf)
        var_w0 = np.zeros([proj_dim[0], proj_dim[1]])


        new_var_impl['var'] = var_impl
        new_var_impl['lb'] = var_lb
        new_var_impl['ub'] = var_ub
        new_var_impl['w0'] = var_w0

        # this is to keep the instance at the same memory position (since it is shared by the OffsetVariable)
        self._impl.clear()
        self._impl.update(new_var_impl)

    def _getVals(self, val_type, nodes):
        """
        wrapper function to get the desired argument from the variable.

        Args:
            val_type: type of the argument to retrieve
            node: desired node at which the argument is retrieved. If not specified, this returns the desired argument at all nodes

        Returns:
            value/s of the desired argument
        """
        if nodes is None:
            checked_nodes = misc.getNodesFromBinary(self._nodes_array).tolist()
        else:
            checked_nodes = misc.checkNodes(nodes, self._nodes_array)
            # checked_nodes_array = [self._nodes_array[i] for i in nodes]
            # checked_nodes = [node for node in enumerate(nodes) if self._nodes_array[node] == 1]
            # checked_nodes = []
            # pos_in_vec = []
            # i = 0
            # for node in nodes:
            #     if self._nodes_array[node] == 1:
            #         checked_nodes.append(node)
                    # pos_in_vec.append(i)
                # i += 1
            # checked_nodes, not_impl_nodes = misc.checkNodes(nodes, self._nodes_array)

        pos_nodes = misc.convertNodestoPos(checked_nodes, self._nodes_array)

        # todo: hack to return matrix with nan values where the variable is not active/implemented
        # if val_type == 'var':
        #     vals = self._casadi_type(self.getDim(), len(nodes))
        #     vals[:] = np.nan
        # else:
        #     vals = np.nan * np.ones([self.getDim(), len(nodes)])
        #
        # vals[:, pos_in_vec] = self._impl[val_type][:, pos_nodes]
        # TODO: before was like this:
        vals = self._impl[val_type][:, pos_nodes]


        return vals

    def getImpl(self, nodes=None):
        """
        Getter for the implemented variable.

        Args:
            node: node at which the variable is retrieved

        Returns:
            implemented instances of the abstract variable
        """
        return self._getVals('var', nodes)

    def getLowerBounds(self, node=None):
        """
        Getter for the lower bounds of the variable.

        Args:
            node: desired node at which the lower bounds are retrieved. If not specified, this returns the lower bounds at all nodes

        Returns:
            value/s of the lower bounds

        """
        return self._getVals('lb', node)

    def getUpperBounds(self, node=None):
        """
        Getter for the upper bounds of the variable.

        Args:
            node: desired node at which the upper bounds are retrieved. If not specified, this returns the upper bounds at all nodes

        Returns:
            value/s of the upper bounds

        """
        return self._getVals('ub', node)

    def getBounds(self, node=None):
        """
        Getter for the bounds of the variable.

        Args:
            node: desired node at which the bounds are retrieved. If not specified, this returns the bounds at all nodes

        Returns:
            value/s of the bounds

        """
        return self.getLowerBounds(node), self.getUpperBounds(node)

    def getInitialGuess(self, node=None):
        """
        Getter for the initial guess of the variable.

        Args:
            node: desired node at which the initial guess is retrieved. If not specified, this returns the lower bounds at all nodes

        Returns:
            value/s of the bounds

        """
        return self._getVals('w0', node)

    def getImplDim(self):
        """
        Getter for the dimension of the implemented variables, considering all the nodes.

        Returns:
            dimension of the variable multiplied by number of nodes
        """
        return self.shape[0] * len(self.getNodes())

    def getNodes(self):
        """
        Getter for the active nodes of the variable.

        Returns:
            the nodes the variable is defined on
        """
        return misc.getNodesFromBinary(self._nodes_array)

    def getName(self):
        """
        getter for the name of the variable

        Returns:
            name of the variable
        """
        return self._tag

    def toParameter(self):
        # self, tag, dim, nodes_array, casadi_type, abstract_casadi_type
        return Parameter(self._tag, self.getDim(), self._nodes_array, self._casadi_type, self._abstract_casadi_type)

    def __getitem__(self, item):
        var_slice = super().__getitem__(item)
        view = VariableView(self, var_slice, item)
        return view

    def __reduce__(self):
        """
        Experimental function to serialize this element.

        Returns:
            instance of this element serialized
        """
        return (self.__class__, (self._tag, self._dim, self._nodes_array,))

class VariableView(AbstractVariableView):
    def __init__(self, parent: Variable, var_slice, indices):
        super().__init__(parent, var_slice, indices)

    def _setVals(self, val_type, val, nodes=None):
        """
        Generic setter.

        Args:
            bounds: desired values to set
            nodes: which nodes the values are applied on
        """
        indices = np.array(range(self._parent._dim))[self._indices]
        self._parent._setVals(val_type, val, nodes, indices)

    def _getVals(self, val_type, nodes):
        """
        wrapper function to get the desired argument from the variable.

        Args:
            val_type: type of the argument to retrieve
            node: desired node at which the argument is retrieved. If not specified, this returns the desired argument at all nodes

        Returns:
            value/s of the desired argument
        """
        indices = np.array(range(self._parent._dim))[self._indices]
        vals = self._parent._getVals(val_type, nodes)[indices]

        return vals

    def getImpl(self, nodes=None):
        """
        Getter for the implemented variable.

        Args:
            node: node at which the variable is retrieved

        Returns:
            implemented instances of the variable
        """
        var_impl = self._parent.getImpl(nodes)[self._indices, :]
        return var_impl

    def getBounds(self, node=None):
        """
        Getter for the bounds of the variable.

        Args:
            node: desired node at which the bounds are retrieved. If not specified, this returns the bounds at all nodes

        Returns:
            value/s of the bounds

        """
        return self.getLowerBounds(node), self.getUpperBounds(node)

    def getLowerBounds(self, node=None):
        """
        Getter for the lower bounds of the variable.

        Args:
            node: desired node at which the lower bounds are retrieved. If not specified, this returns the lower bounds at all nodes

        Returns:
            value/s of the lower bounds

        """
        return self._getVals('lb', node)

    def getUpperBounds(self, node=None):
        """
        Getter for the lower bounds of the variable.

        Args:
            node: desired node at which the lower bounds are retrieved. If not specified, this returns the lower bounds at all nodes

        Returns:
            value/s of the lower bounds

        """
        return self._getVals('ub', node)

    def setLowerBounds(self, bounds, nodes=None):
        """
        Setter for the lower bounds of the variable.

        Args:
            bounds: desired bounds of the variable
            nodes: which nodes the bounds are applied on. If not specified, the variable is bounded along ALL the nodes
        """
        self._setVals('lb', bounds, nodes)

    def setUpperBounds(self, bounds, nodes=None):
        """
        Setter for the upper bounds of the variable.

        Args:
            bounds: desired bounds of the variable
            nodes: which nodes the bounds are applied on. If not specified, the variable is bounded along ALL the nodes
        """
        self._setVals('ub', bounds, nodes)

    def setBounds(self, lb, ub, nodes=None):
        """
        Setter for the bounds of the variable.

        Args:
            lb: desired lower bounds of the variable
            ub: desired upper bounds of the variable
            nodes: which nodes the bounds are applied on. If not specified, the variable is bounded along ALL the nodes
        """
        self.setLowerBounds(lb, nodes)
        self.setUpperBounds(ub, nodes)

    def setInitialGuess(self, val, nodes=None):
        """
        Setter for the initial guess of the variable.

        Args:
            val: desired initial guess of the variable
            nodes: which nodes the bounds are applied on. If not specified, the variable is bounded along ALL the nodes
        """
        self._setVals('w0', val, nodes)

    def getVarOffset(self, node):
        return self._parent.getVarOffset(node)[self._indices, :]

class RecedingVariable(Variable):
    def __init__(self, tag, dim, nodes_array, casadi_type, abstract_casadi_type):
        super().__init__(tag, dim, nodes_array, casadi_type=casadi_type, abstract_casadi_type=abstract_casadi_type)

    def shift(self):

        print(f'============= VARIABLE ================')
        print(f'NAME: {self.getName()}')
        print(f'OLD LB: {self.getLowerBounds()}')
        print(f'OLD UB: {self.getUpperBounds()}')
        # shift bounds
        shift_num = -1

        shifted_lb = misc.shift_array(self.getLowerBounds(), shift_num, -np.inf)
        shifted_ub = misc.shift_array(self.getUpperBounds(), shift_num, np.inf)

        self.setLowerBounds(shifted_lb)
        self.setUpperBounds(shifted_ub)

        print(f'SHIFTED LB: {self.getLowerBounds()}')
        print(f'SHIFTED UB: {self.getUpperBounds()}')


# TODO: should I create the parameter on all the nodes and make it active only on the active_nodes, if nodes= is specified
class RecedingParameter(Parameter):
    def __init__(self, tag, dim, nodes_array, casadi_type, abstract_casadi_type):
        super().__init__(tag, dim, nodes_array, casadi_type=casadi_type, abstract_casadi_type=abstract_casadi_type)

    def shift(self):

        print(f'============= PARAMETER ================')
        print(f'NAME: {self.getName()}')
        print(f'OLD VALUES: {self.getValues()}')
        # shift values
        shift_num = -1
        shifted_vals = misc.shift_array(self.getValues(), shift_num, 0.)
        self.assign(shifted_vals)

        print(f'SHIFTED VALUES: {self.getValues()}')


class InputVariable(Variable):
    """
    Input (Control) Variable of Horizon Problem.
    The variable is abstract, and gets implemented automatically over the horizon nodes except the last one.

    Examples:
        Abstract variable "x". Horizon nodes are N.

        Implemented variable "x" --> x_0, x_1, ... x_N-1
    """
    def __init__(self, tag, dim, nodes, casadi_type, abstract_casadi_type):
        """
        Initialize the Input Variable.

        Args:
            tag: name of the variable
            dim: dimension of the variable
            nodes: should always be N-1, where N is the number of horizon nodes
        """
        super().__init__(tag, dim, nodes, casadi_type=casadi_type, abstract_casadi_type=abstract_casadi_type)

class StateVariable(Variable):
    """
    State Variable of Horizon Problem.
    The variable is abstract, and gets implemented automatically over all the horizon nodes.

    Examples:
        Abstract variable "x". Horizon nodes are N.

        Implemented variable "x" --> x_0, x_1, ... x_N-1, x_N
    """

    def __init__(self, tag, dim, nodes, casadi_type, abstract_casadi_type):
        """
        Initialize the State Variable.

        Args:
            tag: name of the variable
            dim: dimension of the variable
            nodes: should always be N, where N is the number of horizon nodes
        """
        super(StateVariable, self).__init__(tag, dim, nodes, casadi_type=casadi_type, abstract_casadi_type=abstract_casadi_type)

class RecedingInputVariable(RecedingVariable):
    def __init__(self, tag, dim, nodes, casadi_type, abstract_casadi_type):
        """
        Initialize the Receding Input Variable.

        Args:
            tag: name of the variable
            dim: dimension of the variable
            nodes: should always be N-1, where N is the number of horizon nodes
        """
        super().__init__(tag, dim, nodes, casadi_type=casadi_type, abstract_casadi_type=abstract_casadi_type)


class RecedingStateVariable(RecedingVariable):
    def __init__(self, tag, dim, nodes, casadi_type, abstract_casadi_type):
        """
        Initialize the Receding State Variable.

        Args:
            tag: name of the variable
            dim: dimension of the variable
            nodes: should always be N-1, where N is the number of horizon nodes
        """
        super().__init__(tag, dim, nodes, casadi_type=casadi_type, abstract_casadi_type=abstract_casadi_type)



class AbstractAggregate(ABC):
    """
    Abstract Aggregate of the Horizon Problem.
    Used to store more variables of the same nature.
    """
    def __init__(self, *args: AbstractVariable):
        """
        Initialize the Abstract Aggregate.

        Args:
            *args: abstract variables of the same nature
        """
        self.var_list : List[AbstractVariable] = [item for item in args]

    def getVars(self, abstr=False):
        """
        Getter for the variable stored in the aggregate.

        Returns:
            a casadi vector of all the variables stored
        """

        if abstr:
            aggr_vars = self.var_list
        else:
            aggr_vars = cs.vertcat(*self.var_list)
        return aggr_vars

    def __iter__(self):
        """
        Aggregate can be treated as an iterable.
        """
        yield from self.var_list

    def __getitem__(self, ind):
        """
        Aggregate can be accessed with indexing.
        """
        return self.var_list[ind]

class OffsetAggregate(AbstractAggregate):
    """
        Offset Aggregate of the Horizon Problem.
        Used to store more offset variables of the same nature.
        """

    def __init__(self, *args):
        """
        Initialize the Aggregate.

        Args:
            *args: instances of abstract variables of the same nature
        """
        super().__init__(*args)

    def getVarIndex(self, name):
        """
        Return offset and dimension for the variable with given name,
        that must belong to this aggregate. The resulting pair is such
        that the following code returns the variable's SX value
            off, dim = self.getVarIndex(name='myvar')
            v = self.getVars()[off:off+dim]

        Args:
            name ([type]): [description]
        """
        names = [v.getName() for v in self.var_list]
        i = names.index(name)
        offset = sum(v.getDim() for v in self.var_list[:i])
        return offset, self.var_list[i].getDim()

class Aggregate(AbstractAggregate):
    """
    Aggregate of the Horizon Problem.
    Used to store more variables of the same nature.
    """
    def __init__(self, *args):
        """
        Initialize the Aggregate.

        Args:
            *args: instances of abstract variables of the same nature
        """
        super().__init__(*args)

    def getVarOffset(self, offset):
        """
        Getter for the offset variables contained in the aggregate.

        Returns:
            an abstract aggregate with all the offset variables referring to the relative abstract variables
        """
        var_list = list()
        for var in self.var_list:
            var_list.append(var.getVarOffset(offset))

        return OffsetAggregate(*var_list)

    def getVarIndex(self, name):
        """
        Return offset and dimension for the variable with given name, 
        that must belong to this aggregate. The resulting pair is such
        that the following code returns the variable's SX value
            off, dim = self.getVarIndex(name='myvar')
            v = self.getVars()[off:off+dim]

        Args:
            name ([type]): [description]
        """
        names = [v.getName() for v in self.var_list]
        i = names.index(name)
        offset = sum(v.getDim() for v in self.var_list[:i])
        return offset, self.var_list[i].getDim()

    def addVariable(self, var):
        """
        Adds a Variable to the Aggregate.

        Todo:
            Should check if variable type belongs to the aggregate type (no mixing!)

        Args:
            var: variable to be added to the aggregate
        """
        self.var_list.append(var)

    def removeVariable(self, var_name):
        """
        Remove a Variable from the Aggregate.

        Todo:
            Should check if variable type belongs to the aggregate type (no mixing!)

        Args:
            var: variable to be removed from the aggregate
        """

        # todo make this a little better
        for i in range(len(self.var_list)):
            if var_name == self.var_list[i].getName():

                del self.var_list[i]
                break


    def setBounds(self, lb, ub, nodes=None):
        """
        Setter for the bounds of the variables in the aggregate.

        Args:
            lb: desired lower bounds of the variable
            ub: desired upper bounds of the variable
            nodes: which nodes the bounds are applied on. If not specified, the variable is bounded along ALL the nodes
        """
        self.setLowerBounds(lb, nodes)
        self.setUpperBounds(ub, nodes)

    def setLowerBounds(self, lb, nodes=None):
        """
        Setter for the lower bounds of the variables in the aggregate.

        Args:
            bounds: list of desired bounds of the all the variables in the aggregate
            nodes: which nodes the bounds are applied on. If not specified, the variable is bounded along ALL the nodes
        """
        idx = 0
        for var in self:
            nv = var.shape[0]
            var.setLowerBounds(lb[idx:idx+nv], nodes)
            idx += nv

    def setUpperBounds(self, ub, nodes=None):
        """
        Setter for the upper bounds of the variables in the aggregate.

        Args:
            bounds: list of desired bounds of the all the variables in the aggregate
            nodes: which nodes the bounds are applied on. If not specified, the variable is bounded along ALL the nodes
        """
        idx = 0
        for var in self:
            nv = var.shape[0]
            var.setUpperBounds(ub[idx:idx+nv], nodes)
            idx += nv

    def setInitialGuess(self, v0, nodes=None):
        """
        Args:
            v0 ([type]): [description]
            nodes ([type], optional): [description]. Defaults to None.
        """
        idx = 0
        for var in self:
            nv = var.shape[0]
            var.setInitialGuess(v0[idx:idx+nv], nodes)
            idx += nv    
    
    def getBounds(self, node=None):
        """
        Getter for the bounds of the variables in the aggregate.

        Args:
            node: which nodes the bounds are applied on. If not specified, the variable is bounded along ALL the nodes

        Returns:
            array of bound values of each variable in the aggregate

        todo:
            test this!
        """
        lb = self.getLowerBounds(node)
        ub = self.getUpperBounds(node)

        return lb, ub

    def getLowerBounds(self, node):
        """
        Getter for the lower bounds of the variables in the aggregate.

        Args:
            node: which nodes the lower bounds are applied on. If not specified, the variable is bounded along ALL the nodes

        Returns:
            array of lower bound values of each variable in the aggregate

        todo:
            test this!
        """
        return np.vstack([var.getLowerBounds(node) for var in self])

    def getUpperBounds(self, node):
        """
        Getter for the upper bounds of the variables in the aggregate.

        Args:
            node: which nodes the upper bounds are applied on. If not specified, the variable is bounded along ALL the nodes

        Returns:
            array of upper bound values of each variable in the aggregate

        todo:
            test this!
        """
        return np.vstack([var.getUpperBounds(node) for var in self])

    def getInitialGuess(self, node=None) -> np.array:
        """
        [summary]

        Args:
            node ([type]): [description]

        Returns:
            [type]: [description]
        """

        ig_list = list()

        for var in self:
            num_nodes = len(var.getNodes()) if node is None else len(node)
            ig = var.getInitialGuess(node)
            ig = ig.reshape((var.getDim(), num_nodes), order='F')
            ig_list.append(ig)

        return np.vstack(ig_list)

class StateAggregate(Aggregate):
    """
    State Aggregate of the Horizon Problem.
    Used to store all the state variables.
    """
    def __init__(self, *args: StateVariable):
        """
        Initialize the State Aggregate.

        Args:
            *args: instances of state variables
        """
        super().__init__(*args)

class InputAggregate(Aggregate):
    """
    Input (Control) Aggregate of the Horizon Problem.
    Used to store all the control variables.
    """
    def __init__(self, *args: InputVariable):
        """
        Initialize the Input (Control) Aggregate.

        Args:
            *args: instances of input (control) variables
        """
        super().__init__(*args)

# todo what if this is completely useless? at the end of the day, I need this Container for:
#   .getVarAbstrDict() --> get all abstract variables (special attention for the past variables)
#   .getVarImpl(): ---> get implemented variable at node
#   .getVarImplList() ---> get all the implemented variables as list
#   .getVarImplDict() ---> get all the implemented variables as dict
#   Can I do something else? Right now .build() orders them like as follows:
#            (nNone: [vars..]
#                n0: [vars..],
#                n1: [vars..], ...)
#   but since order is everything I care about, is there a simpler way?
#   for var in self.vars:
#    all_impl += var.getAllImpl
#   this is ordered with the variables and I probably don't need build?
#            (x: [n0, n1, n2 ...]
#             y: [n0, n1, n2, ...]
#             z: [nNone, n0, n1, ...])

#

default_abstract_casadi_type = cs.SX
default_casadi_type = cs.MX

class ItemContainer:
    def __init__(self, item):
        self.item = item

class VariablesContainer:
    """
    Container of all the variables of Horizon.
    It is used internally by the Problem to get the abstract and implemented variables.
    """
    def __init__(self, is_receding, logger=None):
        """
        Initialize the Variable Container.

        Args:
           nodes: the number of nodes of the problem
           logger: a logger reference to log data
        """
        self.is_receding = is_receding
        self._logger = logger

        self._vars = OrderedDict()
        self._pars = OrderedDict()

    def createVar(self, var_type, name, dim, nodes_array, casadi_type, abstract_casadi_type):
        """
        Create a variable and adds it to the Variable Container.

        Args:
            var_type: type of variable
            name: name of variable
            dim: dimension of variable
            nodes_array: nodes the variable is defined on
            abstract_casadi_type: type of abstract variable of horizon (casadi SX or MX)
            casadi_type: type of implemented variable of horizon (casadi SX or MX)

        """
        var = var_type(name, dim, nodes_array, casadi_type=casadi_type, abstract_casadi_type=abstract_casadi_type)
        self._vars[name] = var

        if self._logger:
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug('Creating variable {} as {}'.format(name, var_type))

        return var

    def setVar(self, name, dim, nodes_array, casadi_type, abstract_casadi_type):
        """
        Creates a generic variable.

        Args:
            name: name of the variable
            dim: dimension of the variable
            active_nodes: nodes the variable is defined on. If not specified, a Single Variable is generated
            abstract_casadi_type: type of abstract variable of horizon (casadi SX or MX)
            casadi_type: type of implemented variable of horizon (casadi SX or MX)
        """

        if nodes_array is None:
            var_type = SingleVariable
        else:
            if self.is_receding:
                var_type = RecedingVariable
            else:
                var_type = Variable

        var = self.createVar(var_type, name, dim, nodes_array, casadi_type=casadi_type, abstract_casadi_type=abstract_casadi_type)
        return var

    def setStateVar(self, name, dim, nodes_array, casadi_type, abstract_casadi_type):
        """
        Creates a State variable.

        Args:
            name: name of the variable
            dim: dimension of the variable
            nodes_array: binary array of nodes specifying which node is active
            abstract_casadi_type: type of abstract variable of horizon (casadi SX or MX)
            casadi_type: type of implemented variable of horizon (casadi SX or MX)
        """

        if self.is_receding:
            var_type = RecedingStateVariable
        else:
            var_type = StateVariable

        var = self.createVar(var_type, name, dim, nodes_array, casadi_type=casadi_type, abstract_casadi_type=abstract_casadi_type)
        return var

    def setInputVar(self, name, dim, nodes_array, casadi_type, abstract_casadi_type):
        """
        Creates a Input (Control) variable.

        Args:
            name: name of the variable
            dim: dimension of the variable
            nodes_array: binary array of nodes specifying which node is active
            abstract_casadi_type: type of abstract variable of horizon (casadi SX or MX)
            casadi_type: type of implemented variable of horizon (casadi SX or MX)
        """

        if self.is_receding:
            var_type = RecedingInputVariable
        else:
            var_type = InputVariable


        var = self.createVar(var_type, name, dim, nodes_array, casadi_type=casadi_type, abstract_casadi_type=abstract_casadi_type)
        return var

    def setSingleVar(self, name, dim, nodes_array, casadi_type, abstract_casadi_type):
        """
        Creates a Single variable.

        Args:
            name: name of the variable
            dim: dimension of the variable
            nodes_array: binary array of nodes specifying which node is active
            abstract_casadi_type: type of abstract variable of horizon (casadi SX or MX)
            casadi_type: type of implemented variable of horizon (casadi SX or MX)
        """

        var = self.createVar(SingleVariable, name, dim, nodes_array, casadi_type=casadi_type, abstract_casadi_type=abstract_casadi_type)
        return var

    def setParameter(self, name, dim, nodes_array, casadi_type, abstract_casadi_type):
        """
        Creates a Parameter.

        Args:
            name: name of the variable
            dim: dimension of the variable
            nodes_array: binary array of nodes specifying which node is active. If not specified, all the horizon nodes are considered
            abstract_casadi_type: type of abstract variable of horizon (casadi SX or MX)
            casadi_type: type of implemented variable of horizon (casadi SX or MX)
        """

        if self.is_receding:
            par_type = RecedingParameter
        else:
            par_type = Parameter

        par = par_type(name, dim, nodes_array, casadi_type=casadi_type, abstract_casadi_type=abstract_casadi_type)
        self._pars[name] = par

        if self._logger:
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(f'Creating parameter "{name}"')

        return par

    def setSingleParameter(self, name, dim, nodes_array, casadi_type, abstract_casadi_type):
        """
        Creates a Single Variable.

        Args:
            name: name of the variable
            dim: dimension of the variable
            nodes_array: binary array of nodes specifying which node is active.
            abstract_casadi_type: type of abstract variable of horizon (casadi SX or MX)
            casadi_type: type of implemented variable of horizon (casadi SX or MX)
        """

        par = SingleParameter(name, dim, nodes_array, casadi_type=casadi_type, abstract_casadi_type=abstract_casadi_type)
        self._pars[name] = par

        if self._logger:
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(f'Creating single parameter "{name}"')

        return par

    def getStateVars(self):
        """
        Getter for the state variables in the Variable Container.

        Returns:
            a dict with all the state variables
        """
        state_vars = dict()
        for name, var in self._vars.items():
            if isinstance(var, StateVariable):
                state_vars[name] = var

        return state_vars

    def getInputVars(self):
        """
        Getter for the input (control) variables in the Variable Container.

        Returns:
            a dict with all the input (control) variables
        """
        input_vars = dict()
        for name, var in self._vars.items():
            if isinstance(var, InputVariable):
                input_vars[name] = var

        return input_vars

    def getVarList(self, offset=True):
        """
        Getter for the abstract variables in the Variable Container. Used by the Horizon Problem.

        Args:
            offset: if True, get also the offset_variable

        Returns:
            a list with all the abstract variables
        """
        var_abstr_list = list()
        for name, var in self._vars.items():
            var_abstr_list.append(var)
            if offset:
                for var_offset in var.getVarOffsetDict().values():
                    var_abstr_list.append(var_offset)

        return var_abstr_list

    def getParList(self, offset=True):
        """
        Getter for the abstract parameters in the Variable Container. Used by the Horizon Problem.

        Returns:
            a list with all the abstract parameters
        """
        par_abstr_list = list()
        for name, par in self._pars.items():
            par_abstr_list.append(par)
            if offset:
                for par_offset in par.getParOffsetDict().values():
                    par_abstr_list.append(par_offset)

        return par_abstr_list

    def getVar(self, name=None):
        """
        Getter for the abstract variables in the Variable Container.

        Args:
            name: name of the variable to be retrieve

        Returns:
            a dict with all the abstract variables
        """
        if name is None:
            var_dict = self._vars
        else:
            var_dict = self._vars[name]
        return var_dict

    def getPar(self, name=None):
        """
        Getter for the abstract parameters in the Variable Container.

        Returns:
            a dict with all the abstract parameters
        """
        if name is None:
            par_dict = self._pars
        else:
            par_dict = self._pars[name]
        return par_dict

    def removeVar(self, var_name):
        if var_name in self._vars:
            del self._vars[var_name]
            return True
        else:
            return False

    def setNNodes(self, n_nodes):
        """
        set a desired number of nodes to the Variable Container.

        Args:
            n_nodes: the desired number of nodes to be set
        """
        self._nodes = n_nodes

        for var in self._vars.values():
            if isinstance(var, SingleVariable):
                pass
            elif isinstance(var, InputVariable):
                var._setNNodes(list(range(self._nodes-1))) # todo is this right?
            elif isinstance(var, StateVariable):
                var._setNNodes(list(range(self._nodes)))
            elif isinstance(var, Variable):
                # todo Right now i'm only changing the number of nodes.
                #  There is not the notion of positional nodes, i.e.  injecting new nodes between two old nodes.
                #  this is not correct. For example:
                #  assume the variable is defined from node n to m.
                #  assume the nodes i'm injecting are inside this interval [n, m]. Just by changing the number of nodes
                #  is not enough.
                #  should add a .injectNodes(nodes, position)/.removeNodes(nodes, positon) so that I can expand/suppress the variables correctly
                var._setNNodes([node for node in var.getNodes() if node in list(range(self._nodes))])

        for par in self._pars.values():
            if isinstance(par, SingleParameter):
                pass
            elif isinstance(par, Parameter):
                par._setNNodes([node for node in par.getNodes() if node in list(range(self._nodes))])

    def serialize(self):
        """
        Serialize the Variable Container. Used to save it.

        Returns:
           instance of serialized Variable Container
        """

        # todo how to do? I may use __reduce__ but I don't know how
        for name, var in self._vars.items():
            self._vars[name] = var.serialize()

        for name, par in self._pars.items():
            self._pars[name] = par.serialize()

    def deserialize(self):
        """
        Deserialize the Variable Container. Used to load it.

        Returns:
           instance of deserialized Variable Container
        """
        pass
        # for name, var in self._vars.items():
        #     self._vars[name] = cs.SX.deserialize(var)
        #
        # for name, par in self._pars.items():
        #     self._pars[name] = cs.SX.deserialize(par)

    # def __reduce__(self):
    #     return (self.__class__, (self.nodes, self.logger, ))

if __name__ == '__main__':

    x = Variable('x', 4, [1, 1, 0, 1, 1, 1], casadi_type=cs.SX, abstract_casadi_type=cs.SX)
    x_par = x.toParameter()

    print(isinstance(x, SingleVariable), x)
    print(isinstance(x_par, SingleParameter), x_par)


    exit()
    # N = 10
    # # a = cs.MX.sym('a', 3, 1)
    # a = 5 * np.ones([3, 10])
    # a[:, 3] = 3
    # b = 2 * np.ones([3, 10])
    # print(a)
    # print(b)
    #
    # X = cs.MX.sym('X', 3, 1)
    # Y = cs.MX.sym('Y', 3, 1)
    # F = cs.Function('F', [X, Y], [X*Y])
    #
    # F.map(N, 'thread', 10)
    # tic = time.time()
    # res = F(a, b)
    # toc = time.time() - tic
    #
    # print(res)
    # print('time elapsed:', toc)
    # exit()
    ## PARAMETER
    # a = Parameter('p', 3, [0, 1, 2, 3, 4, 5])
    # print(a[2:4], f'type: {type(a[2:4])}')
    # a.assign([1, 1, 1])
    # a.assign([7, 7, 7], nodes=3)
    # a[1:3].assign([2, 3])
    # print(a.getValues())
    #
    # print(a[2].getValues(3))
    #
    # a_prev = a.getOffset(-1)
    # print(a.getImpl())
    # print(a_prev.getImpl([2]))
    #
    # fun = a_prev + a
    #
    # print(fun)
    #
    #
    # exit()
    # print(a_prev[0])
    #
    #
    #
    # x = Variable('x', 4, [0, 1, 2, 3, 4, 5])
    #
    # x_prev = x.getVarOffset(-2)
    # print(x.getImpl())
    # print(x_prev.getImpl())
    #
    # exit()
    ## INPUT
    # i = InputVariable('u', 6, [0, 1, 2, 3, 4, 5])
    # print(i[2:4], f'type: {type(i[2:4])}')
    # i[2:4].setLowerBounds([1, 1])
    # print(i.getLowerBounds())
    # exit()
    # STATE VARIABLE
    # p = StateVariable('x', 6, [0, 1, 2, 3, 4, 5])
    # print(p, f'type: {type(p)}')
    # print(p[0:2], f'type: {type(p[0:2])}')
    # print(p[0:2]+2)
    # p_sliced = p[0:2]
    # print(p_sliced[1])
    # print(p_sliced[0], f'type: {type(p_sliced[0])}')

    # p_sliced[0].setLowerBounds(5)

    # exit()

    # p[-1].setLowerBounds(0)
    # print(p.getLowerBounds())
    # print(p[0].setInitialGuess(10, [1]))
    # print(p.getInitialGuess())

    # SINGLE VARIABLE
    # p = SingleVariable('x', 6, [0, 1, 2])
    # print(p, f'type: {type(p)}')
    # print(p[0:2], f'type: {type(p[0:2])}')
    # print(p[0:2]+2)
    # print(f'type: {type(p[-1])}')
    # p.setUpperBounds([2,2,2,2,2,2])
    # p[-1].setLowerBounds(0)
    # p[2].setUpperBounds(3)
    # print(p.getLowerBounds())
    # print(p.getUpperBounds())

    # VARIABLE
    # x = Variable('x', 3, [0,1,2,3,4,5,6])
    # print(x, f'type: {type(x)}')
    # print(x[0:2], f'type: {type(x[0:2])}')
    # print(x[0:2]+2)
    # print(f'type: {type(x[-1])}')
    # x.setUpperBounds([2,2,2,2,2,2])
    # x = Variable('x', 4, [1, 1, 0, 1, 1, 1], casadi_type=cs.SX)
    # x_prev = x.getVarOffset(-1)
    # print(x)

    # print(x.getImpl(4))
    # print(x_prev)

    # print(x_prev.getImpl([2, 5]))

    slice_x = x[[0, 1]]
    print(slice_x.getVarOffset(-1))

    exit()
    x.setBounds([-1,-1,-1], [1,1,1], nodes=3)
    print(x.getBounds())

    # ub = np.array([[1,2,3], [1,2,3]])
    # print(ub)
    # x[0:2].setUpperBounds(ub, [2,3,4])
    # # x[2].setUpperBounds(3)
    # print(x.getLowerBounds())
    # print(x.getUpperBounds())

    # ig = np.array([[1,2,3], [1,2,3], [1,2,4]])
    # ig = np.array([1,2,3])
    # x.setInitialGuess(ig, [2,5,6])
    # x.setInitialGuess(ig)
    # print(x.getInitialGuess())

    #### SINGLE VARIABLE
    # x = SingleVariable('x', 3, [0, 1, 2, 3, 4, 5, 6])
    # print(x)
    # print(x.getImpl(0))
    # print(x.getImpl(1))
    # print(x.getImpl(2))
    # x_prev = x.getVarOffset(-3)



    # VARIABLE
    # x = Variable('x', 3, [0, 1, 2, 3, 4, 5, 6])
    # print(x)
    # print(x.getImpl(0))
    # print(x.getImpl(1))
    # print(x.getImpl(2))
    # x_prev = x.getVarOffset(-3)



