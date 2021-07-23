import casadi as cs
from collections import OrderedDict
import logging
import numpy as np
import pickle
import horizon.misc_function as misc

'''
now the StateVariable is only abstract at the very beginning.
Formerly
'''

# todo create function checker to check if nodes are in self.nodes and if everything is ok with the input (no dict, no letters...)
class StateVariable(cs.SX):
    def __init__(self, tag, dim, nodes):
        super(StateVariable, self).__init__(cs.SX.sym(tag, dim))

        self.tag = tag
        self.dim = dim
        self.nodes = nodes

        # self.var = cs.SX.sym(tag, dim)
        self.var_impl = dict()

        # todo project it as soon as I create the variable. Ok?
        self._project()

    def setLowerBounds(self, bounds, nodes=None):

        nodes = misc.checkNodes(nodes, range(self.nodes))

        if isinstance(bounds, (list, int, float)):
            bounds = np.array(bounds)
        else:
            bounds = bounds.flatten()

        dim = bounds.shape[0] if bounds.shape else 1
        if dim != self.dim:
            raise Exception('Wrong dimension of lower bounds inserted.')

        for node in nodes:
            self.var_impl['n' + str(node)]['lb'] = bounds

    def setUpperBounds(self, bounds, nodes=None):

        nodes = misc.checkNodes(nodes, range(self.nodes))

        if isinstance(bounds, (list, int, float)):
            bounds = np.array(bounds)
        else:
            bounds = bounds.flatten()

        dim = bounds.shape[0] if bounds.shape else 1
        if dim != self.dim:
            raise Exception('Wrong dimension of upper bounds inserted.')

        for node in nodes:
            self.var_impl['n' + str(node)]['ub'] = bounds

    def setBounds(self, lb, ub, nodes=None):
        self.setLowerBounds(lb, nodes)
        self.setUpperBounds(ub, nodes)

    def setInitialGuess(self, val, nodes=None):

        nodes = misc.checkNodes(nodes, range(self.nodes))

        if isinstance(val, (list, int, float)):
            val = np.array(val)
        else:
            val = val.flatten()

        dim = val.shape[0] if val.shape else 1
        if dim != self.dim:
            raise Exception('Wrong dimension of initial guess inserted.')

        for node in nodes:
            self.var_impl['n' + str(node)]['w0'] = val


    def _setNNodes(self, n_nodes):

        self.nodes = n_nodes
        self._project()

    def _project(self):
        # state_var_impl --> dict
        #  - key: nodes (n0, n1, ...)
        #  - val: dict with name and value of implemented variable
        # old_var_impl = copy.deepcopy(self.var_impl)
        # self.var_impl.clear()
        new_var_impl = dict()

        for n in range(self.nodes):
            if 'n' + str(n) in self.var_impl:
                new_var_impl['n' + str(n)] = self.var_impl['n' + str(n)]
            else:
                var_impl = cs.SX.sym(self.tag + '_' + str(n), self.dim)
                new_var_impl['n' + str(n)] = dict()
                new_var_impl['n' + str(n)]['var'] = var_impl
                new_var_impl['n' + str(n)]['lb'] = np.full(self.dim, -np.inf)
                new_var_impl['n' + str(n)]['ub'] = np.full(self.dim, np.inf)
                new_var_impl['n' + str(n)]['w0'] = np.zeros(self.dim)

        self.var_impl = new_var_impl

    # todo project only at node n (it is used if I want to reproject at each node)
    # def _projectN(self, n):
    #
    #     # state_var_impl --> dict
    #     #  - key: nodes (n0, n1, ...)
    #     #  - val: dict with name and value of implemented variable
    #     var_impl = cs.SX.sym(self.tag + '_' + str(n), self.dim)
    #     self.var_impl['n' + str(n)] = dict()
    #     self.var_impl['n' + str(n)]['var'] = var_impl
    #     self.var_impl['n' + str(n)]['lb'] = [-np.inf] * self.dim
    #     self.var_impl['n' + str(n)]['ub'] = [np.inf] * self.dim
    #     self.var_impl['n' + str(n)]['w0'] = [0] * self.dim
    #
    #     return var_impl


    def getImpl(self, node):
        # todo this is another option: reproject everytime one asks for .getImpl
        # var_impl = self._projectN(node)
        var_impl = self.var_impl['n' + str(node)]['var']
        return var_impl

    def getBoundMin(self, node):
        bound_min = self.var_impl['n' + str(node)]['lb']
        return bound_min

    def getBoundMax(self, node):
        bound_max = self.var_impl['n' + str(node)]['ub']
        return bound_max

    def getBounds(self, node):
        return [self.getBoundMin(node), self.getBoundMax(node)]

    def getInitialGuess(self, node):
        initial_guess = self.var_impl['n' + str(node)]['w0']
        return initial_guess

    def getNNodes(self):
        return self.nodes

    def __reduce__(self):
        return (self.__class__, (self.tag, self.dim, self.nodes, ))


class InputVariable(StateVariable):
    def __init__(self, tag, dim, nodes):
        super(InputVariable, self).__init__(tag, dim, nodes)
        self.nodes = nodes-1

class StateVariables:
    def __init__(self, nodes, logger=None):

        self.logger = logger
        self.nodes = nodes

        self.state_var = OrderedDict()
        self.state_var_prev = OrderedDict()
        self.state_var_impl = OrderedDict()

    def setVar(self, var_type, name, dim, prev_nodes):

        # todo what if 'prev_nodes' it is a list
        createTag = lambda name, node: name + str(node) if node is not None else name
        checkExistence = lambda name, node: True if prev_nodes is None else True if name in self.state_var else False
        tag = createTag(name, prev_nodes)

        if self.logger:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug('Setting variable {} with tag {} as {}'.format(name, tag, var_type))

        if checkExistence(name, prev_nodes):
            var = var_type(tag, dim, self.nodes)
            if prev_nodes is None:
                self.state_var[tag] = var
            else:
                self.state_var_prev[tag] = var
            return var
        else:
            raise Exception('Yet to declare the present variable!')

    def setStateVar(self, name, dim, prev_nodes=None):
        var = self.setVar(StateVariable, name, dim, prev_nodes)
        return var

    def setInputVar(self, name, dim, prev_nodes=None):
        var = self.setVar(InputVariable, name, dim, prev_nodes)
        return var

    def getVarsDim(self):
        var_dim_tot = 0
        for var in self.state_var.values():
            if isinstance(var, StateVariable):
                var_dim_tot += var.shape[0] * var.getNNodes()
        return var_dim_tot


    def getVarImpl(self, name, k):
        node_name = 'n' + str(k)

        if name.find('-') == -1:
            if node_name in self.state_var_impl:
                var = self.state_var_impl[node_name][name]['var']
            else:
                var = None
        else:
            var_name = name[:name.index('-')]
            k_prev = int(name[name.index('-'):])
            node_prev = 'n' + str(k+k_prev)
            if node_name in self.state_var_impl:
                var = self.state_var_impl[node_prev][var_name]['var']
            else:
                var = None

        return var

    def getVarImplAtNode(self, k):
        return self.state_var_impl['n' + str(k)]

    def getVarImplDict(self):
        return self.state_var_impl

    def getVarImplList(self):

        state_var_impl_list = list()
        for node, val in self.state_var_impl.items():
            for var_abstract in val.keys():
                # get from state_var_impl the relative var

                # todo right now, if a variable in state_var_impl is NOT in state_var, it won't be considered in state_var_impl_list

                state_var_impl_list.append(val[var_abstract]['var'])

        return cs.vertcat(*state_var_impl_list)

    def getBoundsMinList(self):
        # todo right now, if a variable in state_var_impl is NOT in state_var, it won't be considered in state_var_impl_list
        state_var_bound_list = np.zeros(self.getVarsDim())

        j = 0
        for node, val in self.state_var_impl.items():
            for var_abstract in val.keys():
                var = val[var_abstract]['lb']
                dim = val[var_abstract]['lb'].shape[0]
                state_var_bound_list[j:j+dim] = var
                j = j + dim

        return state_var_bound_list

    def getBoundsMaxList(self):
        # todo right now, if a variable in state_var_impl is NOT in state_var, it won't be considered in state_var_impl_list

        state_var_bound_list = np.zeros(self.getVarsDim())

        j = 0
        for node, val in self.state_var_impl.items():
            for var_abstract in val.keys():
                var = val[var_abstract]['ub']
                dim = val[var_abstract]['ub'].shape[0]
                state_var_bound_list[j:j + dim] = var
                j = j + dim


        return state_var_bound_list

    def getVarAbstrDict(self, past=True):
        # this is used to check the variable existing in the function. It requires all the variables and the previous variables
        if past:
            var_abstr_dict = {**self.state_var, **self.state_var_prev}
        else:
            var_abstr_dict = self.state_var

        return var_abstr_dict

    def getInitialGuessList(self):

        initial_guess_list = np.zeros(self.getVarsDim())

        j = 0
        for node, val in self.state_var_impl.items():
            for var_abstract in val.keys():
                var = val[var_abstract]['w0']
                dim = val[var_abstract]['w0'].shape[0]
                initial_guess_list[j:j + dim] = var
                j = j + dim

        return initial_guess_list

    def update(self, k):
        # state_var_impl --> dict
        #  - key: nodes (n0, n1, ...)
        #  - val: dict with name and value of implemented variable

        self.state_var_impl['n' + str(k)] = dict()
        # implementation of current state variable
        for name, val in self.state_var.items():
            if isinstance(val, InputVariable) and k == self.nodes-1:
                continue

            var_impl = self.state_var[name].getImpl(k)

            if self.logger:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug('Implemented {} of type {}: {}'.format(name, type(val), var_impl))

            # todo bounds are not necessary here
            var_bound_min = self.state_var[name].getBoundMin(k)
            var_bound_max = self.state_var[name].getBoundMax(k)
            initial_guess = self.state_var[name].getInitialGuess(k)
            var_dict = dict(var=var_impl, lb=var_bound_min, ub=var_bound_max, w0=initial_guess)
            self.state_var_impl['n' + str(k)].update({name: var_dict})

    def updateBounds(self):

        for node in self.state_var_impl.keys():
            for name, state_var in self.state_var_impl[node].items():
                k = node[node.index('n') + 1:]
                state_var['lb'] = self.state_var[name].getBoundMin(k)
                state_var['ub'] = self.state_var[name].getBoundMax(k)
            # self.state_var_impl

    def updateInitialGuess(self):

        for node in self.state_var_impl.keys():
            for name, state_var in self.state_var_impl[node].items():
                k = node[node.index('n') + 1:]
                state_var['w0'] = self.state_var[name].getInitialGuess(k)

    def setNNodes(self, n_nodes):

        # this is required to update the self.state_var_impl EACH time a new number of node is set
        # removed_nodes = [node for node in range(self.nodes) if node not in range(n_nodes)]
        # for node in removed_nodes:
        #     if 'n' + str(node) in self.state_var_impl:
        #         del self.state_var_impl['n' + str(node)]

        self.nodes = n_nodes
        for var in self.state_var.values():
            var._setNNodes(self.nodes)
        for var in self.state_var_prev.values():
            var._setNNodes(self.nodes)


    def clear(self):
        self.state_var_impl.clear()


    def serialize(self):

        # todo how to do? I may use __reduce__ but I don't know how
        # for name, value in self.state_var.items():
        #     print('state_var', type(value))
        #     self.state_var[name] = value.serialize()

        # for name, value in self.state_var_prev.items():
        #     print('state_var_prev', type(value))
        #     self.state_var_prev[name] = value.serialize()

        for node, item in self.state_var_impl.items():
            for name, elem in item.items():
                self.state_var_impl[node][name]['var'] = elem['var'].serialize()

    def deserialize(self):

        # for name, value in self.state_var.items():
        #     self.state_var[name] = cs.SX.deserialize(value)
        #
        # for name, value in self.state_var_prev.items():
        #     self.state_var_prev[name] = cs.SX.deserialize(value)

        for node, item in self.state_var_impl.items():
            for name, elem in item.items():
                self.state_var_impl[node][name]['var'] = cs.SX.deserialize(elem['var'])

    # def __reduce__(self):
    #     return (self.__class__, (self.nodes, self.logger, ))

if __name__ == '__main__':

    # x = StateVariable('x', 2, 4)
    # x._project()
    # print('before serialization:', x)
    # print('bounds:', x.getBounds(2))
    # x.setBounds(2,2)
    # print('bounds:', x.getBounds(2))
    # print('===PICKLING===')
    # a = pickle.dumps(x)
    # print(a)
    # print('===DEPICKLING===')
    # x_new = pickle.loads(a)
    #
    # print(type(x_new))
    # print(x_new)
    # print(x_new.tag)
    #
    # print('bounds:', x.getBounds(2))
    # print(x.var_impl)
    # exit()

    # x = StateVariable('x', 2, 15)
    # print([id(val['var']) for val in x.var_impl.values()])
    # x._setNNodes(20)
    # print([id(val['var']) for val in x.var_impl.values()])

    n_nodes = 10
    sv = StateVariables(n_nodes)
    sv.setStateVar('x', 2)
    sv.setStateVar('y', 2)
    for k in range(n_nodes):
        sv.update(k)

    print(sv.state_var)
    print(sv.state_var_prev)
    print(sv.state_var_impl)
    print(sv.getVarAbstrDict())
    print(sv.getVarImplDict())
    # x_prev = sv.setVar('x', 2, -2)
    #
    # for i in range(4):
    #     sv.update(i)
    #
    # print('state_var_prev', sv.state_var_prev)
    # print('state_var_impl', sv.state_var_impl)
    #
    # print('sv.getVarAbstrDict()', sv.getVarAbstrDict())
    # print('sv.getVarAbstrList()', sv.getVarAbstrList())
    # print('sv.getVarImplList()', sv.getVarImplList())
    # print('sv.getVarImpl()', sv.getVarImpl('x-2', 0))

    print('===PICKLING===')
    sv_serialized = pickle.dumps(sv)
    print(sv_serialized)
    print('===DEPICKLING===')
    sv_new = pickle.loads(sv_serialized)

    print(sv_new.state_var)
    print(sv_new.state_var_prev)
    print(sv_new.state_var_impl)
    print(sv_new.getVarAbstrDict())
    print(sv_new.getVarImplDict())
