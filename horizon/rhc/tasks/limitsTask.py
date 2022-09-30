from horizon.rhc.tasks.task import Task
import casadi as cs
from horizon.problem import Problem
import numpy as np
from horizon.utils.utils import barrier as barrier_fun


# todo set only minimum or maximum individually

class JointLimitsTask(Task):
    def __init__(self, bound_scaling=None, *args, **kwargs):  # bound_scaling, fun_type, *args, **kwargs

        self._bound_scaling = 1.0 if bound_scaling is None else bound_scaling

        super().__init__(*args, **kwargs)

        if self.fun_type == 'constraint':
            self._initialize_bounds()
        elif self.fun_type == 'cost':
            self._initialize_cost()

    def _getQMin(self):
        q_min = np.array(self.kin_dyn.q_min())
        q_max = np.array(self.kin_dyn.q_max())

        with np.errstate(invalid='ignore'):
            q_min_bound = 0.5 * (q_min + q_max) - 0.5 * (q_max - q_min) * self._bound_scaling
        return q_min_bound

    def _getQMax(self):
        q_min = np.array(self.kin_dyn.q_min())
        q_max = np.array(self.kin_dyn.q_max())

        with np.errstate(invalid='ignore'):
            q_max_bound = 0.5 * (q_min + q_max) + 0.5 * (q_max - q_min) * self._bound_scaling
        return q_max_bound

    def _initialize_bounds(self):

        self.q = self.prb.getVariables('q')[self.indices]
        self.q_min = self._getQMin()[self.indices]
        self.q_max = self._getQMax()[self.indices]

        self.q.setBounds(self.q_min, self.q_max, self.initial_nodes)

    def _initialize_cost(self):

        self.q = self.prb.getVariables('q')[self.indices]
        self.q_min = self._getQMin()[self.indices]
        self.q_max = self._getQMax()[self.indices]

        # joint limits
        self.q_min_cost = barrier_fun(self.q - self.q_min)
        self.q_max_cost = barrier_fun(- self.q + self.q_max)

        self.prb.createCost(f'j_lim_min', self.weight * self.q_min_cost)
        self.prb.createCost(f'j_lim_max', self.weight * self.q_max_cost)

    def setRef(self, qmin, qmax):

        qmin_matrix = np.atleast_2d(qmin)
        if qmin_matrix.shape[0] != self.q.getDim():
            raise ValueError(f'Wrong q_min dimension inserted: ({qmin_matrix.shape[0]} != {self.q.getDim()})')

        qmax_matrix = np.atleast_2d(qmax)
        if qmax_matrix.shape[0] != self.q.getDim():
            raise ValueError(f'Wrong q_max dimension inserted: ({qmax_matrix.shape[0]} != {self.q.getDim()})')

        self.q_min = qmin
        self.q_max = qmax

        if self.fun_type == 'constraint':
            self.q.setBounds(self.q_min, self.q_max, self.initial_nodes)
        elif self.fun_type == 'cost':
            self.q_min_cost.setBounds(self.q_min, self.q_max, self.nodes)
            self.q_max_cost.setBounds(self.q_min, self.q_max, self.nodes)

    def setNodes(self, nodes):
        super().setNodes(nodes)

        if not nodes:
            self.nodes = []

        self.nodes = nodes

        if self.fun_type == 'constraint':
            self.q.setBounds(self.q_min, self.q_max, self.initial_nodes)
        elif self.fun_type == 'cost':
            self.q_min_cost.setBounds(self.q_min, self.q_max, self.nodes)
            self.q_max_cost.setBounds(self.q_min, self.q_max, self.nodes)

        # print(f'task {self.name} nodes: {self.pos_constr.getNodes().tolist()}')
        # print(f'param task {self.name} nodes: {self.pos_tgt.getValues()[:, self.pos_constr.getNodes()].tolist()}')
        # print('===================================')


class VelocityLimitsTask(Task):
    def __init__(self, name, prb: Problem, kin_dyn, frame, nodes=None, indices=None, weight=None, kd_frame=None,
                 fun_type=None, bound_scaling=None):
        super().__init__(name, prb, kin_dyn, frame, nodes, indices, weight, kd_frame)

        self._bound_scaling = 1.0 if bound_scaling is None else bound_scaling

        self.fun_type = 'constraint' if fun_type is None else fun_type

        if self.fun_type == 'constraint':
            self._initialize()
        elif self.fun_type == 'cost':
            self._initialize()

    # def _getQDotMax(self):
    #     pass

    def _initialize(self):
        raise NotImplementedError()
        # self.qdot = self.prb.getVariables('v')[self.indices]
        # self.qdot.setBounds(self.q_max, self.initial_nodes)

    def setRef(self, qmin, qmax):
        pass

    def setNodes(self, nodes):
        pass


class TorqueLimitsTask(Task):
    def __init__(self, var, bound_scaling=None, *args, **kwargs):  # bound_scaling, fun_type, *args, **kwargs

        self.tau = var
        self._bound_scaling = 1.0 if bound_scaling is None else bound_scaling

        super().__init__(*args, **kwargs)

        # if self.fun_type == 'constraint':
        #     self._initialize_bounds()
        # elif self.fun_type == 'cost':
        #     self._initialize_cost()

    def _getTauMin(self):
        pass

    def _getTauMax(self):
        pass

    def _initialize_bounds(self):
        pass
        # self.var.setBounds()

    def _initialize_cost(self):
        pass
        # self.prb.createCost(f'j_lim_min', self.weight * self.q_min_cost)
        # self.prb.createCost(f'j_lim_max', self.weight * self.q_max_cost)

    def setRef(self, var_min, var_max):

        if self.fun_type == 'constraint':
            self.tau.setBounds(var_min, var_max, self.nodes)
        # elif self.fun_type == 'cost':
        #     self.q_min_cost.setBounds(self.q_min, self.q_max, self.nodes)
        #     self.q_max_cost.setBounds(self.q_min, self.q_max, self.nodes)

    def setNodes(self, nodes):
        pass
        # super().setNodes(nodes)
        #
        # if not nodes:
        #     self.nodes = []
        #
        # self.nodes = nodes
        #
        # if self.fun_type == 'constraint':
        #     self.tau.setBounds(var_min, var_max, self.nodes)
        #
        # print(f'task {self.name} nodes: {self.pos_constr.getNodes().tolist()}')
        # print(f'param task {self.name} nodes: {self.pos_tgt.getValues()[:, self.pos_constr.getNodes()].tolist()}')
        # print('===================================')
