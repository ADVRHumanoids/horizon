from horizon.rhc.tasks.task import Task
import casadi as cs
from horizon.problem import Problem
import numpy as np

# todo: why the args and kwargs? because they are part of the super class, which may change. If I explicitly define every arguments,
#   if the base class changes I have to change all the derived classes. This, instead, prevent to change sub-classes when changing parent classes

class PosturalTask(Task):
    def __init__(self, postural_ref, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.indices = np.array(list(range(self.kin_dyn.nq() - 7))).astype(int) if self.indices is None else np.array(self.indices).astype(int)
        self.q = self.prb.getVariables('q')[7:]
        self.q0_joints_ref = postural_ref[7:]
        self.q0_ref = self.q0_joints_ref[self.indices]

        # if 'postural_ref' not in task_node:
        #     raise ValueError('Postural reference is not set')

        if self.fun_type == 'constraint':
            self.instantiator = self.prb.createConstraint
        elif self.fun_type == 'cost':
            self.instantiator = self.prb.createCost
        elif self.fun_type == 'residual':
            self.instantiator = self.prb.createResidual

        self._initialize()

    def _initialize(self):
        # get only the joint positions

        name_fun = f'postural_{self.name}'# '_'.join(map(str, self.indices))
        self.q0 = self.prb.createParameter(f'{name_fun}_tgt', self.indices.size)
        self.fun = self.instantiator(f'{name_fun}_task', self.weight * (self.q[self.indices] - self.q0), nodes=self.nodes)
        self.q0.assign(self.q0_ref)

    def getConstraint(self):
        return self.fun

    def setRef(self, postural_ref):

        ref_matrix = np.atleast_2d(postural_ref)

        if ref_matrix.shape[0] != self.fun.getDim():
            raise ValueError(f'Wrong goal dimension inserted: ({ref_matrix.shape[0]} != {self.fun.getDim()})')

        self.q0_ref = ref_matrix

        self.q0.assign(self.q0_ref)

    def setNodes(self, nodes):
        super().setNodes(nodes)

        if not nodes:
            self.nodes = []

        self.fun.setNodes(self.nodes)  # <==== SET NODES

        # print(f'task {self.name} nodes: {self.pos_constr.getNodes().tolist()}')
        # print(f'param task {self.name} nodes: {self.pos_tgt.getValues()[:, self.pos_constr.getNodes()].tolist()}')
        # print('===================================')