# from cmath import sqrt
from horizon.rhc.tasks.task import Task
from horizon.utils.utils import quat_to_rot
import casadi as cs
# from horizon.problem import Problem
import numpy as np
from scipy.spatial.transform import Rotation as scipy_rot

class TestTask(Task):
    def __init__(self, parameter_1, *args, **kwargs):

        self.parameter_1 = parameter_1

        super().__init__(*args, **kwargs)

        self._createWeightParam()

        self.indices = np.array([0, 1, 2]).astype(
            int) if self.indices is None else np.array(self.indices).astype(int)

        if self.fun_type == 'constraint':
            self.instantiator = self.prb.createConstraint
        elif self.fun_type == 'cost':
            self.instantiator = self.prb.createCost
        elif self.fun_type == 'residual':
            self.instantiator = self.prb.createResidual

        print(f'task {self.name} is: {self.fun_type}')

