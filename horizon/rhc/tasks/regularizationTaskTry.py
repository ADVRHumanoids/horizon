from horizon.rhc.tasks.task import Task
from horizon.variables import Variable, InputVariable, RecedingInputVariable
import casadi as cs
from horizon.problem import Problem
import numpy as np

# todo: create aggregate task

class RegularizationTask(Task):
    def __init__(self, opt_variable_name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.opt_variable = self.prb.getVariables(opt_variable_name)

        if self.opt_variable is None:
            raise ValueError(f'variable inserted is not in the problem.')

        var_dim = self.prb.getVariables(opt_variable_name).getDim()
        self.indices = np.array(list(range(var_dim))).astype(int) if self.indices is None else np.array(self.indices).astype(int)

        self.opt_ref = self.prb.createParameter(opt_variable_name + '_reg_ref')
        self._initialize()

    def _initialize(self):

            if isinstance(self.opt_variable, (InputVariable, RecedingInputVariable)):
                nodes = [node for node in list(self.nodes) if node != self.prb.getNNodes()-1]
            else:
                nodes = self.nodes

            self.reg_fun = self.prb.createResidual(f'reg_{self.name}_{self.opt_variable.getName()}', self.weight * (self.opt_variable - self.opt_ref), nodes)

    def setRef(self, ref):
        self.opt_ref.assign(ref)

    def getRef(self):
        return self.opt_ref.getValues()

    def setNodes(self, nodes):
        super().setNodes(nodes)
        self.reg_fun.setNodes(nodes)


# class RegularizationTaskInterface:n
