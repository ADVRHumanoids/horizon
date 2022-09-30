from horizon.rhc.tasks.task import Task
from horizon.variables import Variable, InputVariable, RecedingInputVariable
import casadi as cs
from horizon.problem import Problem
import numpy as np

# todo: create aggregate task
class RegularizationTask(Task):

    @classmethod
    def from_dict(cls, task_description):
        opt_variable = [] if 'variable' not in task_description else task_description['variable']

        if 'weight' in task_description and isinstance(task_description['weight'], dict):
            weights = task_description.pop('weight')
            for opt_var, weight in weights.items():
                task_description['weight'] = weight
                cls(opt_var, **task_description)

        for var in opt_variable:
            task_list = cls(opt_variable, **task_description)
        return task

    def __init__(self, opt_variable_name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print(opt_variable_name)

        self.opt_variable = self.prb.getVariables(opt_variable_name)

        if self.opt_variable is None:
            raise ValueError(f'variable inserted is not in the problem.')

        # todo: what to do with this one?
        self.opt_ref = self.prb.createParameter(opt_variable_name + '_reg_ref', self.prb.getVariables(opt_variable_name).getDim())

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
