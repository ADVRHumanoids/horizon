from horizon.rhc.tasks.task import Task
from horizon.variables import Variable, InputVariable, RecedingInputVariable
import casadi as cs
from horizon.problem import Problem
import numpy as np


# todo: better to do an aggregate
class RegularizationTask(Task):

    @classmethod
    def from_dict(cls, task_description):
        opt_variable = [] if 'variable' not in task_description else task_description['variable']

        if 'weight' in task_description and isinstance(task_description['weight'], dict):
            opt_variable.extend([item for item in list(task_description['weight'].keys()) if item not in opt_variable])

        if 'indices' in task_description and isinstance(task_description['indices'], dict):
            opt_variable.extend([item for item in list(task_description['indices'].keys()) if item not in opt_variable])

        task = cls(opt_variable, **task_description)
        return task

    def __init__(self, opt_variable_names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._createWeightParam()

        # get target variables
        if not isinstance(opt_variable_names, list):
            self.opt_variable_list = [opt_variable_names]
        else:
            self.opt_variable_list = [self.prb.getVariables(name) for name in opt_variable_names]

        self.opt_reference_map = dict()
        self.indices_dict = dict()

        if None in self.opt_variable_list:
            raise ValueError(f'variable inserted is not in the problem.')

        if self.fun_type == 'constraint':
            self.instantiator = self.prb.createConstraint
        elif self.fun_type == 'cost':
            self.instantiator = self.prb.createCost
        elif self.fun_type == 'residual':
            self.instantiator = self.prb.createResidual

        self.reg_fun = []

        self._initialize()

    def _initialize(self):

        if self.indices is None:
            for v in self.opt_variable_list:
                self.indices_dict[v.getName()] = np.array(list(range(v.getDim()))).astype(int)

        if isinstance(self.indices, np.ndarray):
            for v in self.opt_variable_list:
                self.indices_dict[v.getName()] = self.indices.astype(int)

        if isinstance(self.indices, dict):
            self.indices_dict = self.indices

        for v in self.opt_variable_list:

            # get indices if they are specified, otherwise take all
            if v.getName() in self.indices_dict:
                v_indices = self.indices_dict[v.getName()]
            else:
                v_indices = np.array(list(range(v.getDim()))).astype(int)

            self.opt_reference_map[v.getName()] = self.prb.createParameter(f'{v.getName()}_ref', v_indices.size)
            # todo hack about nodes
            if isinstance(v, (InputVariable, RecedingInputVariable)):
                nodes = [node for node in list(self.nodes) if node != self.prb.getNNodes() - 1]
            else:
                nodes = self.nodes

            self.reg_fun.append(self.instantiator(f'reg_{self.name}_{v.getName()}', self.weight_param[v.getName()] * (
                        v[v_indices] - self.opt_reference_map[v.getName()]), nodes))

        for f in self.reg_fun:
            print(f"{f.getName()} (dim: {f.getDim()}): {f.getFunction()}")


    # todo: temporary
    def setRef(self, name, ref, nodes=None):
        self.opt_reference_map[name].assign(ref, nodes)

    def getRef(self):
        return self.opt_reference_map

    def setNodes(self, nodes, erasing=True):
        super().setNodes(nodes)
        for reg in self.reg_fun:
            reg.setNodes(nodes)
