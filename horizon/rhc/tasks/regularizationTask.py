from horizon.rhc.tasks.task import Task
from horizon.variables import Variable, InputVariable, RecedingInputVariable
import casadi as cs
from horizon.problem import Problem
import numpy as np


# todo: better to do an aggregate
class RegularizationTask(Task):

    def __init__(self, variable_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._createWeightParam()

        self.opt_reference = dict()
        self.indices_dict = dict()

        try:
            self.opt_variable = self.prb.getVariables(variable_name)
        except:
            raise Exception(f"variable '{variable_name}' not found.")

        if self.fun_type == 'constraint':
            self.instantiator = self.prb.createConstraint
        elif self.fun_type == 'cost':
            self.instantiator = self.prb.createCost
        elif self.fun_type == 'residual':
            self.instantiator = self.prb.createResidual

        self.reg_fun = None

        self._initialize()

    def _initialize(self):

        if self.indices is None:
            self.indices = np.array(range(self.opt_variable.getDim()))


        self.opt_reference = self.prb.createParameter(f'{self.opt_variable.getName()}_ref', self.indices.size)
        # todo hack about nodes

        if isinstance(self.opt_variable, (InputVariable, RecedingInputVariable)):
            nodes = [node for node in list(self.nodes) if node != self.prb.getNNodes() - 1]
        else:
            nodes = self.nodes

        self.reg_fun = self.instantiator(f'reg_{self.opt_variable.getName()}',
                                         self.weight_param * (self.opt_variable[self.indices] - self.opt_reference), nodes)

    # todo: temporary
    def setRef(self, ref, nodes=None):
        self.opt_reference.assign(ref, nodes)

    def getRef(self):
        return self.opt_reference

    def setNodes(self, nodes, erasing=True):
        super().setNodes(nodes)
        self.reg_fun.setNodes(nodes)
