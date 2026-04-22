from docutils.io import Input

from horizon.rhc.tasks.task import Task
from horizon.variables import Variable, InputVariable, RecedingInputVariable
import casadi as cs
from horizon.problem import Problem
import numpy as np


# todo: better to do an aggregate
class RegularizationTask(Task):

    def __init__(self, variable_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._createWeightParam()

        self.opt_reference = None
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

        # if self.last_node is None, the variable is an input, so it does not exist on the last node
        self.last_node = None
        if isinstance(self.opt_variable, (InputVariable, RecedingInputVariable)):
            self.last_node = self.prb.getNNodes() - 1

        self.reg_fun = None

        self._initialize()

    def _initialize(self):

        name = f'{self.opt_variable.getName()}_ref'

        if self.indices is None:
            self.indices = np.array(range(self.opt_variable.getDim()))
        else:
            if self.indices.size != self.opt_variable.getDim():
                name = name + f'_indices_' + '_'.join(map(str, self.indices))  # "1234"


        self.opt_reference = self.prb.createParameter(name, self.indices.size)
        # todo hack about nodes

        if self.last_node:
            nodes = [node for node in list(self.nodes) if node != self.last_node]
        else:
            nodes = self.nodes

        self._createWeightParam(self.indices.size)

        name_fun = f'reg_{self.opt_variable.getName()}'

        if self.indices is not None and self.indices.size != self.opt_variable.getDim():
            name_fun = name_fun + f'_indices_' + '_'.join(map(str, self.indices))   # "1234"

        self.reg_fun = self.instantiator(name_fun,
                                         self.weight_param * (self.opt_variable[self.indices] - self.opt_reference), nodes)

        # todo: temporary
    def setRef(self, ref, nodes=None):
        self.opt_reference.assign(ref, nodes)

    def getRef(self):
        return self.opt_reference

    def getDim(self):
            return self.indices.size

    def assign(self, val, nodes=None):
        # necessary method for using this task as an item + reference in phaseManager
        self.opt_reference.assign(val, nodes)
        return 1

    def getValues(self):
        # necessary method for using this task as an item + reference in phaseManager
        return self.opt_reference.getValues()

    def setNodes(self, nodes, erasing=True):
        super().setNodes(nodes, erasing=erasing)

        if self.last_node:
            nodes = [node for node in list(self.nodes) if node != self.last_node]

        if not nodes:
            self.nodes = []
            self.reg_fun.setNodes(self.nodes, erasing=erasing)
            return 0

        # core
        self.reg_fun.setNodes(self.nodes[0:], erasing=erasing)  # <==== SET NODES


    def getInfo(self):

        info = dict(
            function=self.reg_fun.getName(),
            weight_param=self.weight_param.getName(),
            ref_param=self.opt_reference.getName()
        )

        return info