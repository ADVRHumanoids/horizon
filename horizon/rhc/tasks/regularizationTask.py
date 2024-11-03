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
            opt_variable = list(task_description['weight'].keys())
            task_description['weight'] = list(task_description['weight'].values())

        task = cls(opt_variable, **task_description)
        return task

    def __init__(self, opt_variable_names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._createWeightParam()

        self.opt_varariable_dim = []
        self.opt_variable_list = []
        if not isinstance(opt_variable_names, list):
            var=self.prb.getVariables(opt_variable_names)
            if var is not None:
                self.opt_variable_list.append(var)
                self.opt_varariable_dim.append(var.getDim())
            else:
                self.opt_varariable_dim.append(None)
        else:
            for name in opt_variable_names:
                var=self.prb.getVariables(name)
                var_dim=None
                if var is not None:
                    var_dim=var.getDim()
                self.opt_variable_list.append(var)
                self.opt_varariable_dim.append(var_dim)
        if None in self.opt_variable_list:
            raise ValueError(f'variable inserted is not in the problem.')
        
        if None in self.opt_variable_list:
            raise ValueError(f'variable inserted is not in the problem.')
        
        var_dim_match=all(x == self.opt_varariable_dim[0] for x in self.opt_varariable_dim)
        if not var_dim_match:
            incorrect_dims=list(map(str, self.opt_varariable_dim))
            raise ValueError(f'Dimensions of variables do not match! -> {", ".join(incorrect_dims)}')
        
        self.indices = np.array(list(range(self.opt_varariable_dim[0]))).astype(int) if self.indices is None else np.array(self.indices).astype(int)

        indices_within_bounds=all(x<self.opt_varariable_dim[0] for x in self.indices)
        if not indices_within_bounds:
            raise ValueError(f'Indeces are not within allowed range (0, {self.opt_varariable_dim[0]-1})!')
    
        # todo: what to do with this one?
        self.opt_reference_list = [self.prb.createParameter(f'{self.name}_{name}_ref', self.indices.size) for name in opt_variable_names]

        if self.fun_type == 'constraint':
            self.instantiator = self.prb.createConstraint
        elif self.fun_type == 'cost':
            self.instantiator = self.prb.createCost
        elif self.fun_type == 'residual':
            self.instantiator = self.prb.createResidual

        self.reg_fun = []

        self._initialize()

    def _initialize(self):

        for v, w, r in zip(self.opt_variable_list, self.weight_param, self.opt_reference_list):
            # todo hack about nodes
            if isinstance(v, (InputVariable, RecedingInputVariable)):
                nodes = [node for node in list(self.nodes) if node != self.prb.getNNodes()-1]
            else:
                nodes = self.nodes

            self.reg_fun.append(self.instantiator(f'reg_{self.name}_{v.getName()}', w * (v[self.indices] - r), nodes))

    # todo: temporary
    def setRef(self, index, ref, nodes=None):
        self.opt_reference_list[index].assign(ref, nodes)

    def getRef(self):
        return self.opt_reference_list

    def setNodes(self, nodes, erasing=True):
        super().setNodes(nodes)
        for reg in self.reg_fun:
            reg.setNodes(nodes)


# class RegularizationTaskInterface:n
