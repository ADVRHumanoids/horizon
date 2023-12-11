from re import sub
import casadi as cs
from typing import List, Iterable, Union, Sequence, Dict, Type, Tuple
import random, string
from horizon.problem import Problem
import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn
from dataclasses import dataclass, field



def generate_id() -> str:
    id_len = 6
    return ''.join(
        random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(id_len))

@dataclass
class Task:
    
    # todo this is context: transform to context
    prb: Problem
    kin_dyn: pycasadi_kin_dyn.CasadiKinDyn
    kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
    model: 'ModelDescription'

    # todo: how to initialize?
    type: str
    name: str
    fun_type: str = 'constraint'
    weight: Union[List, float] = 1.0
    nodes: Sequence = field(default_factory=list)
    indices: Union[List, np.ndarray] = None
    id: str = field(init=False, default_factory=generate_id)

    @classmethod
    def from_dict(cls, task_dict):
        return cls(**task_dict)

    @classmethod
    def subtask_by_class(cls, subtask: Dict, classname: Union[Type, Tuple[Type]]) -> 'classname':
        ret = []
        for _, v in subtask.items():
            if isinstance(v, classname):
                ret.append(v)
        return ret[0] if len(ret) == 1 else ret

    def __post_init__(self):
        # todo: this is for simplicity
        if isinstance(self.indices, dict):
            for elem in self.indices:
                self.indices[elem] = np.array(self.indices[elem])
        else:
            self.indices = np.array(self.indices) if self.indices is not None else None
        # self.nodes = list(range(self.prb.getNNodes()))

    def _createWeightParam(self):

        # weight and dim must be the same dimension
        if isinstance(self.weight, (float, int)):
            self.weight_param = self.prb.createParameter(f'{self.name}_weight', 1)
            self.weight_param.assign(self.weight)

        elif isinstance(self.weight, List):
            self.weight_param = []
            for i_dim in range(len(self.weight)):

                temp_par = self.prb.createParameter(f'{self.name}_weight_{i_dim}', 1)
                temp_par.assign(self.weight[i_dim])
                self.weight_param.append(temp_par)

        elif isinstance(self.weight, dict):
            self.weight_param = dict()
            for elem, w in self.weight.items():
                temp_par = self.prb.createParameter(f'{self.name}_weight_{elem}', 1)
                temp_par.assign(self.weight[elem])
                self.weight_param[elem] = temp_par

    def setNodes(self, nodes, erasing=True):
        self.nodes = nodes
        self.n_active = len(self.nodes)

    def setWeight(self, val, nodes=None, indices=None):
        self.weight_param.assign(val, nodes, indices)

    def setIndices(self, indices):
        self.indices = indices

    def _reset(self):
        pass

    def getNodes(self):
        return self.nodes

    def getName(self):
        return self.name

    def getType(self):
        return self.type

    def getWeight(self):
        return self.weight_param.getValues()
