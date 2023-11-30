from horizon.rhc.tasks.task import Task
import casadi as cs
import numpy as np
class ZmpTask(Task):
    def __init__(self, *args, **kwargs):


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

        self.__initialize()

    def __initialize(self):

        input_zmp = []
        input_zmp.append(self.model.q)
        input_zmp.append(self.model.v)
        input_zmp.append(self.model.a)

        for f_var in self.model.fmap.values():
            input_zmp.append(f_var)

        c_mean = cs.SX([0, 0, 0])
        for c_name, f_var in self.model.fmap.items():
            fk_c_pos = self.kin_dyn.fk(c_name)(q=self.model.q)['ee_pos']
            c_mean += fk_c_pos

        c_mean /= len(self.model.cmap.keys())

        zmp_fun = self._zmp_fun()(*input_zmp)

        self.fun = self.instantiator('zmp', self.weight_param * (zmp_fun[0:2] - c_mean[0:2]), self.nodes)

    def _zmp_fun(self):

        # formulation in forces
        num = cs.SX([0, 0])
        den = cs.SX([0])
        pos_contact = dict()
        force_val = dict()

        q = cs.SX.sym('q', self.model.nq)
        v = cs.SX.sym('v', self.model.nv)
        a = cs.SX.sym('a', self.model.nv)

        com = self.model.kd.centerOfMass()(q=q, v=v, a=a)['com']

        n = cs.SX([0, 0, 1])
        for c in self.model.fmap.keys():
            pos_contact[c] = self.model.kd.fk(c)(q=q)['ee_pos']
            force_val[c] = cs.SX.sym('force_val', 3)
            num += (pos_contact[c][0:2] - com[0:2]) * cs.dot(force_val[c], n)
            den += cs.dot(force_val[c], n)

        zmp = com[0:2] + (num / den)
        input_list = [q, v, a]

        for elem in force_val.values():
            input_list.append(elem)

        f = cs.Function('zmp', input_list, [zmp])

        return f

    def getFunction(self):
        return self.fun

    def setNodes(self, nodes, erasing=True):
        super().setNodes(nodes, erasing=erasing)

        # print(f"cartesian task '{self.getName()}': ", self.nodes)
        if not nodes:
            self.nodes = []
            self.fun.setNodes(self.nodes, erasing=erasing)
