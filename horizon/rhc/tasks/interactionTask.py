from abc import abstractmethod
from horizon.utils.utils import barrier as barrier_fun
from horizon.rhc.tasks.task import Task
from horizon.problem import Problem
import numpy as np
from typing import Callable, Dict, Any, List
import casadi as cs
import time

class InteractionTask(Task):
    def __init__(self,
                 frame,
                 friction_coeff=None,
                 fn_min=0.0,
                 enable_fc=False,
                 *args, **kwargs):
        self.frame = frame
        self.friction_coeff = friction_coeff
        self.fn_min = fn_min
        self.enable_fc = enable_fc

        super().__init__(*args, **kwargs)

        # TODO what to do with it?
        self.indices = np.array([0, 1, 2, 3, 4, 5]).astype(int) if self.indices is None else np.array(
            self.indices).astype(int)

        # TODO: add initialize now conflicts
    #     self.__initialize()
    #
    # def __initialize(self):
    #
    #     self.setNodes(self.nodes)

    @abstractmethod
    def getWrench(self):
        pass

    def make_cop(self):
        pass

    def make_friction_cone(self):
        pass

    def setContact(self, nodes, erasing=True):
        pass

    def getFrame(self):
        return self.frame

    # todo crazy misleading name
    def setNodes(self, nodes, erasing=True):
        super().setNodes(nodes, erasing=erasing)

        self.nodes = nodes
        self._reset()
        self._set_zero(nodes)

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _set_zero(self, nodes):
        pass


class SurfaceContact(InteractionTask):
    def __init__(self,
                 frame,
                 shape='box',
                 dimensions=None,
                 enable_cop=True,
                 *args, **kwargs):

        # init base
        self.shape = shape
        self.dimensions = dimensions
        self.enable_cop = enable_cop

        super().__init__(frame, *args, **kwargs)

        # ask model to create 6d wrench
        self.wrench = self.model.setContactFrame(frame, 'surface')

        self.all_nodes = self.wrench.getNodes()

        self.cop_constr = self.make_cop() if enable_cop else None
        self.fc_constr = self.make_friction_cone() if self.enable_fc else None

        self.fn_barrier = self.make_fn_barrier()

    def make_fn_barrier(self):
        fn_barrier_cost = barrier_fun(self.wrench[2] - self.fn_min)
        fn_barrier = self.prb.createCost(f'{self.frame}_unil_barrier', 1e1 * fn_barrier_cost, self.all_nodes)
        return fn_barrier

    def make_cop(self):

        # get wrench from child
        f = self.getWrench()

        # compute rotation matrix
        frame = self.getFrame()
        _, R = self.model.fk(frame)

        # turn to local coord
        f_local = cs.vertcat(
            R.T @ f[0:3],
            R.T @ f[3:6]
        )

        # write constraint
        xmin, xmax = -self.dimensions[0], self.dimensions[0]
        ymin, ymax = -self.dimensions[1], self.dimensions[1]

        M_cop = cs.DM.zeros(4, 6)
        M_cop[:, 2] = [xmin, -xmax, ymin, -ymax]
        M_cop[[0, 1], 4] = [1, -1]
        M_cop[[2, 3], 3] = [-1, 1]

        rot_M_cop = M_cop @ f_local

        # add constraint
        cop_consrt = self.prb.createIntermediateConstraint(f'cop_{frame}', rot_M_cop)
        cop_consrt.setLowerBounds(-np.inf * np.ones(4))

        return cop_consrt

    def make_friction_cone(self):
        f = self.wrench
        mu = self.friction_coeff
        fcost = barrier_fun(f[2] ** 2 * mu ** 2 - cs.sumsqr(f[:2]))
        fc = self.prb.createIntermediateResidual(f'{self.frame}_fc', 3e-1 * fcost)
        return fc

    def setContact(self, nodes, erasing=True):

        good_nodes = [n for n in nodes if n <= self.all_nodes[-1]]

        # start with all swing
        if erasing:
            self._set_zero(self.all_nodes)

        # no force bounds when in contact
        self.wrench.setBounds(lb=np.full(self.wrench.getDim(), -np.inf),
                              ub=np.full(self.wrench.getDim(), np.inf),
                              nodes=good_nodes)

        # add normal force constraint
        self.fn_barrier.setNodes(good_nodes, erasing=erasing)

        # add cop constraint
        if self.cop_constr:
            # note: this resets bounds to (0, 0) !!!!!! EVIL!!!!!! SATAN!!!!!!
            self.cop_constr.setNodes(good_nodes, erasing=erasing)
            # todo: this should be unnecessary, change behaviour of setNodes?
            # self.cop_constr.setLowerBounds(-np.inf * np.ones(4))

        if self.fc_constr:
            self.fc_constr.setNodes(good_nodes, erasing=erasing)

    def getWrench(self):
        return self.wrench

    def _reset(self):
        # todo reset only on given nodes
        self.wrench.setBounds(lb=np.full(self.wrench.getDim(), -np.inf),
                              ub=np.full(self.wrench.getDim(), np.inf))

    def _set_zero(self, nodes):
        self.wrench.setBounds(lb=np.full(self.wrench.getDim(), 0),
                              ub=np.full(self.wrench.getDim(), 0),
                              nodes=nodes)


class VertexContact(InteractionTask):

    def __init__(self, frame, vertex_frames, *args, **kwargs):

        # init base
        super().__init__(frame, *args, **kwargs)

        self.vertex_frames=vertex_frames
        
        # ask model to create vertex forces
        self.forces = self.model.setContactFrame(frame,
                                                 'vertex',
                                                 {
                                                     'vertex_frames': vertex_frames
                                                 })

        self.all_nodes = self.forces[0].getNodes()

        self.__initialize()

    def __initialize(self):

        self.fn_barrier = self.make_fn_barrier()
        self.fc_constr = self.make_friction_cone() if self.enable_fc else None

        # self.forces_vec = dict()
        # for f in self.forces:
        #     self.forces_vec[f.getName()] = np.zeros(f.getDim())
        #
        # initialize everything with nodes specified
        self.setNodes(self.nodes)


    def make_fn_barrier(self):

        if not self.fn_min < -1e3:
            fn_barrier_cost = []
            for f in self.forces:
                fn_barrier_cost.append(barrier_fun(f[2] - self.fn_min))
            fn_barrier_cost = cs.vertcat(*fn_barrier_cost)
            fn_barrier = self.prb.createResidual(f'{self.frame}_unil_barrier', 1e1 * fn_barrier_cost, self.all_nodes)
            return fn_barrier
        else:
            return None
        
    def make_friction_cone(self):
        fcost = []
        for f in self.forces:
            mu = self.friction_coeff
            fcost_f = barrier_fun(f[2] ** 2 * mu ** 2 - cs.sumsqr(f[:2]))
            fcost.append(fcost_f)

        fcost = cs.vertcat(*fcost)
        fc = self.prb.createIntermediateResidual(f'{self.frame}_fc', 3e-1 * fcost, self.all_nodes)
        return fc

    def setContact(self, nodes, erasing=True):

        # start_time1 = time.time()

        if erasing:
            self._set_zero(self.all_nodes)

        # end_time1 = time.time() - start_time1
        # start_time2 = time.time()

        if nodes:
            good_nodes = [n for n in nodes if n <= self.all_nodes[-1]]

            for f in self.forces:
                # no force bounds when in contact
                f.setBounds(lb=np.full(f.getDim(), -np.inf),
                            ub=np.full(f.getDim(), np.inf),
                            nodes=good_nodes)
        else:
            good_nodes = []

        # end_time2 = time.time() - start_time2
        # start_time3 = time.time()

        # add normal force constraint
        if self.fn_barrier is not None:
            self.fn_barrier.setNodes(good_nodes, erasing=erasing)

        # end_time3 = time.time() - start_time3
        # start_time4 = time.time()

        # set friction cone
        if self.fc_constr:
            self.fc_constr.setNodes(good_nodes, erasing=erasing)

        # end_time4 = time.time() - start_time4
        # print(f"    -subtime-: time for erasing: {end_time1}")
        # print(f"    -subtime-: time for setting forces: {end_time2}")
        # print(f"    -subtime-: time for setting barrier: {end_time3}")
        # print(f"    -subtime-: time for setting fc: {end_time4}")

        # end_time6 = time.time() - start_time1
        # print(f"time for setting nodes at {self.getName()}: {end_time6}")

    def getWrench(self):
        return self.forces

    def _reset(self):
        # todo reset only on given nodes
        for f in self.forces:
            f.setBounds(lb=np.full(f.getDim(), -np.inf),
                        ub=np.full(f.getDim(), np.inf))

    def _set_zero(self, nodes):
        for f in self.forces:
            f.setBounds(lb=np.full(f.getDim(), 0),
                        ub=np.full(f.getDim(), 0),
                        nodes=nodes)

