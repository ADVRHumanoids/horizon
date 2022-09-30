from abc import abstractmethod
from horizon.utils.utils import barrier as barrier_fun
from horizon.rhc.tasks.task import Task
from horizon.rhc.tasks.interactionTask import VertexContact
from horizon.problem import Problem
import numpy as np
from typing import Callable, Dict, Any, List
import casadi as cs


class InteractionTaskMirror(VertexContact):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vertical_to = self.make_vertical_takeoff()

    def make_vertical_takeoff(self):

        dfk = self.kin_dyn.frameVelocity(self.frame, self.kd_frame)
        ee_v = dfk(q=self.prb.getVariables('q'), qdot=self.prb.getVariables('v'))['ee_vel_linear']
        ee_v_ang = dfk(q=self.prb.getVariables('q'), qdot=self.prb.getVariables('v'))['ee_vel_angular']
        lat_vel = cs.vertcat(ee_v[0:2], ee_v_ang)
        vert = self.prb.createConstraint(f"{self.frame}_vert", lat_vel)
        vert.setNodes([])

        return vert

    def setContact(self, nodes):
        super(InteractionTaskMirror, self).setContact(nodes)

        good_nodes = [n for n in nodes if n <= self.all_nodes[-1]]
        off_nodes = [k for k in list(range(self.all_nodes[-1])) if k not in good_nodes]

        if off_nodes:
            nodes_ver = list()
            nodes_ver_temp = [off_nodes[0], off_nodes[-1]]
            [nodes_ver.append(n) for n in nodes_ver_temp if n not in nodes_ver]
            self.vertical_to.setNodes(nodes_ver)


def register_task_plugin(factory) ->None:
    factory.register("VertexForceMirror", InteractionTaskMirror)