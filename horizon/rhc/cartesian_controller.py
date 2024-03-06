#!/usr/bin/python3

from xbot_interface import xbot_interface as xbot
import numpy as np
import yaml
from cartesian_interface.pyci_all import *

class CartesioSolver:

    def __init__(self, urdf, srdf, ti, ee_name, mpc_task) -> None:



        self.base_fk = ti.model.kd.fk('base_link')
        self.xbot_model = xbot.ModelInterface(get_xbot_config(urdf=urdf, srdf=srdf))

        self.ee_name = ee_name
        self.mpc_task = mpc_task

        self.ti = ti

        self.set_model_from_solution(self.ti.solver_bs)

        # define problem
        pb_dict = {
            'stack': [['ee']],

            'ee': {'type': 'Cartesian',
                   'distal_link': ee_name}
        }

        pb = yaml.dump(pb_dict)


        self.ci = pyci.CartesianInterface.MakeInstance(
            solver='',
            problem=pb,
            model=self.xbot_model,
            dt=0.01)

        self.ciros = pyci.RosServerClass(self.ci)

        self.task = self.ci.getTask(self.ee_name)


    def getCartesianInterface(self):

        return self.ci

    def set_model_from_solution(self, solver):

        q = solver.getSolutionDict()['q'][:, 1]
        # dq = self.solver_rti.getSolutionDict()['v'][:, 1]
        base_pos = self.base_fk(q=q)['ee_pos'].toarray()
        base_rot = self.base_fk(q=q)['ee_rot'].toarray()

        qmdl = np.zeros(self.xbot_model.getJointNum())

        qmdl[6:] = q[7:]
        xbot_base_pose = Affine3(pos=base_pos)
        xbot_base_pose.linear = base_rot
        self.xbot_model.setJointPosition(qmdl)
        self.xbot_model.setFloatingBasePose(xbot_base_pose)
        self.xbot_model.update()

    def update_ci(self):

        self.ciros.run()
        Tref = self.task.getPoseReference()[0]
        xy_ref = np.array([Tref.translation.tolist() + [0, 0, 0, 1]]).T

        self.mpc_task.setRef(xy_ref)
