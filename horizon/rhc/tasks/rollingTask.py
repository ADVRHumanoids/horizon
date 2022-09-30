from horizon.rhc.tasks.task import Task
import casadi as cs
import numpy as np


class RollingTask(Task):
    def __init__(self, frame, radius, *args, **kwargs):

        self.frame = frame
        self.radius = radius

        super().__init__(*args, **kwargs)

        self.indices = np.array([0, 1, 2]).astype(
            int) if self.indices is None else np.array(self.indices).astype(int)

        if self.fun_type == 'constraint':
            self.instantiator = self.prb.createConstraint
        elif self.fun_type == 'cost':
            self.instantiator = self.prb.createCost
        elif self.fun_type == 'residual':
            self.instantiator = self.prb.createResidual

        self._initialize()

    # def _acceleration_formulation(self):

        # q = self.prb.getVariables('q')
        # v = self.prb.getVariables('v')
        # a = self.prb.getVariables('a')

        # dfk_distal = self.kin_dyn.frameVelocity(self.frame, self.kd_frame)
        # ee_v_distal_t = dfk_distal(q=q, qdot=v)['ee_vel_linear']
        # ee_v_distal_r = dfk_distal(q=q, qdot=v)['ee_vel_angular']

        # ddfk_distal = self.kin_dyn.frameAcceleration(self.frame, self.kd_frame)
        # ee_a_distal_t = ddfk_distal(q=q, qdot=v)['ee_acc_linear']
        # ee_a_distal_r = ddfk_distal(q=q, qdot=v)['ee_acc_angular']

        # normal of plane where rolling happens
        # world_contact_plane_normal = np.array([0.0001, 0., 1.])

        # rot velocity (in world) of wheel
        # omega = ee_v_distal_r

        # radius of wheel
        # r = - world_contact_plane_normal * self.radius

        # velocity of point contact between wheel and plane
        # v_contact_point = ee_v_distal_t + cs.cross(omega, r)

        # k = .1
        # omega_a = ee_a_distal_r
        # a_contact_point = ee_a_distal_t + cs.cross(omega_a, r) + k * v_contact_point

    # def _trial_formulation(self):
        #
        # q = self.prb.getVariables('q')
        # v = self.prb.getVariables('v')
        #
        # if self.frame == 'wheel_1':
        #     index = 7+ 4
        # elif self.frame == 'wheel_2':
        #     index = 7 + 10
        # elif self.frame == 'wheel_3':
        #     index = 7 + 16
        # elif self.frame == 'wheel_4':
        #     index = 7 + 22
        #
        # ankle_yaw = q[index]
        # contact_at_wheel_frame = self.frame.replace('wheel', 'contact')
        #
        # wheel_dfk_distal = self.kin_dyn.frameVelocity(self.frame, self.kd_frame)
        # wheel_v_distal_t = wheel_dfk_distal(q=q, qdot=v)['ee_vel_linear']
        # wheel_v_distal_r = wheel_dfk_distal(q=q, qdot=v)['ee_vel_angular']
        #
        # contact_dfk_distal = self.kin_dyn.frameVelocity(contact_at_wheel_frame, self.kd_frame)
        # contact_v_distal_t = contact_dfk_distal(q=q, qdot=v)['ee_vel_linear']
        # contact_v_distal_r = contact_dfk_distal(q=q, qdot=v)['ee_vel_angular']
        #
        #
        # print(self.frame, "ankle:", ankle_yaw)
        # fun = contact_v_distal_t[0] * cs.sin(ankle_yaw) - contact_v_distal_t[1] * cs.cos(ankle_yaw)

        # return fun

    def _velocity_formulation(self):

        q = self.prb.getVariables('q')
        v = self.prb.getVariables('v')

        dfk_distal = self.kin_dyn.frameVelocity(self.frame, self.kd_frame)
        ee_v_distal_t = dfk_distal(q=q, qdot=v)['ee_vel_linear']
        ee_v_distal_r = dfk_distal(q=q, qdot=v)['ee_vel_angular']

        # normal of plane where rolling happens
        world_contact_plane_normal = np.array([0.0001, 0., 1.])

        # rot velocity (in world) of wheel
        omega = ee_v_distal_r

        # radius of wheel
        r = - world_contact_plane_normal * self.radius

        # velocity of point contact between wheel and plane
        v_contact_point = ee_v_distal_t + cs.cross(omega, r)

        fun = v_contact_point
        return fun

    def _initialize(self):

        fun = self._velocity_formulation()
        self.constr = self.instantiator(f'{self.name}_rolling_task', self.weight * fun, nodes=self.nodes)


    def setNodes(self, nodes):
        self.constr.setNodes(nodes)

