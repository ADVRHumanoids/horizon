from cmath import sqrt
from horizon.rhc.tasks.task import Task
import casadi as cs
from horizon.problem import Problem
import numpy as np
from scipy.spatial.transform import Rotation as scipy_rot

# todo name is useless


class CartesianTask(Task):
    def __init__(self, distal_link, base_link=None, cartesian_type=None, *args, **kwargs):

        self.distal_link = distal_link

        self.base_link = 'world' if base_link is None else base_link
        self.cartesian_type = 'position' if cartesian_type is None else cartesian_type

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

    # TODO
    # def _fk(self, frame, q, derivative=0):
    #     if ...
    #     fk = cs.Function.deserialize(self.kin_dyn.fk(frame))
    #     ee_p_t = fk(q=q)['ee_pos']
    #     ee_p_r = fk(q=q)['ee_rot']
    #     return ee_p_t, ee_p_r

    def _rot_to_quat(self, R):

        quat = cs.SX(4, 1)  # todo is this ok?
        quat[0] = 1 / 2 * np.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1)
        quat[1] = np.sign(R[2, 1] - R[1, 2] *
                          np.sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1))
        quat[2] = np.sign(R[0, 2] - R[2, 0] *
                          np.sqrt(R[1, 1] - R[2, 2] - R[0, 0] + 1))
        quat[3] = np.sign(R[1, 0] - R[0, 1] *
                          np.sqrt(R[2, 2] - R[0, 0] - R[1, 1] + 1))

        return quat

    def _quat_to_rot(self, quat):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param quat: A 4 element array representing the quaternion (im(quat), re(quat))

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                 This rotation matrix converts a point in the local reference
                 frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q1 = quat[0]
        q2 = quat[1]
        q3 = quat[2]
        q0 = quat[3]

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        r0 = cs.horzcat(r00, r01, r02)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        r1 = cs.horzcat(r10, r11, r12)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        r2 = cs.horzcat(r20, r21, r22)

        # 3x3 rotation matrix
        rot_matrix = cs.vertcat(r0, r1, r2)

        return rot_matrix

    def _skew(self, vec):
        skew_op = np.zeros([3, 3])

        skew_op[0, 1] = - vec[2]
        skew_op[0, 2] = vec[1]
        skew_op[1, 0] = vec[2]
        skew_op[1, 2] = - vec[0]
        skew_op[2, 0] = - vec[1]
        skew_op[2, 1] = vec[0]

        return skew_op

    def _compute_orientation_error1(self, val1, val2):

        # siciliano's method
        quat_1 = self._rot_to_quat(val1)
        if val2.shape == (3, 3):
            quat_2 = self._rot_to_quat(val2)
        elif val2.shape == (4, 1):
            quat_2 = val2

        rot_err = quat_1[3] * quat_2[0:3] - \
            quat_2[3] * quat_1[0:3] - \
            cs.mtimes(self._skew(quat_2[0:3]), quat_1[0:3])

        return rot_err

    def _compute_orientation_error(self, R_0, R_1):

        R_err = R_0 @ R_1.T
        M_err = np.eye(3) - R_err

        # rot_err = cs.trace(M_err)
        rot_err = cs.vertcat(M_err[0, 0], M_err[1, 1], M_err[2, 2])

        return rot_err

    def _compute_orientation_error2(self, R_0, R_1, epsi=1e-5):

        # not well digested by IPOPT // very well digested by ilqr
        R_err = R_0 @ R_1.T
        R_skew = (R_err - R_err.T) / 2

        r = cs.vertcat(R_skew[2, 1], R_skew[0, 2], R_skew[1, 0])
        
        sqrt_arg = 1 + cs.trace(R_err)
        sqrt_arg = cs.if_else(sqrt_arg > epsi, sqrt_arg, epsi)
        div = cs.sqrt(sqrt_arg)
        rot_err = r / div

        return rot_err

    def _initialize(self):
        # todo this is wrong! how to get these variables?
        q = self.prb.getVariables('q')
        v = self.prb.getVariables('v')

        # todo the indices here represent the position and the orientation error
        # kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        if self.cartesian_type == 'position':

            # TODO: make this automatic
            fk_distal = self.kin_dyn.fk(self.distal_link)
            ee_p_distal = fk_distal(q=q)
            ee_p_distal_t = ee_p_distal['ee_pos']
            ee_p_distal_r = ee_p_distal['ee_rot']

            fk_base = self.kin_dyn.fk(self.base_link)
            ee_p_base = fk_base(q=q)
            ee_p_base_t = ee_p_base['ee_pos']
            ee_p_base_r = ee_p_base['ee_rot']

            ee_p_rel = ee_p_distal_t 
            ee_r_rel = ee_p_distal_r

            frame_name = f'{self.name}_{self.distal_link}_pos'
            # TODO: right now this is slightly unintuitive:
            #  if the function is receding, there are two important concepts to stress:
            #    - function EXISTS: the function exists only on the nodes where ALL the variables and parameters of the function are defined.
            #    - function is ACTIVE: the function can be activated/disabled on the nodes where it exists
            #  so, basically, given the following parameter (self.pose_tgt):
            #    - if the parameter exists only on the node n, the whole function will only exists in the node n
            #    - if the parameter exists on all the nodes, the whole function will exists on all the nodes
            self.pose_tgt = self.prb.createParameter(
                f'{frame_name}_tgt', 7)  # 3 position + 4 orientation

            self.pose_tgt.assign([0, 0, 0, 0, 0, 0, 1])

            self.ref = self.pose_tgt

            fun_trans = ee_p_rel - self.pose_tgt[:3]
            # todo check norm_2 with _compute_orientation_error2

            # fun_lin = cs.norm_2(self._compute_orientation_error(ee_p_r, self._quat_to_rot(self.pose_tgt[3:])))
            fun_lin = self._compute_orientation_error2(
                ee_r_rel, self._quat_to_rot(self.pose_tgt[3:]))
            # fun_lin = self._compute_orientation_error(ee_r_rel, self._quat_to_rot(self.pose_tgt[3:]))

            # todo this is ugly, but for now keep it
            #   find a way to check if also rotation is involved
            fun = cs.vertcat(fun_trans, fun_lin)[self.indices]

        elif self.cartesian_type == 'velocity':

            # pose info
            fk_distal = self.kin_dyn.fk(self.distal_link)
            ee_p_distal = fk_distal(q=q)
            ee_p_distal_t = ee_p_distal['ee_pos']
            ee_p_distal_r = ee_p_distal['ee_rot']

            dfk_distal = self.kin_dyn.frameVelocity(
            self.distal_link, self.kd_frame)
            ee_v_distal_t = dfk_distal(q=q, qdot=v)['ee_vel_linear']
            ee_v_distal_r = dfk_distal(q=q, qdot=v)['ee_vel_angular']
            ee_v_distal = cs.vertcat(ee_v_distal_t, ee_v_distal_r)

            if self.base_link == 'world':
                ee_rel = ee_v_distal
            else:

                fk_base = self.kin_dyn.fk(self.base_link)
                ee_p_base = fk_base(q=q)
                ee_p_base_t = ee_p_base['ee_pos']
                ee_p_base_r = ee_p_base['ee_rot']

                ee_p_rel = ee_p_distal_t - ee_p_base_t
                # ========================================================================
                # vel info

                dfk_base = self.kin_dyn.frameVelocity(
                    self.base_link, self.kd_frame)
                ee_v_base_t = dfk_base(q=q, qdot=v)['ee_vel_linear']
                ee_v_base_r = dfk_base(q=q, qdot=v)['ee_vel_angular']

                
                ee_v_base = cs.vertcat(ee_v_base_t, ee_v_base_r)

                # express this velocity from world to base
                m_w = cs.SX.eye(6)
                m_w[[0, 1, 2], [3, 4, 5]] = - cs.skew(ee_p_rel)

                r_adj = cs.SX(6, 6)
                r_adj[[0, 1, 2], [0, 1, 2]] = ee_p_base_r.T
                r_adj[[3, 4, 5], [3, 4, 5]] = ee_p_base_r.T

                # express the base velocity in the distal frame
                ee_v_base_distal = m_w @ ee_v_base

                # rotate in the base frame the relative velocity (ee_v_distal - ee_v_base_distal)
                ee_rel = r_adj @ (ee_v_distal - ee_v_base_distal)

            frame_name = f'{self.name}_{self.distal_link}_vel'
            self.vel_tgt = self.prb.createParameter(
                f'{frame_name}_tgt', self.indices.size)
            self.ref = self.vel_tgt
            fun = ee_rel[self.indices] - self.vel_tgt

        elif self.cartesian_type == 'acceleration':
            ddfk = self.kin_dyn.frameAcceleration(
                self.distal_link, self.kd_frame)
            ee_a_t = ddfk(q=q, qdot=v)['ee_acc_linear']
            ee_a_r = ddfk(q=q, qdot=v)['ee_acc_angular']
            ee_a = cs.vertcat(ee_a_t, ee_a_r)
            frame_name = f'{self.name}_{self.distal_link}_acc'
            self.acc_tgt = self.prb.createParameter(
                f'{frame_name}_tgt', self.indices.size)
            self.ref = self.acc_tgt
            fun = ee_a[self.indices] - self.acc_tgt

        self.constr = self.instantiator(
            f'{frame_name}_cartesian_task', self.weight * fun, nodes=self.nodes)

        # todo should I keep track of the nodes here?
        #  in other words: should be setNodes resetting?

    def getConstraint(self):
        return self.constr

    def setRef(self, ref_traj):

        # todo shouldn't just ignore None, right?
        if ref_traj is None:
            return False

        ref_matrix = np.array(ref_traj)

        # if ref_matrix.ndim == 2 and ref_matrix.shape[1] != len(self.nodes):
        #     raise ValueError(f'Wrong nodes dimension inserted: ({self.ref.shape[1]} != {len(self.nodes)})')
        # elif ref_matrix.ndim == 1 and len(self.nodes) > 1:
        #     raise ValueError(f'Wrong nodes dimension inserted: ({self.ref.shape[1]} != {len(self.nodes)})')

        self.ref.assign(ref_matrix, self.nodes)  # <==== SET TARGET

        return True

    def setNodes(self, nodes):
        super().setNodes(nodes)

        if not nodes:
            self.nodes = []
            self.constr.setNodes(self.nodes)
            return 0

        # print('=============================================')

        # core
        self.constr.setNodes(self.nodes[1:])  # <==== SET NODES

        # print(f'task {self.name} nodes: {self.pos_constr.getNodes().tolist()}')
        # print(f'param task {self.name} nodes: {self.pos_tgt.getValues()[:, self.pos_constr.getNodes()].tolist()}')
        # print('===================================')

    def addReference(self, ref_traj):

        '''
        possible alternative implementation of reference:
            - adding a reference to the task addReference()
            -
        '''
        if ref_traj is None:
            return False

        ref_matrix = np.array(ref_traj)

        return True

