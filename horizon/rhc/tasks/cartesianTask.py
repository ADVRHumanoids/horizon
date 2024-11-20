# from cmath import sqrt
from horizon.rhc.tasks.task import Task
from horizon.utils.utils import quat_to_rot
import casadi as cs
# from horizon.problem import Problem
import numpy as np
from scipy.spatial.transform import Rotation as scipy_rot


# todo name is useless


class CartesianTask(Task):
    def __init__(self, distal_link, base_link=None, cartesian_type=None, *args, **kwargs):

        self.distal_link = distal_link

        self.base_link = 'world' if base_link is None else base_link
        self.cartesian_type = 'position' if cartesian_type is None else cartesian_type

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

    # TODO
    # def _fk(self, frame, q, derivative=0):
    #     if ...
    #     fk = cs.Function.deserialize(self.kin_dyn.fk(frame))
    #     ee_p_t = fk(q=q)['ee_pos']
    #     ee_p_r = fk(q=q)['ee_rot']
    #     return ee_p_t, ee_p_r

    def _rot_to_quat(self, R):

        '''
        Covert a rotation matrix into a quaternion
        [method specific for casadi function]
        WRONG
        '''

        rot_mat = cs.SX.sym('rot_mat', 3, 3)
        quat = cs.SX.sym('rot_quat', 4, 1)  # todo is this ok?

        r11, r12, r13 = rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2]
        r21, r22, r23 = rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2]
        r31, r32, r33 = rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2]

        quat[3] = 0.5 * cs.sqrt(1 + r11 + r22 + r33)
        quat[0] = (r32 - r23) / (4 * quat[3])
        quat[1] = (r13 - r31) / (4 * quat[3])
        quat[2] = (r21 - r12) / (4 * quat[3])

        rot_to_fun_quat = cs.Function('rot_to_quat', [rot_mat], [quat])

        quat = rot_to_fun_quat(R)
        return quat

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

    def __init_position_sym(self, frame_name):
        q = cs.SX.sym('q', self.kin_dyn.nq())
        p_tgt = cs.SX.sym('p_tgt', 7)
        fk_distal = self.kin_dyn.fk(self.distal_link)
        ee_p_distal = fk_distal(q=q)
        ee_p_distal_t = ee_p_distal['ee_pos']
        ee_p_distal_r = ee_p_distal['ee_rot']

        err_pos = ee_p_distal_t - p_tgt[:3]
        err_lin = self._compute_orientation_error2(ee_p_distal_r, quat_to_rot(p_tgt[3:]))

        err_fun = cs.Function(f"position_error_{frame_name}", [q, p_tgt], [err_pos, err_lin])
        return err_fun

    def __init_position(self, frame_name):

        # TODO: make this automatic
        fk_distal = self.kin_dyn.fk(self.distal_link)
        ee_p_distal = fk_distal(q=self.q)
        ee_p_distal_t = ee_p_distal['ee_pos']
        ee_p_distal_r = ee_p_distal['ee_rot']

        if self.base_link == 'world':
            ee_p_rel = ee_p_distal_t
            ee_r_rel = ee_p_distal_r
        else:
            fk_base = self.kin_dyn.fk(self.base_link)
            ee_p_base = fk_base(q=self.q)
            ee_p_base_t = ee_p_base['ee_pos']
            ee_p_base_r = ee_p_base['ee_rot']

            ee_p_rel = ee_p_distal_t - ee_p_base_t
            ee_r_rel = cs.inv(ee_p_base_r) * ee_p_distal_r

        # TODO: right now this is slightly unintuitive:
        #  if the function is receding, there are two important concepts to stress:
        #    - function EXISTS: the function exists only on the nodes where ALL the variables and parameters of the function are defined.
        #    - function is ACTIVE: the function can be activated/disabled on the nodes where it exists
        #  so, basically, given the following parameter (self.pose_tgt):
        #    - if the parameter exists only on the node n, the whole function will only exists in the node n
        #    - if the parameter exists on all the nodes, the whole function will exists on all the nodes
        self.pose_tgt = self.prb.createParameter(f'{frame_name}_tgt', 7)  # 3 position + 4 orientation

        ee_p_distal_0 = fk_distal(q=self.model.q0)

        # start cartesian task target at frame current position
        self.pose_tgt[:3].assign(ee_p_distal_0['ee_pos'].full())
        self.initial_rot = scipy_rot.from_matrix((ee_p_distal_0['ee_rot'].full()))
        self.pose_tgt[3:].assign(self.initial_rot.as_quat())

        self.ref = self.pose_tgt

        # self.initial_ref_matrix = self.ref.getValues().copy()

        fun_trans = ee_p_rel - self.pose_tgt[:3]
        # todo check norm_2 with _compute_orientation_error2

        # fun_lin = cs.norm_2(self._compute_orientation_error(ee_p_r, quat_to_rot(self.pose_tgt[3:])))
        fun_lin = self._compute_orientation_error2(ee_r_rel, quat_to_rot(self.pose_tgt[3:]))
        # fun_lin = self._compute_orientation_error(ee_r_rel, quat_to_rot(self.pose_tgt[3:]))

        # todo this is ugly, but for now keep it
        #   find a way to check if also rotation is involved
        fun = cs.vertcat(fun_trans, fun_lin)[self.indices]

        return fun

    def __init_velocity(self, frame_name):

        # pose info
        fk_distal = self.kin_dyn.fk(self.distal_link)
        ee_p_distal = fk_distal(q=self.q)
        ee_p_distal_t = ee_p_distal['ee_pos']
        ee_p_distal_r = ee_p_distal['ee_rot']

        # vel info
        dfk_distal = self.kin_dyn.frameVelocity(self.distal_link, self.kd_frame)
        ee_v_distal_t = dfk_distal(q=self.q, qdot=self.v)['ee_vel_linear']
        ee_v_distal_r = dfk_distal(q=self.q, qdot=self.v)['ee_vel_angular']
        ee_v_distal = cs.vertcat(ee_v_distal_t, ee_v_distal_r)

        if self.base_link == 'world':
            ee_rel = ee_v_distal
        else:

            fk_base = self.kin_dyn.fk(self.base_link)
            ee_p_base = fk_base(q=self.q)
            ee_p_base_t = ee_p_base['ee_pos']
            ee_p_base_r = ee_p_base['ee_rot']

            ee_p_rel = ee_p_distal_t - ee_p_base_t
            # ========================================================================


            dfk_base = self.kin_dyn.frameVelocity(self.base_link, self.kd_frame)
            ee_v_base_t = dfk_base(q=self.q, qdot=self.v)['ee_vel_linear']
            ee_v_base_r = dfk_base(q=self.q, qdot=self.v)['ee_vel_angular']

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

        self.vel_tgt = self.prb.createParameter(f'{frame_name}_tgt', self.indices.size)
        self.ref = self.vel_tgt
        fun = ee_rel[self.indices] - self.vel_tgt

        return fun

    def __init_acceleration(self, frame_name):

        ddfk = self.kin_dyn.frameAcceleration(self.distal_link, self.kd_frame)

        ee_a_t = ddfk(self.q, qdot=self.v)['ee_acc_linear']
        ee_a_r = ddfk(self.q, qdot=self.v)['ee_acc_angular']

        ee_a = cs.vertcat(ee_a_t, ee_a_r)

        self.acc_tgt = self.prb.createParameter(f'{frame_name}_tgt', self.indices.size)
        self.ref = self.acc_tgt
        fun = ee_a[self.indices] - self.acc_tgt

        return fun


        
    def __initialize(self):
        self.ref_matrix = None
        # todo this is wrong! how to get these variables?
        self.q = self.prb.getVariables('q')
        self.v = self.prb.getVariables('v')

        frame_name = f'{self.name}_{self.distal_link}'
        # todo the indices here represent the position and the orientation error
        # kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        if self.cartesian_type == 'position':

            frame_name = frame_name + '_pos'
            fun = self.__init_position(frame_name)
            self.fun_sym = self.__init_position_sym(frame_name)

        elif self.cartesian_type == 'velocity':

            frame_name = frame_name + '_vel'
            fun = self.__init_velocity(frame_name)

        elif self.cartesian_type == 'acceleration':

            frame_name = frame_name + '_acc'
            fun = self.__init_acceleration(frame_name)

        else:
            raise NameError(f"wrong cartesian type inserted: {self.cartesian_type}")

        final_name = f'{frame_name}_cartesian_task'

        self.constr = self.instantiator(final_name, self.weight_param * fun, nodes=self.nodes)

        # todo should I keep track of the nodes here?
        #  in other words: should be setNodes resetting?

    def getConstraint(self):
        return self.constr

    def setRef(self, ref_traj):

        '''
        possible alternative implementation of reference:
            - adding a reference to the task addReference()
            -
        '''
        if ref_traj is None:
            return False

        # horrible
        self.ref_matrix = np.atleast_2d(np.array(ref_traj))
        self.addReference()

        return True

    def setNodes(self, nodes, erasing=True):
        super().setNodes(nodes, erasing=erasing)

        # print(f"cartesian task '{self.getName()}': ", self.nodes)
        if not nodes:
            self.nodes = []
            self.constr.setNodes(self.nodes, erasing=erasing)
            return 0

        # print('=============================================')

        # core
        self.constr.setNodes(self.nodes[0:], erasing=erasing)  # <==== SET NODES
        if self.ref_matrix is not None:
            self.addReference()

        # print(f'task {self.name} nodes: {self.pos_constr.getNodes().tolist()}')
        # print(f'param task {self.name} nodes: {self.pos_tgt.getValues()[:, self.pos_constr.getNodes()].tolist()}')
        # print('===================================')

    def addReference(self):  # , ref_traj):

        # todo shouldn't just ignore None, right?
        # if ref_traj is None:
        #     return False

        # ref_matrix = np.array(ref_traj)

        # if ref_matrix.ndim == 2 and ref_matrix.shape[1] != len(self.nodes):
        #     raise ValueError(f'Wrong nodes dimension inserted: ({self.ref.shape[1]} != {len(self.nodes)})')
        # elif ref_matrix.ndim == 1 and len(self.nodes) > 1:
        #     raise ValueError(f'Wrong nodes dimension inserted: ({self.ref.shape[1]} != {len(self.nodes)})')
        # what if self.nodes is empty?
        if self.nodes:
            if hasattr(self.nodes, "__iter__"):
                self.ref.assign(self.ref_matrix[:, :len(self.nodes)], self.nodes)  # <==== SET TARGET
            else:
                self.ref.assign(self.ref_matrix, self.nodes)

        return True

    def getError(self, q):
        err = np.empty([6, q.shape[1]])
        for i in range(q.shape[1]):
            [err_pos, err_lin] = self.fun_sym(q[:, i], self.pose_tgt.getValues(0))
            err[:3, i] = np.atleast_2d(err_pos.full().T)
            err[3:, i] = np.atleast_2d(err_lin.full().T)

        return err

    def getDim(self):
        if self.cartesian_type == 'position':
            return 7
        else:
            return self.indices.size

    def getValues(self):
        # necessary method for using this task as an item + reference in phaseManager
        return self.ref.getValues()

    def assign(self, val, nodes=None):
        # necessary method for using this task as an item + reference in phaseManager
        self.ref.assign(val, nodes)
        return 1

    def getCartesianType(self):
        return self.cartesian_type
    