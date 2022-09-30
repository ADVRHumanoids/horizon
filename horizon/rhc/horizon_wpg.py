import threading
from casadi_kin_dyn import pycasadi_kin_dyn
import rospkg
import casadi as cs
import numpy as np
from typing import List, Dict
import time
import math
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from horizon.rhc.taskInterface import TaskInterface
from horizon.rhc.tasks.cartesianTask import CartesianTask
from horizon.rhc.model_description import *
from horizon.problem import Problem
from horizon.solvers import Solver
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.utils import utils, kin_dyn, mat_storer, resampler_trajectory
from horizon.ros import replay_trajectory


# helper class representing a step
class Step:
    def __init__(self, frame: str, k_start: int, k_goal: int,
                 start=np.array([]), goal=np.array([]),
                 clearance=0.10):
        self.frame = frame
        self.k_start = k_start
        self.k_goal = k_goal
        self.goal = np.array(goal)
        self.start = np.array(start)
        self.clearance = clearance


# main wpg class
class HorizonWpg:

    def __init__(self,
                 urdf,
                 contacts: List[str],
                 fixed_joints: List[str],
                 q_init: Dict[str, float],
                 base_init: np.array,
                 opt: Dict[str, any]) -> None:

        ## robot description
        self.N = opt.get('N', 50)
        self.tf = opt.get('tf', 10.0)
        self.dt = self.tf / self.N
        self.nf = 3

        self.prb = Problem(self.N, receding=True)  # logging_level=logging.DEBUG
        self.prb.setDt(self.dt)

        self.contacts = contacts
        self.nc = len(self.contacts)

        self.kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        self.kd = pycasadi_kin_dyn.CasadiKinDyn(urdf)

        self.model = FullModelInverseDynamics(problem=self.prb,
                                         kd=self.kd,
                                         q_init=q_init,
                                         base_init=base_init)


        self.ti = TaskInterface(self.prb, self.model)
        # register my plugin 'Contact'
        self.ti.loadPlugins(['horizon.rhc.plugins.interactionTaskMirror'])

        for frame in self.contacts:
            # add contact task (by dict)

            subtask_force = {'type': 'VertexForceMirror',
                             'name': f'interaction_{frame}',
                             'frame': frame,
                             'fn_min': 10.0,
                             'vertex_frames': [frame]
                             }

            subtask_cartesian = {'type': 'Cartesian',
                                 'name': f'zero_velocity_{frame}',
                                 'distal_link': frame,
                                 'indices': [0, 1, 2, 3, 4, 5],
                                 'cartesian_type': 'velocity',
                                 'nodes': 'all'
                                 }

            contact_task = {'type': 'Contact',
                            'subtask': [subtask_force, subtask_cartesian],
                            'name': 'foot_contact_' + frame}

            z_task = {'type': 'Cartesian',
                      'name': f'foot_z_{frame}',
                      'distal_link': frame,
                      'indices': [2],
                      'fun_type': 'constraint',
                      'cartesian_type': 'position'}

            foot_tgt_task = {'type': 'Cartesian',
                             'name': f'foot_xy_{frame}',
                             'distal_link': frame,
                             'indices': [0, 1],
                             'fun_type': 'constraint',
                             'cartesian_type': 'position'}

            self.ti.setTaskFromDict(contact_task)
            self.ti.setTaskFromDict(z_task)
            self.ti.setTaskFromDict(foot_tgt_task)


        self.forces = [self.prb.getVariables('f_' + c) for c in contacts]

        ## gait params
        f0 = opt.get('f0', np.array([0, 0, 250]))
        self.fmin = opt.get('fmin', 0)
        self.steps = list()

        ## problem description

        # final goal (a.k.a. integral velocity control)
        self.ptgt_final = [0, 0, 0]
        self.vmax = [0.05, 0.05, 0.05]
        # self.ptgt = self.ti.prb.createParameter('ptgt', 3)

        # goalx = self.ti.prb.createFinalConstraint("final_x", self.ti.model.q[0] - self.ptgt[0])
        goalx = {'type': 'Cartesian',
                 'distal_link': 'base_link',
                 'name': 'final_base_x',
                 'indices': [0],
                 'nodes': [self.N]}

        # goaly = self.ti.prb.createFinalResidual("final_y", 1e3 * (self.ti.model.q[1] - self.ptgt[1]))
        goaly = {'type': 'Cartesian',
                 'distal_link': 'base_link',
                 'name': 'final_base_y',
                 'indices': [1],
                 'nodes': [self.N],
                 'fun_type': 'residual',
                 'weight': 1e3}

        # goalrz = self.ti.prb.createFinalResidual("final_rz", 1e3 * (self.ti.model.q[5] - self.ptgt[2]))
        # TODO this does not work
        goalrz = {'type': 'Cartesian',
                  'name': 'final_base_rz',
                  'distal_link': 'base_link',
                  'indices': [5],
                  'nodes': [self.N],
                  'fun_type': 'residual',
                  'weight': 1e3}

        self.ti.setTaskFromDict(goalx)
        self.ti.setTaskFromDict(goaly)
        self.ti.setTaskFromDict(goalrz)

        f_b_x = self.ti.getTask('final_base_x')
        f_b_y = self.ti.getTask('final_base_y')
        f_b_rz = self.ti.getTask('final_base_rz')

        f_b_x.setRef([self.model.q0[0], 0, 0, 0, 0, 0, 1])
        f_b_y.setRef([0, self.model.q0[1], 0, 0, 0, 0, 1])
        f_b_rz.setRef([0, 0, 0, 0, 0, self.model.q0[5], 1])

        self.base_goal_tasks = [f_b_x, f_b_y, f_b_rz]

        q0 = self.model.q0
        v0 = self.model.v0
        f0 = np.array([0, 0, 250])

        # final velocity
        self.model.v.setBounds(v0, v0, nodes=self.N)

        # regularization costs

        # base rotation
        # self.ti.prb.createResidual("min_rot", 1e-4 * (self.ti.model.q[3:5] - q0[3:5]))

        minrot = {'type': 'Cartesian',
                  'distal_link': 'base_link',
                  'name': 'min_rot',
                  'indices': [3, 4],
                  'nodes': list(range(self.N+1)),
                  'fun_type': 'residual',
                  'weight': 1e-4}

        self.ti.setTaskFromDict(minrot)

        min_rot = self.ti.getTask('min_rot')
        min_rot.setRef([0, 0, 0, self.model.q0[3], self.model.q0[4], self.model.q0[5], 1])

        # joint posture
        # self.ti.prb.createResidual("min_q", 1e-1 * (self.ti.model.q[7:] - q0[7:]))
        minq = {'type': 'Postural',
                  'name': 'min_q',
                  'nodes': list(range(self.N+1)),
                  'fun_type': 'residual',
                  'weight': 1e-1}

        self.ti.setTaskFromDict(minq)

        # joint velocity
        self.ti.prb.createResidual("min_v", 1e-2 * self.ti.model.v)

        # final posture
        # self.ti.prb.createFinalResidual("min_qf", 1e1 * (self.ti.model.q[7:] - q0[7:]))
        minqf = {'type': 'Postural',
                  'name': 'min_qf',
                  'nodes': [50],
                  'fun_type': 'residual',
                  'weight': 1e1}

        self.ti.setTaskFromDict(minqf)

        # joint limits

        # jlim_cost = HorizonWpg._barrier(self.ti.model.q[8:10] - (-2.55)) + \
        #             HorizonWpg._barrier(self.ti.model.q[14:16] - (-2.55)) + \
        #             HorizonWpg._barrier(self.ti.model.q[20:22] - (-2.55))
        #
        # self.ti.model.prb.createCost(f'jlim', 10 * jlim_cost)
        jlimMin = {'type': 'JointLimits',
                  'name': 'jlim_min',
                  'indices': [8, 9, 14, 15, 20, 21],
                  'nodes': list(range(self.N+1)),
                  'fun_type': 'residual',
                  'weight': 10,
                  'bound_scaling': 0.95}

        self.ti.setTaskFromDict(jlimMin)

        # regularize input acceleration
        self.ti.prb.createIntermediateResidual("min_q_ddot", 1e-1 * self.ti.model.a)

        # regularize input forces
        for f in self.forces:
            self.ti.prb.createIntermediateResidual(f"min_{f.getName()}", 1e-3 * (f - f0))

        self.fk_fn = list()
        self.com_fn = self.kd.centerOfMass()

        # save default foot height
        self.default_foot_z = dict()
        self.contact_task = dict()
        self.z_task = dict()
        self.foot_tgt_task = dict()

        # contact velocity is zero, and normal force is positive
        for i, frame in enumerate(self.contacts):
            # fk functions and evaluated vars
            fk = self.kd.fk(frame)
            self.fk_fn.append(fk)

            ee_rot = fk(q=self.ti.model.q)['ee_rot']

            # save foot height
            self.default_foot_z[frame] = fk(q=q0)['ee_pos'][2]

            # vertical contact frame
            rot_err = cs.sumsqr(ee_rot[2, :2])
            self.ti.prb.createIntermediateCost(f'{frame}_rot', 1e1 * rot_err)

        ## transcription method
        solver_type = opt.get('solver_type', 'ilqr')

        if solver_type != 'ilqr':
            Transcriptor.make_method('multiple_shooting', self.ti.prb)

        ## solver

        # ilqr options (offline stage)
        ilqr_opts = {
            'ilqr.max_iter': 300,
            'ilqr.alpha_min': 0.01,
            'ilqr.use_filter': False,
            'ilqr.step_length': 1.0,
            'ilqr.enable_line_search': False,
            'ilqr.hxx_reg': 1.0,
            'ilqr.integrator': 'RK4',
            'ilqr.merit_der_threshold': 1e-6,
            'ilqr.step_length_threshold': 1e-9,
            'ilqr.line_search_accept_ratio': 1e-4,
            'ilqr.kkt_decomp_type': 'qr',
            'ilqr.constr_decomp_type': 'qr',
            'ilqr.codegen_enabled': opt.get('codegen', True),
            'ilqr.codegen_workdir': opt.get('workdir', '/tmp/horizon_wpg'),
            'ilqr.verbose': True,
            'ipopt.linear_solver': 'ma57',
        }

        ilqr_opts_rti = ilqr_opts.copy()
        ilqr_opts_rti['ilqr.enable_line_search'] = False
        ilqr_opts_rti['ilqr.max_iter'] = 1

        # set initial condition and initial guess
        self.ti.model.q.setBounds(q0, q0, nodes=0)
        self.ti.model.v.setBounds(v0, v0, nodes=0)

        self.ti.model.q.setInitialGuess(q0)

        (f.setInitialGuess(f0) for f in self.forces)

        # set initial gait pattern
        self._set_gait_pattern(k0=0)

        # create solver and solve initial seed
        self.solver_bs = Solver.make_solver(solver_type, self.ti.prb, ilqr_opts)
        self.solver_rti = Solver.make_solver('ilqr', self.ti.prb, ilqr_opts_rti)

        try:
            self.solver_bs.set_iteration_callback()
            self.solver_rself.ti.set_iteration_callback()
        except:
            pass

        self.bootstrap()

        self.repl = replay_trajectory.replay_trajectory(0.01,
                                                        self.kd.joint_names()[2:], np.array([]),
                                                        {k: None for k in self.ti.model.contacts}, self.ti.kd_frame,
                                                        self.kd)
        self.fixed_joint_pub = rospy.Publisher('joint_states', JointState, queue_size=10)

    def set_target_position(self, x, y, rotz, k=0):
        self.ptgt_final = [x, y, math.sin(rotz / 2.0)]
        kf_x = int(abs((x - self.solution['q'][0, 0]) / self.vmax[0] / self.dt) + 0.5)
        kf_y = int(abs((y - self.solution['q'][1, 0]) / self.vmax[1] / self.dt) + 0.5)
        kf_rotz = int(abs((math.sin(rotz / 2.0) - self.solution['q'][5, 0]) / self.vmax[2] / self.dt) + 0.5)
        kf = k + max((kf_x, kf_y, kf_rotz))
        print(f'[k = {k}] target {self.ptgt_final} to be reached at kf = {kf}')


        self.ti.getTask('final_base_x').setRef([self.ptgt_final[0], 0, 0, 0, 0, 0, 1])
        self.ti.getTask('final_base_y').setRef([0, self.ptgt_final[1], 0, 0, 0, 0, 1])
        self.ti.getTask('final_base_rz').setRef([0, 0, 0, 0, 0, self.ptgt_final[2], 1])
        # self.ptgt.assign(self.ptgt_final, nodes=self.N)

    def set_mode(self, mode: str):
        if mode == 'base_ctrl':
            pass
        elif mode == 'step_ctrl':
            pass
        else:
            raise KeyError(f'invalid mode {mode}: choose between base_ctrl and step_ctrl')

    def bootstrap(self):
        t = time.time()
        self.solver_bs.solve()
        elapsed = time.time() - t
        print(f'bootstrap solved in {elapsed} s')
        self.solution = self.solver_bs.getSolutionDict()
        self.update_initial_guess(dk=0)

    def save_solution(self, filename):
        from horizon.utils import mat_storer
        ms = mat_storer.matStorer(filename)
        ms.store(self.solution)

    def load_solution(self, filename):
        from horizon.utils import mat_storer
        ms = mat_storer.matStorer(filename)
        ig = ms.load()
        self.update_initial_guess(dk=0, from_dict=ig)

    def rti(self, k):

        self._set_gait_pattern(k0=k)

        t = time.time()
        self.solver_rti.solve()
        elapsed = time.time() - t
        print(f'rti solved in {elapsed} s')

        self.solution = self.solver_rti.getSolutionDict()

    def toggle_base_dof(self, id=2, enabled=True):
        self.base_goal_tasks[id].setNodes([self.N] if enabled else [])

    def publish_solution(self):

        self.repl.frame_force_mapping = {self.contacts[i]: self.solution[self.forces[i].getName()] for i in
                                         range(3)}
        self.repl.publish_joints(self.solution['q'][:, 0])
        self.repl.publishContactForces(rospy.Time.now(), self.solution['q'][:, 0], 0)

        msg = JointState()
        msg.name = self.ti.model.fixed_joints
        msg.position = self.ti.model.fixed_joints_pos
        msg.header.stamp = rospy.Time.now()
        self.fixed_joint_pub.publish(msg)

    def update_initial_guess(self, dk=1, from_dict=None):
        if from_dict is None:
            from_dict = self.solution

        x_opt = from_dict['x_opt']
        u_opt = from_dict['u_opt']
        xig = np.roll(x_opt, -dk, axis=1)
        for i in range(abs(dk)):
            xig[:, -1 - i] = x_opt[:, -1]
        self.ti.prb.getState().setInitialGuess(xig)

        uig = np.roll(u_opt, -dk, axis=1)
        for i in range(abs(dk)):
            uig[:, -1 - i] = u_opt[:, -1]
        self.ti.prb.getInput().setInitialGuess(uig)

        self.ti.prb.setInitialState(x0=xig[:, 0])

    def resample(self, dt_res):

        contact_map = {self.ti.model.contacts[i]: self.solution[self.forces[i].getName()] for i in range(3)}

        dae = {'x': self.ti.prb.getState().getVars(), 'p': self.ti.model.a, 'ode': self.ti.model.xdot, 'quad': 0.0}

        q_res, v_res, a_res, f_res, tau_res = resampler_trajectory.resample_torques(
            self.solution['q'], self.solution['v'], self.solution['a'],
            self.dt, dt_res,
            dae,
            contact_map,
            self.kd, self.kd_frame
        )

        self.solution['q_res'] = q_res
        self.solution['v_res'] = v_res
        self.solution['a_res'] = a_res
        self.solution['tau_res'] = tau_res
        self.solution['f_res'] = f_res

        return q_res, v_res, a_res, tau_res, f_res

    def replay(self, dt_res=0.01):

        q_res, v_res, a_res, tau_res, f_res = self.resample(dt_res)
        repl = replay_trajectory.replay_trajectory(dt_res, self.kd.joint_names()[2:], q_res, f_res, self.ti.kd_frame,
                                                   self.kd)
        repl.replay()

    def compute_trj_msg(self, resample=False, dt_res=0.01):

        suffix = ''
        N = self.N
        dt = self.dt

        if resample:
            self.resample(dt_res=dt_res)
            N = self.solution['a_res'].shape[1]
            suffix = '_res'
            dt = dt_res

        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = [f'base_joint_{i}' for i in range(7)] + self.kd.joint_names()[2:]
        for i in range(self.N):
            p = JointTrajectoryPoint()
            p.positions = self.solution['q' + suffix][:, i]
            p.velocities = self.solution['v' + suffix][:, i]
            p.accelerations = self.solution['a' + suffix][:, i]
            if 'tau' + suffix in self.solution.keys():
                p.effort = self.solution['tau' + suffix][:, i]
            p.time_from_start = rospy.Duration(dt * i)
            msg.points.append(p)

        return msg

    def compute_path_msg(self, resample=False, dt_res=0.01):

        suffix = ''
        N = self.N
        dt = self.dt

        if resample:
            self.resample(dt_res=dt_res)
            N = self.solution['a_res'].shape[1]
            suffix = '_res'
            dt = dt_res

        msgs = dict()

        for i, c in enumerate(self.contacts):

            msg = Path()
            msg.header.frame_id = 'world'
            msg.header.stamp = rospy.Time.now()

            for k in range(N):
                q = self.solution['q' + suffix][:, k]
                cpos = self.fk_fn[i](q=q)['ee_pos']
                posmsg = PoseStamped()
                posmsg.header = msg.header
                posmsg.pose.orientation.w = 1
                posmsg.pose.position.x = cpos[0]
                posmsg.pose.position.y = cpos[1]
                posmsg.pose.position.z = cpos[2]
                msg.poses.append(posmsg)

            msgs[c] = msg

        msg = Path()
        msg.header.frame_id = 'world'
        msg.header.stamp = rospy.Time.now()

        for k in range(N):
            q = self.solution['q' + suffix][:, k]
            cpos = self.com_fn(q=q)['com']
            posmsg = PoseStamped()
            posmsg.header = msg.header
            posmsg.pose.orientation.w = 1
            posmsg.pose.position.x = cpos[0]
            posmsg.pose.position.y = cpos[1]
            posmsg.pose.position.z = cpos[2]
            msg.poses.append(posmsg)

        msgs['com'] = msg

        return msgs

    # z trajectory
    def _z_trj(tau):
        return 64. * tau ** 3 * (1 - tau) ** 3

    def _set_step_ctrl(self):

        for t in self.base_goal_tasks:
            t.setNodes([])

    def _set_base_ctrl(self):

        for t in self.base_goal_tasks:
            t.setNodes([self.N])

    def _set_gait_pattern(self, k0: int):
        """
        Set the correct nodes to wpg costs and bounds according
        to a specified gait pattern and initial horizon time (absolute)
        """

        # reset bounds
        for f in self.forces:
            f.setBounds(lb=np.full(self.nf, -np.inf),
                        ub=np.full(self.nf, np.inf))

        # reset contact indices for all legs

        contact_nodes = {key: list(range(self.N + 1)) for key in self.contacts}
        # unilat_nodes = [list(range(self.N)) for _ in range(self.nc)]
        clea_nodes = {key: [] for key in self.contacts}
        contact_k = {key: [] for key in self.contacts}
        # liftoff_touchdown_k = [list() for _ in range(self.nc)]

        z_ref = {key: None for key in self.contacts}
        xy_ref = {key: None for key in self.contacts}

        for s in self.steps:
            s: Step = s
            frame = s.frame
            k_start = s.k_start - k0
            k_goal = s.k_goal - k0
            swing_nodes = list(range(k_start, k_goal))
            swing_nodes_in_horizon_x = [k for k in swing_nodes if k >= 0 and k <= self.N]
            n_swing = len(swing_nodes)

            # this step is outside the horizon!
            if n_swing == 0:
                continue

            # update nodes contact constraint
            contact_nodes[frame] = [k for k in contact_nodes[frame] if k not in swing_nodes and k < self.N]

            # update nodes for unilateral constraint
            # unilat_nodes[l] = [k for k in unilat_nodes[l] if k not in swing_nodes]

            # update zero force constraints
            # fzero = np.zeros(self.nf)
            # self.forces[l].setBounds(lb=fzero, ub=fzero, nodes=swing_nodes_in_horizon_u)

            # update z trajectory constraints
            # for all swing nodes + first stance node
            k_trj = swing_nodes_in_horizon_x[:]

            # compute swing trj
            z_temp = np.zeros((1, len(k_trj)))
            k_elem = 0
            for k in k_trj:
                tau = (k - k_start) / n_swing
                z_start = self.default_foot_z[frame] if s.start.size == 0 else s.start[2]
                z_goal = self.default_foot_z[frame] if s.goal.size == 0 else s.goal[2]
                zk = HorizonWpg._z_trj(tau) * s.clearance
                zk += (1 - tau) * z_start + tau * z_goal
                z_temp[:, k_elem] = zk
                k_elem += 1

            if z_ref[frame] is None:
                z_ref[frame] = z_temp
            else:
                z_ref[frame] = np.append(z_ref[frame], z_temp)

            # foot_tgt_params[l].assign([0, 0, zk], nodes=k)

            # liftoff_touchdown_k[l].append(k_start)

            # if k_goal <= self.N:
            #     liftoff_touchdown_k[l].append(k_goal)

            if len(k_trj) > 0:
                clea_nodes[frame].extend(k_trj)

            # assign xy goal
            xy_ref[frame] = np.array((2, 1))
            if k_goal <= self.N and k_goal > 0 and s.goal.size > 0:
                contact_k[frame].append(k_goal)
                xy_ref[frame] = np.zeros((2, 1))
                xy_ref[frame][0] = s.goal[0]
                xy_ref[frame][1] = s.goal[1]

        # update contact nodes

        # TODO: this is very ugly.
        #  this is because I require now to set the whole vector [pos (3) + orientation (4)] to the cartesian Task
        for frame in self.contacts:
            self.ti.getTask(f'foot_contact_{frame}').setNodes(contact_nodes[frame])
            self.ti.getTask(f'foot_z_{frame}').setNodes(clea_nodes[frame])

            if z_ref[frame] is not None:
                z_ref_mat = np.zeros([7, z_ref[frame].size])
                z_ref_mat[2, :] = z_ref[frame]
            else:
                z_ref_mat = None

            self.ti.getTask(f'foot_z_{frame}').setRef(z_ref_mat)
            self.ti.getTask(f'foot_xy_{frame}').setNodes(contact_k[frame])

            xy_ref_mat = None if xy_ref[frame] is None else [xy_ref[frame][0], xy_ref[frame][1], 0, 0, 0, 0, 1]
            self.ti.getTask(f'foot_xy_{frame}').setRef(xy_ref_mat)
            # self.contact_constr[i].setNodes(contact_nodes[i], erasing=True)
            # self.unilat_constr[i].setNodes(unilat_nodes[i], erasing=True)  # fz > 10
            # friction_constr[i].setNodes(unilat_nodes[i], erasing=True)
            # self.foot_z_constr[i].setNodes(clea_nodes[i], erasing=True)
            # self.foot_tgt_constr[i].setNodes(contact_k[i], erasing=True)
            # self.foot_vert_takeoff_landing[i].setNodes(liftoff_touchdown_k[i], erasing=True)
