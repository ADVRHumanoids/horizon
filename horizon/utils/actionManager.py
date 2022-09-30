import copy

import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn
import casadi as cs

from horizon.problem import Problem
from horizon.utils import utils, kin_dyn
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers.solver import Solver
from horizon.rhc.tasks.cartesianTask import CartesianTask
from horizon.ros import replay_trajectory
from horizon.rhc.taskInterface import TaskInterface
import rospy
import os
import subprocess


# barrier function
def _barrier(x):
    return cs.sum1(cs.if_else(x > 0, 0, x ** 2))


def _trj(tau):
    return 64. * tau ** 3 * (1 - tau) ** 3


class Action:
    """
    simple class representing a generic action
    """

    def __init__(self, frame: str, k_start: int, k_goal: int, start=np.array([]), goal=np.array([])):
        self.frame = frame
        self.k_start = k_start
        self.k_goal = k_goal
        self.goal = np.array(goal)
        self.start = np.array(start)


class Step(Action):
    """
    simple class representing a step, contains the main info about the step
    """

    def __init__(self, frame: str, k_start: int, k_goal: int, start=np.array([]), goal=np.array([]), clearance=0.08, indices=None):
        super().__init__(frame, k_start, k_goal, start, goal)
        self.clearance = clearance
        self.indices = indices


# what if the action manager provides only the nodes? but for what?
# - for the contacts
# - for the variables and the constraints
class ActionManager:
    """
    set of actions which involves combinations of constraints and bounds
    """

    def __init__(self, task_interface: TaskInterface, opts=None):
        # prb: Problem, urdf, kindyn, contacts_map, default_foot_z

        self.ti = task_interface

        self.opts = opts

        self.prb = self.ti.prb
        self.contact_map = self.ti.model.cmap

        self.N = self.prb.getNNodes() - 1
        # todo list of contact is fixed?

        self.constraints = list()
        self.current_cycle = 0  # what to do here?

        self.contacts = self.contact_map.keys()
        self.nc = len(self.contacts)

        self.kd = self.ti.model.kd

        # todo: how to set these options?
        # f0 = opts.get('f0', np.array([0, 0, 250]))
        # self.fmin = opts.get('fmin', 0)
        self.fmin = 10.

        # todo set default foot automatically, not using opts
        # contact velocity is zero, and normal force is positive
        self.default_foot_z = dict()
        for i, frame in enumerate(self.contacts):
            # fk functions and evaluated vars
            fk = self.ti.model.kd.fk(frame)
            dfk = self.ti.model.kd.frameVelocity(frame, self.ti.model.kd_frame)

            # save foot height
            self.default_foot_z[frame] = (fk(q=self.ti.model.q0)['ee_pos'][2]).toarray()

        self.k0 = 0

        # TODO: action manager requires some tasks from the ti. It searches them by NAME.
        self.required_tasks = dict()
        tasks_foot_contact = [f"foot_contact_{contact}" for contact in self.contacts]
        tasks_foot_z = [f"foot_z_{contact}" for contact in self.contacts]
        tasks_foot_xy = [f"foot_xy_{contact}" for contact in self.contacts]

        for task in tasks_foot_contact:
            if self.ti.getTask(task) is None:
                print(f'task "{task}" not found.')

        for task in tasks_foot_z:
            if self.ti.getTask(task) is None:
                print(f'task "{task}" not found.')

        for task in tasks_foot_xy:
            if self.ti.getTask(task) is None:
                print(f'task "{task}" not found.')


        self.required_tasks['foot_contact'] = {contact: self.ti.getTask(f"foot_contact_{contact}") for contact in self.contacts}
        self.required_tasks['foot_z'] = {contact: self.ti.getTask(f"foot_z_{contact}") for contact in self.contacts}
        self.required_tasks['foot_xy'] = {contact: self.ti.getTask(f"foot_xy_{contact}") for contact in self.contacts}
        # TODO: merge the foot_z and the foot_xy?
        # self.required_tasks['foot_cartesian'] = {contact: self.ti.getTask(f"foot_cartesian_{contact}") for contact in self.contacts}

        self.task_type = self._check_required_tasks_type(['Cartesian', 'Contact'])
        self.init_constraints()
        self._set_default_action()

        self.action_list = []

    def compute_polynomial_trajectory(self, k_start, nodes, nodes_duration, p_start, p_goal, clearance, dim=None):

        if dim is None:
            dim = [0, 1, 2]

        # todo check dimension of parameter before assigning it

        traj_array = np.zeros(len(nodes))

        start = p_start[dim]
        goal = p_goal[dim]

        index = 0
        for k in nodes:
            tau = (k - k_start) / nodes_duration
            trj = _trj(tau) * clearance
            trj += (1 - tau) * start + tau * goal
            traj_array[index] = trj
            index = index + 1

        return np.array(traj_array)

    def _set_default_action(self):
        # todo for now the default is robot still, in contact

        for frame, nodes in self.contact_constr_nodes.items():
            self.contact_constr_nodes[frame] = list(range(self.N + 1))
        for frame, nodes in self.z_constr_nodes.items():
            self.z_constr_nodes[frame] = []
        for frame, nodes in self.foot_tgt_constr_nodes.items():
            self.foot_tgt_constr_nodes[frame] = []

        for frame, param in self._foot_z_param.items():
            self._foot_z_param[frame] = np.empty((7, self.N + 1))
            self._foot_z_param[frame][:] = np.NaN

        for frame, param in self._foot_tgt_params.items():
            self._foot_tgt_params[frame] = np.empty((7, self.N + 1))
            self._foot_tgt_params[frame][:] = np.NaN

        # clearance nodes
        for frame, z_constr in self.z_constr.items():
            z_constr.setNodes(self.z_constr_nodes[frame])

        # xy trajectory nodes
        for frame, tgt_constr in self.foot_tgt_constr.items():
            tgt_constr.setNodes(self.foot_tgt_constr_nodes[frame])

        # contact nodes
        for frame, c_constr in self.contact_constr.items():
            c_constr.setNodes(self.contact_constr_nodes[frame])

    def _check_required_tasks_type(self, required_tasks):
        # actionManager requires some tasks for working. It asks the TaskInterface for tasks.
        task_type = dict()
        for task in required_tasks:
            found_task = self.ti.getTasksType(task)
            if found_task is None:
                raise Exception(
                    'Task {} not found. ActionManager requires this task, please provide your implementation.'.format(
                        task))
            else:
                task_type[task] = found_task

        return task_type

    def setRequiredTasks(self, task_dict):
        # TODO: add some logic
        self.required_tasks = task_dict

    def init_constraints(self):

        # todo ISSUE: I need to initialize all the constraints that I will use in the problem
        # self._zero_vel_constr = dict()
        # self._unil_constr = dict()
        # self._friction_constr = dict()
        # self._cartesian_constr = dict()
        # self._target_constr = dict()
        # self._foot_z_constr = dict()
        # self._foot_tgt_params = dict()
        # self._foot_z_param = dict()

        # todo do i keep track of the nodes here or I implement it in the receding_task?
        # ===================================================================================
        self.contact_constr_nodes = dict()
        self.z_constr_nodes = dict()
        self.foot_tgt_constr_nodes = dict()

        for frame in self.contacts:
            self.contact_constr_nodes[frame] = []
            self.z_constr_nodes[frame] = []
            self.foot_tgt_constr_nodes[frame] = []

        # ... and params
        self._foot_tgt_params = dict()
        self._foot_z_param = dict()

        for frame in self.contacts:
            self._foot_tgt_params[frame] = None
            self._foot_z_param[frame] = None

            # ==================================================================================

        self.contact_constr = dict()
        self.z_constr = dict()
        self.foot_tgt_constr = dict()

        for frame in self.contacts:

            # TODO should specify these from outside or inside?
            #   if inside: hidden from the user, easier. But what if I want to change the costs?
            #   if outside:
            self.contact_constr[frame] = self.required_tasks['foot_contact'][frame] # this is a plugin!
            self.z_constr[frame] = self.required_tasks['foot_z'][frame]
            self.foot_tgt_constr[frame] = self.required_tasks['foot_xy'][frame]


    def setContact(self, frame, nodes):
        """
        establish/break contact
        """
        # todo reset all the other "contact" constraints on these nodes
        # self._reset_task_constraints(frame, nodes_in_horizon_x)

        # todo what to do with z_constr?
        # self.z_constr.reset()

        self.contact_constr[frame].setNodes(nodes)

    def _append_nodes(self, node_list, new_nodes):
        for node in new_nodes:
            if node not in node_list:
                node_list.append(node)

        return node_list

    def _append_params(self, params_array, new_params, nodes):

        params_array[nodes] = np.append(params_array, new_params)

        return params_array

    def setStep(self, step):
        self.action_list.append(step)
        self._step(step)

    def _step(self, step: Step):
        """
        add step to horizon stack
        """
        s = copy.deepcopy(step)
        # todo temporary

        # todo how to define current cycle
        frame = s.frame
        k_start = s.k_start
        k_goal = s.k_goal
        all_contact_nodes = self.contact_constr_nodes[frame]
        swing_nodes = list(range(k_start, k_goal))
        stance_nodes = [k for k in all_contact_nodes if k not in swing_nodes]
        n_swing = len(swing_nodes)

        swing_nodes_in_horizon = [k for k in swing_nodes if k >= 0 and k <= self.N]
        stance_nodes_in_horizon = [k for k in stance_nodes if k >= 0 and k <= self.N]
        n_swing_in_horizon = len(swing_nodes_in_horizon)

        # this step is outside the horizon!
        # todo what to do with default action?

        if n_swing_in_horizon == 0:
            print(f'========= skipping step {s.frame}. Not in horizon: {swing_nodes_in_horizon} ==========')
            return 0
        print(f'========= activating step {s.frame}: {swing_nodes_in_horizon} ==========')
        # adding nodes to the current ones (if any)
        self.contact_constr_nodes[frame] = stance_nodes
        self.foot_tgt_constr_nodes[frame] = self._append_nodes(self.foot_tgt_constr_nodes[frame], [k_goal])
        self.z_constr_nodes[frame] = self._append_nodes(self.z_constr_nodes[frame], swing_nodes_in_horizon)

        # break contact at swing nodes + z_trajectory + (optional) xy goal
        # contact
        self.setContact(frame, self.contact_constr_nodes[frame])

        # xy goal
        # TODO: refactor this
        # if self.N >= k_goal > 0 and step.goal.size > 0:
            # adding param:
            # self._foot_tgt_params[frame][:, swing_nodes_in_horizon] = s.goal[:2]
            #
            # self.foot_tgt_constr[frame].setNodes(self.foot_tgt_constr_nodes[frame])  # [k_goal]
            # self.foot_tgt_constr[frame].setRef(self._foot_tgt_params[frame][self.foot_tgt_constr_nodes[frame]])  # s.goal[:2]

        if step.goal.size > 0:
            s.start = np.array([0, 0, s.goal[2]])

        # z goal
        start = np.array([0, 0, self.default_foot_z[frame]]) if s.start.size == 0 else s.start
        goal = np.array([0, 0, self.default_foot_z[frame]]) if s.goal.size == 0 else s.goal

        z_traj = self.compute_polynomial_trajectory(k_start, swing_nodes_in_horizon, n_swing, start, goal, s.clearance,
                                                    dim=2)

        cart_mask_z = np.zeros([7, len(swing_nodes_in_horizon)])
        cart_mask_z[2, :] = z_traj[:len(swing_nodes_in_horizon)]

        # adding param
        self._foot_z_param[frame][:, swing_nodes_in_horizon] = cart_mask_z
        self.z_constr[frame].setNodes(self.z_constr_nodes[frame])  # swing_nodes_in_horizon
        self.z_constr[frame].setRef(self._foot_z_param[frame][:, self.z_constr_nodes[frame]])  # z_traj

    # todo unify the actions below, these are just different pattern of actions
    def _jump(self, nodes):

        # todo add parameters for step
        for contact in self.contacts:
            k_start = nodes[0]
            k_end = nodes[-1]
            s = Step(contact, k_start, k_end)
            self.setStep(s)

    def _walk(self, nodes, step_pattern=None, step_nodes_duration=None):

        # todo add parameters for step
        step_list = list()
        k_step_n = 5 if step_nodes_duration is None else step_nodes_duration  # default duration (in nodes) of swing step
        k_start = nodes[0]  # first node to begin walk
        k_end = nodes[1]

        n_step = (k_end - k_start) // k_step_n  # integer divide
        # default step pattern of classic walking (crawling)
        pattern = step_pattern if step_pattern is not None else list(range(len(self.contacts)))
        # =========================================
        for n in range(n_step):
            l = list(self.contacts)[pattern[n % len(pattern)]]
            k_end_rounded = k_start + k_step_n
            s = Step(l, k_start, k_end_rounded)
            print(l, k_start, k_end_rounded)
            k_start = k_end_rounded
            step_list.append(s)

        for s_i in step_list:
            self.setStep(s_i)

    def _trot(self, nodes):

        # todo add parameters for step
        k_start = nodes[0]
        k_step_n = 5  # default swing duration
        k_end = nodes[1]

        n_step = (k_end - k_start) // k_step_n  # integer divide
        step_list = []
        for n in range(n_step):
            if n % 2 == 0:
                l1 = 'lf_foot'
                l2 = 'rh_foot'
            else:
                l1 = 'lh_foot'
                l2 = 'rf_foot'
            k_end = k_start + k_step_n
            s1 = Step(l1, k_start, k_end, clearance=0.03)
            s2 = Step(l2, k_start, k_end, clearance=0.03)
            k_start = k_end
            step_list.append(s1)
            step_list.append(s2)

        for s_i in step_list:
            self.setStep(s_i)

    def execute(self, bootstrap_solution):
        """
        set the actions and spin
        """
        self._update_initial_state(bootstrap_solution, -1)

        self._set_default_action()
        k0 = 1

        for action in self.action_list:
            action.k_start = action.k_start - k0
            action.k_goal = action.k_goal - k0
            action_nodes = list(range(action.k_start, action.k_goal))
            action_nodes_in_horizon = [k for k in action_nodes if k >= 0]
            self._step(action)

        # for cnsrt_name, cnsrt in self.prb.getConstraints().items():
        #     print(cnsrt_name)
        #     print(cnsrt.getNodes().tolist())
        # remove expired actions
        self.action_list = [action for action in self.action_list if
                            len([k for k in list(range(action.k_start, action.k_goal)) if k >= 0]) != 0]
        # todo right now the non-active nodes of the parameter gets dirty,
        #  because .assing() only assign a value to the current nodes, the other are left with the old value
        #  better to reset?
        # self.pos_tgt.reset()
        # return 0

        ## todo should implement --> removeNodes()
        ## todo should implement a function to reset to default values

    def _update_initial_state(self, bootstrap_solution, shift_num):

        x_opt = bootstrap_solution['x_opt']
        u_opt = bootstrap_solution['u_opt']

        xig = np.roll(x_opt, shift_num, axis=1)

        for i in range(abs(shift_num)):
            xig[:, -1 - i] = x_opt[:, -1]
        self.prb.getState().setInitialGuess(xig)

        uig = np.roll(u_opt, shift_num, axis=1)

        for i in range(abs(shift_num)):
            uig[:, -1 - i] = u_opt[:, -1]
        self.prb.getInput().setInitialGuess(uig)

        self.prb.setInitialState(x0=xig[:, 0])


if __name__ == '__main__':

    # set up problem
    ns = 50
    tf = 2.0  # 10s
    dt = tf / ns

    # set up solver
    solver_type = 'ilqr'

    # set up model
    path_to_examples = os.path.dirname('../examples/')
    urdffile = os.path.join(path_to_examples, 'urdf', 'spot.urdf')
    urdf = open(urdffile, 'r').read()

    contacts = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']

    base_init = np.array([0.0, 0.0, 0.505, 0.0, 0.0, 0.0, 1.0])
    q_init = {'lf_haa_joint': 0.0,
              'lf_hfe_joint': 0.9,
              'lf_kfe_joint': -1.52,

              'lh_haa_joint': 0.0,
              'lh_hfe_joint': 0.9,
              'lh_kfe_joint': -1.52,

              'rf_haa_joint': 0.0,
              'rf_hfe_joint': 0.9,
              'rf_kfe_joint': -1.52,

              'rh_haa_joint': 0.0,
              'rh_hfe_joint': 0.9,
              'rh_kfe_joint': -1.52}

    problem_opts = {'ns': ns, 'tf': tf, 'dt': dt}
    model_description = 'whole_body'

    # todo for now, there are three ways to add contacts:
    #  contacts=contacts CORRECT
    #  setContactFrame(contact) WRONG, does not update dynamics
    #  interactionTask WRONG, does not update dynamics
    ti = TaskInterface(urdf, q_init, base_init, problem_opts, model_description, contacts=contacts) #
    ti.loadPlugins(['horizon.rhc.plugins.contactTaskSpot'])

    # [ti.model.setContactFrame(contact) for contact in contacts]

    for contact in contacts:
        contact_task = {
            'type': 'Force',
            'name': f'{contact}_interaction_task',
            'frame': contact,
        }
        ti.setTaskFromDict(contact_task)

    q0 = ti.q0
    v0 = ti.v0

    f0 = np.array([0, 0, 55])
    nc = 4  # todo: only required for replay

    # final goal (a.k.a. integral velocity control)
    ptgt_final = [0., 0., 0.]
    vmax = [0.05, 0.05, 0.05]
    ptgt = ti.prb.createParameter('ptgt', 3)

    q = ti.prb.getVariables('q')
    v = ti.prb.getVariables('v')
    a = ti.prb.getVariables('a')

    forces = [ti.prb.getVariables('f_' + c) for c in contacts]

    # goalx = prb.createFinalResidual("final_x",  1e3*(q[0] - ptgt[0]))
    goalx = ti.prb.createFinalConstraint("final_x", q[0] - ptgt[0])
    goaly = ti.prb.createFinalResidual("final_y", 1e3 * (q[1] - ptgt[1]))
    # goalrz = prb.createFinalResidual("final_rz", 1e3 * (q[5] - ptgt[2]))
    # base_goal_tasks = [goalx, goaly, goalrz]

    # final velocity
    # v.setBounds(v0, v0, nodes=ns)
    # regularization costs

    # base rotation
    ti.prb.createResidual("min_rot", 1e-3 * (q[3:5] - q0[3:5]))

    # joint posture
    ti.prb.createResidual("min_q", 1e0 * (q[7:] - q0[7:]))

    # joint velocity
    ti.prb.createResidual("min_v", 1e-2 * v)

    # final posture
    ti.prb.createFinalResidual("min_qf", 1e0 * (q[7:] - q0[7:]))

    # regularize input
    # ti.prb.createIntermediateResidual("min_q_ddot", 1e-1 * a)
    ti.prb.createIntermediateResidual("min_q_ddot", 1e-2 * a)

    # ti.prb.createFinalConstraint('q_fb', q[:6] - q0[:6])

    for f in forces:
        ti.prb.createIntermediateResidual(f"min_{f.getName()}", 1e-2 * (f - f0))

    for frame in contacts:
        subtask_force = {'type': 'Force',
                         'name': f'interaction_{frame}',
                         'frame': frame,
                         'indices': [0, 1, 2]}

        subtask_cartesian = {'type': 'Cartesian',
                             'name': f'zero_velocity_{frame}',
                             'distal_link': frame,
                             'indices': [0, 1, 2],
                             'cartesian_type': 'velocity'}

        contact_task_node = {'type': 'Contact',
                             'name': f'foot_contact_{frame}',
                             'subtask': [subtask_force, subtask_cartesian]}

        z_task_node = {'type': 'Cartesian',
                       'name': f'foot_z_{frame}',
                       'distal_link': frame,
                       'indices': [2],
                       'fun_type': 'constraint',
                       'cartesian_type': 'position'}

        foot_tgt_task_node = {'type': 'Cartesian',
                              'name': f'foot_xy_{frame}',
                              'distal_link': frame,
                              'indices': [0, 1],
                              'fun_type': 'constraint',
                              'cartesian_type': 'position'}

        ti.setTaskFromDict(contact_task_node)
        ti.setTaskFromDict(z_task_node)
        ti.setTaskFromDict(foot_tgt_task_node)

    opts = dict()
    am = ActionManager(ti, opts)


    # set the required tasks from action manager
    # k_start = 15
    # k_end = 25
    # s_1 = Step('lf_foot', k_start, k_end)
    #
    # k_start = 10
    # k_end = 20
    # s_2 = Step('rf_foot', k_start, k_end)
    #
    # k_start = 25
    # k_end = 35
    # f_k = cs.Function.deserialize(ti.kd.fk('lh_foot'))
    # initial_lh_foot = f_k(q=q0)['ee_pos']
    # step_len = np.array([0.2, 0.1, 0])
    # print(f'step target: {initial_lh_foot + step_len}')
    # s_3 = Step('lh_foot', k_start, k_end, goal=initial_lh_foot + step_len)
    #
    # k_start = 45
    # k_end = 55
    # s_4 = Step('rh_foot', k_start, k_end)

    # k_start = 8
    # k_end = 15
    # s_2 = Step('rf_foot', k_start, k_end)
    if solver_type != 'ilqr':
        Transcriptor.make_method('multiple_shooting', ti.prb)

    # set initial condition and initial guess
    q.setBounds(q0, q0, nodes=0)
    v.setBounds(v0, v0, nodes=0)

    q.setInitialGuess(q0)

    for f in forces:
        f.setInitialGuess(f0)

    # k_start = 25
    # k_end = 36
    # s_lf = Step('lf_foot', k_start, k_end)
    # s_rf = Step('rf_foot', k_start, k_end)
    # s_lh = Step('lh_foot', k_start, k_end)
    # s_rh = Step('rh_foot', k_start, k_end)
    # ============== add steps!!!!!!!!! =======================
    # am.setStep(s_lf)
    # am.setStep(s_rf)
    # am.setStep(s_lh)
    # am.setStep(s_rh)

    # am._jump(list(range(25, 36)))
    # am._jump(list(range(40, 51)))
    # am._jump(list(range(55, 66)))
    # am._jump(list(range(70, 81)))
    # am.setStep(s_1)
    # am.setStep(s_2)
    # am._jump(list(range(40, 51)))
    # am.setStep(s_3)

    # all_nodes = ti.prb.getNNodes()
    # swing_nodes = [k for k in list(range(all_nodes)) if k not in list(range(k_start, k_end))]
    # am.setContact('rf_foot', nodes=swing_nodes)
    # am.setContact('lf_foot', nodes=swing_nodes)
    # am.setContact('rh_foot', nodes=swing_nodes)
    # am.setContact('lh_foot', nodes=swing_nodes)

    am._walk([10, 200], [0, 2, 1, 3])
    # am._trot([50, 100])
    # am._jump([55, 60])

    # =========================================

    # k_start = 20
    # k_end = 25
    # s_lf = Step('lf_foot', k_start, k_end)
    # s_rh = Step('rh_foot', k_start, k_end)
    # k_start = 25
    # k_end = 30
    # s_lh = Step('lh_foot', k_start, k_end)
    # s_rf = Step('rf_foot', k_start, k_end)
    #
    # am.setStep(s_lf)
    # am.setStep(s_rf)
    # am.setStep(s_lh)
    # am.setStep(s_rh)

    # create solver and solve initial seed
    # print('===========executing ...========================')

    opts = {'ipopt.tol': 0.001,
            'ipopt.constr_viol_tol': 1e-3,
            'ipopt.max_iter': 1000,
            'error_on_fail': True,
            'ilqr.max_iter': 200,
            'ilqr.alpha_min': 0.01,
            'ilqr.use_filter': False,
            'ilqr.hxx_reg': 0.0,
            'ilqr.integrator': 'RK4',
            'ilqr.merit_der_threshold': 1e-6,
            'ilqr.step_length_threshold': 1e-9,
            'ilqr.line_search_accept_ratio': 1e-4,
            'ilqr.kkt_decomp_type': 'qr',
            'ilqr.constr_decomp_type': 'qr',
            'ilqr.verbose': True,
            'ipopt.linear_solver': 'ma57',
            }

    opts_rti = opts.copy()
    opts_rti['ilqr.enable_line_search'] = False
    opts_rti['ilqr.max_iter'] = 4

    solver_bs = Solver.make_solver(solver_type, ti.prb, opts)
    solver_rti = Solver.make_solver(solver_type, ti.prb, opts_rti)

    # ptgt.assign(ptgt_final, nodes=ns)
    solver_bs.solve()
    solution = solver_bs.getSolutionDict()

    os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
    subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])
    rospy.loginfo("'spot' visualization started.")

    # single replay
    # q_sol = solution['q']
    # frame_force_mapping = {contacts[i]: solution[forces[i].getName()] for i in range(nc)}
    # repl = replay_trajectory.replay_trajectory(dt, kd.joint_names()[2:], q_sol, frame_force_mapping, kd_frame, kd)
    # repl.sleep(1.)
    # repl.replay(is_floating_base=True)
    # exit()
    # =========================================================================
    repl = replay_trajectory.replay_trajectory(dt, ti.kd.joint_names()[2:], np.array([]), {k: None for k in contacts},
                                               ti.kd_frame, ti.kd)
    iteration = 0

    solver_rti.solution_dict['x_opt'] = solver_bs.getSolutionState()
    solver_rti.solution_dict['u_opt'] = solver_bs.getSolutionInput()

    flag_action = 1
    while True:

        if flag_action == 1 and iteration > 50:
            flag_action = 0
            am._trot([40, 80])
        #
        # if iteration > 100:
        #     ptgt.assign([1., 0., 0], nodes=ns)

        # if iteration > 160:
        #     ptgt.assign([0., 0., 0], nodes=ns)

        # if iteration % 20 == 0:
        #     am.setStep(s_1)
        #     am.setContact(0, 'rh_foot', range(5, 15))
        #
        iteration = iteration + 1
        print(iteration)
        #
        am.execute(solver_rti)

        # if iteration == 10:
        #     am.setStep(s_lol)
        # for cnsrt_name, cnsrt in ti.prb.getConstraints().items():
        #     print(cnsrt_name)
        #     print(cnsrt.getNodes())
        # if iteration == 20:
        #     am._jump(list(range(25, 36)))
        # solver_bs.solve()
        solver_rti.solve()
        solution = solver_rti.getSolutionDict()

        repl.frame_force_mapping = {contacts[i]: solution[forces[i].getName()][:, 0:1] for i in range(nc)}
        repl.publish_joints(solution['q'][:, 0])
        repl.publishContactForces(rospy.Time.now(), solution['q'][:, 0], 0)
    #

    # set ROS stuff and launchfile
    plot = True
    #
    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        for contact in contacts:
            FK = cs.Function.deserialize(kd.fk(contact))
            pos = FK(q=solution['q'])['ee_pos']

            plt.title(f'feet position - plane_xy')
            plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[1, :]).flatten(), linewidth=2.5)
            plt.scatter(np.array(pos[0, 0]), np.array(pos[1, 0]))
            plt.scatter(np.array(pos[0, -1]), np.array(pos[1, -1]), marker='x')

        plt.figure()
        for contact in contacts:
            FK = cs.Function.deserialize(kd.fk(contact))
            pos = FK(q=solution['q'])['ee_pos']

            plt.title(f'feet position - plane_xz')
            plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[2, :]).flatten(), linewidth=2.5)
            plt.scatter(np.array(pos[0, 0]), np.array(pos[2, 0]))
            plt.scatter(np.array(pos[0, -1]), np.array(pos[2, -1]), marker='x')

        hplt = plotter.PlotterHorizon(prb, solution)
        hplt.plotVariables([elem.getName() for elem in forces], show_bounds=True, gather=2, legend=False)
        hplt.plotVariables(['q'], show_bounds=True, gather=2, legend=False)
        matplotlib.pyplot.show()
