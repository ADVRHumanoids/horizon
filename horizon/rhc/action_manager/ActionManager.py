import copy

import numpy as np
import phase_manager.pymanager as pm
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

class Action:
    """
    an action is a desired (ordered) sequence of phases for a contact/set of contacts
    """

    def __init__(self, name: str, phases: list):
        self.name = name
        self.phases = phases

    def addPhase(self, phase):
        self.phases

class Step(Action):
    """
    simple class representing a step, contains the main info about the step
    """

    def __init__(self):
        pass

class ActionManager():
    """
    set of actions which involves combinations of constraints and bounds
    action manager should work on top of the phase manager - task interface

    given the correct tasks it should:
        - set a default action (standing still or walking?)
        - add actions (walk, jump, trot..)

    """

    def __init__(self, task_interface: TaskInterface, phase_manager: pm.PhaseManager, opts=None):
        # prb: Problem, urdf, kindyn, contacts_map, default_foot_z

        self.ti = task_interface

        self.prb = self.ti.prb

        self.opts = opts

        self.contact_map = self.ti.model.cmap
        self.contacts = self.contact_map.keys()
        self.nc = len(self.contacts)

        self.phase_manager_map = dict()  #: [str, pm.SinglePhaseManager]
        for c in self.contacts:
            self.phase_manager_map[c] = phase_manager.getTimeline(c)

        self.ns = self.prb.getNNodes()
        # todo list of contact is fixed?

        # self.constraints = list()
        # self.current_cycle = 0  # what to do here?
        #

        # initialize default action dict with action for each contact (init --> None)
        self.default_action = {key: None for key in self.contacts}

        self.kd = self.ti.model.kd

        # searches in the phase manager the required tasks
        # TODO: action manager requires phases from phase manager. It searches them by NAME.
        # for now this is ugly, take a phase that is called stance + name of the contact
        self.stance_phases = dict()
        self.swing_phases = dict()
        for c in self.contacts:
            self.stance_phases[c] = self.phase_manager_map[c].getRegisteredPhase(f'stance_{c}')
            self.swing_phases[c] = self.phase_manager_map[c].getRegisteredPhase(f'flight_{c}')

        # todo: remove from here
        stance_action = Action('standing_still')
        for c in self.contacts:
            stance_action.addPhase(c) = self.stance_phase[c]

        self.setDefaultAction(stance_action)
        self._add_default_action()

        self.action_list = []

    def setDefaultAction(self, action):
        # todo for now the default is robot still, in contact

        for c in self.contacts:
            if action.getPhases(c) is None:
                raise Exception(f"Contact {c} not set. ActionManager requires a default action for every contact ({self.contacts}).")

            self.default_action[c] = action.getPhases(c)


    def _add_default_action(self):


        for c in self.contacts:
            self.phase_manager_map[c].addPhase(self.default_action[c])











            # def _check_required_tasks_type(self, required_tasks):
            #     # actionManager requires some tasks for working. It asks the TaskInterface for tasks.
            #     task_type = dict()
            #     for task in required_tasks:
            #         found_task = self.ti.getTasksType(task)
            #         if found_task is None:
            #             raise Exception(
            #                 'Task {} not found. ActionManager requires this task, please provide your implementation.'.format(
            #                     task))
            #         else:
            #             task_type[task] = found_task
            #
            #     return task_type

    def setRequiredTasks(self, task_dict):
        # TODO: add some logic
        self.required_tasks = task_dict

    def setContact(self, frame, nodes):
        """
        establish/break contact
        """
        # todo reset all the other "contact" constraints on these nodes
        # self._reset_task_constraints(frame, nodes_in_horizon_x)

        # todo what to do with z_constr?
        # self.z_constr.reset()

        self.contact_constr[frame].setNodes(nodes)

    # def _append_nodes(self, node_list, new_nodes):

    # def _append_params(self, params_array, new_params, nodes):

    # def setStep(self, step):
        # self.action_list.append(step)
        # self._step(step)

    def _step(self, step: Step):
        """
        add step to horizon stack
        """
        s = copy.deepcopy(step)
        # todo temporary

        # todo how to define current cycle
        # frame = s.frame
        # k_start = s.k_start
        # k_goal = s.k_goal
        # all_contact_nodes = self.contact_constr_nodes[frame]
        # swing_nodes = list(range(k_start, k_goal))
        # stance_nodes = [k for k in all_contact_nodes if k not in swing_nodes]
        # n_swing = len(swing_nodes)
        #
        # swing_nodes_in_horizon = [k for k in swing_nodes if k >= 0 and k <= self.N]
        # stance_nodes_in_horizon = [k for k in stance_nodes if k >= 0 and k <= self.N]
        # n_swing_in_horizon = len(swing_nodes_in_horizon)

        # this step is outside the horizon!
        # todo what to do with default action?

        # if n_swing_in_horizon == 0:
        #     print(f'========= skipping step {s.frame}. Not in horizon: {swing_nodes_in_horizon} ==========')
        #     return 0
        # print(f'========= activating step {s.frame}: {swing_nodes_in_horizon} ==========')
        # adding nodes to the current ones (if any)
        # self.contact_constr_nodes[frame] = stance_nodes
        # self.foot_tgt_constr_nodes[frame] = self._append_nodes(self.foot_tgt_constr_nodes[frame], [k_goal])
        # self.z_constr_nodes[frame] = self._append_nodes(self.z_constr_nodes[frame], swing_nodes_in_horizon)
        #
        # break contact at swing nodes + z_trajectory + (optional) xy goal
        # contact
        # self.setContact(frame, self.contact_constr_nodes[frame])

        # xy goal
        # TODO: refactor this
        # if self.N >= k_goal > 0 and step.goal.size > 0:
            # adding param:
            # self._foot_tgt_params[frame][:, swing_nodes_in_horizon] = s.goal[:2]
            #
            # self.foot_tgt_constr[frame].setNodes(self.foot_tgt_constr_nodes[frame])  # [k_goal]
            # self.foot_tgt_constr[frame].setRef(self._foot_tgt_params[frame][self.foot_tgt_constr_nodes[frame]])  # s.goal[:2]

        # if step.goal.size > 0:
        #     s.start = np.array([0, 0, s.goal[2]])
        #
        # # z goal
        # start = np.array([0, 0, self.default_foot_z[frame]]) if s.start.size == 0 else s.start
        # goal = np.array([0, 0, self.default_foot_z[frame]]) if s.goal.size == 0 else s.goal
        #
        # z_traj = self.compute_polynomial_trajectory(k_start, swing_nodes_in_horizon, n_swing, start, goal, s.clearance,
        #                                             dim=2)
        #
        # cart_mask_z = np.zeros([7, len(swing_nodes_in_horizon)])
        # cart_mask_z[2, :] = z_traj[:len(swing_nodes_in_horizon)]
        #
        # # adding param
        # self._foot_z_param[frame][:, swing_nodes_in_horizon] = cart_mask_z
        # self.z_constr[frame].setNodes(self.z_constr_nodes[frame])  # swing_nodes_in_horizon
        # self.z_constr[frame].setRef(self._foot_z_param[frame][:, self.z_constr_nodes[frame]])  # z_traj

    # todo unify the actions below, these are just different pattern of actions
    def _jump(self, nodes):
        pass

        # # todo add parameters for step
        # for contact in self.contacts:
        #     k_start = nodes[0]
        #     k_end = nodes[-1]
        #     s = Step(contact, k_start, k_end)
        #     self.setStep(s)

    def _walk(self, nodes, step_pattern=None, step_nodes_duration=None):

        pass
        # # todo add parameters for step
        # step_list = list()
        # k_step_n = 5 if step_nodes_duration is None else step_nodes_duration  # default duration (in nodes) of swing step
        # k_start = nodes[0]  # first node to begin walk
        # k_end = nodes[1]
        #
        # n_step = (k_end - k_start) // k_step_n  # integer divide
        # # default step pattern of classic walking (crawling)
        # pattern = step_pattern if step_pattern is not None else list(range(len(self.contacts)))
        # # =========================================
        # for n in range(n_step):
        #     l = list(self.contacts)[pattern[n % len(pattern)]]
        #     k_end_rounded = k_start + k_step_n
        #     s = Step(l, k_start, k_end_rounded)
        #     print(l, k_start, k_end_rounded)
        #     k_start = k_end_rounded
        #     step_list.append(s)
        #
        # for s_i in step_list:
        #     self.setStep(s_i)

    def _trot(self, nodes):
        pass
        # # todo add parameters for step
        # k_start = nodes[0]
        # k_step_n = 5  # default swing duration
        # k_end = nodes[1]
        #
        # n_step = (k_end - k_start) // k_step_n  # integer divide
        # step_list = []
        # for n in range(n_step):
        #     if n % 2 == 0:
        #         l1 = 'lf_foot'
        #         l2 = 'rh_foot'
        #     else:
        #         l1 = 'lh_foot'
        #         l2 = 'rf_foot'
        #     k_end = k_start + k_step_n
        #     s1 = Step(l1, k_start, k_end, clearance=0.03)
        #     s2 = Step(l2, k_start, k_end, clearance=0.03)
        #     k_start = k_end
        #     step_list.append(s1)
        #     step_list.append(s2)
        #
        # for s_i in step_list:
        #     self.setStep(s_i)

    def execute(self, bootstrap_solution):
        """
        set the actions and spin
        """
        self._update_initial_state(bootstrap_solution, -1)

        self._set_default_action()

        k0 = 1

        # for action in self.action_list:
            # action.k_start = action.k_start - k0
            # action.k_goal = action.k_goal - k0
            # action_nodes = list(range(action.k_start, action.k_goal))
            # action_nodes_in_horizon = [k for k in action_nodes if k >= 0]
            # self._step(action)

        # for cnsrt_name, cnsrt in self.prb.getConstraints().items():
        #     print(cnsrt_name)
        #     print(cnsrt.getNodes().tolist())
        # remove expired actions
        # self.action_list = [action for action in self.action_list if
        #                     len([k for k in list(range(action.k_start, action.k_goal)) if k >= 0]) != 0]
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
    pass