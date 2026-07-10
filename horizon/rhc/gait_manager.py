import numpy as np
from horizon.rhc.taskInterface import TaskInterface
from phase_manager import pyphase, pymanager, pytimeline
import colorama
from horizon.utils import trajectoryGenerator
from horizon.utils import logger
from functools import partial
from dataclasses import dataclass, field
from typing import Callable, Dict
from scipy.spatial.transform import Rotation

import time

# how to operate:
# ~/forest_ws/src/unitree_mujoco/simulate/build  ./unitree_mujoco
# mon launch cogimon_controller g1_experimental.launch  xbot:=true joy:=true

@dataclass
class ActionPlugin:
    def __init__(self, task_interface: TaskInterface):
        """Initialize with task interface and name."""

        self.__logger = logger.Logger(self)
        self.__task_interface = task_interface
        self.__action_dict = dict()
        self.__action_status = dict()

    def register_action(self, action_name: str, action_func: Callable):
        """Register an action dynamically."""
        self.__action_dict[action_name] = action_func
        self.__action_status[action_name] = 'Stopped'

    def get_actions(self):
        """Return all registered actions."""
        return self.__action_dict

    def getTaskInterface(self):

        return self.__task_interface

    def getLogger(self):

        return self.__logger

    def getStatus(self):

        return self.__action_status

    def setStatus(self, action_name, status):

        if status == 'Started' or status == 'Stopped' or status == 'Running':
            self.__action_status[action_name] = status

            return True
        return False



class SwingTrajectory:
    def __init__(self, task_interface: TaskInterface, task_list):

        self.__logger = logger.Logger(self)

        self.__trajectory_generator = trajectoryGenerator.TrajectoryGenerator()

        self.__z_task_list = task_list.copy()

        self.__flight_nodes_list = None

        self.__task_interface = task_interface
        self.__model = self.__task_interface.model

        self.__default_height = 0.05

        self.__contact_z_position_initial = dict()
        self.__contact_z_position_final = dict()
        self.__contact_z_height = dict()

        self.__contact_list = self.__init_contacts()
        self.__init_swing_trajectory()

    def __init_contacts(self):

        # get z_tasks from taskInterface
        self.__z_task_dict = {}
        self.__fk_dict = {}
        for z_task_name in self.__z_task_list:

            z_task = self.__task_interface.getTask(z_task_name)

            if self.__task_interface.getTask(z_task_name) is None:
                raise Exception(f'Task name "{z_task_name}" not found in horizon stack.')

            self.__logger.log(f'Found task "{z_task_name}" in horizon task')
            self.__z_task_dict[z_task.getDistalLink()] = z_task_name
            self.__fk_dict[z_task.getDistalLink()] = self.__model.kd.fk(z_task.getDistalLink())

            if z_task.getDistalLink() in self.__model.getContacts():
                self.__logger.log(f'Task {z_task_name} linked to contact: {z_task.getDistalLink()}')
            else:
                raise Exception(f'Task {z_task_name} is not linked to any defined contact ({self.__model.getContacts()})')

    def __init_swing_trajectory(self):

        for contact_link in self.__z_task_dict.keys():
            contact_initial_pose = self.__model.kd.fk(contact_link)(q=self.__model.q0)['ee_pos'].elements()

            self.__contact_z_position_initial[contact_link] = contact_initial_pose[2]
            self.__contact_z_position_final[contact_link] = contact_initial_pose[2]
            self.__contact_z_height[contact_link] = self.__default_height

    def __update_swing_trajectory(self, solution, contact=None):
        z_task_dict = dict()
        if contact is not None:
            z_task_dict = {contact: self.__z_task_dict[contact]}
        else:
            z_task_dict = self.__z_task_dict

        for contact_link in z_task_dict.keys():
            contact_initial_pose = self.__fk_dict[contact_link](q=solution['q'][:, 0])['ee_pos'].elements()

            self.__contact_z_position_initial[contact_link] = contact_initial_pose[2]
            self.__contact_z_position_final[contact_link] = contact_initial_pose[2]
            self.__contact_z_height[contact_link] = self.__default_height

    def updateReferenceTrajectory(self, solution, phases, contact, z_height):
        self.__update_swing_trajectory(solution, contact)
        self.setSwingTrajectoryToPhases(phases, contact, z_height)

    def setSwingTrajectoryToPhases(self, phases, contact_name, z_height):

        flight_duration = len(phases)
        ref_trj_z = np.zeros(shape=[7, 1])

        # self.__logger.log(f'{[phase.getName() for phase in phases]}')
        # self.__logger.log(f'setting swing trajectory of contact {contact_name}:')
        # self.__logger.log(f' --> step_duration: {flight_duration}')
        # self.__logger.log(f' --> step_height: {z_height}')

        temp_traj = self.__trajectory_generator.from_derivatives(flight_duration,
                                                                 self.__contact_z_position_initial[contact_name],
                                                                 self.__contact_z_position_final[contact_name],
                                                                 z_height,
                                                                 [None, 0, 0]
                                                                 )
        for phase_i in range(len(phases)):
            ref_trj_z[2, :] = temp_traj[phase_i]
            # self.__logger.log(f'setting reference to phase {phases[phase_i].getName()} ({contact_name}):')
            # self.__logger.log(f'{ref_trj_z.T}')
            phases[phase_i].setItemReference(self.__z_task_dict[contact_name], ref_trj_z)


class PhaseGaitWrapper:
    def __init__(self, task_interface: TaskInterface, phase_manager:pymanager.PhaseManager, contact_list, swing_task_list=None, plugin_actions=None):

        self.__logger = logger.Logger(self)

        self.__contact_list = contact_list
        self.__task_interface = task_interface
        self.__model = self.__task_interface.model

        self.__phase_manager = phase_manager

        self.__swing_flag = False
        if swing_task_list is not None:
            self.__swing_flag = True
            self.__swing_trajectory_manager = SwingTrajectory(self.__task_interface, swing_task_list)

        # MAP -> contact name : timeline
        self.__contact_timelines = dict()
        self.__stance_phases = dict()
        self.__flight_phases = dict()

        self.__last_added_phases = dict()

        self.__init_actions()
        self.__init_timelines(contact_list)

        self.__plugin_dict = dict()

        if plugin_actions:
            self.__init_plugin_actions(plugin_actions)

    def __init_plugin_actions(self, plugin_actions):

        for plugin_action_name, action_plugin in plugin_actions.items():
            self.__plugin_dict.update({plugin_action_name: action_plugin(task_interface=self.__task_interface)})

    def __init_actions(self):

        self.__action_list = {
            'walk':  partial(self.__bipedal_walk_cycle),
            'crawl': partial(self.__crawl),
            'trot': partial(self.__trot),
            'stand': partial(self.__add_cycles, [[1] * len(self.__contact_list)], duration=1)
        }

    def getActionList(self):

        return list(self.__action_list.keys())

    def getPluginDict(self):

        return self.__plugin_dict

    def getContacts(self):

        return self.__contact_list

    def getTaskInterface(self):

        return self.__task_interface

    def getSwingTrajectoryManager(self):
        
        return self.__swing_trajectory_manager

    def __init_timelines(self, contact_list):

        experimental_duration = 1
        for contact in contact_list:

            self.__contact_timelines[contact] = self.__phase_manager.createTimeline(f'{contact}_timeline')

            if self.__contact_timelines[contact] is None:
                raise Exception(f'Failed to create timeline for contact {contact}')

            self.__logger.log(f'created timeline for contact: "{contact}"')

            self.__stance_phases[contact] = self.__contact_timelines[contact].createPhase(experimental_duration, f'stance_phase_{contact}')
            self.__flight_phases[contact] = self.__contact_timelines[contact].createPhase(experimental_duration, f'flight_phase_{contact}')
            
    def getContactTimelines(self):
        return self.__contact_timelines

    def getStancePhases(self):
        return self.__stance_phases

    def getFlightPhases(self):
        return self.__flight_phases

    def __add_phase(self, timeline: pytimeline, phase: pyphase, duration: int):

        for i in range(duration):
            timeline.addPhase(phase)

    def __add_cycle(self, cycle_list, *args, **kwargs):

        for contact_flag, (contact_name, contact_timeline) in zip(cycle_list, self.__contact_timelines.items()):
            if contact_flag == 0:
                self.__add_phase(contact_timeline, self.__flight_phases[contact_name], duration=kwargs['duration'])
                if self.__swing_flag:
                    self.__swing_trajectory_manager.updateReferenceTrajectory(kwargs['solution'], contact_timeline.getPhases()[-kwargs['duration']:], contact_name, kwargs['height'])
                    # self.__swing_trajectory_manager.setSwingTrajectoryToPhases(contact_timeline.getPhases()[-kwargs['duration']:], contact_name, kwargs['height'])
                    
            else:
                self.__add_phase(contact_timeline, self.__stance_phases[contact_name], duration=kwargs['duration'])

            # self.__last_added_phases[contact_name] = contact_timeline.getPhases()[-kwargs['duration']:]
            # self.__logger.log(f'adding {[phase.getName() for phase in self.__last_added_phases[contact_name]]} to timeline: {contact_timeline.getName()}')

        # return self.__last_added_phases

    def __add_cycles(self, cycle_lists, **kwargs):

        for cycle_i in cycle_lists:
            self.__add_cycle(cycle_i, **kwargs)

        # todo do this here, or in the main loop?
        # self.__phase_manager.update()

    def updateReferenceTrajectory(self, solution, phases, contact, z_height):
        self.__swing_trajectory_manager.updateReferenceTrajectory(solution, phases, contact, z_height)

    def action(self, action_name, *args, **kwargs):

        # self.__logger.log(f'action called: {action_name}')
        # self.__logger.log(f'args: {args}')
        # self.__logger.log(f'kwargs: {kwargs}')

        self.__action_list[action_name](**kwargs)


    def call_plugin(self, plugin_name, action_name, *args, **kwargs):
        """Call a method on a registered action dynamically."""
        plugin = self.__plugin_dict.get(plugin_name)

        if not plugin:
            raise ValueError(f"Plugin '{plugin_name}' not found.")

        action = getattr(plugin, action_name, None)
        if not action or not callable(action):
            raise ValueError(f"Action '{action_name}' not found in '{plugin_name}'.")

        return action(*args, **kwargs)

    def initializeTimeline(self):

        for contact_name, contact_timeline in self.__contact_timelines.items():
            self.__logger.log(f"initializing timeline {contact_name}:")
            phase_i = 0
            while contact_timeline.getEmptyNodes() > 0:
                contact_timeline.addPhase(self.__stance_phases[contact_name])
                phase_i += 1

            self.__logger.log(f" --> added {phase_i} '{self.__stance_phases[contact_name].getName()}' phases.")

        self.__phase_manager.update()

    def __bipedal_walk_cycle(self, **kwargs):

        step_duration = kwargs['step_duration']
        step_height = kwargs['step_height']
        double_stance = kwargs['double_stance']

        self.__add_cycle([1, 0], duration=step_duration, height=step_height)
        self.__add_cycle([1, 1], duration=double_stance)
        self.__add_cycle([0, 1], duration=step_duration, height=step_height)
        self.__add_cycle([1, 1], duration=double_stance)

    def __crawl(self, **kwargs):
        step_duration = kwargs['step_duration']
        step_height = kwargs['step_height']
        double_stance = kwargs['double_stance']
        solution = kwargs['solution']

        q = solution['q'][:, 0]
        w_R_b = Rotation.from_quat([q[3], q[4], q[5], q[6]]).as_matrix()

        vref = self.__task_interface.getTask('final_base_xy').ref.getValues()[:, 0]

        vref= w_R_b[:2, :2].T @ vref.T
        vx, vy = vref[0], vref[1]
        omega = self.__task_interface.getTask('base_yaw_orientation').ref.getValues()[0, 0] 

        if vx**2 + vy**2 <= omega**2:

            # turning gait
            if omega > 0:
                self.__add_cycle([0, 1, 1, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 1, 0, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 1, 1, 0], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 0, 1, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
            else:
                self.__add_cycle([0, 1, 1, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 0, 1, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 1, 1, 0], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 1, 0, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)   
        else:
            # forward
            if vx > abs(vy): 
                self.__add_cycle([1, 1, 0, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([0, 1, 1, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 1, 1, 0], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 0, 1, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)   

            elif vx < -abs(vy):
                # backward
                self.__add_cycle([0, 1, 1, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 1, 0, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 0, 1, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 1, 1, 0], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)   

            elif vy > abs(vx):
                # left
                self.__add_cycle([1, 0, 1, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([0, 1, 1, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 1, 1, 0], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 1, 0, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)   
                
            else:
                # right
                self.__add_cycle([0, 1, 1, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 0, 1, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 1, 0, 1], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)
                self.__add_cycle([1, 1, 1, 0], duration=step_duration, height=step_height, solution=solution)
                self.__add_cycle([1, 1, 1, 1], duration=double_stance)


    def __trot(self, **kwargs):

        step_duration = kwargs['step_duration']
        step_height = kwargs['step_height']
        double_stance = kwargs['double_stance']
        solution = kwargs['solution']

        self.__add_cycle([0, 1, 1, 0], duration=step_duration, height=step_height, solution=solution)
        self.__add_cycle([1, 1, 1, 1], duration=double_stance)
        self.__add_cycle([1, 0, 0, 1], duration=step_duration, height=step_height, solution=solution)
        self.__add_cycle([1, 1, 1, 1], duration=double_stance)


    def save(self):
        """
        Return a dict describing the elements registered in each stance/flight phase,
        including actual stored values for item_references, item_weights, parameters,
        and variable_bounds — so C++ can reconstruct the phase configuration exactly.
        """
        import numpy as np

        def _ref_entry(item):
            """name + current values as nested list for YAML serialisation."""
            vals = item.getValues()
            return {'name': item.getName(), 'values': vals.tolist()}

        def _weight_entry(item):
            w = item.getWeight()
            return {'name': item.getName(), 'weight': w.tolist()}

        def _bounds_entry(item):
            lb, ub = item.getBounds()
            return {'name': item.getName(), 'lower': lb.tolist(), 'upper': ub.tolist()}

        def _phase_dict(phase):
            return {
                'items':        [i.getName() for i in phase.getItems()],
                'costs':        [i.getName() for i in phase.getCosts()],
                'constraints':  [i.getName() for i in phase.getConstraints()],
                'item_references': [_ref_entry(i) for i in phase.getItemReferences()],
                'item_weights':    [_weight_entry(i) for i in phase.getItemWeights()],
                'parameters':      [_ref_entry(i) for i in phase.getParameters()],
                'variables':       [_bounds_entry(i) for i in phase.getVariables()],
            }

        saved = {}
        for contact_name in self.__contact_list:
            saved[contact_name] = {
                'stance': _phase_dict(self.__stance_phases[contact_name]),
                'flight': _phase_dict(self.__flight_phases[contact_name]),
            }

        task_interface_data = self.__task_interface.save()

        return {
            **task_interface_data,
            'gait_manager': saved,
        }

class GaitManager:
    def __init__(self, task_interface: TaskInterface, phase_manager: pymanager.PhaseManager, contact_map):

        # TODO: preserve the order given by the contact_map
        # contact_map is not necessary if contact name is the same as the timeline name
        self.__task_interface = task_interface
        self.__phase_manager = phase_manager

        self.__contact_timelines = dict()

        # contact map links 'contact_name' with 'contact_timeline'

        # register each timeline of the phase manager as the contact phases
        for contact_name, timeline_name in contact_map.items():
            self.__contact_timelines[contact_name] = self.__phase_manager.getTimelines()[timeline_name]

        # self.zmp_timeline = self.phase_manager.getTimelines()['zmp_timeline']

        self.__flight_phases = dict()
        self.__stance_phases = dict()
        self.__stance_phases_crawl = dict()
        self.__crawl_phases = dict()

        self.__flight_short_phases = dict()
        self.__stance_short_phases = dict()

        self.__flight_recovery_phases = dict()
        self.__stance_recovery_phases = dict()

        self.__init_tasks(contact_map)

    def __init_tasks(self, contact_map):

        # retrieve manually (for now) the correct tasks if present
        for contact_name, timeline_name in contact_map.items():
            self.__flight_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'flight_{contact_name}')
            self.__crawl_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'crawl_{contact_name}')
            self.__stance_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'stance_{contact_name}')
            self.__stance_phases_crawl[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'stance_crawl_{contact_name}')

            # different duration (todo: flexible implementation?)
            self.__flight_short_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'flight_{contact_name}_short')
            self.__stance_short_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'stance_{contact_name}_short')

            self.__flight_recovery_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'flight_{contact_name}_recovery')
            self.__stance_recovery_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'stance_{contact_name}_recovery')

            # # hardcoded
            # contact_task_dict = {'l_sole': 'foot_contact_l',
            #                      'r_sole': 'foot_contact_r'}
            #
            # self.__stance_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'stance_{contact_task_dict[contact_name]}')
            # self.__flight_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'flight_{contact_task_dict[contact_name]}')
            #
            # self.__stance_short_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'short_stance_{contact_task_dict[contact_name]}')
            # self.__flight_short_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'short_flight_{contact_task_dict[contact_name]}')

    def getContactTimelines(self):

        return self.__contact_timelines

    def getTaskInterface(self):

        return self.__task_interface

    def cycle_short(self, cycle_list):

        for flag_contact, contact_name in zip(cycle_list, self.__contact_timelines.keys()):
            timeline_i = self.__contact_timelines[contact_name]

            if flag_contact == 1:
                timeline_i.addPhase(self.__stance_short_phases[contact_name])
            else:
                timeline_i.addPhase(self.__flight_short_phases[contact_name])

    def cycle(self, cycle_list):

        for flag_contact, contact_name in zip(cycle_list, self.__contact_timelines.keys()):
            timeline_i = self.__contact_timelines[contact_name]

            if flag_contact == 1:
                timeline_i.addPhase(self.__stance_phases[contact_name])
                timeline_i.addPhase(self.__stance_short_phases[contact_name])
                print(f'adding {self.__stance_phases[contact_name]} to phase: {contact_name}')
                print(f'adding {self.__stance_short_phases[contact_name]} to phase: {contact_name}')
            else:
                timeline_i.addPhase(self.__flight_phases[contact_name])
                timeline_i.addPhase(self.__stance_short_phases[contact_name])
                print(f'adding {self.__stance_phases[contact_name]} to phase: {contact_name}')
                print(f'adding {self.__stance_short_phases[contact_name]} to phase: {contact_name}')


    def cycle_recovery(self, cycle_list):

        for flag_contact, contact_name in zip(cycle_list, self.__contact_timelines.keys()):
            timeline_i = self.__contact_timelines[contact_name]

            if flag_contact == 1:
                # timeline_i.addPhase(self.__stance_recovery_phases[contact_name])
                timeline_i.addPhase(self.__stance_short_phases[contact_name])
                timeline_i.addPhase(self.__stance_short_phases[contact_name])
                timeline_i.addPhase(self.__stance_short_phases[contact_name])
                timeline_i.addPhase(self.__stance_short_phases[contact_name])
            else:
                timeline_i.addPhase(self.__flight_recovery_phases[contact_name])

    def step(self, swing_contact):

        cycle_list = [[True if contact_name != swing_contact else False for contact_name in self.__contact_timelines.keys()]]
        self.__add_cycles(cycle_list)

    def diagonal_pair(self, val=0):

        cycle_lists = [[0, 1, 1, 0]] # fr-rl

        if val == 1:
            cycle_lists = [[1, 0, 0, 1]] # fl-rr

        self.__add_cycles(cycle_lists)

    def diagonal_pair_recovery(self, val=0):

        cycle_lists = [[0, 1, 1, 0]] # fr-rl

        if val == 1:
            cycle_lists = [[1, 0, 0, 1]] # fl-rr

        self.__add_cycles_recovery(cycle_lists)

    def trot(self):

        self.diagonal_pair(0)
        self.diagonal_pair(1)
        # self.zmp_timeline.addPhase(self.zmp_timeline.getRegisteredPhase('zmp_empty_phase'))
        # self.zmp_timeline.addPhase(self.zmp_timeline.getRegisteredPhase('zmp_empty_phase'))

    def crawl(self, vref=[0, 0, 1]):

        vx, vy, omega = vref
        Rmax = 1
        
        if vx**2 + vy**2 <= Rmax*omega**2:

            # turning gait
            if omega > 0:

                cycle_lists = [[1, 1, 0, 1],
                               [1, 1, 1, 0],
                               [1, 0, 1, 1],
                               [0, 1, 1, 1]]
            else:

                cycle_lists = [[1, 1, 0, 1],
                               [0, 1, 1, 1],
                               [1, 0, 1, 1],
                               [1, 1, 1, 0]]

        else:

            if vx > abs(vy): 
                
                # forward
                cycle_lists = [[1, 1, 0, 1],  # rl
                               [0, 1, 1, 1],  # fl
                               [1, 1, 1, 0],  # rr
                               [1, 0, 1, 1]]  # fr

            elif vx < -abs(vy):

                # backward
                cycle_lists = [[0, 1, 1, 1],  # fl
                               [1, 1, 0, 1],  # rl
                               [1, 0, 1, 1],
                               [1, 1, 1, 0]]

            elif vy > abs(vx):

                # left
                cycle_lists = [[1, 0, 1, 1],
                               [0, 1, 1, 1],
                               [1, 1, 1, 0],
                               [1, 1, 0, 1]]
                
            else:

                # right
                cycle_lists = [[0, 1, 1, 1],
                               [1, 0, 1, 1],
                               [1, 1, 0, 1],
                               [1, 1, 1, 0]]

        self.__add_cycles(cycle_lists)

    def leap(self):

        cycle_lists = [[0, 0, 1, 1], [1, 1, 0, 0]]
        self.__add_cycles(cycle_lists)

    def walk(self):

        cycle_lists = [[1, 0, 1, 0], [0, 1, 0, 1]]
        self.__add_cycles(cycle_lists)

    def jump(self):

        cycle_list = [[0, 0, 0, 0]]
        self.__add_cycles(cycle_list)

    def wheelie(self):

        cycle_list = [[0, 0, 1, 1]]
        self.__add_cycles(cycle_list)

    def give_paw(self):

        cycle_list = [[0, 1, 1, 1]]
        self.__add_cycles(cycle_list)

    def stand(self):

        cycle_list = [[1, 1, 1, 1]]
        self.__add_cycles(cycle_list)

    def drag(self):

        cycle_list = [[1, 0, 1, 1], [0, 1, 1, 1]]

        self.__add_cycles(cycle_list)

    def walk2(self):

        cycle_list = [[0, 1],
                      [1, 0]]

        self.__add_cycles(cycle_list)

    def __add_cycles(self, cycle_lists):

        for cycle_i in cycle_lists:
            self.cycle(cycle_i)


    def __add_cycles_recovery(self, cycle_lists):

        for cycle_i in cycle_lists:
            self.cycle_recovery(cycle_i)



