import numpy as np
from networkx.algorithms.bipartite.basic import color

from horizon.rhc.taskInterface import TaskInterface
from phase_manager import pyphase, pymanager, pytimeline
import colorama
from horizon.utils import trajectoryGenerator
from horizon.utils import logger
from functools import partial

class PhaseGaitWrapper:
    def __init__(self, task_interface: TaskInterface, phase_manager:pymanager.PhaseManager, contact_list):

        self.__logger = logger.Logger(self)

        self.__trajectory_generator = trajectoryGenerator.TrajectoryGenerator()

        self.__contact_list = contact_list
        self.__task_interface = task_interface
        self.__model = self.__task_interface.model

        self.__phase_manager = phase_manager

        # todo: this is now hardcoded
        self.__contact_task_dict = {'l_sole': 'foot_contact_l',
                                    'r_sole': 'foot_contact_r'}

        self.__z_task_dict = {'l_sole': 'foot_z_l',
                              'r_sole': 'foot_z_r'}

        # MAP -> contact name : timeline
        self.__contact_timelines = dict()
        self.__stance_phases = dict()
        self.__flight_phases = dict()

        self.__last_added_phases = dict()

        self.__contact_z_position_initial = dict()
        self.__contact_z_position_final = dict()
        self.__contact_z_height = dict()


        self.__init_actions()
        self.__init_swing_trajectory(contact_list)
        self.__init_timelines(contact_list)

    def __init_actions(self):

        self.__action_list = {
            'walk':  partial(self.__walk_cycle),
            'stand': partial(self.__add_cycles, [[1, 1]], duration=1)
        }
    def __init_swing_trajectory(self, contact_list):

        default_height = 0.05
        for contact in contact_list:

            contact_initial_pose = self.__model.kd.fk(contact)(q=self.__model.q0)['ee_pos'].elements()

            self.__contact_z_position_initial[contact] = contact_initial_pose[2]
            self.__contact_z_position_final[contact] = contact_initial_pose[2]
            self.__contact_z_height[contact] = default_height

    def getContacts(self):

        return self.__contact_list

    def getTaskInterface(self):

        return self.__task_interface

    def __init_timelines(self, contact_list):

        experimental_duration = 1
        for contact in contact_list:
            self.__logger.log(f'creating timeline for contact: {contact}')

            self.__contact_timelines[contact] = self.__phase_manager.createTimeline(f'{contact}_timeline')

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
                self.setSwingTrajectory(contact_timeline.getPhases()[-kwargs['duration']:], contact_name, kwargs['height'])
            else:
                self.__add_phase(contact_timeline, self.__stance_phases[contact_name], duration=kwargs['duration'])

            self.__last_added_phases[contact_name] = contact_timeline.getPhases()[-kwargs['duration']:]
            # self.__logger.log(f'adding {[phase.getName() for phase in self.__last_added_phases[contact_name]]} to timeline: {contact_timeline.getName()}')

        return self.__last_added_phases

    def __add_cycles(self, cycle_lists, **kwargs):

        for cycle_i in cycle_lists:
            self.__add_cycle(cycle_i, **kwargs)

        self.__phase_manager.update()

    def action(self, action_name, *args, **kwargs):

        self.__logger.log(f'action called: {action_name}')
        # self.__logger.log(f'args: {args}')
        # self.__logger.log(f'kwargs: {kwargs}')

        self.__action_list[action_name](**kwargs)


    def initializeTimeline(self):

        for contact_name, contact_timeline in self.__contact_timelines.items():
            self.__logger.log(f"initializing timeline {contact_name}:")
            phase_i = 0
            while contact_timeline.getEmptyNodes() > 0:
                contact_timeline.addPhase(self.__stance_phases[contact_name])
                phase_i += 1

            self.__logger.log(f" --> added {phase_i} '{self.__stance_phases[contact_name].getName()}' phases.")

        self.__phase_manager.update()

    def setSwingTrajectory(self, phases, contact_name, z_height):

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
                                                                 [None, 0, None]
                                                                 )

        for phase_i in range(len(phases)):
            ref_trj_z[2, :] = temp_traj[phase_i]
            # self.__logger.log(f'setting reference to phase {phases[phase_i].getName()} ({contact_name}):')
            # self.__logger.log(f'{ref_trj_z.T}')
            phases[phase_i].setItemReference(self.__z_task_dict[contact_name], ref_trj_z)

    def __walk_cycle(self, **kwargs):

        step_duration = kwargs['step_duration']
        step_height = kwargs['step_height']
        double_stance = kwargs['double_stance']

        self.__add_cycle([1, 0], duration=step_duration, height=step_height)
        self.__add_cycle([1, 1], duration=double_stance)
        self.__add_cycle([0, 1], duration=step_duration, height=step_height)
        self.__add_cycle([1, 1], duration=double_stance)

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

            # hardcoded
            contact_task_dict = {'l_sole': 'foot_contact_l',
                                 'r_sole': 'foot_contact_r'}

            self.__stance_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'stance_{contact_task_dict[contact_name]}')
            self.__flight_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'flight_{contact_task_dict[contact_name]}')

            self.__stance_short_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'short_stance_{contact_task_dict[contact_name]}')
            self.__flight_short_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'short_flight_{contact_task_dict[contact_name]}')

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



