import numpy as np
from horizon.rhc.taskInterface import TaskInterface
from phase_manager import pyphase, pymanager
import colorama

class GaitManager:
    def __init__(self, task_interface: TaskInterface, phase_manager: pymanager.PhaseManager, contact_map):

        # TODO: preserve the order given by the contact_map
        # contact_map is not necessary if contact name is the same as the timeline name
        self.__task_interface = task_interface
        self.__phase_manager = phase_manager

        self.__contact_timelines = dict()

        # register each timeline of the phase manager as the contact phases
        for contact_name, timeline_name in contact_map.items():
            self.__contact_timelines[contact_name] = self.__phase_manager.getTimelines()[timeline_name]

        # self.zmp_timeline = self.phase_manager.getTimelines()['zmp_timeline']

        self.__flight_phases = dict()
        self.__stance_phases = dict()

        self.__flight_short_phases = dict()
        self.__stance_short_phases = dict()

        self.__flight_recovery_phases = dict()
        self.__stance_recovery_phases = dict()

        self.__init_tasks(contact_map)

    def __init_tasks(self, contact_map):

        # retrieve manually (for now) the correct tasks if present
        for contact_name, timeline_name in contact_map.items():
            self.__flight_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'flight_{contact_name}')
            self.__stance_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'stance_{contact_name}')

            # different duration (todo: flexible implementation?)
            self.__flight_short_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'flight_{contact_name}_short')
            self.__stance_short_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'stance_{contact_name}_short')

            self.__flight_recovery_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'flight_{contact_name}_recovery')
            self.__stance_recovery_phases[contact_name] = self.__contact_timelines[contact_name].getRegisteredPhase(f'stance_{contact_name}_recovery')

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
                # for i in range(8):
                timeline_i.addPhase(self.__stance_phases[contact_name])
                    # timeline_i.addPhase(self.__stance_short_phases[contact_name])


                timeline_i.addPhase(self.__stance_short_phases[contact_name])
            else:
                timeline_i.addPhase(self.__flight_phases[contact_name])
                timeline_i.addPhase(self.__stance_short_phases[contact_name])


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

    def __add_cycles(self, cycle_lists):

        for cycle_i in cycle_lists:
            self.cycle(cycle_i)


    def __add_cycles_recovery(self, cycle_lists):

        for cycle_i in cycle_lists:
            self.cycle_recovery(cycle_i)



