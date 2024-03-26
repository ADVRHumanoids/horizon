import numpy as np
from horizon.rhc.taskInterface import TaskInterface
from phase_manager import pyphase, pymanager
from sensor_msgs.msg import Joy
import rospy
import math
from visualization_msgs.msg import Marker

class GaitManager:
    def __init__(self, task_interface: TaskInterface, phase_manager: pymanager.PhaseManager, contact_map):

        # contact_map is not necessary if contact name is the same as the timeline name
        self.__task_interface = task_interface
        self.__phase_manager = phase_manager

        self.__contact_phases = dict()

        # register each timeline of the phase manager as the contact phases
        for contact_name, phase_name in contact_map.items():
            self.__contact_phases[contact_name] = self.__phase_manager.getTimelines()[phase_name]

        # self.zmp_timeline = self.phase_manager.getTimelines()['zmp_timeline']

    def getContactTimelines(self):

        return self.__contact_phases

    def getTaskInterface(self):

        return self.__task_interface

    def cycle_short(self, cycle_list):
        # how do I know that the stance phase is called stance_{c} or flight_{c}?
        for flag_contact, contact_name in zip(cycle_list, self.__contact_phases.keys()):
            phase_i = self.__contact_phases[contact_name]
            if flag_contact == 1:
                phase_i.addPhase(phase_i.getRegisteredPhase(f'stance_{contact_name}_short'))
            else:
                phase_i.addPhase(phase_i.getRegisteredPhase(f'flight_{contact_name}_short'))

    def cycle(self, cycle_list):

        # how do I know that the stance phase is called stance_{c} or flight_{c}?
        for flag_contact, contact_name in zip(cycle_list, self.__contact_phases.keys()):
            phase_i = self.__contact_phases[contact_name]
            if flag_contact == 1:
                phase_i.addPhase(phase_i.getRegisteredPhase(f'stance_{contact_name}'))
                phase_i.addPhase(phase_i.getRegisteredPhase(f'stance_{contact_name}_short'))
            else:
                phase_i.addPhase(phase_i.getRegisteredPhase(f'flight_{contact_name}'))
                phase_i.addPhase(phase_i.getRegisteredPhase(f'stance_{contact_name}_short'))

    def step(self, swing_contact):
        cycle_list = [True if contact_name != swing_contact else False for contact_name in self.__contact_phases.keys()]
        self.cycle(cycle_list)

    def trot(self):
        cycle_list_1 = [0, 1, 1, 0]
        cycle_list_2 = [1, 0, 0, 1]
        self.cycle(cycle_list_1)
        self.cycle(cycle_list_2)
        # self.zmp_timeline.addPhase(self.zmp_timeline.getRegisteredPhase('zmp_empty_phase'))
        # self.zmp_timeline.addPhase(self.zmp_timeline.getRegisteredPhase('zmp_empty_phase'))

    def crawl(self, vref=[0, 0, 1]):

        vx, vy, omega = vref
        Rmax = 1
        
        if vx**2 + vy**2 <= Rmax*omega**2:

            # turning gait
            if omega > 0:

                cycle_list_1 = [0, 1, 1, 1]
                cycle_list_2 = [1, 1, 0, 1]
                cycle_list_3 = [1, 1, 1, 0]
                cycle_list_4 = [1, 0, 1, 1]
            else:

                cycle_list_1 = [0, 1, 1, 1]
                cycle_list_2 = [1, 0, 1, 1]
                cycle_list_3 = [1, 1, 1, 0]
                cycle_list_4 = [1, 1, 0, 1]

        else:

            if vx > abs(vy): 
                
                # forward
                cycle_list_1 = [1, 1, 0, 1]  # rl
                cycle_list_2 = [0, 1, 1, 1]  # fl
                cycle_list_3 = [1, 1, 1, 0]  # rr
                cycle_list_4 = [1, 0, 1, 1]  # fr

            elif vx < -abs(vy):

                # backward
                cycle_list_1 = [0, 1, 1, 1]  # fl
                cycle_list_2 = [1, 1, 0, 1]  # rl
                cycle_list_3 = [1, 0, 1, 1]
                cycle_list_4 = [1, 1, 1, 0]

            elif vy > abs(vx):

                # left
                cycle_list_1 = [1, 0, 1, 1]
                cycle_list_2 = [0, 1, 1, 1]
                cycle_list_3 = [1, 1, 1, 0]
                cycle_list_4 = [1, 1, 0, 1]
                
            else:

                # right
                cycle_list_1 = [0, 1, 1, 1]
                cycle_list_2 = [1, 0, 1, 1]
                cycle_list_3 = [1, 1, 0, 1]
                cycle_list_4 = [1, 1, 1, 0]

        self.cycle(cycle_list_1)
        self.cycle(cycle_list_2)
        self.cycle(cycle_list_3)
        self.cycle(cycle_list_4)

    def leap(self):
        cycle_list_1 = [0, 0, 1, 1]
        cycle_list_2 = [1, 1, 0, 0]
        self.cycle(cycle_list_1)
        self.cycle(cycle_list_2)

    def walk(self):
        cycle_list_1 = [1, 0, 1, 0]
        cycle_list_2 = [0, 1, 0, 1]
        self.cycle(cycle_list_1)
        self.cycle(cycle_list_2)

    def jump(self):
        cycle_list = [0, 0, 0, 0]
        self.cycle(cycle_list)

    def wheelie(self):
        cycle_list = [0, 0, 1, 1]
        self.cycle(cycle_list)

    def give_paw(self):
        cycle_list = [0, 1, 1, 1]
        self.cycle(cycle_list)

    def stand(self):
        cycle_list = [1, 1, 1, 1]
        self.cycle(cycle_list)
        # self.zmp_timeline.addPhase(self.zmp_timeline.getRegisteredPhase('zmp_phase'))
