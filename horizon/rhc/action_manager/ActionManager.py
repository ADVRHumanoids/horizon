from horizon.rhc.taskInterface import TaskInterface
from phase_manager.pymanager import PhaseManager
from phase_manager.pyphase import Phase

class ActionManager:
    def __init__(self, task_interface: TaskInterface, phase_manager: PhaseManager, contact_map):

        # contact_map is not necessary if contact name is the same as the timeline name
        self.task_interface = task_interface
        self.phase_manager = phase_manager

        self.contact_phases = dict()

        # register each timeline of the phase manager as the contact phases
        for contact_name, phase_name in contact_map.items():
            self.contact_phases[contact_name] = self.phase_manager.getTimelines()[phase_name]

        self.zmp_timeline = self.phase_manager.getTimelines()['zmp_timeline']

    def cycle_short(self, cycle_list):
        # how do I know that the stance phase is called stance_{c} or flight_{c}?
        for flag_contact, contact_name in zip(cycle_list, self.contact_phases.keys()):
            phase_i = self.contact_phases[contact_name]
            if flag_contact == 1:
                phase_i.addPhase(phase_i.getRegisteredPhase(f'stance_{contact_name}_short'))
            else:
                phase_i.addPhase(phase_i.getRegisteredPhase(f'flight_{contact_name}_short'))

    def cycle(self, cycle_list):
        # how do I know that the stance phase is called stance_{c} or flight_{c}?
        for flag_contact, contact_name in zip(cycle_list, self.contact_phases.keys()):
            phase_i = self.contact_phases[contact_name]
            if flag_contact == 1:
                phase_i.addPhase(phase_i.getRegisteredPhase(f'stance_{contact_name}'))
            else:
                phase_i.addPhase(phase_i.getRegisteredPhase(f'flight_{contact_name}'))

# class GaitManager:
#     def __init__(self, task_interface: TaskInterface, phase_manager: pymanager.PhaseManager, contact_map):
#
#         # contact_map is not necessary if contact name is the same as the timeline name
#         self.task_interface = task_interface
#         self.phase_manager = phase_manager
#
#         self.contact_phases = dict()
#
#         # register each timeline of the phase manager as the contact phases
#         for contact_name, phase_name in contact_map.items():
#             self.contact_phases[contact_name] = self.phase_manager.getTimelines()[phase_name]
#
#         self.zmp_timeline = self.phase_manager.getTimelines()['zmp_timeline']
#
#     def cycle_short(self, cycle_list):
#         # how do I know that the stance phase is called stance_{c} or flight_{c}?
#         for flag_contact, contact_name in zip(cycle_list, self.contact_phases.keys()):
#             phase_i = self.contact_phases[contact_name]
#             if flag_contact == 1:
#                 phase_i.addPhase(phase_i.getRegisteredPhase(f'stance_{contact_name}_short'))
#             else:
#                 phase_i.addPhase(phase_i.getRegisteredPhase(f'flight_{contact_name}_short'))
#
#     def cycle(self, cycle_list):
#         # how do I know that the stance phase is called stance_{c} or flight_{c}?
#         for flag_contact, contact_name in zip(cycle_list, self.contact_phases.keys()):
#             phase_i = self.contact_phases[contact_name]
#             if flag_contact == 1:
#                 phase_i.addPhase(phase_i.getRegisteredPhase(f'stance_{contact_name}'))
#             else:
#                 phase_i.addPhase(phase_i.getRegisteredPhase(f'flight_{contact_name}'))
#
#
#     def step(self, swing_contact):
#         cycle_list = [True if contact_name != swing_contact else False for contact_name in self.contact_phases.keys()]
#         self.cycle(cycle_list)
#
#
#     def trot_jumped(self):
#
#         #  diagonal 1 duration 4
#         self.contact_phases['ball_2'].addPhase(self.contact_phases['ball_2'].getRegisteredPhase(f'flight_ball_2'))
#         self.contact_phases['ball_3'].addPhase(self.contact_phases['ball_3'].getRegisteredPhase(f'flight_ball_3'))
#
#         # diagonal 2 short stance 1 (3 times)
#         self.contact_phases['ball_1'].addPhase(self.contact_phases['ball_1'].getRegisteredPhase(f'stance_ball_1_short'))
#         self.contact_phases['ball_1'].addPhase(self.contact_phases['ball_1'].getRegisteredPhase(f'stance_ball_1_short'))
#         self.contact_phases['ball_1'].addPhase(self.contact_phases['ball_1'].getRegisteredPhase(f'stance_ball_1_short'))
#         self.contact_phases['ball_4'].addPhase(self.contact_phases['ball_4'].getRegisteredPhase(f'stance_ball_4_short'))
#         self.contact_phases['ball_4'].addPhase(self.contact_phases['ball_4'].getRegisteredPhase(f'stance_ball_4_short'))
#         self.contact_phases['ball_4'].addPhase(self.contact_phases['ball_4'].getRegisteredPhase(f'stance_ball_4_short'))
#
#         #  diagonal 2 duration 4
#         self.contact_phases['ball_1'].addPhase(self.contact_phases['ball_1'].getRegisteredPhase(f'flight_ball_1'))
#         self.contact_phases['ball_4'].addPhase(self.contact_phases['ball_4'].getRegisteredPhase(f'flight_ball_4'))
#
#         # diagonal 1 short stance 1 (3 times)
#         self.contact_phases['ball_2'].addPhase(self.contact_phases['ball_2'].getRegisteredPhase(f'stance_ball_2_short'))
#         self.contact_phases['ball_2'].addPhase(self.contact_phases['ball_2'].getRegisteredPhase(f'stance_ball_2_short'))
#         self.contact_phases['ball_2'].addPhase(self.contact_phases['ball_2'].getRegisteredPhase(f'stance_ball_2_short'))
#         self.contact_phases['ball_3'].addPhase(self.contact_phases['ball_3'].getRegisteredPhase(f'stance_ball_3_short'))
#         self.contact_phases['ball_3'].addPhase(self.contact_phases['ball_3'].getRegisteredPhase(f'stance_ball_3_short'))
#         self.contact_phases['ball_3'].addPhase(self.contact_phases['ball_3'].getRegisteredPhase(f'stance_ball_3_short'))
#
#
#         # self.contact_phases['ball_1'].addPhase(self.contact_phases['ball_1'].getRegisteredPhase(f'stance_ball_1'))
#         # self.contact_phases['ball_2'].addPhase(self.contact_phases['ball_2'].getRegisteredPhase(f'flight_ball_2'))
#         # self.contact_phases['ball_3'].addPhase(self.contact_phases['ball_3'].getRegisteredPhase(f'stance_ball_3'))
#         # self.contact_phases['ball_4'].addPhase(self.contact_phases['ball_4'].getRegisteredPhase(f'flight_ball_4'))
#
#
#
#     def trot(self):
#         cycle_list_1 = [0, 1, 1, 0]
#         cycle_list_2 = [1, 0, 0, 1]
#         self.cycle(cycle_list_1)
#         self.cycle(cycle_list_2)
#         self.zmp_timeline.addPhase(self.zmp_timeline.getRegisteredPhase('zmp_empty_phase'))
#         self.zmp_timeline.addPhase(self.zmp_timeline.getRegisteredPhase('zmp_empty_phase'))
#
#     def crawl(self):
#         cycle_list_1 = [0, 1, 1, 1]
#         cycle_list_2 = [1, 1, 1, 0]
#         cycle_list_3 = [1, 0, 1, 1]
#         cycle_list_4 = [1, 1, 0, 1]
#         self.cycle(cycle_list_1)
#         self.cycle(cycle_list_2)
#         self.cycle(cycle_list_3)
#         self.cycle(cycle_list_4)
#
#     def leap(self):
#         cycle_list_1 = [0, 0, 1, 1]
#         cycle_list_2 = [1, 1, 0, 0]
#         self.cycle(cycle_list_1)
#         self.cycle(cycle_list_2)
#
#     def walk(self):
#         cycle_list_1 = [1, 0, 1, 0]
#         cycle_list_2 = [0, 1, 0, 1]
#         self.cycle(cycle_list_1)
#         self.cycle(cycle_list_2)
#
#     def jump(self):
#         cycle_list = [0, 0, 0, 0]
#         self.cycle(cycle_list)
#
#     def wheelie(self):
#         cycle_list = [0, 0, 1, 1]
#         self.cycle(cycle_list)
#
#     def stand(self):
#         cycle_list = [1, 1, 1, 1]
#         self.cycle(cycle_list)
#         self.zmp_timeline.addPhase(self.zmp_timeline.getRegisteredPhase('zmp_phase'))

