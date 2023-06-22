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

