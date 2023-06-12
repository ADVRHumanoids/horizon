from horizon.rhc.tasks.cartesianTask import CartesianTask
from horizon.rhc.tasks.rollingTask import RollingTask
from horizon.rhc.tasks.interactionTask import InteractionTask
from horizon.rhc.tasks.task import Task

# todo this is a composition of atomic tasks: how to do?

class ContactTask(Task):
    def __init__(self, subtask,
                 *args, **kwargs):
        """
        establish/break contact
        """

        # todo : default interaction or cartesian task ?
        # todo : make tasks discoverable by name?  subtask: {'interaction': force_contact_1}
        self.interaction_task: InteractionTask = Task.subtask_by_class(subtask, InteractionTask)
        self.cartesian_task: CartesianTask = Task.subtask_by_class(subtask, RollingTask) # CartesianTask RollingTask

        # initialize data class
        super().__init__(*args, **kwargs)

        self.initialize()

    def initialize(self):

        self.setNodes(self.nodes)

    def setNodes(self, nodes, erasing=True):

        self.interaction_task.setContact(nodes, erasing=erasing)  # this is from taskInterface
        self.cartesian_task.setNodes(nodes, erasing=erasing)  # state + starting from node 1  # this is from taskInterface
