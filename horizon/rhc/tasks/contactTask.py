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

        self.dynamics_tasks: InteractionTask = Task.subtask_by_class(subtask, InteractionTask)
        # allowed tasks are cartesian and rolling
        self.kinematics_tasks: CartesianTask = Task.subtask_by_class(subtask, (CartesianTask, RollingTask)) # CartesianTask RollingTask

        # initialize data class
        super().__init__(*args, **kwargs)

        self.__initialize()

    def __initialize(self):

        self.setNodes(self.nodes)

    def setNodes(self, nodes, erasing=True):
        
        for task in self.dynamics_tasks:
            task.setContact(nodes, erasing=erasing)  # this is from taskInterface
        for task in self.kinematics_tasks:
            task.setNodes(nodes, erasing=erasing)  # state + starting from node 1  # this is from taskInterface
