import casadi as cs
import numpy as np
from horizon.rhc.tasks.cartesianTask import CartesianTask
from horizon.rhc.tasks.rollingTask import RollingTask
from horizon.rhc.tasks.interactionTask import InteractionTask
from horizon.functions import RecedingConstraint, RecedingCost
from horizon.utils.utils import barrier as barrier_fun
from horizon.rhc.tasks.task import Task

# todo this is a composition of atomic tasks: how to do?

class ContactTask(Task):
    def __init__(self, subtask,
                 *args, **kwargs):
        """
        establish/break contact
        """

        # todo: default interaction or cartesian task ?
        # todo: make tasks discoverable by name?  subtask: {'interaction': force_contact_1}
        self.interaction_task: InteractionTask = Task.subtask_by_class(subtask, InteractionTask)
        self.cartesian_task: CartesianTask = Task.subtask_by_class(subtask, CartesianTask) # CartesianTask RollingTask

        # initialize data class
        super().__init__(*args, **kwargs)


    def setNodes(self, nodes):

        self.interaction_task.setContact(nodes)  # this is from taskInterface
        self.cartesian_task.setNodes(nodes)  # state + starting from node 1  # this is from taskInterface
