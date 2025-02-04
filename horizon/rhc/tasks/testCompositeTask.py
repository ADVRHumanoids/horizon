from horizon.rhc.tasks.testTask import TestTask
from horizon.rhc.tasks.task import Task

# todo this is a composition of atomic tasks: how to do?

class TestCompositeTask(Task):
    def __init__(self, subtask,
                 *args, **kwargs):
        """
        establish/break contact
        """

        # todo : default interaction or cartesian task ?
        # todo : make tasks discoverable by name?  subtask: {'interaction': force_contact_1}
        self.task_1: TestTask = Task.subtask_by_class(subtask, TestTask)
        self.task_2: TestTask = Task.subtask_by_class(subtask, TestTask) # CartesianTask RollingTask

        # initialize data class
        super().__init__(*args, **kwargs)
