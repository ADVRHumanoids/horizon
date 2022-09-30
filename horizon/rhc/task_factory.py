from horizon.rhc.tasks.task import Task

from typing import Callable, Dict, Any, List

task_creation_funcs: Dict[str, Callable[..., Task]] = {}


def get_registered_tasks(task_type=None) -> (List, Callable[..., Task]):
    """
    Get a list of all registered task types.
    """
    if task_type is None:
        return list(task_creation_funcs.values())
    else:
        return None if task_type not in task_creation_funcs else task_creation_funcs[task_type]


def register(task_type: str, func: Callable[..., Task]) -> None:
    """
    Register a new task class.
    """
    task_creation_funcs[task_type] = func


def unregister(task_type: str) -> None:
    """
    Unregister a task class.
    """
    task_creation_funcs.pop(task_type, None)


def create(args: Dict[str, Any]) -> Task:
    """
    Create a new task of a specific type.
    """
    args_copy = args.copy()
    task_type = args_copy['type'] # todo: why pop?

    try:
        creation_func = task_creation_funcs[task_type]
    except KeyError:
        raise ValueError(f'Unknown task type: {task_type}; available types {list(task_creation_funcs.keys())}') from None

    return creation_func.from_dict(args_copy)

