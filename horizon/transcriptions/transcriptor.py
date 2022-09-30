import casadi as cs
import horizon.transcriptions.integrators as integ
from abc import ABC
from horizon.problem import Problem
from horizon.variables import SingleVariable, Variable

class Transcriptor(ABC):

    @classmethod
    def make_method(cls, type: str, prb: Problem, opts=None, logger=None):
        """
        Construct a transcription method. The optimal control problem is an infinite-dimensional optimization problem, since the decision variables are functions, rather than real numbers. All solution techniques perform transcription, a process by which the trajectory optimization problem (optimizing over functions) is converted into a constrained parameter optimization problem (optimizing over real numbers).

        Args:
            type (str): type of transcription method (multiple shooting / direct collocation)
            prb (Problem): the horizon's problem instance to be solved
            dt (float): the discretization step
            opts (Dict, optional): A solver-dependent Dict of options. Defaults to None.
            logger (Logger, optional): pointer to a logger
        """

        # handle inputs
        if opts is None:
            opts = dict()

        import horizon.transcriptions.methods as tm
        if type == 'multiple_shooting':
            default_integrator = 'RK4'
            integrator = opts.get('integrator', default_integrator)
            return tm.MultipleShooting(prob=prb, integrator=prb.getIntegrator())
        elif type == 'direct_collocation':
            default_degree = 3
            degree = opts.get('degree', default_degree)
            return tm.DirectCollocation(prob=prb, degree=degree)
        else:
            raise KeyError(f'unsupported transcription method type "{type}"')

    def __init__(self, prb: Problem, logger=None):

        self.logger = logger
        self.problem = prb
        self.integrator = None

        self.dt = prb.getDt()

        state_list = self.problem.getState()
        state_prev_list = list()

        for var in state_list:
            state_prev_list.append(var.getVarOffset(-1))

        self.state = cs.vertcat(*state_list)
        self.state_prev = cs.vertcat(*state_prev_list)

        input_list = self.problem.getInput()
        input_prev_list = list()
        for var in input_list:
            input_prev_list.append(var.getVarOffset(-1))

        self.input = cs.vertcat(*input_list)
        self.input_prev = cs.vertcat(*input_prev_list)


