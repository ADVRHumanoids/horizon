import casadi as cs
import horizon.problem as prb
from horizon.variables import AbstractVariable, SingleVariable, Variable, SingleParameter, Parameter
import numpy as np
from horizon.transcriptions.transcriptor import Transcriptor
import horizon.transcriptions.integrators as integ

class DirectCollocation(Transcriptor):
    """
    Add auxiliary input variables and constraints that implement
    the Direct Collocation transcription scheme. This could also be
    seen as multiple shooting with a different integration strategy
    that is based on approximating polynomials.
    """

    def __init__(self,
                 prob: prb.Problem,
                 degree: int) -> None:

        """
        Initialize the direct collocation method.

        Parameters:
            prob (prb.Problem): the horizon problem
            degree (int): degree of approximating polynomial
        """

        super().__init__(prob)

        # todo some of this is a duplicate of the Transcriptor initialization
        # handle input
        d = degree

        dt = prob.getDt()
        # some constants
        N = prob.getNNodes() - 1

        # get dynamics, state at current and previous iter
        xdot = prob.getDynamics()
        x = prob.getState().getVars()
        x_prev = prob.getState().getVarOffset(-1).getVars()

        # create additional variables (states at collocation points)
        collo = [prob.createInputVariable(f'collo_x_{i}', dim=x.shape[0]) for i in range(d)]
        collo_prev = [var.getVarOffset(-1) for var in collo]
        collo.insert(0, x)  # append state at beginning of interval
        collo_prev.insert(0, x_prev)  # append state at beginning of previous interval

        # generate coefficients from magic polynomial
        collo_points = cs.collocation_points(d, 'legendre')  # or 'radau'

        # roots of basis polynomials (0 + collocation points)
        tau_root = np.append(0, collo_points)

        # coefficients of the collocation equation (i.e., dynamics)
        # note: first row is not used (dynamics constraint at beginning of interval),
        # as dynamics is only enforced at collocation points (excluding t = 0)
        C = np.zeros((d + 1, d + 1))

        # coefficients of the continuity equation
        D = np.zeros(d + 1)

        # coefficients of the quadrature function
        B = np.zeros(d + 1)

        # loop over polynomial basis
        for j in range(d + 1):

            # construct j-th Lagrange polynomial, s.t. pj(tj) = 1 && pj(ti) = 0 if j != i
            # note: these d+1 polynomials form a basis for all d-th degree polynomials
            p = np.poly1d([1])
            for r in range(d + 1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

            # evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = p(1.0)

            # evaluate the time derivative of the polynomial at all collocation points to get the coefficients
            # of the collocation equation
            pder = np.polyder(p)
            for r in range(d + 1):
                C[r, j] = pder(tau_root[r])  # note: r = 0 is actually not used!

            # evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            B[j] = pint(1.0)

        # continuity constraint
        x_prev_int = 0  # this will be the previous state, integrated according to the previous polynomial

        for i in range(d + 1):
            # loop on basis polynomials
            x_prev_int += collo_prev[i] * D[i]

        cc = prob.createConstraint('collo_continuity', x_prev_int - x, nodes=range(1, N + 1))

        # dynamics constraint (one per collocation point)
        for i in range(d):
            # loop on collocation points
            xder = 0
            for j in range(d + 1):
                # loop on basis polynomials
                xder += C[i + 1, j] * collo[j]
            dyn = prob.createConstraint(f'collo_dyn_{i}', xder - xdot * dt, nodes=range(N))


class MultipleShooting(Transcriptor):
    """
    Add auxiliary input variables and constraints that implement
    the Multiple Shooting transcription scheme. A defect constraint is defined between all the segments of the trajectory.
    A desired integrator yields its evolution along the segments.
    """
    def __init__(self,
                 prob: prb.Problem,
                 integrator):
        """
        Initialize the multiple shooting method.

        Args:
            prob (prb.Problem): the horizon problem
            dt (float|StateVariable): discretization interval (can be a control input)
            integrator (string|any): name of the default integrator or custom integrator
        """
        super().__init__(prob)
        # logic to pick a default integrator or keep a custom integrator
        # todo could be done dividing the arguments, for instance default_integrator_type and integrator
        if isinstance(integrator, str):
            if integrator in integ.__dict__:
                self.setDefaultIntegrator(integrator)
            else:
                raise Exception('Selected default integrator is not implemented.')
        else:
            self.integrator = integrator

        # todo add support for dt that is not a variable/parameter of the problem but depends on problem variables/parameters
        #   this could be solved by using state_next, not state:
        #   state_next_int = self.__integrate(self.state, self.input, self.dt)
        #   ms = self.problem.createConstraint('multiple_shooting', state_next_int - self.state.getVarOffset(), nodes=range(0, self.problem.getNNodes() - 1))
        # if dt is a single SX (which means that it involves ALL nodes):
        if isinstance(self.dt, (cs.SX, cs.MX)):
            # for var in self.problem.getVariables().values():
            #     if cs.depends_on(self.dt, var):
            #         if var.getNodes() != -1 or len(var.getNodes()) != self.problem.getNNodes():
            #             raise Exception('dt must depend on variables that are defined over all the nodes.')
            #         else:
            if isinstance(self.dt, Variable) or isinstance(self.dt, SingleVariable):
                self.dt_prev = self.dt.getVarOffset(-1)
            elif isinstance(self.dt, Parameter) or isinstance(self.dt, SingleParameter):
                self.dt_prev = self.dt.getParOffset(-1)
            else:
                raise Exception('dt not supported: not a variable/parameter of the problem. Please build your own transcriptor.')

            state_int = self.__integrate(self.state_prev, self.input_prev, self.dt_prev)
            ms = self.problem.createConstraint('multiple_shooting', state_int - self.state, nodes=range(1, self.problem.getNNodes()))
        # if dt is a single value (which means that it involves ALL nodes):
        elif isinstance(self.dt, (float, int)):
            state_int = self.__integrate(self.state_prev, self.input_prev, self.dt)
            ms = self.problem.createConstraint('multiple_shooting', state_int - self.state, nodes=range(1, self.problem.getNNodes()))
        # if dt is an array of elements, that may be values or parameters or variables:
        elif hasattr(self.dt, '__iter__'):
            for node_n in range(1, self.problem.getNNodes()):
                state_int = self.__integrate(self.state_prev, self.input_prev, self.dt[node_n - 1])
                ms = self.problem.createConstraint(f'multiple_shooting_node_{node_n}', state_int - self.state, nodes=node_n)

    def setDefaultIntegrator(self, integrator_type):

        # todo if the dt is not a abstract value, add option['tf']
        #  the integrator should be refactored a bit
        opts = dict()
        dae = dict()

        dae['x'] = self.state
        dae['p'] = self.input
        dae['ode'] = self.state_dot
        dae['quad'] = 0  # note: we don't use the quadrature fn here

        self.integrator = integ.__dict__[integrator_type](dae=dae, opts=opts, casadi_type=cs.SX)

    def __integrate(self, state, input, dt):

        state_int = self.integrator(state, input, dt)[0]

        return state_int




