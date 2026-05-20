import unittest

from horizon.solvers import Solver
from horizon.problem import Problem
from horizon.transcriptions.transcriptor import Transcriptor
import casadi as cs
import numpy as np
np.set_printoptions(suppress=True, precision=3)

class ParamTest(unittest.TestCase):
    def setUp(self) -> None:
        A11 = np.array([[1, 2], [3, 4]])
        A13 = np.array([[1, -2], [-3, 4]])
        A21 = np.array([[3, 2], [5, 4]])
        A32 = np.array([[-3, 2], [-5, 4]])
        B21 = np.array([[1, -1], [3, 2]])
        B32 = np.array([[1, -2], [-3, 4]])
        self.matrices = [A11, A13, A21, A32, B21, B32]

    def _test_a_vs_b(self, a: Solver, b: Solver):
        a.solve()
        print(a.x_opt)
        b.solve()
        xerr = a.x_opt - b.x_opt
        uerr = a.u_opt - b.u_opt
        self.assertLess(np.abs(xerr).max(), 1e-4)
        self.assertLess(np.abs(uerr).max(), 1e-4)
        for i in [1, 2, 3, 4]:
            self.assertAlmostEqual(a.x_opt[2, i], 4)
            self.assertAlmostEqual(a.x_opt[3, i], 5)
        
        self.assertTrue(np.allclose(a.x_opt[:, -1], np.array([1, 2, 2, 3, 3, 4])))

    def test_ipopt_vs_ilqr(self):
        ilqr = make_problem('ilqr', *self.matrices)
        ipopt = make_problem('ipopt', *self.matrices)
        self._test_a_vs_b(ilqr, ipopt)

    def test_gnsqp_vs_ilqr(self):
        ilqr = make_problem('ilqr', *self.matrices)
        blocksqp = make_problem('gnsqp', *self.matrices)
        self._test_a_vs_b(ilqr, blocksqp)


def make_problem(solver_type, A11, A13, A21, A32, B21, B32):
    # on a linear-quadratic problem, all solvers should agree on the solution
    N = 5
    dt = 0.1
    prob = Problem(N)

    # a random linear dynamics
    x1 = prob.createStateVariable('x1', dim=2)
    x2 = prob.createStateVariable('x2', dim=2)
    x3 = prob.createStateVariable('x3', dim=2)
    u1 = prob.createInputVariable('u1', dim=2)
    u2 = prob.createInputVariable('u2', dim=2)
    p1 = prob.createParameter('p1', dim=2)
    time = prob.createParameter('t', dim=1)
    prob.setDt(time)
    x = prob.getState().getVars()

    xdot = cs.vertcat(
        A11@x1 + A13@x3,
        B21@u1, 
        A32@x1 + B32@u2
    )
    prob.setDynamics(xdot)

    # a random cost
    prob.createIntermediateResidual('c12', x2 - p1)
    prob.createIntermediateResidual('c23', x1 + x3)
    prob.createIntermediateResidual('c13', x1 - x3)
    prob.createIntermediateResidual('u', u2)

    # a final constraint
    xtgt = prob.createParameter('xtgt', 6)
    xtgt.assign(np.array([1, 2, 2, 3, 3, 4]))
    prob.createFinalConstraint('xtgt', x - xtgt)

    # an initial state
    x0 = -xtgt.getValues(nodes=0)
    prob.getState().setBounds(lb=x0, ub=x0, nodes=0)
    prob.getState().setInitialGuess(x0)

    # param value
    p1.assign(np.array([4, 5]))
    time.assign(dt)

    # solve first with ilqr
    if solver_type == 'ilqr':
        ilqrsol = Solver.make_solver('ilqr', prob, 
                opts={'max_iter': 3, 
                'ilqr.hxx_reg': 0.0,
                'ilqr.alpha_min': 1e-6,
                'ilqr.integrator': 'EULER'})
        return ilqrsol

    # solver with sqp or ipopt need a dynamic constraint
    Transcriptor.make_method('multiple_shooting', prob, opts=dict(integrator='EULER'))

    # blocksqp needs exact hessian to be accurate
    opts = None 
    if solver_type == 'blocksqp':
        opts = {'hess_update': 4}
    elif solver_type == 'gnsqp':
        opts = dict()
        opts['gnsqp.qp_solver'] = 'osqp'
        opts['merit_derivative_tolerance'] = 1e-6
        opts['constraint_violation_tolerance'] = 1e-12
        opts['osqp.polish'] = True # without this
        opts['osqp.delta'] = 1e-9 # and this, it does not converge!
        opts['osqp.verbose'] = False
        opts['osqp.rho'] = 0.02
        opts['osqp.scaled_termination'] = False
        
    bsqpsol = Solver.make_solver(solver_type, prob, opts)
    return bsqpsol

if __name__ == '__main__':
    unittest.main()