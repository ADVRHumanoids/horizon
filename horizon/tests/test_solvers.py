from horizon.problem import Problem
from horizon.solvers import Solver
from horizon.transcriptions.transcriptor import Transcriptor
import casadi as cs
import numpy as np

import unittest


class TestSolvers(unittest.TestCase):

    def setUp(self) -> None:
        self.nodes = 50
        self.dt = 0.1
        self.mu = 0.2
        self.grav = 9.81

        self.prb = Problem(self.nodes, casadi_type=cs.MX)

        p = self.prb.createStateVariable('pos', dim=2)
        v = self.prb.createStateVariable('vel', dim=2)
        f = self.prb.createInputVariable('force', dim=2)
        u = self.prb.createInputVariable('independent_input', dim=2)

        state = self.prb.getState()
        state_prev = state.getVarOffset(-1)
        x = state.getVars()

        xdot = cs.vertcat(v, f)  # - mu*grav*np.sign(v)
        self.prb.setDynamics(xdot)
        self.prb.setDt(self.dt)


        # set initial state (rest in zero)
        p.setBounds(lb=[0, 0], ub=[0, 0], nodes=0)
        v.setBounds(lb=[0, 0], ub=[0, 0], nodes=0)

        p.setInitialGuess([0, 0])

        # final constraint
        # p.setBounds(lb=[1, 1], ub=[1, 1], nodes=self.nodes)
        v.setBounds(lb=[0, 0], ub=[0, 0], nodes=self.nodes)

        # obs_center = np.array([0.5, 0.5])
        # obs_r = 0.4
        # obs = cs.sumsqr(p - obs_center) - obs_r ** 2
        #
        # obs_cnsrt = self.prb.createIntermediateConstraint('obstacle', obs)
        # obs_cnsrt.setUpperBounds(np.inf)
        self.prb.createIntermediateResidual('cost', f)
        self.prb.createIntermediateCost('c12', cs.sumsqr(p - u))

        self.prb.createFinalConstraint('final_pos', p - [1, 1])

    #
    def testIPOPT(self):

        th = Transcriptor.make_method('multiple_shooting', self.prb)
        slvr = Solver.make_solver('ipopt', self.prb)
        sol = slvr.solve()
        self.assertTrue(sol)

    def testILQR(self):

        opts = {
            'ilqr.integrator': 'RK4',
            'ilqr.line_search_accept_ratio': 1e-9,
            'ilqr.svd_threshold': 1e-12,
            'ilqr.alpha_min': 0.1,
            'ilqr.hxx_reg': 1000.,
        }
    #
        slvr = Solver.make_solver('ilqr', self.prb, opts)
        sol = slvr.solve()
        self.assertTrue(sol)

    def testGNSQP(self):
        opts = dict()
        opts['gnsqp.qp_solver'] = 'osqp'
        opts['warm_start_primal'] = True
        opts['warm_start_dual'] = True
        opts['merit_derivative_tolerance'] = 1e-6
        opts['constraint_violation_tolerance'] = self.nodes * 1e-12
        opts['osqp.polish'] = True  # without this
        opts['osqp.delta'] = 1e-9  # and this, it does not converge!
        opts['osqp.verbose'] = False
        opts['osqp.rho'] = 0.02
        opts['osqp.scaled_termination'] = False

        th = Transcriptor.make_method('multiple_shooting', self.prb)
        slvr = Solver.make_solver('gnsqp', self.prb, opts)

        sol = slvr.solve()
        self.assertTrue(sol)

if __name__ == '__main__':
    unittest.main()