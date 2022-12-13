import unittest

from horizon.solvers import Solver
from horizon.problem import Problem
from horizon.transcriptions.transcriptor import Transcriptor
import casadi as cs
import numpy as np
np.set_printoptions(suppress=True, precision=3)

def _test_a_vs_b(self, a: Solver, b: Solver):
    a.solve()
    b.solve()
    xerr = a.x_opt - b.x_opt
    uerr = a.u_opt - b.u_opt
    self.assertLess(np.abs(xerr).max(), 1e-6)
    self.assertLess(np.abs(uerr).max(), 1e-6)

class InitialStateOpt(unittest.TestCase):

    def make_problem(self, solver_type):
        
        N = 10
        tf = 1.0
        prob = Problem(N)
        prob.setDt(tf/N)

        # mass-damper falling ball with rocket
        p = prob.createStateVariable('p', dim=3)
        v = prob.createStateVariable('v', dim=3)
        fz = prob.createInputVariable('fz', dim=1)

        x = prob.getState().getVars()
        g = np.array([0, 0, -10])
        F = cs.vertcat(0, 0, fz)

        xdot = cs.vertcat(
            v,
            F + g - v
        )

        prob.setDynamics(xdot)

        # initial and final constraints
        p0 = np.array([0, 0, 0])
        prob.createConstraint('initial', p - p0, nodes=0)

        ptgt = np.array([1, 1, 1])
        prob.createFinalConstraint('final', p - ptgt)

        prob.createFinalConstraint('z_vel', v[2])

        # min effort
        prob.createIntermediateResidual('min_effort', fz)

        # solve first with ilqr
        if solver_type == 'ilqr':
            ilqrsol = Solver.make_solver('ilqr', prob,  
                    opts={'max_iter': 1, 'ilqr.integrator': 'RK4'})
            ilqrsol.set_iteration_callback()
            return ilqrsol

        # solver with sqp or ipopt need a dynamic constraint
        th = Transcriptor.make_method('multiple_shooting', prob, opts=dict(integrator='RK4'))

        # blocksqp needs exact hessian to be accurate
        opts = None 
        if solver_type == 'blocksqp':
            opts = {'hess_update': 4}
            
        bsqpsol = Solver.make_solver(solver_type, prob, opts)
        return bsqpsol

    def test_ilqr_vs_ipopt(self):
        print(self.__class__.__name__, self._testMethodName)
        ilqr = self.make_problem('ilqr')
        ipopt = self.make_problem('ipopt')
        _test_a_vs_b(self, ipopt, ilqr)
        

        


class SolverConsistency(unittest.TestCase):
    def setUp(self) -> None:
        A11 = np.array([[1, 2], [3, 4]])
        A13 = np.array([[1, -2], [-3, 4]])
        A21 = np.array([[3, 2], [5, 4]])
        A32 = np.array([[-3, 2], [-5, 4]])
        B21 = np.array([[1, -1], [3, 2]])
        B32 = np.array([[1, -2], [-3, 4]])
        self.matrices = [A11, A13, A21, A32, B21, B32]

    def test_blocksqp_vs_ipopt(self):
        print(self.__class__.__name__, self._testMethodName)
        ipopt = make_problem('ipopt', *self.matrices)
        blocksqp = make_problem('blocksqp', *self.matrices)
        _test_a_vs_b(self, ipopt, blocksqp)

    def test_gnsqp_vs_ipopt(self):
        print(self.__class__.__name__, self._testMethodName)
        ipopt = make_problem('ipopt', *self.matrices)
        gnsqp = make_problem('gnsqp', *self.matrices)
        _test_a_vs_b(self, ipopt, gnsqp)
    
    def test_blocksqp_vs_ilqr(self):
        print(self.__class__.__name__, self._testMethodName)
        ilqr = make_problem('ilqr', *self.matrices)
        blocksqp = make_problem('blocksqp', *self.matrices)
        _test_a_vs_b(self, ilqr, blocksqp)

def make_problem(solver_type, A11, A13, A21, A32, B21, B32):
    # on a linear-quadratic problem, all solvers should agree on the solution
    N = 5
    dt = 0.1
    prob = Problem(N)
    prob.setDt(dt)

    # a random linear dynamics
    x1 = prob.createStateVariable('x1', dim=2)
    x2 = prob.createStateVariable('x2', dim=2)
    x3 = prob.createStateVariable('x3', dim=2)
    u1 = prob.createInputVariable('u1', dim=2)
    u2 = prob.createInputVariable('u2', dim=2)
    u3 = prob.createInputVariable('u3', dim=2)

    x = prob.getState().getVars()

    xdot = cs.vertcat(
        A11@x1 + A13@x3,
        A21@x1 + B21@u1, 
        A32@x2 + B32@u2
    )
    prob.setDynamics(xdot)

    # a random cost
    cost_weight = prob.createParameter('cost_weight', 1)
    cost_weight.assign(2.)
    prob.createIntermediateResidual('c12', x1 + cost_weight*x2)
    prob.createIntermediateResidual('c23', x2 + x3)
    prob.createIntermediateResidual('c13', x1 + x3)
    prob.createIntermediateResidual('u', cs.vertcat(u1, u2))

    # a final constraint
    xtgt_value = np.array([1, 2, 2, 3, 3, 4])
    xtgt = prob.createParameter('xtgt', 6)
    xtgt.assign(xtgt_value)
    prob.createFinalConstraint('xtgt', x - xtgt)

    # an initial state
    x0 = -xtgt_value
    prob.getState().setBounds(lb=x0, ub=x0, nodes=0)
    # prob.getState().setBounds(lb=xtgt, ub=xtgt, nodes=N)
    prob.getState().setInitialGuess(x0)

    # solve first with ilqr
    if solver_type == 'ilqr':
        ilqrsol = Solver.make_solver('ilqr', prob,  
                opts={'max_iter': 3, 'ilqr.integrator': 'EULER'})
        ilqrsol.set_iteration_callback()
        return ilqrsol

    # solver with sqp or ipopt need a dynamic constraint
    th = Transcriptor.make_method('multiple_shooting', prob, opts=dict(integrator='EULER'))

    # blocksqp needs exact hessian to be accurate
    opts = None 
    if solver_type == 'blocksqp':
        opts = {'hess_update': 4}
    else:
        opts = {'gnsqp.osqp.polish': True}
        
    bsqpsol = Solver.make_solver(solver_type, prob, opts)
    return bsqpsol

if __name__ == '__main__':
    unittest.main()