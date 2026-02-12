from horizon.solvers import Solver
from horizon.problem import Problem
from horizon.transcriptions.transcriptor import Transcriptor
import casadi as cs
import numpy as np
import yaml
import os
np.set_printoptions(suppress=True, precision=3)


        # particle subject to a force about +x axis,
        # and constrained to a x2 = x1^2 parabola
        # m*ddx + m*g = [tau, 0] + Jc'*f

N = 10
tf = 1.0
dt = tf / N
prb = Problem(N)
x = prb.createStateVariable('x', 2)
v = prb.createStateVariable('v', 2)
a = prb.createInputVariable('a', 2)
f = prb.createInputVariable('f', 1)

xdot = cs.vertcat(v, a)
prb.setDynamics(xdot)
prb.setDt(dt)

# set bounds
x0 = np.array([1, 2, 3, 4])
prb.setInitialState(x0)
print(x.setInitialGuess([1, 2]))
print(v.setInitialGuess([3, 4]))

print(x.getInitialGuess())
print(v.getInitialGuess())
a.setInitialGuess([1, 2])
f.setInitialGuess(1)


contact = x[1] - x[0] ** 2
Jc = cs.horzcat(-2 * x[0], 1)
contact = prb.createIntermediateConstraint('contact', contact, nodes=range(1, N + 1))

underactuation = a[1] + 1.0 - Jc[1] * f
under = prb.createIntermediateConstraint('underactuation', underactuation)

minacc = prb.createIntermediateCost('min_acc', cs.sumsqr(a))

goal = 2.0
goal = prb.createFinalConstraint('goal', x[0] - goal)

vf = prb.createFinalConstraint('vf', v)

# print(prb.getState().getBounds())
# print(prb.getInput().getBounds())
# print(contact.getNodes())
# print(under.getNodes())
# print(minacc.getNodes())
# print(goal.getNodes())
# print(vf.getNodes())

solver = Solver.make_solver('ilqr',
                            prb,
                            opts={'ilqr.enable_gn': True}
                            # opts={
                            #     'ilqr.integrator': 'RK4',
                            #     'ilqr.line_search_accept_ratio': 1e-9,
                            # }
                            )


data = prb.save()
filename = "test_problem_save_2.yaml"

with open(filename, "w") as file:
    yaml.dump(data, file)

full_path = os.path.abspath(filename)
print("File saved at:", full_path)

solver.set_iteration_callback()
ret = solver.solve()

np.set_printoptions(suppress=True, precision=4, linewidth=2000)
for item, sol in solver.getSolutionDict().items():
    if item != 'x_opt' or item != 'u_opt':
        print(item, sol.shape)
        print(sol)


# class Unicycle(unittest.TestCase):
#     def setUp(self) -> None:
#         # particle subject to a force about +x axis,
#         # and constrained to a x2 = x1^2 parabola
#         # m*ddx + m*g = [tau, 0] + Jc'*f
#
#         N = 1000
#         tf = 1.0
#         dt = tf / N
#         prb = Problem(N)
#
#         p = prb.createStateVariable('p', 2)
#         theta = prb.createStateVariable('theta', 1)
#         v = prb.createStateVariable('v', 1)
#         omega = prb.createStateVariable('omega', 1)
#         a = prb.createInputVariable('a', 1)
#         omegadot = prb.createInputVariable('omegadot', 1)
#
#         xdot = cs.vertcat(v * cs.cos(theta),
#                           v * cs.sin(theta),
#                           omega,
#                           a,
#                           omegadot)
#
#         x = prb.getState().getVars()
#         u = prb.getInput().getVars()
#
#         omegadot.setBounds(-20, 20)
#
#         prb.setDynamics(xdot)
#         prb.setDt(dt)
#
#         x0 = np.array([0, 0, 0, 0, 0])
#         prb.setInitialState(x0)
#
#         # we need to set an appropriate initial guess to break symmetry
#         theta.setInitialGuess(cs.pi / 2.0)
#
#         # regularize state
#         prb.createIntermediateResidual('u_reg', u)
#
#         # x-y-theta goal
#         prb.createFinalConstraint('goal_p', p - np.array([0, 1.0]))
#         prb.createFinalConstraint('goal_theta', theta)
#
#         # end at zero velocity
#         prb.createFinalConstraint('vf', cs.vertcat(v, omega))
#
#         # save prb
#         self.prb = prb
#
#     def test_simple_constrained(self):
#         solver = Solver.make_solver('ilqr',
#                                     self.prb,
#                                     opts={
#                                         'ilqr.rho_base': 1.0,
#                                         'ilqr.enable_auglag': False,
#                                         'ilqr.max_iter': 20,
#                                     })
#
#         solver.set_iteration_callback()
#         ret = solver.solve()
#         self.assertTrue(ret)
#
#     def test_bound_constrained(self):
#         solver = Solver.make_solver('ilqr',
#                                     self.prb,
#                                     opts={
#                                         'ilqr.rho_base': 1.0,
#                                         'ilqr.enable_auglag': True,
#                                         'ilqr.max_iter': 200,
#                                     })
#
#         solver.set_iteration_callback()
#         ret = solver.solve()
#         self.assertTrue(ret)