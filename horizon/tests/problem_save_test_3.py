from horizon.solvers import Solver
from horizon.problem import Problem
from horizon.transcriptions.transcriptor import Transcriptor
import casadi as cs
import numpy as np
import yaml

np.set_printoptions(suppress=True, precision=3)


N = 20
tf = 1.0
dt = tf / N
prb = Problem(N)

p = prb.createStateVariable('p', 2)
theta = prb.createStateVariable('theta', 1)
v = prb.createStateVariable('v', 1)
omega = prb.createStateVariable('omega', 1)
a = prb.createInputVariable('a', 1)
omegadot = prb.createInputVariable('omegadot', 1)

xdot = cs.vertcat(v * cs.cos(theta),
                  v * cs.sin(theta),
                  omega,
                  a,
                  omegadot)

x = prb.getState().getVars()
u = prb.getInput().getVars()

omegadot.setBounds(-20, 20)

prb.setDynamics(xdot)
prb.setDt(dt)

x0 = np.array([0, 0, 0, 0, 0])
prb.setInitialState(x0)

# we need to set an appropriate initial guess to break symmetry
theta.setInitialGuess(cs.pi / 2.0)

# regularize state
prb.createIntermediateResidual('u_reg', u)

# x-y-theta goal
prb.createFinalConstraint('goal_p', p - np.array([0, 1.0]))
prb.createFinalConstraint('goal_theta', theta)

# end at zero velocity
prb.createFinalConstraint('vf', cs.vertcat(v, omega))

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

#                                     opts={
#                                         'ilqr.rho_base': 1.0,
#                                         'ilqr.enable_auglag': True,
#                                         'ilqr.max_iter': 200,

data = prb.save()

with open("test_problem_save_3.yaml", "w") as file:
    yaml.dump(data, file)

solver.set_iteration_callback()
ret = solver.solve()

np.set_printoptions(suppress=True, precision=4, linewidth=2000)
for item, sol in solver.getSolutionDict().items():
    if item != 'x_opt' or item != 'u_opt':
        print(item, sol.shape)
        print(sol)