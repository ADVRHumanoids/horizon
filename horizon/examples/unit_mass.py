#!/usr/bin/env python3

import horizon.problem as prb
import horizon.utils.plotter as plotter
import casadi as cs
import numpy as np
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers import solver
import matplotlib.pyplot as plt

n_nodes = 50
dt = 0.1
mu = 0.2
grav = 9.81
prob = prb.Problem(n_nodes, casadi_type=cs.SX)

p = prob.createStateVariable('pos', dim=2)
v = prob.createStateVariable('vel', dim=2)
F = prob.createInputVariable('force', dim=2)


state = prob.getState()
state_prev = state.getVarOffset(-1)
x = state.getVars()


xdot = cs.vertcat(v, F) #- mu*grav*np.sign(v)
prob.setDynamics(xdot)
prob.setDt(dt)

use_ms = False
if use_ms:
    th = Transcriptor.make_method('multiple_shooting', prob)
else:
    th = Transcriptor.make_method('direct_collocation', prob) # opts=dict(degree=5)

# set initial state (rest in zero)
p.setBounds(lb=[0, 0], ub=[0, 0], nodes=0)
v.setBounds(lb=[0, 0], ub=[0, 0], nodes=0)

# final constraint
p.setBounds(lb=[1, 1], ub=[1, 1], nodes=n_nodes)
v.setBounds(lb=[0, 0], ub=[0, 0], nodes=n_nodes)


obs_center = np.array([0.5, 0.5])
obs_r = 0.4
obs = cs.sumsqr(p - obs_center) - obs_r**2

obs_cnsrt = prob.createIntermediateConstraint('obstacle', obs)
obs_cnsrt.setUpperBounds(np.inf)
# intermediate cost ( i want to minimize the force! )
prob.createIntermediateCost('cost', cs.sumsqr(F))

# solve
solver = solver.Solver.make_solver('ipopt', prob)
solver.solve()
solution = solver.getSolutionDict()

# plot
plot_all = True

if plot_all:
    hplt = plotter.PlotterHorizon(prob, solution)
    hplt.plotVariables(['pos', 'vel', 'force'], grid=True)
    # hplt.plotFunctions(grid=True)


fig, ax = plt.subplots()
ax.set_title('xy plane')
ax.plot(solution['pos'][0], solution['pos'][1])
ax.plot([0, 0], [0, 0], 'bo', markersize=12)
ax.plot([1, 1], [1, 1], 'g*', markersize=12)
circle = plt.Circle(obs_center, radius=obs_r, fc='r')
ax.add_patch(circle)
ax.legend(['traj', 'start', 'goal', 'obstacle'])
plt.gca().add_patch(circle)

plt.show()
