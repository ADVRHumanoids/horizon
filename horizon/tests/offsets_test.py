from horizon.problem import Problem
from horizon.solvers import Solver
import pprint
import numpy as np
import logging
import casadi as cs
from horizon.transcriptions.transcriptor import Transcriptor


nodes = 10
dt = 0.01
prb = Problem(nodes)
x = prb.createStateVariable('x', 1)
u = prb.createInputVariable('u', 1)

prb.setDynamics(x)
prb.setDt(dt)

cost1 = prb.createIntermediateConstraint('const', x - u)

cost1 = prb.createIntermediateCost('cost', x - x.getVarOffset(-1) - x.getVarOffset(-2) + u, nodes=1)

solver = Solver.make_solver('ipopt', prb)

print(cost1.getFunction())
print(cost1.getImpl())

exit()
sol = solver.solve()
print(sol)


