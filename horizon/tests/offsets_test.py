from horizon.problem import Problem
from horizon.solvers import Solver
import pprint
import numpy as np
import logging
import casadi as cs
from horizon.transcriptions.transcriptor import Transcriptor


n_nodes = 10
dt = 0.01
prb = Problem(n_nodes, receding=True, casadi_type=cs.SX)
x = prb.createStateVariable('x', 1)
u = prb.createInputVariable('u', 1)

# x_offset = x.getVarOffset(-9)
# cost_strange = prb.createCost('cost_strange', x_offset, nodes=[9])
# print(cost_strange.getFunction())
# print(cost_strange.getImpl())
#
constr1 = prb.createIntermediateConstraint('const', x - u)

prb.setDynamics(x)
prb.setDt(dt)

cost_strange = prb.createIntermediateCost('cost_strange', x + u, nodes=[2, 5])
cost_normal = prb.createIntermediateCost('cost_normal', x - x.getVarOffset(-1) - x.getVarOffset(-2) + u, nodes=[2, 5])
# constr1 = prb.createIntermediateConstraint('constr', x - x.getVarOffset(-1) - x.getVarOffset(-2) + u, nodes=[2, 3])


print(cost_strange.getFunction())
print(cost_strange.getImpl())
print("=====================")
print(cost_normal.getFunction())
print(cost_normal.getImpl())
#
# print(constr1.getFunction())
# print(constr1.getImpl())

solver = Solver.make_solver('ipopt', prb)


exit()

sol = solver.solve()
print(sol)


