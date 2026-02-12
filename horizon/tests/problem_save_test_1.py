from horizon.problem import Problem
import numpy as np
import casadi as cs
from horizon.solvers import Solver
from horizon.transcriptions.transcriptor import Transcriptor
import yaml

N = 10
nodes_vec = np.array(range(N + 1))  # nodes = 10
dt = 0.01
prb = Problem(N, receding=True, casadi_type=cs.SX)

prb.setDt(dt)

x1 = prb.createStateVariable('x1', 2)
x2 = prb.createStateVariable('x2', 2)
y = prb.createInputVariable('y', 2)

x1.setBounds([1, 0], [1, 0], nodes=0)
x2.setBounds([1, 2], [1, 2], nodes=0)
y.setBounds([50, 10], [50, 10], nodes=0)
y.setBounds([20, 20], [20, 20], nodes=9)

constr1 = prb.createFinalConstraint('constr1', x1[0] - x2[1])
constr2 = prb.createIntermediateConstraint('constr2', x1[1] - 5)
# cost1 = prb.createIntermediateCost('cost1', y[1] - x1[0])
cost2 = prb.createIntermediateResidual('cost2', y[1] - x1[0])
cost3 = prb.createIntermediateCost('cost3', cs.sumsqr(y))


prb.setDynamics(cs.vertcat(x1, x2 - y))


# print(cost1.getFunction())
# print(cost1.getDim())

ilqrsol = Solver.make_solver('ilqr', prb, opts={'ilqr.enable_gn': True, 'ilqr.max_iter': 1}) #opts={'max_iter': 1, 'ilqr.integrator': 'RK4'}
ilqrsol.solve()

np.set_printoptions(suppress=True, precision=4, linewidth=2000)
for item, sol in ilqrsol.getSolutionDict().items():
    if item != 'x_opt' or item != 'u_opt':
        print(item, sol.shape)
        print(sol)

# solver with sqp or ipopt need a dynamic constraint

#
# exit()
data = prb.save()

with open("test_problem_save_1.yaml", "w") as file:
    yaml.dump(data, file)

exit()