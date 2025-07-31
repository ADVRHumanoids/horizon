from horizon.problem import Problem
import numpy as np
import casadi as cs
import yaml
from collections import OrderedDict

N = 10
nodes_vec = np.array(range(N + 1))  # nodes = 10
dt = 0.01
prb = Problem(N, receding=True, casadi_type=cs.SX)

prb.setDt(dt)

x1 = prb.createStateVariable('x1', 2)
x2 = prb.createStateVariable('x2', 2)
y = prb.createInputVariable('y', 2)

par1 = prb.createParameter('par1', 3)

constr1 = prb.createIntermediateConstraint('constr1', y[1] - x1[1])
constr2 = prb.createIntermediateConstraint('constr2', y[1] - x1[0])
constr3 = prb.createIntermediateConstraint('constr3', y - x2)
constr4 = prb.createIntermediateConstraint('constr4', y[1] - x1[0] / par1[2])
constr5 = prb.createFinalConstraint('constr5', x1 - par1[2:])
constr6 = prb.createIntermediateConstraint('constr6', y[1] - x1[0] / par1[2], nodes=[0, 5, 9])


prb.setDynamics(cs.vertcat(x1, x2))


# constr1.setBounds([-np.inf, -np.inf], [np.inf, np.inf], nodes=4)
x1.setBounds([0, 0], [0, 0])
x2.setBounds([0, 1], [0, 1], nodes=4)

# print(mimmo.getNodes())
# print(mimmo.getBounds())

par_vals = np.matrix([[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
                      [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
                      [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]])
par1.assign(par_vals)


constr1_vals_lb = np.matrix([[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]])
constr1_vals_ub = np.matrix([[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]])
constr1.setBounds(constr1_vals_lb, constr1_vals_ub)

print(constr1.getNodes())
print(constr1.getLowerBounds())

constr1.setNodes([0, 1, 2, 3, 4])


print(constr1.getNodes())
print(constr1.getLowerBounds())

exit()
data = prb.save()

with open("data.yaml", "w") as file:
    yaml.dump(data, file)

exit()
# ============================================================
# ============================================================
# ============================================================

# N = 10
# nodes_vec = np.array(range(N + 1))  # nodes = 10
# dt = 0.01
# prb = Problem(N, receding=True, casadi_type=cs.SX)
# x = prb.createStateVariable('x', 2)
# y = prb.createInputVariable('y', 2)
# dan = prb.createParameter('dan', 2)
# x.setBounds([-2, -2], [2, 2])
# y.setBounds([-5, -5], [5, 5])
# prb.createCost('cost_x', x)
# mimmo = prb.createIntermediateConstraint('cost_y', y - dan)
#
# print(mimmo.getUpperBounds())
# print(mimmo.getLowerBounds())
# exit()
#
# print(dan.getValues())
# print(dan.getNodes())
# dan.assign([[2, 3, 4]], [])
# print(dan.getValues())
# exit()
# # for i in range(500):
# #     cnsrt = prb.createConstraint(f'cnsrt_{i}', x - i * y, nodes=[])
# #     print(cnsrt.getBounds())
#
# prb.setDynamics(x)
# prb.setDt(dt)
#
# opts = dict()
# opts['ipopt.linear_solver'] = 'ma27'
# # opts['ipopt.check_derivatives_for_naninf'] = 'yes'
# # opts['ipopt.jac_c_constant'] = 'yes'
# # opts['ipopt.jac_d_constant'] = 'yes'
# # opts['ipopt.hessian_constant'] = 'yes'
# solv = Solver.make_solver('ipopt', prb, opts)
# tic = time.time()
# solv.solve()
# toc = time.time() - tic
# print(toc)
# print(solv.getSolutionDict()['x'])
# exit()