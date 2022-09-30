from horizon.problem import Problem
from horizon.solvers import Solver
import pprint
import numpy as np
import logging
import casadi as cs
from horizon.transcriptions.transcriptor import Transcriptor

def test_singleParameter():
    nodes = 10
    dt = 0.01
    prb = Problem(nodes)
    x = prb.createStateVariable('x', 6)
    u = prb.createInputVariable('u', 2)
    p = prb.createSingleParameter('p', 6)

    prb.setDynamics(x)
    prb.setDt(dt)
    x.setBounds([1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2])

    constr1 = prb.createIntermediateConstraint('constr', x[2:4] ** 2 + p[2:4] - u)

    constr1.setBounds([1, 1], [1, 1])
    solver = Solver.make_solver('ipopt', prb)

    all_sol = dict()
    for i in range(100):
        p.assign(6 * [2*i])
        solver.solve()
        sol = solver.getSolutionDict()
        all_sol[i] = sol

    pprint.pprint(all_sol)

def test_parameters():
    #  check if bounds are correct when setting them AFTER building the problem
    nodes = 10
    dt = 0.01
    prb = Problem(nodes)
    x = prb.createStateVariable('x', 6)
    u = prb.createInputVariable('u', 2)
    p = prb.createParameter('p', 1)
    prb.setDynamics(x)
    prb.setDt(dt)
    constr1 = prb.createIntermediateConstraint('constr', x[2] ** 2 + p - u[1])

    constr1.setBounds(1, 1)

    solver = Solver.make_solver('ipopt', prb)

    x.setBounds([1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2])

    cos_fun = np.cos(np.linspace(0, 2*np.pi, 11))

    for n in range(nodes+1):
        p.assign(cos_fun[n], n)

    solver.solve()
    sol = solver.getSolutionDict()

def test_singleVariable():
    #  check if bounds are correct when setting them AFTER building the problem
    nodes = 10
    dt = 0.01
    prb = Problem(nodes)
    x = prb.createStateVariable('x', 6)
    u = prb.createSingleVariable('u', 2)
    prb.setDynamics(x)
    prb.setDt(dt)
    constr1 = prb.createIntermediateConstraint('constr', x[2] ** 2 + u[1:2])

    constr1.setBounds(1, 1)

    solver = Solver.make_solver('ipopt', prb)

    x.setBounds([1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2])

    cos_fun = np.cos(np.linspace(0, 2*np.pi, 11))

    solver.solve()
    sol = solver.getSolutionDict()

def test_constraintBounds():
    #  check if bounds are correct when setting them AFTER building the problem
    nodes = 10
    dt = 0.01
    prb = Problem(nodes)
    x = prb.createStateVariable('x', 2)
    u = prb.createSingleVariable('u', 2)
    prb.setDynamics(x)
    prb.setDt(dt)
    constr1 = prb.createIntermediateConstraint('constr1', x+u, bounds=dict(lb=[0, 0]))
    constr2 = prb.createIntermediateConstraint('constr2', x + u, bounds=dict(ub=[0, 0]))
    constr3 = prb.createIntermediateConstraint('constr3', x + u, bounds=dict(lb=[0, 0], ub=[0, 0]))

    solver = Solver.make_solver('ipopt', prb)
    solver.solve()
    sol = solver.getSolutionDict()

def test_intermediateConstraint():
    nodes = 10
    dt = 0.01
    prb = Problem(nodes, logging_level=logging.DEBUG)
    x = prb.createStateVariable('x', 2)
    prb.setDynamics(x)
    prb.setDt(dt)

    cnsrt = prb.createIntermediateConstraint('cnsrt', x)

    solver = Solver.make_solver('ipopt', prb)
    solver.solve()
    sol = solver.getSolutionDict()

def test_variables():
    nodes = 10
    prb = Problem(nodes, logging_level=logging.DEBUG)
    x = prb.createStateVariable('x', 2)
    v = prb.createStateVariable('v', 2)
    k = prb.createVariable('k', 2, range(2, 8))
    u = prb.createInputVariable('u', 2)
    t_tot = prb.createVariable('t', 2)
    p = prb.createSingleVariable('p', 2)
    t_tot.setBounds([100, 100], [100, 100])
    prb.setDynamics(cs.vertcat(x, v))

def test_prev():
    nodes = 10
    prb = Problem(nodes, logging_level=logging.DEBUG)
    x = prb.createStateVariable('x', 2)
    v = prb.createStateVariable('v', 2)
    k = prb.createVariable('k', 2, range(2, 8))
    u = prb.createInputVariable('u', 2)
    t_tot = prb.createVariable('t', 2)
    t_tot.setBounds([100, 100], [100, 100])

    cnsrt1 = prb.createConstraint('cnsrt1', x-k, nodes=range(2, 8))
    cnsrt2 = prb.createConstraint('cnsrt2', x - t_tot)
    cnsrt2.setBounds([100, 100], [100, 100])
    xprev = x.getVarOffset(-1)
    print(f'id xprev: {id(xprev)}')
    xprev_copy = x.getVarOffset(-1)
    print(f'id xprev once called again (should be the same as the previous): {id(xprev_copy)}')
    xnext = x.getVarOffset(+1)

    opts = {'ipopt.tol': 1e-4,
            'ipopt.max_iter': 2000}

    state = prb.getState()
    state_prev = state.getVarOffset(-1)

    print(state.getVars())
    state.setBounds([0, 0, 0, 0], [0, 0, 0, 0], 2)
    print(state.getBounds(2))


    dt = 0.01
    state_dot = cs.vertcat(v, u)
    prb.setDynamics(state_dot)
    prb.setDt(dt)
    th = Transcriptor.make_method('multiple_shooting', prb, opts=dict(integrator='RK4'))
    solver = Solver.make_solver('ipopt', prb, opts)
    solver.solve()
    sol = solver.getSolutionDict()

def test_bounds_input():
    nodes = 10
    dt = 0.01
    prb = Problem(nodes, logging_level=logging.DEBUG)
    x = prb.createStateVariable('x', 2)
    y = prb.createStateVariable('y', 2)
    z = prb.createInputVariable('z', 2)
    cnsrt = prb.createConstraint('cnsrt', x + y, range(4, 10))
    prb.setDynamics(cs.vertcat(x,y))
    prb.setDt(dt)
    x.setBounds([2, 2], [2, 2])
    cnsrt.setBounds([12, 12], [12, 12], list(range(4, 10)))

    solver = Solver.make_solver('ipopt', prb)
    solver.solve()

def test_bounds_2():
    nodes = 8
    dt = 0.01
    prb = Problem(nodes)
    x = prb.createStateVariable('x', 2)
    y = prb.createStateVariable('y', 2)
    cnsrt = prb.createConstraint('cnsrt', x+y)
    prb.setDynamics(cs.vertcat(x,y))
    prb.setDt(dt)
    cnsrt.setBounds([12, 12],[12, 12], 4)
    solver = Solver.make_solver('ipopt', prb)
    solver.solve()

def test_boundsarray():
    nodes = 8
    dt = 0.01
    prb = Problem(nodes)
    x = prb.createStateVariable('x', 1)
    y = prb.createStateVariable('y', 1)
    prb.setDynamics(cs.vertcat(x, y))
    cnsrt = prb.createConstraint('cnsrt', x + y)

    bounds = cnsrt.getBounds()
    print(bounds[0].shape, bounds[0])
    print(bounds[0].shape, bounds[1])

    lb = cnsrt.getLowerBounds()
    print(lb.shape)


def test_view():
    nodes = 10
    prb = Problem(nodes)
    x = prb.createStateVariable('x', 5)
    y = prb.createInputVariable('y', 5)
    z = prb.createVariable('z', 5, [3, 4, 5])
    p = prb.createParameter('p', 4)
    b = prb.createSingleParameter('b', 6)

    x_prev = x.getVarOffset(-2)

    cnsrt1 = prb.createIntermediateConstraint('cnsrt1', x + y)
    cnsrt2 = prb.createConstraint('cnsrt2', x[0:4] + p)
    cnsrt3 = prb.createConstraint('cnsrt3', z + b[1:6], [3, 4, 5])

    state = prb.getState()
    state[0][2].setLowerBounds([2])
    print(state[0].getLowerBounds())
    print('==============================')

    print(f'type: {type(x_prev[2:5])}')
    print('==============================')

    x_slice = x[2:5]
    print(f'{type(x)} --> {type(x_slice)}')
    x_slice.setLowerBounds([10, 10, 10])
    print(x.getLowerBounds())
    print('==============================')

    y_slice = y[2:5]
    print(f'{type(y)} --> {type(y_slice)}')
    y_slice.setLowerBounds([6, 6, 6])
    print(y.getLowerBounds())
    print('==============================')

    z_slice = z[2]
    print(f'{type(z)} --> {type(z_slice)}')
    z_slice.setLowerBounds(23)
    print(z.getLowerBounds())
    print('==============================')

    p_slice = p[2:4]
    print(f'{type(p)} --> {type(p_slice)}')
    p_slice.assign([52, 52])
    print(p.getValues())
    print('==============================')

    b_slice = b[2:5]
    print(f'{type(b)} --> {type(b_slice)}')
    b_slice.assign([12, 12, 12])
    print(b.getValues())


if __name__ == '__main__':
    test_singleParameter()
    test_parameters()
    test_singleVariable()
    test_constraintBounds()
    test_intermediateConstraint()
    test_variables()
    test_prev()
    test_bounds_input()
    test_bounds_2()
    test_boundsarray()
    test_view()


#
#
# print('===PICKLING===')
# prb = prb.serialize()
# prb_serialized = pickle.dumps(prb)
#
# print('===DEPICKLING===')
# prb_new = pickle.loads(prb_serialized)
# prb_new.deserialize()
#
# sv_new = prb_new.getStateVariables()
# cnstr_new = prb_new.getConstraints()
#
# # these two are different
# print('x', x)
# print('new x', sv_new['x'])
#

# print('===PICKLING===')
#
# prb = prb.serialize()
# prb_serialized = pickle.dumps(prb)
#
#
# print('===DEPICKLING===')
# prb_new = pickle.loads(prb_serialized)
# prb_new.deserialize()
#
# prb_new.createProblem()
# print(prb_new.prob)
#
# exit()
# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================

# nodes = 8
# prb = Problem(nodes, logging_level=logging.INFO)
# x = prb.createStateVariable('x', 2)
# y = prb.createStateVariable('y', 2)
#
# x.setBounds([-2, -2], [2, 2])

# todo this is allright but I have to remember that if I update the nodes (from 3 to 6 for example) i'm not updating the constraint nodes
# todo so if it was active on all the node before, then it will be active only on the node 1, 2, 3 (not on 4, 5, 6)


# scoping_node = nodes
# print('var at nodes {}  BEFORE creating the problem: '.format(scoping_node), prb.scopeNodeVars(scoping_node))
# print('number of nodes of {}: {}'.format(x, x.getNNodes()))
# print('bounds of function {} at node {} are: {}'.format(x, scoping_node, x.getBounds(scoping_node)))

# print('getVarImplList way before:', prb.state_var_container.getVarImplList())
# danieli = prb.createConstraint('danieli', x + y)
# sucua = prb.createCostFunction('sucua', x * y, nodes=list(range(3, 15)))
# pellico = prb.createCostFunction('pellico', x - y, nodes=[0, 4, 6])

# danieli.setBounds(lb=[-1, -1], ub=[1, 1], nodes=3)

# prb.createProblem({"nlpsol.ipopt": True})

# for i in range(nodes + 1):
#     print(x.getBounds(i))

# print('var at nodes {}  AFTER creating the problem: '.format(scoping_node), prb.scopeNodeVars(scoping_node))
# print('getVarImplList before:', prb.state_var_container.getVarImplList())
# new_n_nodes = 5
# print('================== Changing n. of nodes to {} =================='.format(new_n_nodes))
# prb.setNNodes(new_n_nodes)
# scoping_node = 8
# print('var at nodes {} AFTER changing the n. of nodes but BEFORE rebuilding: {}'.format(scoping_node, prb.scopeNodeVars(scoping_node)))
# print('number of nodes of {}: {}'.format(x, x.getNNodes()))
# print('bounds of function {} at node {} are: {}'.format(x, scoping_node, x.getBounds(scoping_node)))
# print('getVarImplList after but before create:', prb.state_var_container.getVarImplList())
# prb.createProblem()

# todo check why this is so
# print('after:', prb.state_var_container.getVarImplList())

# scoping_node = 8
# print('var at nodes {} AFTER changing the n. of nodes but AFTER rebuilding: {}'.format(scoping_node, prb.scopeNodeVars(scoping_node))) # should not work
# print('number of nodes of {}: {}'.format(x, x.getNNodes()))
# print('bounds of function {} at node {} are: {}'.format(x, scoping_node, x.getBounds(scoping_node)))
# x.setBounds(10)
# danieli.setNodes([1,6])
# prb.scopeNodeVars(2)

# x.setBounds([2, 8], [2, 8], 5)
#
# for i in range(new_n_nodes + 1):
#     print(x.getBounds(i))

# todo what do I do?
# is it better to project the abstract variable as soon as it is created, to set the bounds and everything?
# or is it better to wait for the buildProblem to generate the projection of the abstract value along the horizon line?
