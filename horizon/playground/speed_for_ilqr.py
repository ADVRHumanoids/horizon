from horizon import problem
from horizon import solvers
import casadi as cs
prb = problem.Problem(10)

dt = 0.001
use_ms = True
solver_type = 'ilqr'

# Create problem STATE variables
q = prb.createStateVariable("q", 1)
q_dot = prb.createStateVariable("q_dot", 1)
# Create problem CONTROL variables
u = prb.createInputVariable("u", 1)

# Create dynamics
prb.setDynamics(cs.vertcat(q, q_dot))
prb.setDt(dt)

# Limits

# Constraints
prb.createFinalCost("up", q)

# Creates problem
solver = solvers.Solver.make_solver(solver_type, prb)

# solver.plot_iter = True
solver.max_iter = 5000
solver.solve()

