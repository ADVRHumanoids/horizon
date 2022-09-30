import time

from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer, collision
from horizon.transcriptions import integrators
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.ros.replay_trajectory import *
from horizon.solvers import solver
import matplotlib.pyplot as plt
import os, math
from itertools import filterfalse
import rospkg

# mat storer
filename_with_ext = __file__
filename, _ = os.path.splitext(filename_with_ext)
ms = mat_storer.matStorer(f'{filename}.mat')

# options
solver_type = 'ilqr'
transcription_method = 'multiple_shooting'
transcription_opts = dict(integrator='RK4')
load_initial_guess = False
tf = 2.0
n_nodes = 100
ilqr_plot_iter = False

# load urdf
urdffile = os.path.join(rospkg.RosPack().get_path('teleop_urdf'), 'urdf', 'teleop_capsules.rviz')
urdf = open(urdffile, 'r').read()
srdffile = os.path.join(rospkg.RosPack().get_path('teleop_srdf'), 'srdf', 'teleop_capsules.srdf')
srdf = open(srdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

# collision
ch = cas_kin_dyn.CollisionHandler(kindyn, srdf)

# joint names
joint_names = kindyn.joint_names()
if 'universe' in joint_names: joint_names.remove('universe')
if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')

# parameters
n_q = kindyn.nq()
n_v = kindyn.nv()
dt = tf/n_nodes

# define dynamics
prb = problem.Problem(n_nodes)
q = prb.createStateVariable('q', n_q)
q_dot = prb.createStateVariable('q_dot', n_v)
q_ddot = prb.createInputVariable('q_ddot', n_v)
x = cs.vertcat(q, q_dot)
x_dot = cs.vertcat(q_dot, q_ddot)
prb.setDynamics(x_dot)

# initial state and initial guess
q_init = np.array([1.0, 1.0, 1.0, 0, 0])
q.setBounds(q_init, q_init, 0)
q_dot.setBounds(np.zeros(n_v), np.zeros(n_v), 0)
q.setInitialGuess(q_init)

# bounds
# q_dot.setBounds(lb=np.full(n_v, -1), ub=np.full(n_v, 1), nodes=range(1, n_nodes))
# q_ddot.setBounds(lb=np.full(n_v, -10), ub=np.full(n_v, 10), nodes=range(1, n_nodes))

# barrier
def barrier(x):
    return cs.if_else(x > 0, 0, x)

# transcription
if solver_type != 'ilqr':
    th = Transcriptor.make_method(transcription_method, prb, dt, opts=transcription_opts)

if solver_type == 'gnsqp':
    def residual_to_cost(r):
        return r
else:
    def residual_to_cost(r):
        return cs.sumsqr(r)

# pick and place task
fk = cs.Function.deserialize(kindyn.fk('TCP'))
p = fk(q=q)['ee_pos']

pf = fk(q=q_init)['ee_pos'].toarray()
prb.createConstraint('place', q - q_init, nodes=[n_nodes])
prb.createConstraint('placev', q_dot, nodes=[n_nodes])

q0 = q_init.copy()
q0[0] *= -1
prb.createConstraint('pick', q - q0, nodes=[n_nodes//2])
prb.createConstraint('pickv', q_dot, nodes=[n_nodes//2])

# regularize
prb.createIntermediateResidual('rega', q_ddot)
prb.createIntermediateResidual('regv', q_dot)
# prb.createIntermediateCost('regq', residual_to_cost(1e-2*(q - q_init)[1:n_q]))

# collision
from urdf_parser_py import urdf as upp
g1 = upp.Cylinder(length=10.0, radius=0.10)
c1 = upp.Collision(geometry=g1, origin=upp.Pose(xyz=[0.70, 0, 0.0], rpy=[0, 0, 0]))
ch.world['world/caps'] = c1 

ch.addShape('mycaps', 'capsule',
{
    'length': [10.0],
    'radius': [0.10],
    'position': [0.70, 0, 0.0],
})

dfn = ch.getDistanceFunction()
d = dfn(q=q)
nd = d.size1()
# prb.createIntermediateConstraint('coll', d, bounds=dict(lb=np.zeros(nd), ub=np.full(nd, np.inf)))
coll_w = prb.createParameter('coll_w', 1)
coll_w.assign(100)
prb.createIntermediateCost('coll', coll_w*residual_to_cost(barrier(d-0.05)))
# prb.createConstraint('coll', cs.sum1(barrier(d)), bounds=dict(lb=0, ub=np.inf))

# =============
# SOLVE PROBLEM
# =============
opts = dict()
if solver_type == 'ipopt':
    opts['ipopt.tol'] = 1e-6
    opts['ipopt.constr_viol_tol'] = 1e-6
    opts['ipopt.max_iter'] = 2000
    opts['ipopt.linear_solver'] = 'ma57'

if solver_type == 'ilqr':
    opts = {'ilqr.max_iter': 1000,
        'ilqr.integrator': 'RK4', 
        'ilqr.closed_loop_forward_pass': True,
        'ilqr.line_search_accept_ratio': 1e-9,
        'ilqr.svd_threshold': 1e-12,
        'ilqr.kkt_decomp_type': 'ldlt',
        'ilqr.constr_decomp_type': 'qr',
        'ilqr.codegen_enabled': True,
        'ilqr.codegen_workdir': '/tmp/collision_example',
        }

if solver_type == 'gnsqp':
    qp_solver = 'osqp'
    if qp_solver == 'osqp':
        opts['gnsqp.qp_solver'] = 'osqp'
        opts['warm_start_primal'] = True
        opts['warm_start_dual'] = True
        opts['merit_derivative_tolerance'] = 1e-10
        opts['constraint_violation_tolerance'] = 1e-11
        opts['osqp.polish'] = True # without this
        opts['osqp.delta'] = 1e-9 # and this, it does not converge!
        opts['osqp.verbose'] = False
        opts['osqp.rho'] = 0.02
        opts['osqp.scaled_termination'] = False
    if qp_solver == 'qpoases': #does not work!
        opts['gnsqp.qp_solver'] = 'qpoases'
        opts['sparse'] = True
        opts["enableEqualities"] = True
        opts["initialStatusBounds"] = "inactive"
        opts["numRefinementSteps"] = 0
        opts["enableDriftCorrection"] = 0
        opts["terminationTolerance"] = 10e9 * 1e-7
        opts["enableFlippingBounds"] = False
        opts["enableNZCTests"] = False
        opts["enableRamping"] = False
        opts["enableRegularisation"] = True
        opts["numRegularisationSteps"] = 2
        opts["epsRegularisation"] = 5. * 10e3 * 1e-7
        opts['hessian_type'] =  'posdef'
        #opts['printLevel'] = 'high'
        opts['linsol_plugin'] = 'ma57'

solver = solver.Solver.make_solver(solver_type, prb, dt, opts)

try:
    solver.set_iteration_callback()
    solver.plot_iter = ilqr_plot_iter 
except:
    pass

t = time.time()
solver.solve()
elapsed = time.time() - t
print(f'solved in {elapsed} s')

if solver_type == 'gnsqp':
    print(f"mean Hessian computation time: {sum(solver.getHessianComputationTime())/len(solver.getHessianComputationTime())}")
    print(f"mean QP computation time: {sum(solver.getQPComputationTime())/len(solver.getQPComputationTime())}")
    print(f"mean Line Search computation time: {sum(solver.getLineSearchComputationTime()) / len(solver.getLineSearchComputationTime())}")


try:
    solver.print_timings()
except:
    pass

solution = solver.getSolutionDict()
solution_constraints_dict = dict()
coll = ch.get_function()(solution['q']).toarray()

plt.plot(coll.T)
plt.show()

if isinstance(dt, cs.SX):
    ms.store({**solution, **solution_constraints_dict})
else:
    dt_dict = dict(dt=dt)
    ms.store({**solution, **solution_constraints_dict, **dt_dict})


# ========================================================

# resampling
resampling = False
if resampling:

    if isinstance(dt, cs.SX):
        dt_before_res = solution['dt'].flatten()
    else:
        dt_before_res = dt

    dt_res = 0.001
    dae = {'x': x, 'p': q_ddot, 'ode': x_dot, 'quad': 1}
    q_res, qdot_res, qddot_res, contact_map_res, tau_res = resampler_trajectory.resample_torques(
        solution["q"], solution["q_dot"], solution["q_ddot"], dt_before_res, dt_res, dae, contact_map,
        kindyn,
        cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

    repl = replay_trajectory(dt_res, joint_names, q_res, contact_map_res, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)
else:
    # remember to run a robot_state_publisher
    repl = replay_trajectory(dt, joint_names, solution['q'], {}, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)

repl.sleep(1.)
repl.replay(is_floating_base=False)

