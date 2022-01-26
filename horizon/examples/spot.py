import time

from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer
from horizon.transcriptions import integrators
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.ros.replay_trajectory import *
from horizon.solvers import solver
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os, rospkg, argparse
from scipy.io import loadmat
from itertools import filterfalse

parser = argparse.ArgumentParser(
    description='cart-pole problem: moving the cart so that the pole reaches the upright position')
parser.add_argument('--replay', help='visualize the robot trajectory in rviz', action='store_true')
args = parser.parse_args()

rviz_replay = False
resampling = False
plot_sol = True
action = 'leap' #  'wheelie' #  # 'jump_up' # 'jump_forward' # 'jump_on_wall' # 'leap'
solver_type = 'ipopt'  # 'ipopt', 'ilqr'

if args.replay:
    from horizon.ros.replay_trajectory import *
    import roslaunch, rospkg, rospy

    rviz_replay = True
    plot_sol = False

r = rospkg.RosPack()
path_to_examples = r.get_path('horizon_examples')

# mat storer
file_name = os.path.splitext(os.path.basename(__file__))[0]
ms = mat_storer.matStorer(path_to_examples + f'/mat_files/{file_name}.mat')

# options
transcription_method = 'multiple_shooting'
transcription_opts = dict(integrator='RK4')
load_initial_guess = False
ilqr_plot_iter = False

tf = 2.5
n_nodes = 50

disp = [0., 0., 0., 0., 0., 0., 1.]

if action == 'jump_forward' or action == 'leap':
    disp[0] = 2 # [m]
if action == 'jump_twist':
    disp[3:7] = [0, 0, 0.8509035, 0.525322]

if action == 'wheelie':
    node_action = (20, n_nodes)
elif action == 'leap':
    node_action = [(15, 35), (30, 45)]
else:
    node_action = (20, 40)


# load urdf
urdffile = os.path.join(path_to_examples, 'urdf', 'spot.urdf')
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

# joint names
joint_names = kindyn.joint_names()
if 'universe' in joint_names: joint_names.remove('universe')
if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')

# parameters
n_c = 4
n_q = kindyn.nq()
n_v = kindyn.nv()
n_f = 3
dt = tf / n_nodes

# define dynamics
prb = problem.Problem(n_nodes)
q = prb.createStateVariable('q', n_q)
q_dot = prb.createStateVariable('q_dot', n_v)
q_ddot = prb.createInputVariable('q_ddot', n_v)
f_list = [prb.createInputVariable(f'f{i}', n_f) for i in range(n_c)]
x, x_dot = utils.double_integrator_with_floating_base(q, q_dot, q_ddot)
prb.setDynamics(x_dot)
prb.setDt(dt)
# contact map
contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
contact_map = dict(zip(contacts_name, f_list))

# import initial guess if present
if load_initial_guess:
    prev_solution = ms.load()
    q_ig = prev_solution['q']
    q_dot_ig = prev_solution['q_dot']
    q_ddot_ig = prev_solution['q_ddot']
    f_ig_list = list()
    for f in f_list:
        f_ig_list.append(prev_solution[f'f{i}'])

# initial state and initial guess
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0.0, 0.9, -1.5238505,
                   0.0, 0.9, -1.5202315,
                   0.0, 0.9, -1.5300265,
                   0.0, 0.9, -1.5253125])

q.setBounds(q_init, q_init, 0)
q_dot.setBounds(np.zeros(n_v), np.zeros(n_v), 0)

q.setInitialGuess(q_init)

for f in f_list:
    f.setInitialGuess([0, 0, 55])

# transcription
if solver_type != 'ilqr':
    th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

# dynamic feasibility
id_fn = kin_dyn.InverseDynamics(kindyn, contact_map.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
tau = id_fn.call(q, q_dot, q_ddot, contact_map)
prb.createIntermediateConstraint("dynamic_feasibility", tau[:6])

# final velocity is zero
prb.createFinalConstraint('final_velocity', q_dot)


# contact handling
k_all = range(1, n_nodes + 1)
if action == 'leap':
    list_swing = [list(range(*n_range)) for n_range in node_action]
    k_swing = [item for sublist in list_swing for item in sublist]
    k_swing_front = list(range(*[node for node in node_action[0]]))
    k_swing_hind = list(range(*[node for node in node_action[1]]))
    k_stance_front = list(filterfalse(lambda k: k in k_swing_front, k_all))
    k_stance_hind = list(filterfalse(lambda k: k in k_swing_hind, k_all))

else:
    k_swing = list(range(*[node for node in node_action]))

k_stance = list(filterfalse(lambda k: k in k_swing, k_all))

# list of lifted legs
lifted_legs = ['lf_foot', 'rf_foot']

if action != 'wheelie' and action != 'jump_on_wall':
    lifted_legs.extend(['lh_foot', 'rh_foot'])

q_final = q_init
q_final[:3] = q_final[:3] + disp[:3]
q_final[3:7] = disp[3:7]

def barrier(x):
    return cs.if_else(x > 0, 0, x ** 2)

for frame, f in contact_map.items():
    nodes_stance = k_stance if frame in lifted_legs else k_all

    if action == 'leap':
        nodes_stance = k_stance_front if frame in ['lf_foot', 'rf_foot'] else k_stance_hind
        nodes_swing = k_swing_front if frame in ['lf_foot', 'rf_foot'] else k_swing_hind

    FK = cs.Function.deserialize(kindyn.fk(frame))
    DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
    DDFK = cs.Function.deserialize(kindyn.frameAcceleration(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))

    p = FK(q=q)['ee_pos']
    p_start = FK(q=q_init)['ee_pos']
    v = DFK(q=q, qdot=q_dot)['ee_vel_linear']
    a = DDFK(q=q, qdot=q_dot)['ee_acc_linear']

    prb.createConstraint(f"{frame}_vel", v, nodes=nodes_stance)
    prb.createIntermediateCost(f'{frame}_fn', barrier(f[2] - 25.0))


    if action == 'leap':
        gc = prb.createConstraint(f'{frame}_ground', p[2], nodes=nodes_swing)
        gc.setLowerBounds(p_start[2])

    if frame in lifted_legs:
        if action == 'jump_on_wall':
            mu = 1
            p_goal = p_start + [0.3, 0., 0.8]
            rot = -np.pi / 2.
            R_wall = np.array([[np.cos(rot), 0, np.sin(rot)],
                              [0,            1,           0],
                              [-np.sin(rot), 0, np.cos(rot)]])

            fc, fc_lb, fc_ub = kin_dyn.linearized_friciton_cone(f, mu, R_wall)
            prb.createIntermediateConstraint(f"{frame}_friction_cone_wall", fc,
                                             nodes=range(k_swing[-1], n_nodes), bounds=dict(lb=fc_lb, ub=fc_ub))
            prb.createFinalConstraint(f"lift_{frame}_leg", p - p_goal)


# swing force is zero
for leg in lifted_legs:
    if action == 'leap':
        nodes = k_swing_front if leg in ['lf_foot', 'rf_foot'] else k_swing_hind
    else:
        nodes = k_swing

    fzero = np.zeros(n_f)
    contact_map[leg].setBounds(fzero, fzero, nodes=nodes)


if action != 'wheelie' and action != 'jump_on_wall':
    prb.createFinalConstraint(f"final_nominal_pos", q - q_final)

prb.createResidual("min_q_dot", q_dot)
for f in f_list:
    prb.createIntermediateResidual(f"min_{f.getName()}", cs.sqrt(3e-3) * f)

# =============
# SOLVE PROBLEM
# =============
opts = {'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 0.001,
        'ipopt.max_iter': 2000,
        # 'ipopt.linear_solver': 'ma57',
        'ilqr.integrator': 'RK4',
        'ilqr.closed_loop_forward_pass': True,
        'ilqr.line_search_accept_ratio': 1e-9,
        'ilqr.svd_threshold': 1e-12,
        }

solv = solver.Solver.make_solver(solver_type, prb, opts)

try:
    solv.set_iteration_callback()
    solv.plot_iter = ilqr_plot_iter
except:
    pass

solv.solve()

if solver_type == 'ilqr':
    solv.print_timings()

solution = solv.getSolutionDict()
dt_sol = solv.getDt()
cumulative_dt = np.zeros(len(dt_sol) + 1)
for i in range(len(dt_sol)):
    cumulative_dt[i + 1] = dt_sol[i] + cumulative_dt[i]

solution_constraints_dict = dict()

if isinstance(dt, cs.SX):
    ms.store({**solution, **solution_constraints_dict})
else:
    dt_dict = dict(dt=dt)
    ms.store({**solution, **solution_constraints_dict, **dt_dict})

# ========================================================
if plot_sol:

    hplt = plotter.PlotterHorizon(prb, solution)
    # hplt.plotVariables(show_bounds=True, same_fig=True, legend=False)
    hplt.plotVariables(['f0', 'f1', 'f2', 'f3'], show_bounds=True, gather=2, legend=False)
    # hplt.plotFunctions(show_bounds=True, same_fig=True)
    # hplt.plotFunction('inverse_dynamics', show_bounds=True, legend=True, dim=range(6))

    pos_contact_list = list()
    fig = plt.figure()
    fig.suptitle('Contacts')
    gs = gridspec.GridSpec(2, 2)
    i = 0
    for contact in contacts_name:
        ax = fig.add_subplot(gs[i])
        ax.set_title('{}'.format(contact))
        i += 1
        FK = cs.Function.deserialize(kindyn.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']
        for dim in range(n_f):
            ax.plot(np.atleast_2d(cumulative_dt), np.array(pos[dim, :]), marker="x", markersize=3,
                    linestyle='dotted')  # marker="x", markersize=3, linestyle='dotted'

    plt.figure()
    for contact in contacts_name:
        FK = cs.Function.deserialize(kindyn.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'feet position - plane_xy')
        plt.scatter(np.array(pos[0, :]), np.array(pos[1, :]), linewidth=0.1)

    plt.figure()
    for contact in contacts_name:
        FK = cs.Function.deserialize(kindyn.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'feet position - plane_xz')
        plt.scatter(np.array(pos[0, :]), np.array(pos[2, :]), linewidth=0.1)

    plt.show()
# ======================================================
contact_map = dict(zip(contacts_name, [solution['f0'], solution['f1'], solution['f2'], solution['f3']]))

# resampling
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

if rviz_replay:

    # set ROS stuff and launchfile
    r = rospkg.RosPack()
    path_to_examples = r.get_path('horizon_examples')

    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(uuid, [path_to_examples + "/replay/launch/spot.launch"])
    launch.start()
    rospy.loginfo("'spot' visualization started.")

    if resampling:
        repl = replay_trajectory(dt_res, joint_names, q_res, contact_map_res,
                                 cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)
    else:
        # remember to run a robot_state_publisher
        repl = replay_trajectory(dt, joint_names, solution['q'], contact_map,
                                 cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)

    repl.sleep(1.)
    repl.replay(is_floating_base=True)

else:
    print("To visualize the robot trajectory, start the script with the '--replay")

