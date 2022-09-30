#!/usr/bin/env python3
#     <joint name="LHipLat" value="-0.0"/>
#     <joint name="LHipSag" value="-0.363826"/>
#     <joint name="LHipYaw" value="0.0"/>
#     <joint name="LKneePitch" value="0.731245"/>
#     <joint name="LAnklePitch" value="-0.307420"/>
#     <joint name="LAnkleRoll" value="0.0"/>
#     <joint name="RHipLat" value="0.0"/>
#     <joint name="RHipSag" value="-0.363826"/>
#     <joint name="RHipYaw" value="0.0"/>
#     <joint name="RKneePitch" value="0.731245"/>
#     <joint name="RAnklePitch" value="-0.307420"/>
#     <joint name="RAnkleRoll" value="-0.0"/>
#     <joint name="WaistLat" value="0.0"/>
#     <joint name="WaistYaw" value="0.0"/>
#     <joint name="LShSag" value="0.959931"/>
#     <joint name="LShLat" value="0.007266"/>
#     <joint name="LShYaw" value="-0.0"/>
#     <joint name="LElbj" value="-1.919862"/>
#     <joint name="LForearmPlate" value="0.0"/>
#     <joint name="LWrj1" value="-0.523599"/>
#     <joint name="LWrj2" value="-0.0"/>
#     <joint name="RShSag" value="0.959931"/>
#     <joint name="RShLat" value="-0.007266"/>
#     <joint name="RShYaw" value="-0.0"/>
#     <joint name="RElbj" value="-1.919862"/>
#     <joint name="RForearmPlate" value="0.0"/>
#     <joint name="RWrj1" value="-0.523599"/>
#     <joint name="RWrj2" value="-0.0"/>


from horizon import problem
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers.solver import Solver
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import casadi as cs
import os, math
from itertools import filterfalse

# flag to activate the replay on rviz
rviz_replay = True
# flag to activate the resampling of the solution to an higher frequency
resampling = True
# flag to plot the solution
plot_sol = True
# flag to load initial guess (if a previous solution is present)
load_initial_guess = False

# get path to the examples folder and temporary add it to the environment
path_to_examples = os.path.abspath(__file__ + "/../../../examples")
os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples

# mat storer
file_name = os.path.splitext(os.path.basename(__file__))[0]
# if the folder /mat_files does not exist, create it
save_dir = path_to_examples + '/mat_files'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
ms = mat_storer.matStorer(path_to_examples + f'/mat_files/{file_name}.mat')

# options for horizon transcription
solver = 'ipopt'
transcription_method = 'multiple_shooting'  # can choose between 'multiple_shooting' and 'direct_collocation'
transcription_opts = dict(integrator='RK4')  # integrator used by the multiple_shooting


# Create CasADi interface to Pinocchio with casadi_kin_dyn
urdffile = os.path.join(path_to_examples, 'urdf', 'cogimon.urdf')
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

# joint names
joint_names = kindyn.joint_names()
if 'universe' in joint_names:
    joint_names.remove('universe')
if 'reference' in joint_names:
    joint_names.remove('reference')
if 'floating_base_joint' in joint_names:
    joint_names.remove('floating_base_joint')

# Optimization parameters
n_nodes = 50
node_action = (20, 30)
tf = 2.5
# number of contacts (4 feet of spot)
n_c = 2
# Get dimension of pos and vel from urdf
n_q = kindyn.nq()
n_v = kindyn.nv()
# Dimension of the forces (x,y,z) [no torques]
n_f = 6

dt = tf / n_nodes

stance_orientation = [0., 0., 0., 0.] # [0, 0, 0.8509035, 0.525322] # equivalent to 2/3 * pi
# Contact links name
contacts_name = ['l_sole', 'r_sole']

# Create horizon problem
prb = problem.Problem(n_nodes)

# creates the variables for the problem
# design choice: position and the velocity as state and acceleration as input

# STATE variables
q = prb.createStateVariable('q', n_q)
q_dot = prb.createStateVariable('q_dot', n_v)
# CONTROL variables
q_ddot = prb.createInputVariable('q_ddot', n_v)
# set the dt as a variable: the optimizer can choose the dt in between each node
# dt = prb.createInputVariable("dt", 1)
f_list = [prb.createInputVariable(f'force_{i}', n_f) for i in contacts_name]

# Set the contact map: couple the link names at each force
contact_map = dict(zip(contacts_name, f_list))

# logic to load a previous solution as the initial guess of the optimization problem.
# starting the problem with a "good" guess GREATLY simplify the solver's work, reducing the computation time.
# the solution is stored in a .mat file. the 'mat_storer' module load the solution and set each variable to it for each node.
if load_initial_guess:
    prev_solution = ms.load()
    q_ig = prev_solution['q']
    q_dot_ig = prev_solution['q_dot']
    q_ddot_ig = prev_solution['q_ddot']
    f_ig_list = [prev_solution[f.getName()] for f in f_list]
    # dt_ig = prev_solution['dt']


# Creates double integrator taking care of the floating base part
x_dot = utils.double_integrator_with_floating_base(q, q_dot, q_ddot)

# Set dynamics of the system and the relative dt
prb.setDynamics(x_dot)
prb.setDt(dt)

# ===================== Set BOUNDS  ===================================
# joint limits
# q_min = [-10., -10., -10., -1., -1., -1., -1.]  # floating base
# q_min.extend(kindyn.q_min()[7:])
# q_min = np.array(q_min)

# q_max = [10., 10., 10., 1., 1., 1., 1.]  # floating base
# q_max.extend(kindyn.q_max()[7:])
# q_max = np.array(q_max)

# exit()
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   -0.0, -0.363826, 0.0, 0.731245, -0.307420, 0.0,
                    0.0, -0.363826, 0.0, 0.731245, -0.307420, -0.0,
                    0.0, 0.0,
                    0.959931, 0.007266, -0.0, -1.919862, 0.0, -0.523599, -0.0,
                    0.959931, -0.007266, -0.0, -1.919862, 0.0, -0.523599, -0.0])

# velocity limits
# q_dot_lim = 100. * np.ones(n_v)
# acceleration limits
# q_ddot_lim = 100. * np.ones(n_v)
# f bounds
f_min = [-100000., -100000., 0., -10000., -10000., -10000.]
f_max = [100000., 100000., 100000., 10000., 10000., 10000.]

# time bounds
# dt_min = 0.02  # [s]
# dt_max = 0.1  # [s]

# set bounds and of q
# q.setBounds(q_min, q_max)
q.setBounds(q_init, q_init, 0)

# set bounds of q_dot
q_dot_init = np.zeros(n_v)
# q_dot.setBounds(-q_dot_lim, q_dot_lim)
q_dot.setBounds(q_dot_init, q_dot_init, 0)

# set bounds of q_ddot
# q_ddot.setBounds(-q_ddot_lim, q_ddot_lim)

# set bounds of f
[f.setBounds(f_min, f_max) for f in f_list]


# ================== Set TRANSCRIPTION METHOD ===============================
if solver != 'ilqr':
    th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

# ====================== Set CONSTRAINTS ===============================

# the torques are computed using the inverse dynamics, as the input of the problem are the joint acceleration and the forces
id_fn = kin_dyn.InverseDynamics(kindyn, contact_map.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
tau = id_fn.call(q, q_dot, q_ddot, contact_map)

# dynamic feasiblity constraint:
# floating base cannot exert torques (+ torque bounds of other joints)
prb.createIntermediateConstraint("dynamic_feasibility", tau[:6])

# Set final velocity constraint: at the last node, the robot is standing still
prb.createFinalConstraint('final_velocity', q_dot)

# set a final pose of the floating base and robot configuration:
# rotate the robot orientation on the z-axis
q_final = q_init.copy()
# q_final[3:7] = stance_orientation
prb.createFinalResidual(f"final_nominal_pos", q[7:] - q_final[7:])
prb.createFinalConstraint(f"final_nominal_pos_res", q[:6] - q_final[:6])


k_all = range(1, n_nodes + 1)
# gather all the nodes where the robot is NOT touching the ground/jumping
nodes_swing = list(range(*[node for node in node_action]))
nodes_stance = list(filterfalse(lambda k: k in nodes_swing, k_all))

# iterate over the four contacts:
for frame, f in contact_map.items():

    FK = cs.Function.deserialize(kindyn.fk(frame))
    DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
    DDFK = cs.Function.deserialize(kindyn.frameAcceleration(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))

    p = FK(q=q)['ee_pos']
    p_start = FK(q=q_init)['ee_pos']
    v = DFK(q=q, qdot=q_dot)['ee_vel_linear']

    # velocity of each end effector must be zero before and after the jump
    prb.createConstraint(f"{frame}_vel_ground", v, nodes=nodes_stance) # [nodes_swing[0]]

    # the robot cannot exert force in this node:
    # the only possible solution for the solver is a jumping trajectory
    prb.createConstraint(f"{frame}_no_force_during_jump", f, nodes=nodes_swing)
# ====================== Set COSTS ===============================

# minimize the velocity of the robot
prb.createCost("min_q_dot", 3 * cs.sumsqr(q_dot))
prb.createIntermediateCost("min_q_ddot", 0.01 * cs.sumsqr(q_ddot))

# regularization term for the forces of system (input)
for f in f_list:
    prb.createIntermediateCost(f"min_{f.getName()}", 0.1 * cs.sumsqr(f))

# ==================== BUILD PROBLEM =============================

# the chosen solver is IPOPT.
# populating the options, it is possible to access the internal option of the solver.
opts = {'ilqr.verbose': True,
        'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 0.001,
        'ipopt.max_iter': 2000,
        'ilqr.max_iter': 200,
        'ilqr.integrator': 'RK4',
        'ilqr.closed_loop_forward_pass': True,
        'ilqr.line_search_accept_ratio': 1e-9,
        'ilqr.constraint_violation_threshold': 1e-3,
        'ilqr.step_length_threshold': 1e-12,
        'ilqr.alpha_min': 0.2,
        'ilqr.kkt_decomp_type': 'qr',
        'ilqr.constr_decomp_type': 'qr',
        'ilqr.codegen_workdir': '/tmp/spot_motions'}

# the solver class accept different solvers, such as 'ipopt', 'ilqr', 'gnsqp'.
# Different solver are useful (and feasible) in different situations.
solv = Solver.make_solver(solver, prb, opts)

# ==================== SOLVE PROBLEM =============================
solv.solve()

# ====================== SAVE SOLUTION =======================

# the solution is retrieved in the form of a dictionary ('variable_name' = values)
solution = solv.getSolutionDict()
# the dt is retrieved as a vector of values (size: number of intervals --> n_nodes - 1)
dt_sol = solv.getDt()

try:
    solv.set_iteration_callback()
except:
    pass

solv.solve()

if solver == 'ilqr':
    solv.print_timings()

# the value of the constraints functions
# solution_constraints = solv.getConstraintSolutionDict()

# populate what will be the .mat file with the solution
# solution_constraints_dict = dict()
# for name, item in prb.getConstraints().items():
#     lb, ub = item.getBounds()
#     lb_mat = np.reshape(lb, (item.getDim(), len(item.getNodes())), order='F')
#     ub_mat = np.reshape(ub, (item.getDim(), len(item.getNodes())), order='F')
#     solution_constraints_dict[name] = dict(val=solution_constraints[name], lb=lb_mat, ub=ub_mat, nodes=item.getNodes())
#
# info_dict = dict(n_nodes=n_nodes, node_action=node_action, stance_orientation=stance_orientation)
#
# if isinstance(dt, cs.SX):
#     ms.store({**solution, **solution_constraints_dict, **info_dict})
# else:
#     dt_dict = dict(dt=dt)
#     ms.store({**solution, **solution_constraints_dict, **info_dict, **dt_dict})

# ====================== PLOT SOLUTION =======================
if plot_sol:

    # Horizon expose a plotter to simplify the generation of graphs
    # Once instantiated, variables and constraints can be plotted with ease

    hplt = plotter.PlotterHorizon(prb, solution)
    ## plot all the variables, showing the relative bounds
    # hplt.plotVariables(show_bounds=True, legend=False)
    ## plot all the constraints, showing the relative bounds
    # hplt.plotFunctions(show_bounds=True)
    ## plot only the forces, organizing the plots in a 2x2 figure with gather
    hplt.plotVariables([elem.getName() for elem in f_list], show_bounds=False, gather=2, legend=False)
    ## plot the desired dimensions of the constraint 'dynamic_feasibility'
    hplt.plotFunction('dynamic_feasibility', show_bounds=True, legend=True, dim=range(6))

    # some custom plots to visualize the robot motion
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
        for dim in range(3):
            ax.plot(np.array([range(pos.shape[1])]), np.array(pos[dim, :]), marker="x", markersize=3,
                    linestyle='dotted')

        plt.vlines([node_action[0], node_action[1]], plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], linestyles='dashed', colors='k', linewidth=0.4)

    plt.figure()
    for contact in contacts_name:
        FK = cs.Function.deserialize(kindyn.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'plane_xy')
        plt.scatter(np.array(pos[0, :]), np.array(pos[1, :]), linewidth=0.1)

    plt.figure()
    for contact in contacts_name:
        FK = cs.Function.deserialize(kindyn.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'plane_xz')
        plt.scatter(np.array(pos[0, :]), np.array(pos[2, :]), linewidth=0.1)

    plt.show()

# ====================== RESAMPLE SOLUTION =======================
# the solution trajectories are not necessarily at the desired frequency. This is mainly for two legitimate reasons:
# 1. the number of nodes in the problem is kept low to limit the complexity of the problem
# 2. if the dt is an optimization variable, its solution vector will probably have different values of dt, resulting in a varying frequency

# Horizon exposes a resampler to obtain a trajectory with a desired frequency.
# The resampler makes use of the dynamics of the system to integrate the state using the input solution found.
# In other words, the value of the variables at each intermediate node (at the resampled frequency) in between the old nodes (at the old frequency)
# is found by integrating the state at the old nodes with the input found by the optimizer, which is constant in between two old nodes

contact_map = {contacts_name[i]: solution[f_list[i].getName()] for i in range(n_c)}

# resampling
if resampling:

    if isinstance(dt, cs.SX):
        dt_before_res = solution['dt'].flatten()
    else:
        dt_before_res = dt

    dt_res = 0.001
    dae = {'x': prb.getState().getVars(), 'p': q_ddot, 'ode': x_dot, 'quad': 1}
    q_res, qdot_res, qddot_res, contact_map_res, tau_res = resampler_trajectory.resample_torques(
        solution["q"], solution["q_dot"], solution["q_ddot"], dt_before_res, dt_res, dae, contact_map,
        kindyn,
        cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

# ====================== REPLAY SOLUTION =======================

# Horizon exposes a module to replay in RVIZ the optimal trajectory found by the solver.
# It ciclically replay the solution on the robot.

if rviz_replay:

    # set ROS stuff and launchfile
    from horizon.ros.replay_trajectory import *
    import rospy
    import subprocess
    # temporary add the example path to the environment
    os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
    subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=cogimon'])
    rospy.loginfo("'spot' visualization started.")


    # visualize the robot in RVIZ
    # with LOCAL_WORLD_ALIGNED, the forces are rotated in LOCAL frame before being published
    if resampling:
        repl = replay_trajectory(dt_res, joint_names, q_res, contact_map_res, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)
    else:
        repl = replay_trajectory(dt, joint_names, solution['q'], contact_map, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)

    # sleep for 1 second in between each cycle of the replayer
    repl.sleep(1.)
    repl.replay(is_floating_base=True)

