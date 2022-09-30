#!/usr/bin/env python3

import casadi as cs
import numpy as np
from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers import solver
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from horizon.utils.plotter import PlotterHorizon
import matplotlib.pyplot as plt
import os, argparse
from itertools import filterfalse

def str2bool(v):
  #susendberg's function
  return v.lower() in ("yes", "true", "t", "1")

def main(args):

    rviz_replay = args.replay
    plot_sol = args.plot
    resample = True

    if rviz_replay:
        from horizon.ros.replay_trajectory import replay_trajectory
        import rospy
        rviz_replay = True
        plot_sol = False
        resample = True

    path_to_examples = os.path.dirname(os.path.realpath(__file__))

    urdffile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'urdf', 'quadruped_template.urdf')
    urdf = open(urdffile, 'r').read()
    kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

    # OPTIMIZATION PARAMETERS
    n_nodes = 40  # number of nodes
    n_c = 4  # number of contacts
    n_q = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)
    DoF = n_q - 7  # Contacts + anchor_rope + rope
    n_v = kindyn.nv()  # Velocity DoFs
    n_f = 3  # Force DOfs

    lift_node = 20
    touch_down_node = 30

    joint_names = kindyn.joint_names()
    if 'universe' in joint_names: joint_names.remove('universe')
    if 'reference' in joint_names: joint_names.remove('reference')

    contact_names = ['Contact1', 'Contact2', 'Contact3', 'Contact4']

    # Create horizon problem
    prb = problem.Problem(n_nodes)

    # Creates problem STATE variables
    q = prb.createStateVariable("q", n_q)
    qdot = prb.createStateVariable("qdot", n_v)

    # Creates problem CONTROL variables
    qddot = prb.createInputVariable("qddot", n_v)

    f_list = list()
    for i in range(n_c):
        f_list.append(prb.createInputVariable(f'f{i}', n_f))

    dt_const = 0.05
    dt = prb.createSingleVariable("dt", 1)

    dt_list = (lift_node) * [dt_const] + (touch_down_node - lift_node) * [dt] + (n_nodes - touch_down_node) * [dt_const]

    xdot = utils.double_integrator_with_floating_base(q, qdot, qddot)

    prb.setDynamics(xdot)
    prb.setDt(dt_list)
    # Formulate discrete time dynamics
    dae = {'x': prb.getState().getVars(), 'p': qddot, 'ode': xdot, 'quad': 1}

    # Limits
    q_min = kindyn.q_min()
    q_max = kindyn.q_max()
    q_min[:3] = [-10, -10, -10]
    q_max[:3] = [10, 10, 10]

    q_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, # floating base
              0.349999, 0.349999, -0.635, # contact 1
              0.349999, -0.349999, -0.635, # contact 2
              -0.349999, -0.349999, -0.635, # contact 3
              -0.349999, 0.349999, -0.635] # contact 4

    q.setBounds(q_min, q_max)
    q.setBounds(q_init, q_init, 0)
    q.setInitialGuess(q_init)

    qdot_min = -100.*np.ones(n_v)
    qdot_max = -qdot_min
    qdot_init = np.zeros(n_v)
    qdot.setBounds(qdot_min, qdot_max)
    qdot.setBounds(qdot_init, qdot_init, 0)
    qdot.setInitialGuess(qdot_init)

    qddot_min = -100.*np.ones(n_v)
    qddot_max = -qddot_min
    qddot_init = np.zeros(n_v)
    qddot.setBounds(qddot_min, qddot_max)
    qddot.setInitialGuess(qddot_init)

    # f_min = -10000.*np.ones(n_f)
    # f_max = -f_min
    f_init = np.zeros(n_f)

    for f in f_list:
    #     f.setBounds(f_min, f_max)
        f.setInitialGuess(f_init)

    dt_min = 0.03 # [s]
    dt_max = 0.2 # [s]
    dt_init = dt_min
    dt.setBounds(dt_min, dt_max)
    dt.setInitialGuess(dt_init)

    tau_min = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
                -2000., -2000., -2000.,  # Contact 1
                -2000., -2000., -2000.,  # Contact 2
                -2000., -2000., -2000.,  # Contact 3
                -2000., -2000., -2000.])  # Contact 4

    tau_max = -tau_min


    # SET UP COST FUNCTION

    prb.createIntermediateCost("min_qddot", 10 * cs.sumsqr(qddot))
    prb.createIntermediateCost("min_qdot", 100 * cs.sumsqr(qdot))

    # for f in f_list:
    #     prb.createIntermediateResidual(f"min_{f.getName()}", 0.01 * f)

    # Constraints
    transcription_method = 'multiple_shooting'
    transcription_opts = dict(integrator='RK4')
    th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

    contact_map = dict(zip(contact_names, f_list))
    tau = kin_dyn.InverseDynamics(kindyn, contact_map.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(q, qdot, qddot, contact_map)
    prb.createConstraint("dynamic_feasibility", tau, nodes=range(0, n_nodes), bounds=dict(lb=tau_min, ub=tau_max))

    prb.createFinalConstraint('final_velocity', qdot)

    # foot
    all_nodes = range(n_nodes + 1)
    touch_nodes = list(range(0, lift_node)) + list(range(touch_down_node, n_nodes + 1))
    lift_nodes = list(filterfalse(lambda k: k in touch_nodes, all_nodes))

    for frame, f in zip(contact_names, f_list):
        FK = cs.Function.deserialize(kindyn.fk(frame))
        DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
        p = FK(q=q)['ee_pos']
        p_start = FK(q=q_init)['ee_pos']
        v = DFK(q=q, qdot=qdot)['ee_vel_linear']

        # JUMP
        prb.createConstraint(f"{frame}_vel_touch", v, nodes=touch_nodes)
        prb.createConstraint(f"{frame}_no_force_during_jump", f, nodes=lift_nodes)

        prb.createFinalConstraint(f"lift_{frame}_leg", p - p_start)

    # Create problem
    opts = {'ipopt.tol': 0.001,
            'ipopt.constr_viol_tol': 0.001,
            'ipopt.max_iter': 2000}#,
            # 'ipopt.linear_solver': 'ma57'}

    solv = solver.Solver.make_solver('ipopt', prb, opts)
    solv.solve()

    solution = solv.getSolutionDict()
    solution_constraints = solv.getConstraintSolutionDict()
    dt_sol = solv.getDt()

    tau_sol = np.zeros(solution["qddot"].shape)
    ID = kin_dyn.InverseDynamics(kindyn, contact_names, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
    for i in range(n_nodes):
        contact_map_i = dict(zip(contact_names, [solution['f0'][:, i], solution['f1'][:, i], solution['f2'][:, i], solution['f3'][:, i]]))
        tau_sol[:, i] = ID.call(solution["q"][:, i], solution["qdot"][:, i], solution["qddot"][:, i], contact_map_i).toarray().flatten()

    if resample:

        # resampling
        dt_res = 0.001

        contact_map_sol = dict(zip(contact_names, [solution['f0'], solution['f1'], solution['f2'], solution['f3']]))
        q_res, qdot_res, qddot_res, contact_map_sol_res, tau_res = resampler_trajectory.resample_torques(
                                                                                solution["q"], solution["qdot"], solution["qddot"], dt_sol,
                                                                                dt_res, dae, contact_map_sol, kindyn,
                                                                                cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)



    if plot_sol:

        hplt = PlotterHorizon(prb, solution)
        hplt.plotVariables(show_bounds=False, legend=True)
        # hplt.plotFunction('dynamic_feasibility', show_bounds=False)

        plt.figure()
        for i in range(6):
            plt.plot(tau_sol[i, :])
        plt.suptitle('$\mathrm{Base \ Forces}$', size=20)
        plt.xlabel('$\mathrm{sample}$', size=20)
        plt.ylabel('$\mathrm{[N]}$', size=20)

        if resample:

            time = np.zeros(q_res.shape[1])
            for i in range(q_res.shape[1] - 1):
                time[i + 1] = dt_res + time[i]

            plt.figure()
            for i in range(6):
                plt.plot(time[:-1], tau_res[i, :])
            plt.suptitle('$\mathrm{Base \ Forces \ Resampled}$', size=20)
            plt.xlabel('$\mathrm{[sec]}$', size=20)
            plt.ylabel('$\mathrm{[N]}$', size=20)

            plt.figure()
            for i in range(q_res.shape[0]):
                plt.plot(time, q_res[i, :])
            plt.suptitle('$\mathrm{q \ Resampled}$', size=20)
            plt.xlabel('$\mathrm{[sec]}$', size=20)
            plt.ylabel('$\mathrm{q}$', size=20)

            plt.figure()
            for i in range(qdot_res.shape[0]):
                plt.plot(time, qdot_res[i, :])
            plt.suptitle('$\mathrm{\dot{q} \ Resampled}$', size=20)
            plt.xlabel('$\mathrm{[sec]}$', size=20)
            plt.ylabel('$\mathrm{\dot{q}}$', size=20)

            plt.figure()
            for i in range(qddot_res.shape[0]):
                plt.plot(time[:-1], qddot_res[i, :])
            plt.suptitle('$\mathrm{\ddot{q} \ Resampled}$', size=20)
            plt.xlabel('$\mathrm{[sec]}$', size=20)
            plt.ylabel('$\mathrm{\ddot{q}}$', size=20)

        plt.show()

    if rviz_replay:

        try:
            # set ROS stuff and launchfile
            import subprocess
            os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
            subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=quadruped_template'])
            rospy.loginfo("quadruped_jump' visualization started.")
        except:
            print('Failed to automatically run RVIZ. Launch it manually.')

        repl = replay_trajectory(dt_res, joint_names, q_res, contact_map_sol_res, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)
        repl.sleep(1.)
        repl.replay()

    else:
        print("To visualize the robot trajectory, start the script with the '--replay")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='cart-pole problem: moving the cart so that the pole reaches the upright position')
    parser.add_argument('--replay', help='visualize the robot trajectory in rviz', action='store_true')
    parser.add_argument("--plot", '-p', type=str2bool, nargs='?', const=True, default=True, help="plot solutions")

    args = parser.parse_args()
    main(args)