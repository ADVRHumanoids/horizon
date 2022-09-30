#!/usr/bin/env python

import casadi as cs
import numpy as np
from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer
from horizon.transcriptions.transcriptor import Transcriptor
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from horizon.solvers import solver
import os, argparse
from itertools import filterfalse

def str2bool(v):
  #susendberg's function
  return v.lower() in ("yes", "true", "t", "1")

def main(args):

    # Switch between suspended and free fall
    rviz_replay = args.replay
    rope_mode = args.action
    plot_sol = args.plot
    resample = True
    warmstart_flag = args.warmstart

    load_initial_guess = False

    if rviz_replay:
        from horizon.ros.replay_trajectory import replay_trajectory
        import rospy
        resample = True
        plot_sol = False


    path_to_examples = os.path.dirname(os.path.realpath(__file__))

    if warmstart_flag:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        save_dir = path_to_examples + '/mat_files'
        save_file = path_to_examples + f'/mat_files/{file_name}.mat'

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        ms = mat_storer.matStorer(path_to_examples + f'/mat_files/{file_name}.mat')

        if os.path.isfile(save_file):
            print(f'{file_name}.mat file found. Using previous solution as initial guess.')
            load_initial_guess = True
        else:
            print(f'{file_name}.mat file NOT found. The solution will be saved for future warmstarting.')

    # Loading URDF model in pinocchio
    urdffile = os.path.join(path_to_examples, 'urdf', 'roped_template.urdf')
    urdf = open(urdffile, 'r').read()
    kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

    joint_names = kindyn.joint_names()
    if 'universe' in joint_names: joint_names.remove('universe')
    if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')
    if 'reference' in joint_names: joint_names.remove('reference')

    transcription_method = 'multiple_shooting'
    transcription_opts = dict(integrator='RK4')

    # OPTIMIZATION PARAMETERS
    n_nodes = 70  # number of shooting nodes
    nc = 3  # number of contacts
    nq = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)
    DoF = nq - 7  # Contacts + anchor_rope + rope
    nv = kindyn.nv()  # Velocity DoFs
    nf = 3  # 2 feet contacts + rope contact with wall, Force DOfs

    # Create horizon problem
    prb = problem.Problem(n_nodes)

    # Creates problem STATE variables
    q = prb.createStateVariable("q", nq)
    qdot = prb.createStateVariable("qdot", nv)

    # Creates problem CONTROL variables
    qddot = prb.createInputVariable("qddot", nv)
    f1 = prb.createInputVariable("f1", nf)
    f2 = prb.createInputVariable("f2", nf)
    frope = prb.createInputVariable("frope", nf)

    # Node times
    # dt = prb.createVariable("dt", 1)
    dt = 0.02

    # Creates double integrator
    xdot = utils.double_integrator_with_floating_base(q, qdot, qddot)

    prb.setDynamics(xdot)
    prb.setDt(dt)

    # Set bounds and initial guess to variables

    if load_initial_guess:
        if os.path.exists(path_to_examples + f'/mat_files/{file_name}.mat'):
            prev_solution = ms.load()
            variables_dict = {"q": q, "qdot": qdot, "qddot": qddot, "f1": f1, "f2": f2, "frope": frope}
            mat_storer.setInitialGuess(variables_dict, prev_solution)

    q_min = kindyn.q_min()
    q_max = kindyn.q_max()

    foot_z_offset = 0.5

    q_min[:3] = [-10.0, -10.0, -10.0]
    q_min[3:7] = -1 * np.ones(4)
    q_min[-1] = -10
    q_max[:3] = [10.0, 10.0, 10.0]
    q_max[3:7] = 1 * np.ones(4)
    q_max[9] = q_max[9] + foot_z_offset
    q_max[12] = q_max[12] + foot_z_offset
    q_max[-1] = 10

    alpha = 0.3
    rope_lenght = 0.3

    x_foot = rope_lenght * np.sin(alpha)
    q_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
               x_foot, 0., 0.,
               x_foot, 0., 0.,
               0., alpha, 0.,
               rope_lenght]

    q.setInitialGuess(q_init)
    q.setBounds(q_min, q_max)
    q.setBounds(q_init, q_init, 0)

    qdot.setBounds(np.zeros(nv), np.zeros(nv), [0, n_nodes])


    node_action = (10, 60) #20

    # Constraints
    th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

    tau_min = np.array([0., 0., 0., 0., 0., 0.,  # floating base
                        -1000., -1000., -1000.,  # contact 1
                        -1000., -1000., -1000.,  # contact 2
                        0., 0., 0.,  # rope anchor point
                        -1000.])  # rope

    tau_max = - tau_min

    dd = {'Contact1': f1, 'Contact2': f2, 'rope_anchor2': frope}
    tau = kin_dyn.InverseDynamics(kindyn, dd.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(q, qdot, qddot, dd)
    prb.createConstraint("inverse_dynamics", tau, nodes=range(0, n_nodes), bounds=dict(lb=tau_min, ub=tau_max))

    FKRope = cs.Function.deserialize(kindyn.fk('rope_anchor2'))
    p_rope_init = FKRope(q=q_init)['ee_pos']
    p_rope = FKRope(q=q)['ee_pos']

    prb.createConstraint("rope_anchor_point", p_rope - p_rope_init)

    x_distance = -0.4
    prb.createCost("wall_distance", 1e2 * cs.sumsqr(q[0] - x_distance), nodes=range(node_action[0], n_nodes+1))
    prb.createCost("min_qdot", cs.sumsqr(qdot))

    frope_prev = frope.getVarOffset(-1)
    prb.createCost("min_df", 1e-4 * cs.sumsqr(frope-frope_prev), nodes=range(1, n_nodes))
    prb.createFinalCost("final_config", 1e3 * cs.sumsqr(q[7:13] - q_init[7:13]))

    prb.createIntermediateResidual(f"min_{f1.getName()}", 1e-2 * f1)
    prb.createIntermediateResidual(f"min_{f2.getName()}", 1e-2 * f2)
    prb.createIntermediateResidual(f"min_{frope.getName()}", 1e-3 * frope)

    # WALL
    mu = 0.5
    R_wall = np.zeros([3, 3])
    rot = -np.pi/2.
    R_wall[0, 0] = np.cos(rot)
    R_wall[0, 2] = np.sin(rot)
    R_wall[1, 1] = 1.
    R_wall[2, 0] = -np.sin(rot)
    R_wall[2, 2] = np.cos(rot)


    contact_names = ['Contact1', 'Contact2']
    forces = [f1, f2]

    all_nodes = range(n_nodes + 1)
    touch_nodes = list(range(0, node_action[0])) + list(range(node_action[1], n_nodes + 1))
    lift_nodes = list(filterfalse(lambda k: k in touch_nodes, all_nodes))

    for frame, f in zip(contact_names, forces):
        FK = cs.Function.deserialize(kindyn.fk(frame))
        DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
        p = FK(q=q)['ee_pos']
        pd = FK(q=q_init)['ee_pos']
        v = DFK(q=q, qdot=qdot)['ee_vel_linear']

        # STANCE PHASE
        prb.createConstraint(f"zero_{frame}_vel_touch", v, nodes=touch_nodes)

        fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(f, mu, R_wall)
        prb.createConstraint(f"{frame}_friction_cone_before_jump", fc, nodes=touch_nodes[:-1], bounds=dict(lb=fc_lb, ub=fc_ub))

        # FLIGHT PHASE
        prb.createConstraint(f"{frame}_no_force_during_jump", f, nodes=lift_nodes)

        # LAND
        surface_dict = {'a': 1., 'd': -x_foot}
        c, lb, ub = kin_dyn.surface_point_contact(surface_dict, q, kindyn, frame)
        prb.createFinalConstraint(f"{frame}_on_wall", c, bounds=dict(lb=lb, ub=ub))

    # Creates problem
    opts = {'ipopt.tol': 0.01,
            'ipopt.constr_viol_tol': 0.01,
            'ipopt.max_iter': 1000}

    solv = solver.Solver.make_solver('ipopt', prb, opts)
    solv.solve()

    solution = solv.getSolutionDict()

    solution_constraints_dict = dict()

    if warmstart_flag:
        if isinstance(dt, cs.SX):
            ms.store({**solution, **solution_constraints_dict})
        else:
            dt_dict = dict(dt=dt)
            ms.store({**solution, **solution_constraints_dict, **dt_dict})

        variables_dict = {"q": q, "qdot": qdot, "qddot": qddot, "f1": f1, "f2": f2, "frope": frope}
        mat_storer.setInitialGuess(variables_dict, solution)



    tau_hist = np.zeros(solution['qddot'].shape)
    ID = kin_dyn.InverseDynamics(kindyn, ['Contact1', 'Contact2', 'rope_anchor2'], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

    for i in range(n_nodes):
        frame_force_mapping_i = {'Contact1': solution["f1"][:, i], 'Contact2': solution["f2"][:, i], 'rope_anchor2': solution["frope"][:, i]}
        tau_hist[:, i] = ID.call(solution["q"][:, i], solution["qdot"][:, i], solution["qddot"][:, i], frame_force_mapping_i).toarray().flatten()


    if plot_sol:
        import matplotlib.pyplot as plt

        hplt = plotter.PlotterHorizon(prb, solution)
        hplt.plotVariables(['f1', 'f2', 'frope'])
        plt.show()


    # resampling
    if resample:
        dae = {'x': prb.getState().getVars(), 'p': qddot, 'ode': xdot, 'quad': 1}

        dt_res = 0.001
        frame_force_hist_mapping = {'Contact1': solution["f1"], 'Contact2': solution["f2"], 'rope_anchor2': solution["frope"]} #
        q_res, qdot_res, qddot_res, frame_force_res_mapping, tau_res = resampler_trajectory.resample_torques(solution["q"],
                                                                                                             solution["qdot"],
                                                                                                             solution["qddot"],
                                                                                                             dt, dt_res, dae,
                                                                                                             frame_force_hist_mapping,
                                                                                                             kindyn,
                                                                                                             cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)


    if rviz_replay:

        try:
            # set ROS stuff and launchfile
            import subprocess
            os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
            subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=roped_template'])
            rospy.loginfo("'roped_robot' visualization started.")
        except:
            print('Failed to automatically run RVIZ. Launch it manually.')

        repl = replay_trajectory(dt_res, joint_names, q_res, frame_force_res_mapping,
                          cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)

        repl.sleep(1.)
        repl.replay()

    else:
        print("To visualize the robot trajectory, start the script with the '--replay")


if __name__ == '__main__':

    roped_robot_actions = ('rappel')

    parser = argparse.ArgumentParser(
        description='A template model of a robot connected to a wall thorugh a rope.')
    parser.add_argument('--replay', help='visualize the robot trajectory in rviz', action='store_true')
    parser.add_argument('--action', '-a', help='choose which action spot will perform', choices=roped_robot_actions,
                        default=roped_robot_actions[0])
    parser.add_argument("--warmstart", '-w', type=str2bool, nargs='?', const=True, default=False,
                        help="save solutions to mat file")
    parser.add_argument("--plot", '-p', type=str2bool, nargs='?', const=True, default=True, help="plot solutions")

    args = parser.parse_args()
    main(args)






