#!/usr/bin/env python3

import numpy as np
from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from horizon.transcriptions import integrators
from horizon.solvers import solver
import os, argparse
import casadi as cs

def str2bool(v):
  #susendberg's function
  return v.lower() in ("yes", "true", "t", "1")

def main(args):

    # Switch between suspended and free fall
    rviz_replay = args.replay
    rope_mode = args.action
    plot_sol = args.plot
    resample = True

    if rviz_replay:
        from horizon.ros.replay_trajectory import replay_trajectory
        import rospy
        resample = True
        plot_sol = False


    path_to_examples = os.path.dirname(os.path.realpath(__file__))

    # Loading URDF model in pinocchio
    urdffile = os.path.join(path_to_examples, 'urdf', 'roped_template.urdf')
    urdf = open(urdffile, 'r').read()
    kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

    joint_names = kindyn.joint_names()
    if 'universe' in joint_names: joint_names.remove('universe')
    if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')
    if 'reference' in joint_names: joint_names.remove('reference')

    # OPTIMIZATION PARAMETERS

    n_nodes = 3
    nc = 3  # number of contacts
    nq = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)
    DoF = nq - 7  # Contacts + anchor_rope + rope
    nv = kindyn.nv()  # Velocity DoFs
    nf = 3  # 2 feet contacts + rope contact with wall, Force DOfs

    # Create horizon problem
    prb = problem.Problem(n_nodes, casadi_type=cs.MX)

    # Creates problem STATE variables
    q = prb.createStateVariable("q", nq)
    qdot = prb.createStateVariable("qdot", nv)

    # Creates problem CONTROL variables
    qddot = prb.createInputVariable("qddot", nv)
    f1 = prb.createInputVariable("f1", nf)
    f2 = prb.createInputVariable("f2", nf)
    frope = prb.createInputVariable("frope", nf)

    # Creates double integrator
    xdot = utils.double_integrator_with_floating_base(q, qdot, qddot)

    tf = 1.  # [s]
    dt = tf / n_nodes

    # if rope_mode == 'jump':
    #     dt = prb.createVariable('dt', 1)
    #     dt.setBounds(0.05, 0.5)
    #     dt.setInitialGuess(0.1)

    prb.setDynamics(xdot)
    prb.setDt(dt)

    # Formulate discrete time dynamics
    L = 0.5 * cs.sumsqr(qdot)  # Objective term


    dae = {'x': prb.getState().getVars(), 'p': qddot, 'ode': xdot, 'quad': L}

    if rope_mode == 'swing':
        F_integrator_LEAPFROG = integrators.LEAPFROG(dae)
        F_integrator = integrators.RK4(dae)
    else:
        F_integrator = integrators.RK4(dae)

    # limits
    rope_lenght = 1

    q_min = kindyn.q_min()
    q_max = kindyn.q_max()
    q_min[:3] = [-10.0, -10.0, -10.0]
    q_max[:3] = [10.0, 10.0, 10.0]
    q_min[3:7] = -np.ones(4)
    q_max[3:7] = np.ones(4)

    if rope_mode != 'free_fall':
        q_min[-1] = rope_lenght
        q_max[-1] = rope_lenght

    if rope_mode == 'swing':
        q_min[7:13] = np.zeros(6)
        q_max[7:13] = np.zeros(6)

    # if rope_mode == 'jump':
    # foot_z_offset = 0.5
    # q_max[9] = q_max[9] + foot_z_offset
    # q_max[12] = q_max[12] + foot_z_offset
    # q_max[-1] = 1

    q_init = [0., 0., 0., 0., 0., 0., 1.0,
              0., 0., 0.,
              0., 0., 0.,
              0., 0., 0.,
              rope_lenght]

    q.setBounds(q_min, q_max)
    # q.setBounds(q_init, q_init, nodes=0)
    q.setInitialGuess(q_init)

    qdot_min = -1000.*np.ones(nv)
    qdot_max = -qdot_min
    qdot_init = np.zeros(nv)
    qdot.setBounds(qdot_min, qdot_max)
    # qdot.setBounds(qdot_init, qdot_init, nodes=0)
    qdot.setInitialGuess(qdot_init)

    qddot_min = -1000.*np.ones(nv)
    qddot_max = -qddot_min
    qddot_init = np.zeros(nv)
    qddot.setBounds(qddot_min, qddot_max)
    qddot.setInitialGuess(qdot_init)

    if rope_mode == 'swing':
        f_min = np.zeros(nf)
        f_max = f_min
    else:
        f_min = -10000.*np.ones(nf)
        f_max = -f_min

    frope_min = -10000. * np.ones(nf)
    frope_max = -frope_min

    f1.setBounds(f_min, f_max)
    f2.setBounds(f_min, f_max)
    frope.setBounds(frope_min, frope_max)

    if rope_mode == 'swing':
        # starting from a tilted position
        q_init[14] = 0.5 # rope_anchor_y

    f_init = np.zeros(nf)
    f1.setInitialGuess(f_init)
    f2.setInitialGuess(f_init)
    frope.setInitialGuess(f_init)

    state = prb.getState()
    input = prb.getInput()
    state_prev = state.getVarOffset(-1)
    input_prev = input.getVarOffset(-1)

    # x_prev, _ = utils.double_integrator_with_floating_base(state_prev[0], state_prev[1], input_prev[0])
    x_prev = prb.getState().getVarOffset(-1).getVars()
    x = prb.getState().getVars()

    x_int = F_integrator(x0=x_prev, p=input_prev[0], time=dt)


    if rope_mode == 'swing':
        prb.createConstraint("multiple_shooting", x_int["xf"] - x, nodes=1)
        q_pprev = q.getVarOffset(-2)
        qdot_pprev = qdot.getVarOffset(-2)
        qddot_pprev = qddot.getVarOffset(-2)
        # x_pprev, _ = utils.double_integrator_with_floating_base(q_pprev, qdot_pprev, qddot_pprev)
        x_pprev = prb.getState().getVarOffset(-2).getVars()

        x_int2 = F_integrator_LEAPFROG(x0=x_prev, x0_prev=x_pprev, p=input_prev[0], time=dt)
        prb.createConstraint("multiple_shooting2", x_int2["xf"] - x, nodes=range(2, n_nodes + 1))
    else:
        prb.createConstraint("multiple_shooting", x_int["xf"] - x, nodes=range(1, n_nodes + 1))

    # Constraints
    prb.createConstraint("q_init", q - q_init, nodes=0)
    prb.createConstraint("qdot_init", qdot - qdot_init, nodes=0)


    tau_min = np.array([0., 0., 0., 0., 0., 0.,  # floating base
                        -1000., -1000., -1000.,  # contact 1
                        -1000., -1000., -1000.,  # contact 2
                        0., 0., 0.,  # rope anchor point
                        0.])  # rope

    tau_max = - tau_min

    if rope_mode != 'free_fall':
        tau_min[-1] = -10000.

    frame_force_mapping = {'rope_anchor2': frope}

    # if rope_mode == 'jump':
    #     contacts = {'Contact1': f1, 'Contact2': f2}
    #     frame_force_mapping.update(contacts)


    id = kin_dyn.InverseDynamics(kindyn, frame_force_mapping.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
    tau = id.call(q, qdot, qddot, frame_force_mapping)

    prb.createConstraint("inverse_dynamics", tau, nodes=range(0, n_nodes), bounds=dict(lb=tau_min, ub=tau_max))

    FKRope = cs.Function.deserialize(kindyn.fk('rope_anchor2'))

    p_rope_init = FKRope(q=q_init)['ee_pos']
    p_rope = FKRope(q=q)['ee_pos']
    prb.createConstraint("rope_anchor_point", p_rope - p_rope_init)


    # Cost function
    if rope_mode == 'swing':
        weigth_joint_vel = 1.
    else:
        weigth_joint_vel = 100.

    prb.createCost("min_joint_vel", weigth_joint_vel*cs.sumsqr(qdot))

    if rope_mode != 'swing':
        prb.createCost("min_joint_acc", 1000. * cs.sumsqr(qddot[6:-1]), range(1, n_nodes))
        prb.createCost("min_f1", 1000. * cs.sumsqr(f1), range(1, n_nodes))
        prb.createCost("min_f2", 1000. * cs.sumsqr(f2), range(1, n_nodes))

        frope_prev = frope.getVarOffset(-1)
        prb.createCost("min_dfrope", 1000. * cs.sumsqr(frope-frope_prev), range(1, n_nodes))

    # Creates problem
    opts = {'ipopt.tol': 1e-3,
            'ipopt.constr_viol_tol': 1e-3,
            'ipopt.max_iter': 2000} #

    solv = solver.Solver.make_solver('ipopt', prb, opts)
    solv.solve()

    solution = solv.getSolutionDict()
    dt_sol = solv.getDt()
    total_time = sum(dt_sol)

    # ======================================================================================================================
    time = np.arange(0.0, total_time + 1e-6, total_time/n_nodes)

    tau_sol = np.zeros(solution["qddot"].shape)
    ID = kin_dyn.InverseDynamics(kindyn, ['rope_anchor2']) #['Contact1', 'Contact2',
    for i in range(n_nodes):
        # 'Contact1': solution["f1"][:, i], 'Contact2': solution["f2"][:, i],
        frame_force_mapping_i = {'rope_anchor2': solution["frope"][:, i]}
        tau_sol[:, i] = ID.call(solution["q"][:, i], solution["qdot"][:, i], solution["qddot"][:, i], frame_force_mapping_i).toarray().flatten()


    if resample:
        # resampling
        dt_res = 0.001
        frame_force_hist_mapping = {'Contact1': solution["f1"], 'Contact2': solution["f2"], 'rope_anchor2': solution["frope"]} #
        q_res, qdot_res, qddot_res, frame_force_res_mapping, tau_res = resampler_trajectory.resample_torques(solution["q"],
                                                                                                             solution["qdot"],
                                                                                                             solution["qddot"],
                                                                                                             dt, dt_res, dae,
                                                                                                             frame_force_hist_mapping,
                                                                                                             kindyn,
                                                                                                             cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)


        time_res = np.arange(0.0, q_res.shape[1] * dt_res - dt_res, dt_res)

    if plot_sol:
        import matplotlib.pyplot as plt
        # plots raw solution

        hplt = plotter.PlotterHorizon(prb, solution)
        hplt.plotVariables(['f1', 'f2', 'frope'])
        # hplt.plotFunctions()
        plt.show()


        plt.figure()
        for i in range(0, 3):
            plt.plot(time, solution["q"][i,:])
        plt.suptitle('$\mathrm{Base \ Position}$', size = 20)
        plt.xlabel('$\mathrm{[sec]}$', size = 20)
        plt.ylabel('$\mathrm{[m]}$', size = 20)

        plt.figure()
        for i in range(0, 3):
            plt.plot(time[:-1], solution["qddot"][i,:])
        plt.suptitle('$\mathrm{Base \ Acceleration}$', size = 20)
        plt.xlabel('$\mathrm{[sec]}$', size = 20)
        plt.ylabel('$\mathrm{ [m] } /  \mathrm{ [sec^2] } $', size = 20)

        plt.figure()
        for i in range(0, 3):
            plt.plot(time[:-1], solution["f1"][i, :])
            plt.plot(time[:-1], solution["f2"][i, :])
            plt.plot(time[:-1], solution["frope"][i, :])
        plt.suptitle('$\mathrm{force \ feet \ and \ rope}$', size=20)
        plt.xlabel('$\mathrm{[sec]}$', size=20)
        plt.ylabel('$\mathrm{ [N] } /  \mathrm{ [sec^2] } $', size=20)

        plt.figure()
        for i in range(0, 6):
            plt.plot(time[:-1], tau_sol[i, :])
        plt.suptitle('$\mathrm{base \ force}$', size=20)
        plt.xlabel('$\mathrm{[sec]}$', size=20)
        plt.ylabel('$\mathrm{ [N] }} $', size=20)


    if rviz_replay:

        try:
            # set ROS stuff and launchfile
            import subprocess
            os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
            subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=roped_template'])
            rospy.loginfo("'roped_robot' visualization started.")
        except:
            print('Failed to automatically run RVIZ. Launch it manually.')

        replay_trajectory(dt_res, joint_names, q_res, frame_force_res_mapping,
                          cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn).replay()

    else:
        print("To visualize the robot trajectory, start the script with the '--replay")

if __name__ == '__main__':
    roped_robot_actions = ('swing', 'free_fall', 'hang')

    parser = argparse.ArgumentParser(
        description='cart-pole problem: moving the cart so that the pole reaches the upright position')
    parser.add_argument('--replay', help='visualize the robot trajectory in rviz', action='store_true')
    parser.add_argument('--action', '-a', help='choose which action spot will perform', choices=roped_robot_actions,
                        default=roped_robot_actions[1])
    parser.add_argument("--plot", '-p', type=str2bool, nargs='?', const=True, default=True, help="plot solutions")

    args = parser.parse_args()
    main(args)