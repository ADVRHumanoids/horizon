#!/usr/bin/env python3

'''
An example of the cart-pole problem
'''

import argparse
import os
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from horizon import problem
from horizon.solvers import solver
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.utils import utils, kin_dyn
from horizon.utils.plotter import PlotterHorizon


def str2bool(v):
  #susendberg's function
  return v.lower() in ("yes", "true", "t", "1")

def main(args):

    # ipopt, gnsqp, ilqr, blocksqp
    solver_type = 'ipopt' # todo fails with 'ilqr', 'gnsqp'

    rviz_replay = args.replay
    system_input = args.input
    minimize_t = args.minimize_t
    y_reference = args.parametric
    plot_sol = args.plot

    resampling = False

    # ilqr_plot_iter = False

    torque_input = False
    if system_input == 'torque':
        torque_input = True

    if solver_type == 'ilqr' and minimize_t:
        input("'ilqr' solver supports only float and Parameter dt. Press a button to continue.")
        minimize_t = False

    if solver_type == 'gnsqp' and y_reference:
        input("'gnsqp' solver does not support Parameters. Press a button to continue.")
        y_reference = False


    if args.replay:
        from horizon.ros.replay_trajectory import replay_trajectory
        import rospkg, rospy
        plot_sol = False


    path_to_examples = os.path.dirname(os.path.realpath(__file__))

    # Create CasADi interface to Pinocchio
    urdffile = os.path.join(path_to_examples, 'urdf', 'cart_pole_xy.urdf')
    urdf = open(urdffile, 'r').read()
    kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

    # Get dimension of pos and vel
    nq = kindyn.nq()
    nv = kindyn.nv()

    # Optimization parameters
    tf = 5.0  # [s]
    n_nodes = 100  # number of nodes
    dt = tf / n_nodes

    q_tgt = np.array([0.5, 0, np.pi])

    # options for horizon transcription
    transcription_method = 'multiple_shooting'  # can choose between 'multiple_shooting' and 'direct_collocation'
    transcription_opts = dict(integrator='EULER') # integrator used by the multiple_shooting

    # Create horizon problem
    prb = problem.Problem(n_nodes)

    # creates the variables for the problem
    # design choice: position and the velocity as state and acceleration as input

    # STATE variables
    q = prb.createStateVariable("q", nq)
    qdot = prb.createStateVariable("qdot", nv)

    if y_reference:
        # send reference as a parameter
        q_ref = prb.createParameter('q_ref', 1)

    # CONTROL variables
    if torque_input:
        # cart joint is actuated, while the pole joint is unactuated
        tau = prb.createInputVariable("tau", nv)
    else:
        qddot = prb.createInputVariable("qddot", nv)

    # Set dynamics of the system and the relative dt
    if torque_input:
        fd = kindyn.aba()  # this is the forward dynamics function:
        qddot = fd(q=q, v=qdot, tau=tau)['a']  # qddot = M^-1(tau - h)
        xdot = utils.double_integrator(qdot, qddot)  # xdot = [qdot, qddot]
    else:
        xdot = utils.double_integrator(qdot, qddot)

    if minimize_t:
        # Create final time variable
        dt_min = 0.005
        dt_max = 0.1
        dt_init = 0.01
        dt = prb.createVariable("dt", 1)
        dt.setBounds(dt_min, dt_max)
        dt.setInitialGuess(dt_init)


    prb.setDynamics(xdot)
    prb.setDt(dt)

    # Define LIMITS and INITIAL GUESS
    # joint limits + initial pos
    q_min = [-0.5, -1, -2.*np.pi]
    q_max = [0.5, 1, 2.*np.pi]
    q_init = [0., 0., 0.]

    q.setBounds(q_min, q_max)
    q.setBounds(q_init, q_init, nodes=0)
    q.setInitialGuess(q_init)

    # velocity limits + initial vel
    qdot_lims = np.array([100., 100., 100.])
    qdot_init = [0., 0., 0.]

    qdot.setBounds(-qdot_lims, qdot_lims)
    qdot.setBounds(qdot_init, qdot_init, nodes=0)
    qdot.setInitialGuess(qdot_init)

    # input limits

    tau_lims = np.array([1000, 1000, 0])

    if torque_input:
        tau_init = [0, 0, 0]
        tau.setBounds(-tau_lims, tau_lims)
        tau.setInitialGuess(tau_init)
    else:
        qddot_lims = np.array([1000., 1000., 1000.])
        qddot_init = [0., 0., 0.]
        qddot.setBounds(-qddot_lims, qddot_lims)
        qddot.setInitialGuess(qddot_init)

    if solver_type != 'ilqr':
        # Set transcription method
        th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

    if not torque_input:
        # Set dynamic feasibility:
        # the cart can apply a torque, the joint connecting the cart to the pendulum is UNACTUATED
        # the torques are computed using the inverse dynamics, as the input of the problem is the cart acceleration
        tau = kin_dyn.InverseDynamics(kindyn).call(q, qdot, qddot)
        iv = prb.createIntermediateConstraint("dynamic_feasibility", tau, bounds=dict(lb=-tau_lims, ub=tau_lims))

    # Set desired constraints
    # at the last node, the pendulum is upright
    prb.createFinalConstraint("up", q[2] - q_tgt[2])
    # prb.createFinalConstraint("center", q[0] - q_tgt[0])
    # at the last node, the system velocity is zero
    prb.createFinalConstraint("final_qdot", qdot)

    if y_reference:
        # impose a reference to follow for the cart
        cnrst_ref = prb.createConstraint('sinusoidal_ref', q[1] - q_ref, range(10, n_nodes + 1))

    # Set cost functions
    # regularization of the input
    if torque_input:
        if solver_type == 'gnsqp':
            prb.createIntermediateResidual("tau", 1e-1 * tau)
        else:
            prb.createIntermediateCost("tau", cs.sumsqr(tau))
    else:
        if solver_type == 'gnsqp':
            prb.createIntermediateResidual("qddot", qddot)
        else:
            prb.createIntermediateCost("qddot", cs.sumsqr(qddot))

    if solver_type == 'gnsqp':
        prb.createIntermediateResidual("qdot", 1e-1 * qdot)
        prb.createIntermediateResidual("qfinal", 1e-1 * (q[2] - q_tgt[2]))
        prb.createFinalResidual("qdot_f", 1e-1 * qdot)
        prb.createFinalResidual("qfinal_f", 1e-2 * (q - q_tgt))


    if minimize_t:
        if solver_type == 'gnspq':
            prb.createResidual("min_dt", 1e2 * dt)
        else:
            prb.createCost("min_dt", n_nodes * 1e5 * cs.sumsqr(dt))

    # ======================================================================================================================
    opts = dict()
    if solver_type == 'gnsqp':
        qp_solver = 'osqp'
        if qp_solver == 'osqp':
            opts['gnsqp.qp_solver'] = 'osqp'
            opts['warm_start_primal'] = True
            opts['warm_start_dual'] = True
            opts['merit_derivative_tolerance'] = 1e-6
            opts['constraint_violation_tolerance'] = n_nodes * 1e-12
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
            # opts['linsol_plugin'] = 'ma57'

    solv = solver.Solver.make_solver(solver_type, prb, opts=opts)

    if y_reference:
        # choose the reference for the cart. Notice that this can be done even AFTER the problem is built.
        cos_fun = np.cos(np.linspace(np.pi / 2, 2 * np.pi, n_nodes + 1))
        for n in range(n_nodes + 1):
            q_ref.assign(cos_fun[n], n)

    # Solve the problem
    try:
        solv.set_iteration_callback()
        solv.plot_iter = ilqr_plot_iter
    except:
        pass

    solv.solve()
    # Get the solution as a dictionary
    solution = solv.getSolutionDict()
    # Get the array of dt, one for each interval between the nodes
    dt_sol = solv.getDt()

    if solver_type == 'gnsqp':
        print(f"mean Hessian computation time: {sum(solv.getHessianComputationTime())/len(solv.getHessianComputationTime())}")
        print(f"mean QP computation time: {sum(solv.getQPComputationTime())/len(solv.getQPComputationTime())}")
        print(f"mean Line Search computation time: {sum(solv.getLineSearchComputationTime()) / len(solv.getLineSearchComputationTime())}")

    try:
        solver.print_timings()
    except:
        pass


    total_time = sum(dt_sol)
    print(f"total trajectory time: {total_time}")
    ########################################################################################################################

    if plot_sol:
        time = np.arange(0.0, total_time + 1e-6, total_time / n_nodes) # wrong
        plt.figure()
        plt.plot(time, solution['q'][0,:])
        plt.plot(time, solution['q'][1,:])
        plt.suptitle('$\mathrm{Base \ Position}$', size = 20)
        plt.xlabel('$\mathrm{[sec]}$', size = 20)
        plt.ylabel('$\mathrm{[m]}$', size = 20)

        hplt = PlotterHorizon(prb, solution)
        hplt.plotVariables()
        # hplt.plotFunctions()
        plt.show()

    if rviz_replay:
        try:
            # set ROS stuff and launchfile
            import subprocess
            os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
            subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=cart_pole_xy'])
            rospy.loginfo("'cart_pole' visualization started.")
        except:
            print('Failed to automatically run RVIZ. Launch it manually.')

        # visualize the robot in RVIZ
        # joint_list=["cart_joint", "pole_joint"]
        joint_list = ["cart_joint_x", "cart_joint_y", "pole_joint"]
        replay_trajectory(total_time / n_nodes, joint_list, solution['q']).replay(is_floating_base=False)

    else:
        print("To visualize the robot trajectory, start the script with the '--replay")


if __name__ == '__main__':

    cart_pole_actions = ()
    cart_pole_input = ('torque', 'acceleration')
    # cart_pole_solvers = ('ipopt')

    parser = argparse.ArgumentParser(
        description='cart-pole problem: moving the cart so that the pole reaches the upright position')
    parser.add_argument('--replay', '-r', help='visualize the robot trajectory in rviz', action='store_true',
                        default=False)
    parser.add_argument('--input', '-i', help='choose which input of the cart-pole system', choices=cart_pole_input,
                        default='torque')
    parser.add_argument("--minimize_t", '-t', type=str2bool, nargs='?', const=True, default=False,
                        help="minimize the duration of the trajectory.")
    parser.add_argument("--parametric", '-c', type=str2bool, nargs='?', const=True, default=False,
                        help="add a parametric reference for the trajectory.")
    parser.add_argument("--plot", '-p', type=str2bool, nargs='?', const=True, default=True, help="plot solutions")
    # parser.add_argument('--solver', '-s', help='choose which solver will be used', choices=spot_solvers, default=spot_solvers[0])

    args = parser.parse_args()
    main(args)
