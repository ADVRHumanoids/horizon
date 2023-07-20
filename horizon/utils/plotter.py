import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import gridspec
from horizon.problem import Problem
from horizon.variables import InputVariable, Variable, RecedingVariable, RecedingInputVariable
from horizon.functions import Function, RecedingFunction

import math
import numpy as np
import casadi as cs
import random

import matplotlib
matplotlib.use("Qt5Agg")

# TODO:
#   label: how to reduce it
#   actually opts must be set from outside

def create_grid(n_plots, title, n_rows_max, opts=None):
    if opts is None:
        opts = {}

    cols = n_plots if n_plots < n_rows_max else n_rows_max
    rows = int(math.ceil(n_plots / cols))

    gs = gridspec.GridSpec(rows, cols, **opts)

    fig = plt.figure(layout='constrained')
    fig.suptitle(title)

    return fig, gs

class Plotter():
    def __init__(self, prb: Problem, solution):
        self.prb = prb
        self.solution = solution

        # gs.hspace = 0.3

        # w = 7195
        # h = 3841

        # fig = plt.figure(frameon=True)
        # fig.set_size_inches(fig_size[0], fig_size[1])
        # fig.tight_layout()


    def plot(self, ax, item_name, dim, args):

        return self.__plot_element(ax, item_name, dim, args)

    def __get_val(self, item, dim):

        if isinstance(item, (Variable, RecedingVariable)):
            val = self.solution[item.getName()]
        elif isinstance(item, (Function, RecedingFunction)):
            val = self.prb.evalFun(item, self.solution)
        else:
            raise ValueError('item not recognized')

        var_dim_select = set(range(val.shape[0]))

        if dim is not None:
            if not set(dim).issubset(var_dim_select):
                raise Exception('Wrong selected dimension.')
            else:
                var_dim_select = dim

        return val, var_dim_select

    def __plot_element(self, ax, item, dim, args):

        val, var_dim_select = self.__get_val(item, dim)
        nodes_var = val.shape[1]

        if isinstance(val, (InputVariable, RecedingInputVariable)):
            plot_fun = ax.step
        else:
            plot_fun = ax.plot

        for dim in var_dim_select:
            plot_fun(np.array(range(nodes_var)), val[dim, range(nodes_var)], **args)

        return True

    # def plot
    # def __plot_bounds(self, ax, abstract_var, dim, args=None):
    #
    #     val, var_dim_select = self.__get_val(abstract_var, dim)
    #     nodes_var = val.shape[1]
    #
    #     lb, ub = abstract_var.getBounds()
    #
    #     for dim in var_dim_select:
    #         ax.plot(np.array(range(nodes_var)), lb[dim, range(nodes_var)], *args)
    #         ax.plot(np.array(range(nodes_var)), ub[dim, range(nodes_var)], *args)
    #L


class PlotterHorizon:
    def __init__(self, prb: Problem, solution=None, opts=None, logger=None):

        self.solution = solution
        self.prb = prb
        self.logger = logger
        self.opts = opts

    def setSolution(self, solution):
        self.solution = solution

    def _plotVar(self, val, ax, abstract_var, markers, show_bounds, legend, dim):
        var_dim_select = set(range(val.shape[0]))
        nodes_var = val.shape[1]
        if dim is not None:
            if not set(dim).issubset(var_dim_select):
                raise Exception('Wrong selected dimension.')
            else:
                var_dim_select = dim

        if nodes_var == 1:
            markers = True

        baseline = None
        legend_list = list()
        if isinstance(abstract_var, InputVariable):
            for i in var_dim_select: # get i-th dimension

                r = random.random()
                b = random.random()
                g = random.random()
                color = (r, g, b)

                for j in range(nodes_var-1):
                    # ax.plot(np.array(range(val.shape[1])), val[i, :], linewidth=0.1, color=color)
                    # ax.plot(range(val.shape[1])[j:j + 2], [val[i, j]] * 2, color=color)
                    ax.step(np.array(range(nodes_var)), val[i, range(nodes_var)], linewidth=0.1, color=color, label='_nolegend_')

                    if show_bounds:
                        lb, ub = abstract_var.getBounds()

                        if markers:
                            ax.plot(range(nodes_var), lb[i, range(nodes_var)], marker="x", markersize=3, linestyle='dotted',linewidth=1, color=color)
                            ax.plot(range(nodes_var), ub[i, range(nodes_var)], marker="x", markersize=3, linestyle='dotted',linewidth=1, color=color)
                        else:
                            ax.plot(range(nodes_var), lb[i, range(nodes_var)], linestyle='dotted')
                            ax.plot(range(nodes_var), ub[i, range(nodes_var)], linestyle='dotted')

                if legend:
                    legend_list.append(f'{abstract_var.getName()}_{i}')
                    if show_bounds:
                        legend_list.append(f'{abstract_var.getName()}_{i}_lb')
                        legend_list.append(f'{abstract_var.getName()}_{i}_ub')
        else:
            for i in var_dim_select:
                if markers:
                    baseline, = ax.plot(range(nodes_var), val[i, :], marker="o", markersize=2)

                else:
                    baseline, = ax.plot(range(nodes_var), val[i, :])

                if show_bounds:
                    lb, ub = abstract_var.getBounds()
                    lb_mat = np.reshape(lb, (abstract_var.getDim(), len(abstract_var.getNodes())), order='F')
                    ub_mat = np.reshape(ub, (abstract_var.getDim(), len(abstract_var.getNodes())), order='F')

                    if markers:
                        ax.plot(range(nodes_var), lb_mat[i, :], marker="x", markersize=3, linestyle='dotted', linewidth=1, color=baseline.get_color())
                        ax.plot(range(nodes_var), ub_mat[i, :], marker="x", markersize=3, linestyle='dotted', linewidth=1, color=baseline.get_color())
                    else:
                        ax.plot(range(nodes_var), lb_mat[i, :], linestyle='dotted')
                        ax.plot(range(nodes_var), ub_mat[i, :], linestyle='dotted')

                    if legend:
                        legend_list.append(f'{abstract_var.getName()}_{i}')
                        legend_list.append(f'{abstract_var.getName()}_{i}_lb')
                        legend_list.append(f'{abstract_var.getName()}_{i}_ub')

        if legend:
            ax.legend(legend_list)

    def plotVariables(self, names=None, grid=False, gather=None, markers=False, show_bounds=True, legend=True, dim=None):

        if self.solution is None:
            raise Exception('Solution not set. Cannot plot variables.')

        if names is None:
            selected_sol = self.solution
        else:
            if isinstance(names, str):
                names = [names]
            selected_sol = {name: self.solution[name] for name in names}

        if gather:

            fig, gs = create_grid(len(selected_sol), 'Variables', gather)
            i = 0
            for key, val in selected_sol.items():
                ax = fig.add_subplot(gs[i, :])
                if grid:
                    ax.grid(axis='x')
                self._plotVar(val, ax, self.prb.getVariables(key), markers=markers, show_bounds=show_bounds, legend=legend, dim=dim)

                # options
                ax.set_title('{}'.format(key))
                ax.ticklabel_format(useOffset=False, style='plain')
                ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
                # ax.set(xlabel='nodes', ylabel='vals')
                # mplt.xticks(list(range(val.shape[1])))
                i = i+1
        else:
            for key, val in selected_sol.items():
                fig, ax = mplt.subplots(layout='constrained')
                ax.set_title('{}'.format(key))
                if grid:
                    ax.grid(axis='x')
                self._plotVar(val, ax, self.prb.getVariables(key), markers=markers, show_bounds=show_bounds, legend=legend, dim=dim)


        fig.tight_layout()
        plt.show(block=False)

    def plotVariable(self, name, grid=False, markers=None, show_bounds=None, legend=None, dim=None):

        if self.solution is None:
            raise Exception('Solution not set. Cannot plot variable.')

        val = self.solution[name]

        fig, ax = plt.subplots(layout='constrained')
        if grid:
            ax.grid(axis='x')
        self._plotVar(val, ax, self.prb.getVariables(name), markers=markers, show_bounds=show_bounds, legend=legend, dim=dim)

        ax.set_title('{}'.format(name))
        # mplt.xticks(list(range(val.shape[1])))
        ax.set(xlabel='nodes', ylabel='vals')

    def plotFunctions(self, grid=False, gather=None, markers=None, show_bounds=None, legend=None, dim=None):

        if self.solution is None:
            raise Exception('Solution not set. Cannot plot functions.')

        if self.prb.getConstraints():
            if gather:
                fig, gs = create_grid(len(self.prb.getConstraints()), 'Functions', gather)

                i = 0
                for name, fun in self.prb.getConstraints().items():
                    ax = fig.add_subplot(gs[i])
                    if grid:
                        ax.grid(axis='x')
                    fun_evaluated = self.prb.evalFun(fun, self.solution)
                    self._plotVar(fun_evaluated, ax, self.prb.getConstraints(name), markers=markers, show_bounds=show_bounds, legend=legend, dim=dim)

                    ax.set_title('{}'.format(name))
                    plt.xticks(list(range(fun_evaluated.shape[1])))
                    ax.ticklabel_format(useOffset=False, style='plain')
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
                    i = i+1

            else:
                for name, fun in self.prb.getConstraints().items():
                    fig, ax = plt.subplots(layout='constrained')
                    ax.set_title('{}'.format(name))
                    if grid:
                        ax.grid(axis='x')
                    fun_evaluated = self.prb.evalFun(fun, self.solution)
                    self._plotVar(fun_evaluated, ax, self.prb.getConstraints(name), markers=markers, show_bounds=show_bounds, legend=legend, dim=dim)

            fig.tight_layout()
            plt.show(block=False)

    def plotFunction(self, name, grid=False, markers=None, show_bounds=None, legend=None, dim=None):

        if self.solution is None:
            raise Exception('Solution not set. Cannot plot functions.')

        fun = self.prb.getConstraints(name)

        fig, ax = plt.subplots(layout='constrained')
        ax.set_title('{}'.format(name))
        if grid:
            ax.grid(axis='x')
        fun_evaluated = self.prb.evalFun(fun, self.solution)
        self._plotVar(fun_evaluated, ax, self.prb.getConstraints(name), markers=markers, show_bounds=show_bounds, legend=legend, dim=dim)

        fig.tight_layout()
        plt.show(block=False)

if __name__ == '__main__':

    from xbot_interface import config_options as co
    from xbot_interface import xbot_interface as xbot
    from horizon.problem import Problem
    from horizon.rhc.model_description import FullModelInverseDynamics
    from horizon.rhc.taskInterface import TaskInterface
    from horizon.utils import trajectoryGenerator, analyzer, utils
    from horizon.ros import replay_trajectory
    import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn
    import phase_manager.pymanager as pymanager
    import phase_manager.pyphase as pyphase

    from sensor_msgs.msg import Joy
    from trajectory_msgs.msg import JointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from matplotlib import pyplot as plt

    import casadi as cs
    import rospy
    import rospkg
    import numpy as np
    import subprocess
    import os

    global joy_msg

    global base_pose
    global base_twist


    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'


    def gt_pose_callback(msg):
        global base_pose
        base_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                              msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                              msg.pose.orientation.w])


    def gt_twist_callback(msg):
        global base_twist
        base_twist = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
                               msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])


    def joy_callback(msg):
        global joy_msg
        joy_msg = msg


    rospy.init_node('cogimon_walk_srbd')

    solution_publisher = rospy.Publisher('/mpc_solution', JointTrajectory, queue_size=10)

    rospy.Subscriber('/joy', Joy, joy_callback)

    '''
    Load urdf and srdf
    '''
    cogimon_urdf_folder = rospkg.RosPack().get_path('cogimon_urdf')
    cogimon_srdf_folder = rospkg.RosPack().get_path('cogimon_srdf')

    urdf = open(cogimon_urdf_folder + '/urdf/cogimon.urdf', 'r').read()
    srdf = open(cogimon_srdf_folder + '/srdf/cogimon.srdf', 'r').read()

    file_dir = os.getcwd()

    '''
    Build ModelInterface and RobotStatePublisher
    '''
    cfg = co.ConfigOptions()
    cfg.set_urdf(urdf)
    cfg.set_srdf(srdf)
    cfg.generate_jidmap()
    cfg.set_string_parameter('model_type', 'RBDL')
    cfg.set_string_parameter('framework', 'ROS')
    cfg.set_bool_parameter('is_model_floating_base', True)

    global base_pose
    global base_twist

    base_pose = None
    base_twist = None
    try:
        robot = xbot.RobotInterface(cfg)
        rospy.Subscriber('/xbotcore/link_state/base_link/pose', PoseStamped, gt_pose_callback)
        rospy.Subscriber('/xbotcore/link_state/base_link/twist', TwistStamped, gt_twist_callback)
        while base_pose is None or base_twist is None:
            rospy.sleep(0.01)
        robot.sense()
        q_init = robot.getPositionReference()
        q_init = robot.eigenToMap(q_init)

    except:
        print(bcolors.WARNING + 'RobotInterface not created' + bcolors.ENDC)
        base_pose = np.array([0.03, 0., 0.962, 0., -0.029995, 0.0, 0.99955])
        q_init = {"LHipLat": -0.0,
                  "LHipSag": -0.363826,
                  "LHipYaw": 0.0,
                  "LKneePitch": 0.731245,
                  "LAnklePitch": -0.307420,
                  "LAnkleRoll": 0.0,
                  "RHipLat": 0.0,
                  "RHipSag": -0.363826,
                  "RHipYaw": 0.0,
                  "RKneePitch": 0.731245,
                  "RAnklePitch": -0.307420,
                  "RAnkleRoll": -0.0,
                  "WaistLat": 0.0,
                  "WaistYaw": 0.0,
                  "LShSag": 0.959931,
                  "LShLat": 0.007266,
                  "LShYaw": 0.,
                  "LElbj": -1.919862,
                  "LForearmPlate": 0.0,
                  "LWrj1": -0.523599,
                  "LWrj2": -0.0,
                  "RShSag": 0.959931,
                  "RShLat": -0.007266,
                  "RShYaw": 0.,
                  "RElbj": -1.919862,
                  "RForearmPlate": 0.0,
                  "RWrj1": -0.523599,
                  "RWrj2": -0.0}
        base_twist = np.zeros(6)
        robot = None

    '''
    Initialize Horizon problem
    '''
    ns = 20
    T = 1.5
    dt = T / ns

    prb = Problem(ns, receding=True, casadi_type=cs.SX)
    prb.setDt(dt)

    kin_dyn = casadi_kin_dyn.CasadiKinDyn(urdf)

    model = FullModelInverseDynamics(problem=prb,
                                     kd=kin_dyn,
                                     q_init=q_init,
                                     base_init=base_pose,
                                     sys_order_degree=3)

    rospy.set_param('/robot_description', urdf)
    bashCommand = 'rosrun robot_state_publisher robot_state_publisher'
    process = subprocess.Popen(bashCommand.split(), start_new_session=True)

    ti = TaskInterface(prb=prb, model=model)
    ti.setTaskFromYaml("/home/fruscelli/forest_ws/src/walk_me_maybe/config/config_walk_horizon_full_body.yaml")

    # ti.getTask('com_height').setRef(base_init.T)

    cd_fun = ti.model.kd.computeCentroidalDynamics()

    # adding minimization of angular momentum
    h_lin, h_ang, dh_lin, dh_ang = cd_fun(model.q, model.v, model.a)
    ti.prb.createIntermediateResidual('min_angular_mom', 1e-1 * dh_ang)

    # open loop regularizations
    # prb.createResidual('vel_regularization', 1. * model.v)

    # acc_reference = prb.createParameter('acc_reference', model.nv)
    # acc_reference.assign(np.zeros(model.nv))
    # prb.createResidual('acc_regularization', 1e-2 * (model.a - acc_reference))

    # force_reference = dict()
    # for force_name, force_value in model.fmap.items():
    #     force_reference[f'force_reference_{force_name}'] = prb.createParameter(f'force_reference_{force_name}', 3)
    #     force_reference[f'force_reference_{force_name}'].assign(np.array([0, 0, model.kd.mass() / 8]))
    #     prb.createResidual(f'force_regularization_{force_name}', 5e-3 * (force_value[2] - force_reference[f'force_reference_{force_name}'][2]))
    #     prb.createResidual(f'force_regularization_xy_{force_name}', 5e-3 * (force_value[0:2]))

    # prb.createIntermediateResidual('jerk_regularization', 1e-3 * model.j)
    # for force_name in model.fmap.keys():
    #     prb.createIntermediateResidual(f'force_dot_regularization_{force_name}', 1e-3 * model.getInput()[f'fdot_{force_name}'])

    # close loop regularizations
    prb.createResidual('vel_regularization', 1. * model.v)

    acc_reference = prb.createParameter('acc_reference', model.nv)
    acc_reference.assign(np.zeros(model.nv))
    prb.createResidual('acc_regularization', 5e-2 * (model.a - acc_reference))

    force_reference = dict()
    for force_name, force_value in model.fmap.items():
        force_reference[f'force_reference_{force_name}'] = prb.createParameter(f'force_reference_{force_name}', 3)
        force_reference[f'force_reference_{force_name}'].assign(np.array([0, 0, model.kd.mass() / 8]))
        prb.createResidual(f'force_regularization_{force_name}',
                           5e-3 * (force_value[2] - force_reference[f'force_reference_{force_name}'][2]))
        prb.createResidual(f'force_regularization_xy_{force_name}', 5e-3 * (force_value[0:2]))
    #
    prb.createIntermediateResidual('jerk_regularization', 1e-3 * model.j)
    for force_name in model.fmap.keys():
        prb.createIntermediateResidual(f'force_dot_regularization_{force_name}',
                                       1e-3 * model.getInput()[f'fdot_{force_name}'])

    '''
    Foot vertices relative distance constraint
    '''
    pos_lf = model.kd.fk('l_sole')(q=model.q)['ee_pos']
    pos_rf = model.kd.fk('r_sole')(q=model.q)['ee_pos']
    base_ori = utils.toRot(model.kd.fk('base_link')(q=model.q0)['ee_rot'])
    rel_dist = base_ori.T @ (pos_lf - pos_rf)

    # prb.createResidual('relative_distance_lower_x', utils.barrier(rel_dist[0] + 0.3))
    # prb.createResidual('relative_distance_upper_x', utils.barrier1(rel_dist[0] - 0.4))
    prb.createResidual('relative_distance_lower_y', 10 * utils.barrier(rel_dist[1] - 0.25))
    prb.createResidual('relative_distance_upper_y', utils.barrier1(rel_dist[1] + 1.))

    tg = trajectoryGenerator.TrajectoryGenerator()

    pm = pymanager.PhaseManager(ns + 1)
    # phase manager handling
    c_phases = dict()
    for c in model.cmap:
        c_phases[c] = pm.addTimeline(f'{c}_timeline')

    step_time = 0.75
    for c in model.cmap:
        # stance phase
        time_flight = step_time
        stance_duration = int(time_flight / dt)
        stance_phase = pyphase.Phase(stance_duration, f'stance_{c}')
        stance_phase.addItem(ti.getTask(f'foot_contact_{c}'))
        c_phases[c].registerPhase(stance_phase)

        time_double_stance = 0.6
        short_stance_duration = int(time_double_stance / dt)
        short_stance_phase = pyphase.Phase(short_stance_duration, f'short_stance_{c}')
        short_stance_phase.addItem(ti.getTask(f'foot_contact_{c}'))
        c_phases[c].registerPhase(short_stance_phase)

        time_flight = step_time
        flight_duration = int(time_flight / dt)
        flight_phase = pyphase.Phase(flight_duration, f'flight_{c}')
        #
        init_z_foot = model.kd.fk(c)(q=model.q0)['ee_pos'].elements()[2]
        #
        ref_trj = np.zeros(shape=[7, flight_duration])
        ref_trj[2, :] = np.atleast_2d(
            tg.from_derivatives(flight_duration, init_z_foot, init_z_foot, 0.05, [None, 0, None]))
        #
        flight_phase.addItemReference(ti.getTask(f'foot_z_{c}'), ref_trj)
        # flight_phase.addItem(ti.getTask(f'foot_contact_{c}'), nodes=[flight_duration-1]) #, nodes=[flight_duration-1]

        # v_contact = model.kd.frameVelocity(c, model.kd_frame)(q=model.q, qdot=model.v)['ee_vel_linear']
        # p_contact = model.kd.fk(c)(q=model.q)['ee_pos'].elements()
        # last_swing_vel = prb.createConstraint(f'{c}last_swing_vel', v_contact, [])
        # last_swing_zero = prb.createConstraint(f'{c}_last_swing_zero', p_contact[2] - init_z_foot, [])
        # flight_phase.addConstraint(last_swing_vel, nodes=[flight_duration-1])
        # for contact in contact_dict[c]['vertex_frames']:
        #     flight_phase.addVariableBounds(prb.getVariables(f'f_{contact}'), np.array([[0, 0, 0]]).T, np.array([[np.inf, np.inf, np.inf]]).T, nodes=[flight_duration-1])

        c_phases[c].registerPhase(flight_phase)

    for c in model.cmap:
        stance = c_phases[c].getRegisteredPhase(f'stance_{c}')
        flight = c_phases[c].getRegisteredPhase(f'flight_{c}')
        short_stance = c_phases[c].getRegisteredPhase(f'short_stance_{c}')
        while c_phases[c].getEmptyNodes() > 0:
            c_phases[c].addPhase(stance)

    model.q.setBounds(model.q0, model.q0, nodes=0)
    model.v.setBounds(model.v0, model.v0, nodes=0)
    model.a.setBounds(np.zeros(model.nv), np.zeros(model.nv), nodes=0)
    model.q.setInitialGuess(ti.model.q0)

    f0 = [0, 0, kin_dyn.mass() / 8 * 9.8]
    for cname, cforces in ti.model.cmap.items():
        for c in cforces:
            c.setInitialGuess(f0)

    # finalize taskInterface and solve bootstrap problem
    ti.finalize()

    # fig1 = mplt.figure()
    # ax_p = mplt.subplot()
    # ax_v = mplt.subplot()
    # ax_a = mplt.subplot()

    # fig2 = mplt.figure()
    # ax_f = mplt.subplot()

    # fig3 = mplt.figure()
    # ax_t0 = mplt.subplot()
    # ax_t1 = mplt.subplot()
    # ax_t2 = mplt.subplot()
    # ax_t3 = mplt.subplot()
    # ax_t4 = mplt.subplot()
    # ax_t5 = mplt.subplot()
    # ax_tnr0 = mplt.subplot()
    # ax_tnr1 = mplt.subplot()
    # ax_tnr2 = mplt.subplot()
    # ax_tnr3 = mplt.subplot()
    # ax_tnr4 = mplt.subplot()
    # ax_tnr5 = mplt.subplot()

    # anal = analyzer.ProblemAnalyzer(prb)
    # anal.print()
    # anal.printVariables('f_l_foot_lower_left_link')
    # anal.printConstraints('zero_velocity_l_foot_l_sole_vel_cartesian_task')
    ti.bootstrap()
    ti.load_initial_guess()
    solution = ti.solution


    # hplt = Plotter(prb, solution)
    hplt = PlotterHorizon(prb, solution)

    # fig, axes = mplt.subplots(3, 2, sharex=True, gridspec_kw=dict(height_ratios=[3, 1, 1]))
    # fig.suptitle('cazzi')
    # hplt.plot(axes[0][0], prb.getConstraints('dynamics'), [0, 1, 2], args=dict(color='r'))

    hplt.plotVariables(['q', 'v'], gather=1)
    plt.show()

    exit()
    iteration = 0
    rate = rospy.Rate(1 / dt)

    contact_list_repl = list(model.cmap.keys())
    repl = replay_trajectory.replay_trajectory(dt, model.kd.joint_names(), np.array([]),
                                               {k: None for k in model.fmap.keys()},
                                               model.kd_frame, model.kd, trajectory_markers=contact_list_repl)  #


    def step(swing, stance):
        c_phases[swing].addPhase(c_phases[swing].getRegisteredPhase(f'flight_{swing}'))
        c_phases[stance].addPhase(c_phases[stance].getRegisteredPhase(f'stance_{stance}'))
        c_phases[stance].addPhase(c_phases[stance].getRegisteredPhase(f'short_stance_{stance}'))
        c_phases[swing].addPhase(c_phases[swing].getRegisteredPhase(f'short_stance_{swing}'))


    global joy_msg
    np.set_printoptions(precision=3, suppress=True)
    while not rospy.is_shutdown():

        # get initial guess
        x_opt = solution['x_opt']
        u_opt = solution['u_opt']

        # this part shifts variables initial guess and bounds
        shift_num = -1
        xig = np.roll(x_opt, shift_num, axis=1)

        for i in range(abs(shift_num)):
            xig[:, -1 - i] = x_opt[:, -1]
        prb.getState().setInitialGuess(xig)

        uig = np.roll(u_opt, shift_num, axis=1)

        for i in range(abs(shift_num)):
            uig[:, -1 - i] = u_opt[:, -1]
        prb.getInput().setInitialGuess(uig)

        # setting lower and upper bounds of state
        prb.setInitialState(x0=xig[:, 0])

        if robot is not None:
            robot.sense()
            q = robot.getJointPosition()
            q = np.hstack([base_pose, q])
            model.q.setBounds(q, q, nodes=0)

            qdot = robot.getJointVelocity()
            qdot = np.hstack([base_twist, qdot])
            model.v.setBounds(qdot, qdot, nodes=0)

        # update acceleration and force references
        # acc_reference.assign(np.roll(solution['a'], shift_num, axis=1))
        # for force_name in model.fmap.keys():
        #     force_reference[f'force_reference_{force_name}'].assign(solution[f'f_{force_name}'])

        # shift phases of phase manager
        pm._shift_phases()

        # add a new phase when the timeline starts to be empty
        for c in model.cmap:
            if c_phases[c].getEmptyNodes() > 0:
                if joy_msg.buttons[4] == 1:
                    if c_phases['l_sole'].getEmptyNodes() > 0:
                        step('l_sole', 'r_sole')
                        step('r_sole', 'l_sole')
                else:
                    for c in model.cmap:
                        if c_phases[c].getEmptyNodes() > 0:
                            c_phases[c].addPhase(c_phases[c].getRegisteredPhase(f'stance_{c}'))

        if np.abs(joy_msg.axes[1]) > 0.1:
            final_base_x = ti.getTask('final_base_x')
            reference = np.atleast_2d(np.array([solution['q'][0, 0] + 1.5 * joy_msg.axes[1], 0., 0., 0., 0., 0., 0.]))
            final_base_x.setRef(reference.T)
        else:
            final_base_x = ti.getTask('final_base_x')
            reference = np.atleast_2d(np.array([solution['q'][0, 0], 0., 0., 0., 0., 0., 0.]))
            final_base_x.setRef(reference.T)

        if np.abs(joy_msg.axes[0]) > 0.1:
            final_base_y = ti.getTask('final_base_y')
            reference = np.atleast_2d(np.array([0., solution['q'][1, 0] + 0.5 * joy_msg.axes[0], 0., 0., 0., 0., 0.]))
            final_base_y.setRef(reference.T)
        else:
            final_base_y = ti.getTask('final_base_y')
            reference = np.atleast_2d(np.array([0., solution['q'][1, 0], 0., 0., 0., 0., 0.]))
            final_base_y.setRef(reference.T)

        iteration = iteration + 1

        # solve real time iteration
        ti.rti()

        # get new solution and overwrite old one
        solution = ti.solution
        dt_res = 0.01
        ti.resample(dt_res=dt_res, nodes=[0, 1], resample_tau=False)

        tau = list()

        for i in range(solution['q_res'].shape[1] - 1):
            tau.append(
                ti.model.computeTorqueValues(solution['q_res'][:, i], solution['v_res'][:, i], solution['a_res'][:, i],
                                             {name: solution['f_' + name + '_res'][:, i] for name in model.fmap}))

        tau_not_res = list()
        for i in range(solution['q'].shape[1] - 1):
            tau_not_res.append(
                ti.model.computeTorqueValues(solution['q'][:, i], solution['v'][:, i], solution['a'][:, i],
                                             {name: solution['f_' + name][:, i] for name in model.fmap}))

        jt = JointTrajectory()
        for i in range(solution['q_res'].shape[1]):
            jtp = JointTrajectoryPoint()
            jtp.positions = solution['q_res'][:, i].tolist()
            jtp.velocities = solution['v_res'][:, i].tolist()
            if i < len(tau):
                jtp.accelerations = solution['a_res'][:, i].tolist()
                jtp.effort = tau[i].elements()
            else:
                jtp.accelerations = solution['a_res'][:, -1].tolist()
                jtp.effort = tau[-1].elements()

            jt.points.append(jtp)
        jt.joint_names = [elem for elem in kin_dyn.joint_names() if elem not in ['universe', 'reference']]
        jt.header.stamp = rospy.Time.now()

        solution_publisher.publish(jt)

        # mplt.ion()
        # mplt.show()
        # ax_p.scatter(range(ns + 1), solution['q'][10, :])
        # ax_p.plot(np.linspace(0, 1, solution['q_res'].shape[1]), solution['q_res'][10, :])
        # ax_v.scatter(range(ns + 1), solution['v'][9, :])
        # ax_v.plot(np.linspace(0, 1, solution['v_res'].shape[1]), solution['v_res'][9, :])
        # ax_a.scatter(range(ns + 1), solution['a'][9, :])
        # ax_a.plot(np.linspace(0, 1, solution['a_res'].shape[1]), solution['a_res'][9, :])

        # ax_f.plot(range(ns + 1), solution['f_l_foot_lower_left_link'][2, :])
        # ax_f.set_ylim(0, 350)

        # tau_vect0 = list()
        # tau_vect1 = list()
        # tau_vect2 = list()
        # tau_vect3 = list()
        # tau_vect4 = list()
        # tau_vect5 = list()
        # tau_vect_not_res0 = list()
        # tau_vect_not_res1 = list()
        # tau_vect_not_res2 = list()
        # tau_vect_not_res3 = list()
        # tau_vect_not_res4 = list()
        # tau_vect_not_res5 = list()
        # for t in tau:
        #     tau_vect0.append(t[0].elements()[0])
        #     tau_vect1.append(t[1].elements()[0])
        #     tau_vect2.append(t[2].elements()[0])
        #     tau_vect3.append(t[3].elements()[0])
        #     tau_vect4.append(t[4].elements()[0])
        #     tau_vect5.append(t[5].elements()[0])
        # for t in tau_not_res:
        #     tau_vect_not_res0.append(t[0].elements()[0])
        #     tau_vect_not_res1.append(t[1].elements()[0])
        #     tau_vect_not_res2.append(t[2].elements()[0])
        #     tau_vect_not_res3.append(t[3].elements()[0])
        #     tau_vect_not_res4.append(t[4].elements()[0])
        #     tau_vect_not_res5.append(t[5].elements()[0])

        # ax_t0.plot(np.linspace(0, 1, len(tau)), tau_vect0)
        # ax_t1.plot(np.linspace(0, 1, len(tau)), tau_vect1)
        # ax_t2.plot(np.linspace(0, 1, len(tau)), tau_vect2)
        # ax_t3.plot(np.linspace(0, 1, len(tau)), tau_vect3)
        # ax_t4.plot(np.linspace(0, 1, len(tau)), tau_vect4)
        # ax_t5.plot(np.linspace(0, 1, len(tau)), tau_vect5)
        # ax_tnr0.scatter(range(ns), tau_vect_not_res0[:])
        # ax_tnr1.scatter(range(ns), tau_vect_not_res1[:])
        # ax_tnr2.scatter(range(ns), tau_vect_not_res2[:])
        # ax_tnr3.scatter(range(ns), tau_vect_not_res3[:])
        # ax_tnr4.scatter(range(ns), tau_vect_not_res4[:])
        # ax_tnr5.scatter(range(ns), tau_vect_not_res5[:])
        # ax_t0.set_ylim(-50, 50)

        # mplt.pause(0.001)
        # ax_p.clear()
        # ax_v.clear()
        # ax_a.clear()
        # ax_f.clear()
        # ax_t0.clear()
        # ax_t1.clear()
        # ax_t2.clear()
        # ax_t3.clear()
        # ax_t4.clear()
        # ax_t5.clear()
        # ax_tnr0.clear()
        # ax_tnr1.clear()
        # ax_tnr2.clear()
        # ax_tnr3.clear()
        # ax_tnr4.clear()
        # ax_tnr5.clear()

        # replay stuff
        repl.frame_force_mapping = {cname: solution[f.getName()] for cname, f in ti.model.fmap.items()}
        repl.publish_joints(solution['q'][:, 0])
        repl.publishContactForces(rospy.Time.now(), solution['q'][:, 0], 0)
        rate.sleep()



