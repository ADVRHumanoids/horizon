from horizon.utils import plotter
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.ros import replay_trajectory
from horizon.solvers.solver import Solver
from horizon.rhc.taskInterface import TaskInterface
from horizon.rhc.model_description import *
from horizon.utils.utils import barrier, barrier_1
from horizon.utils.patternGenerator import PatternGenerator
from horizon.utils.trajectoryGenerator import TrajectoryGenerator
from horizon.utils.mat_storer import matStorer
import numpy as np
import rospkg
import casadi as cs
import rospy
import subprocess
import time

urdf_path = rospkg.RosPack().get_path('mirror_urdf') + '/urdf/mirror.urdf'
urdf = open(urdf_path, 'r').read()
kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
kd = pycasadi_kin_dyn.CasadiKinDyn(urdf)
rospy.set_param('/robot_description', urdf)
subprocess.Popen(['rosrun', 'robot_state_publisher', 'robot_state_publisher'])

solver_type = 'ipopt'
transcription_method = 'multiple_shooting'
transcription_opts = dict(integrator='RK4')

resample_flag = True
plot_flag = False

init_nodes = 20
end_nodes = 20
cycle_nodes = 30
clearance = 0.3
duty_cycle = 1.
vertical_constraint_nodes = 2
# ns = 100
ns = init_nodes + cycle_nodes + end_nodes
tf = 20.0  # 10s
dt = tf / ns

problem_opts = {'ns': ns, 'tf': tf, 'dt': dt}

q_init = {}

for i in range(3):
    q_init[f'arm_{i + 1}_joint_2'] = -1.2 #-0.9 #  -1.9
    q_init[f'arm_{i + 1}_joint_3'] =-2.10 #- 1.8 # -2.30
    q_init[f'arm_{i + 1}_joint_5'] = -0.9 #- 0.9 # -0.4

base_init = np.array([0, 0, 0.72, 0, 0, 0, 1])

# todo: this should not be in initialization
#  I should add contacts after the initialization, as forceTasks, no?

# set up model
prb = Problem(ns, receding=True)  # logging_level=logging.DEBUG
prb.setDt(dt)

model = FullModelInverseDynamics(problem=prb,
                                 kd=kd,
                                 q_init=q_init,
                                 base_init=base_init)

contacts = [f'arm_{i + 1}_TCP' for i in range(3)]

for contact in contacts:
    model.setContactFrame(contact, 'vertex', dict(vertex_frames=[contact]))

# for contact in contacts:
#     model.setContactFrame(contact, 'vertex', dict(vertex_frames=[contact]))

ptgt_final = base_init.copy()
# ptgt_final[0] += base_disp_x
# ptgt_final[1] += base_disp_y
# ptgt_final[2] = base_rot
# ptgt_final[2] =

# base x position
base_pos_x_param = prb.createParameter('base_pos_x_param', 1)
prb.createFinalConstraint('base_link_x', model.q[0] - base_pos_x_param)

# base y position
base_pos_y_param = prb.createParameter('base_pos_y_param', 1)
prb.createFinalResidual('base_link_y', 1e3 * (model.q[1] - base_pos_y_param))

base_pos_rot_param = prb.createParameter('base_pos_rot_param', 1)
prb.createFinalResidual('base_link_rot', 1e3 * (model.q[5] - base_pos_rot_param))
# ================================================================================
# ================================================================================

gait_matrix = np.array([[1, 1],
                        [0, 1],
                        [0, 0]]).astype(int)

flight_with_duty = int(cycle_nodes / gait_matrix.shape[1] * duty_cycle)
stance_with_duty = int(cycle_nodes / gait_matrix.shape[1] * (1 - duty_cycle))

print('number of nodes: ', ns)
print('init phase duration:', init_nodes * dt, f'({init_nodes})')
print('final phase duration:', end_nodes * dt, f'({end_nodes})')
print('nodes flight duration:', flight_with_duty * dt, f'({flight_with_duty})')
print('nodes stance duration:', stance_with_duty * dt, f'({stance_with_duty})')

# create pattern
pg = PatternGenerator(cycle_nodes, contacts)
stance_nodes, swing_nodes, cycle_nodes = pg.generateCycle_old(gait_matrix, cycle_nodes, duty_cycle)

list_init_nodes = list(range(init_nodes))

list_move_nodes = dict()
for c in contacts:
    list_move_nodes[c] = [init_nodes + elem for elem in stance_nodes[c]]

list_end_nodes = [init_nodes + cycle_nodes + elem for elem in range(end_nodes + 1)]

for c in contacts:
    stance_nodes[c] = list_init_nodes + list_move_nodes[c] + list_end_nodes

    swing_nodes[c] = [init_nodes + elem for elem in swing_nodes[c]]

for name, elem in stance_nodes.items():
    print(f'{name}: {elem}')

for name, elem in swing_nodes.items():
    print(f'{name}: {elem}')

# ================================================================================
# ================================================================================

contact_pos = dict()
z_des = dict()
pos_des = dict()
clea = dict()
pos_constr = dict()
for contact in contacts:
    FK = kd.fk(contact)
    DFK = kd.frameVelocity(contact, kd_frame)
    DDFK = kd.frameAcceleration(contact, kd_frame)

    ee_pos = FK(q=model.q)['ee_pos']
    ee_rot = FK(q=model.q)['ee_rot']
    ee_v = DFK(q=model.q, qdot=model.v)['ee_vel_linear']
    ee_v_ang = DFK(q=model.q, qdot=model.v)['ee_vel_angular']
    a = DDFK(q=model.q, qdot=model.v)['ee_acc_linear']

    contact_pos[contact] = FK(q=model.q0)['ee_pos']

    initial_ee_rotation_z = FK(q=model.q0)['ee_rot'][2, :2]

    if np.any(np.absolute(initial_ee_rotation_z) > 0.0001):
        raise Exception('ee rotation is not zero:', initial_ee_rotation_z)

    if not swing_nodes[contact]:
        # vertical contact frame
        rot_err = cs.sumsqr(ee_rot[2, :2])
        prb.createIntermediateCost(f'{contact}_rot', 1e4 * rot_err)

    # barrier force
    fcost = barrier(model.fmap[contact][2] - 100.0)  # fz > 10
    prb.createResidual(f'{contact}_unil', 1e1 * fcost, nodes=stance_nodes[contact][:-1])  # [1:-1]

    # contact constraint
    prb.createConstraint(f"{contact}_vel", cs.vertcat(ee_v, ee_v_ang), nodes=stance_nodes[contact])

    if swing_nodes[contact]:

        rot_err = cs.sumsqr(ee_rot[2, :2])
        prb.createIntermediateCost(f'{contact}_rot', 1e4 * rot_err)

        # clearance
        contact_pos[contact] = FK(q=model.q0)['ee_pos']
        z_des[contact] = prb.createParameter(f'{contact}_z_des', 1)
        clea[contact] = prb.createConstraint(f"{contact}_clea", ee_pos[2] - z_des[contact], nodes=swing_nodes[contact])

        # xy pos
        # pos_des[contact] = prb.createParameter(f'{contact}_pos_goal', 2)
        # pos_constr[contact] = prb.createConstraint(f"{contact}_pos_goal", ee_pos[:2] - pos_des[contact],
        #                                            nodes=swing_nodes[contact][-1])

        # zero force
        model.fmap[contact].setBounds(np.array([[0, 0, 0]] * len(swing_nodes[contact])).T,
                                      np.array([[0, 0, 0]] * len(swing_nodes[contact])).T,
                                      nodes=swing_nodes[contact])

        # vertical takeoff and touchdown
        lat_vel = cs.vertcat(ee_v[0:2], ee_v_ang)
        vert = prb.createConstraint(f"{contact}_vert", lat_vel,
                                    nodes=swing_nodes[contact][:vertical_constraint_nodes] + swing_nodes[contact][
                                                                                             -vertical_constraint_nodes:])


tg = TrajectoryGenerator()

for contact, z_constr in z_des.items():
    pos_z = contact_pos[contact][2].elements()[0]
    z_trj = np.atleast_2d(tg.from_derivatives(len(swing_nodes[contact]), pos_z, pos_z, clearance, [0, 0, 0]))
    z_des[contact].assign(z_trj, nodes=swing_nodes[contact])


# # assign xy
# dtheta = 2 * np.pi / 3

# theta = 0
# for contact, pos_constr in pos_constr.items():
#     pos_xy = contact_pos[contact].toarray().flatten()[:2]
#     pos_goal = np.array([lenght_disp * cs.cos(theta), lenght_disp * cs.sin(theta)])
#
#     pos_des[contact].assign(pos_xy + pos_goal, nodes=swing_nodes[contact][-1])
#     theta += dtheta
#
#     print(f'contact {contact} moving from {pos_xy} to {pos_xy + pos_goal}')

# base_pos_x_param.assign(ptgt_final[0])
# base_pos_y_param.assign(ptgt_final[1])
# base_pos_rot_param.assign(ptgt_final[2])

q0 = model.q0
v0 = model.v0

f0 = np.array([0, 0, kd.mass() * 9.8 / 3])

# joint limits
model.q.setBounds(kd.q_min(), kd.q_max())

# final velocity
model.v.setBounds(v0, v0, nodes=ns)

# base rotation
prb.createResidual("min_rot", 1e-4 * (model.q[3:5] - q0[3:5]))

# joint posture
prb.createIntermediateResidual("min_q", 1e1 * (model.q[7:] - q0[7:]))

# joint velocity
prb.createResidual("min_v", 1e-3 * model.v)

# final posture
# todo: incredible, this is the problem. if it's FinalResidual, everything goes to whores
# prb.createIntermediateResidual("min_qf", 1e1 * (model.q[7:] - q0[7:]))
prb.createFinalResidual("min_qf_final", 1e2 * (model.q[7:] - model.q0[7:]))
#
# regularize input
prb.createIntermediateResidual("min_q_ddot", 1e0 * model.a)

# regularize forces
for f in model.fmap.values():
    prb.createIntermediateResidual(f"min_{f.getName()}", 1e-3 * (f - f0))

# com_fn = kd.centerOfMass()


# set initial condition and initial guess
model.q.setBounds(q0, q0, nodes=0)
model.v.setBounds(v0, v0, nodes=0)

model.q.setInitialGuess(q0)

for f in model.fmap.values():
    f.setInitialGuess(f0)

model.setDynamics()

if solver_type != 'ilqr':
    Transcriptor.make_method(transcription_method, prb, transcription_opts)

# print('VARIABLES:')
# for var_name, obj in prb.getVariables().items():
#     print(var_name, ':', type(obj))
#     print(obj)
#     print(obj.getNodes().tolist())
#     print(obj.getBounds())
#
# print('CONSTRAINTS:')
# for cnsrt, obj in prb.getConstraints().items():
#     print(cnsrt,':', type(obj))
#     print(obj.getFunction())
#     print(obj._fun_impl)
#     print(obj.getNodes())
#     print(obj.getBounds())
# #
# print('COSTS:')
# for cnsrt, obj in prb.getCosts().items():
#     print(cnsrt,':', obj.getNodes(), type(obj))


opts = {'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 1e-3,
        'ipopt.max_iter': 1000,
        'error_on_fail': True,
        'ilqr.max_iter': 200,
        'ilqr.alpha_min': 0.01,
        'ilqr.use_filter': False,
        'ilqr.hxx_reg': 0.0,
        'ilqr.integrator': 'RK4',
        'ilqr.merit_der_threshold': 1e-6,
        'ilqr.step_length_threshold': 1e-9,
        'ilqr.line_search_accept_ratio': 1e-4,
        'ilqr.kkt_decomp_type': 'qr',
        'ilqr.constr_decomp_type': 'qr',
        'ilqr.verbose': True,
        'ipopt.linear_solver': 'ma57',
        }

opts_rti = opts.copy()
opts_rti['ilqr.enable_line_search'] = False
opts_rti['ilqr.max_iter'] = 4

solver_bs = Solver.make_solver(solver_type, prb, opts)

try:
    solver_bs.set_iteration_callback()
except:
    pass

t = time.time()
solver_bs.solve()
elapsed = time.time() - t
print(f'bootstrap solved in {elapsed} s')

solution = solver_bs.getSolutionDict()

# ================================================================
if resample_flag:
    from horizon.utils import resampler_trajectory

    dt_res = 0.001
    u_res = resampler_trajectory.resample_input(
        solution['u_opt'],
        prb.getDt(),
        dt_res)

    x_res = resampler_trajectory.resampler(
        solution['x_opt'],
        solution['u_opt'],
        prb.getDt(),
        dt_res,
        dae=None,
        f_int=prb.getIntegrator())

    solution['dt_res'] = dt_res
    solution['x_opt_res'] = x_res
    solution['u_opt_res'] = u_res

    for s in prb.getState():
        sname = s.getName()
        off, dim = prb.getState().getVarIndex(sname)
        solution[f'{sname}_res'] = x_res[off:off + dim, :]

    for s in prb.getInput():
        sname = s.getName()
        off, dim = prb.getInput().getVarIndex(sname)
        solution[f'{sname}_res'] = u_res[off:off + dim, :]

    # get tau resampled

    # new fmap with resampled forces
    if model.fmap:

        fmap = dict()
        for frame, wrench in model.fmap.items():
            fmap[frame] = solution[f'{wrench.getName()}']

        fmap_res = dict()
        for frame, wrench in model.fmap.items():
            fmap_res[frame] = solution[f'{wrench.getName()}_res']

        tau = np.zeros([model.tau.shape[0], prb.getNNodes() - 1])
        tau_res = np.zeros([model.tau.shape[0], u_res.shape[1]])

        id = kin_dyn.InverseDynamics(model.kd, fmap_res.keys(), model.kd_frame)

        # id_fn = kin_dyn.InverseDynamics(self.kd, self.fmap.keys(), self.kd_frame)
        # tau = id_fn.call(q, v, a, fmap)
        # prb.createIntermediateConstraint('dynamics', tau[:6])

        # todo: this is horrible. id.call should take matrices, I should not iter over each node

        for i in range(tau.shape[1]):
            fmap_i = dict()
            for frame, wrench in fmap.items():
                fmap_i[frame] = wrench[:, i]
            tau_i = id.call(solution['q'][:, i], solution['v'][:, i], solution['a'][:, i], fmap_i)
            tau[:, i] = tau_i.toarray().flatten()

        for i in range(tau_res.shape[1]):
            fmap_res_i = dict()
            for frame, wrench in fmap_res.items():
                fmap_res_i[frame] = wrench[:, i]
            tau_res_i = id.call(solution['q_res'][:, i], solution['v_res'][:, i], solution['a_res'][:, i],
                                fmap_res_i)
            tau_res[:, i] = tau_res_i.toarray().flatten()

        solution['tau'] = tau
        solution['tau_res'] = tau_res

# ====================store solutions ============================
name_stored = f'mat_files/mirror_balancing_1_leg.mat'
ms = matStorer(name_stored)
info_dict = dict(n_nodes=prb.getNNodes(), dt=prb.getDt(),
                 init_nodes=init_nodes,
                 end_nodes=end_nodes,
                 cycle_node=cycle_nodes,
                 clearance=clearance,
                 duty_cycle=duty_cycle,
                 tf=tf,
                 )


ms.store({**solution, **info_dict})
print('solution stored as', name_stored)
## single replay


if not plot_flag:
    q_sol = solution['q']
    frame_force_mapping = {contacts[i]: solution[f'f_{contacts[i]}'] for i in range(3)}
    repl = replay_trajectory.replay_trajectory(dt, kd.joint_names(), q_sol, frame_force_mapping, kd_frame, kd)
    repl.sleep(1.)
    repl.replay(is_floating_base=True)
# exit()

else:
    import matplotlib.pyplot as plt
    import matplotlib

    plt.figure()
    for contact in contacts:
        FK = kd.fk(contact)
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'feet position - plane_xy')
        plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[1, :]).flatten(), linewidth=2.5)
        plt.scatter(np.array(pos[0, 0]), np.array(pos[1, 0]))
        plt.scatter(np.array(pos[0, -1]), np.array(pos[1, -1]), marker='x')

    plt.figure()
    for contact in contacts:
        FK = kd.fk(contact)
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'feet position - plane_xz')
        plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[2, :]).flatten(), linewidth=2.5)
        plt.scatter(np.array(pos[0, 0]), np.array(pos[2, 0]))
        plt.scatter(np.array(pos[0, -1]), np.array(pos[2, -1]), marker='x')

    hplt = plotter.PlotterHorizon(prb, solution)
    hplt.plotVariables([elem.getName() for elem in model.fmap.values()], show_bounds=True, gather=2, legend=False)
    hplt.plotVariables(['q'], show_bounds=True, gather=2, legend=False)
    matplotlib.pyplot.show()
