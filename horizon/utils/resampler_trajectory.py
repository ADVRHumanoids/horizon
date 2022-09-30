from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from horizon.utils import kin_dyn
from horizon.transcriptions import integrators
import numpy as np
import casadi as cs

def resample_torques(p, v, a, node_time, dt, dae, frame_force_mapping, kindyn, force_reference_frame = cas_kin_dyn.CasadiKinDyn.LOCAL):
    """
        Resample solution to a different number of nodes, RK4 integrator is used for the resampling
        Args:
            p: position
            v: velocity
            a: acceleration
            node_time: previous node time
            dt: resampled period
            dae: a dictionary containing
                    'x': state
                    'p': control
                    'ode': a function of the state and control returning the derivative of the state
                    'quad': quadrature term
            frame_force_mapping: dictionary containing a map between frames and force variables e.g. {'lsole': F1}
            kindyn: object of type casadi_kin_dyn
            force_reference_frame: this is the frame which is used to compute the Jacobian during the ID computation:
                    LOCAL (default)
                    WORLD
                    LOCAL_WORLD_ALIGNED

        Returns:
            p_res: resampled p
            v_res: resampled v
            a_res: resampled a
            frame_res_force_mapping: resampled frame_force_mapping
            tau_res: resampled tau
        """
    p_res, v_res, a_res = second_order_resample_integrator(p, v, a, node_time, dt, dae)

    frame_res_force_mapping = dict()

    for frame, wrench in frame_force_mapping.items():
        frame_res_force_mapping[frame] = resample_input(wrench, node_time, dt)
    tau_res = np.zeros(a_res.shape)

    ID = kin_dyn.InverseDynamics(kindyn, frame_force_mapping.keys(), force_reference_frame)
    ni = a_res.shape[1]
    for i in range(ni):
        frame_force_map_i = dict()
        for frame, wrench in frame_res_force_mapping.items():
            frame_force_map_i[frame] = wrench[:, i]
        tau_i = ID.call(p_res[:, i], v_res[:, i], a_res[:, i], frame_force_map_i)
        tau_res[:, i] = tau_i.toarray().flatten()


    return p_res, v_res, a_res, frame_res_force_mapping, tau_res

def resample_input(input, node_time, dt):
    """
    Resample an input variable according to a new sample time dt.
    NOTE: the resampling is done by considering constant input between nodes
    Args:
        input: input to resample
        node_time: original node time
        dt: new node time
    Returns:
        input_res: resampled input
    """
    number_of_nodes = input.shape[1]+1
    node_time_array = np.zeros([number_of_nodes])
    if hasattr(node_time, "__iter__"):
        for i in range(1, number_of_nodes):
            node_time_array[i] = node_time_array[i - 1] + node_time[i - 1]
    else:
        for i in range(1, number_of_nodes):
            node_time_array[i] = node_time_array[i - 1] + node_time

    n_res = int(round(node_time_array[-1] / dt))

    input_res = np.zeros([input.shape[0], n_res])

    t = 0.
    node = 0
    i = 0
    while i < input_res.shape[1]:
        input_res[:, i] = input[:, node]
        t += dt
        i += 1
        if t > node_time_array[node + 1]:
            node += 1

    return input_res


def second_order_resample_integrator(p, v, u, node_time, dt, dae):
    """
    Resample a solution with the given dt (RK4 integrator is used internally)
    Args:
        p: position
        v: velocity
        u: input
        node_time: previous node time
        dt: resampling time
        dae: dynamic model
    Returns:
        p_res: resampled position
        v_res: resampled velocity
        u_res: resampled input
    """
    number_of_nodes = p.shape[1]
    node_time_array = np.zeros([number_of_nodes])
    if hasattr(node_time, "__iter__"):
        for i in range(1, number_of_nodes):
            node_time_array[i] = node_time_array[i-1] + node_time[i - 1]
    else:
        for i in range(1, number_of_nodes):
            node_time_array[i] = node_time_array[i-1] + node_time

    n_res = int(round(node_time_array[-1]/dt))

    F_integrator = integrators.RK4(dae, cs.SX)

    x_res0 = np.hstack((p[:, 0], v[:, 0]))

    x_res = np.zeros([p.shape[0] + v.shape[0], n_res+1])
    p_res = np.zeros([p.shape[0], n_res+1])
    v_res = np.zeros([v.shape[0], n_res+1])
    u_res = np.zeros([u.shape[0], n_res])

    x_res[:, 0] = x_res0
    p_res[:, 0] = x_res0[0:p.shape[0]]
    v_res[:, 0] = x_res0[p.shape[0]:]
    u_res[:, 0] = u[:, 0]

    t = 0.
    i = 0
    node = 0
    while i < u_res.shape[1]-1:
        x_resi = F_integrator(x0=x_res[:, i], p=u[:, node], time=dt)['xf'].toarray().flatten()

        t += dt
        i += 1

        #print(f"{t} <= {tf-dt} @ node time {(node+1)*node_time} i: {i}")

        x_res[:, i] = x_resi
        p_res[:, i] = x_resi[0:p.shape[0]]
        v_res[:, i] = x_resi[p.shape[0]:]
        u_res[:, i] = u[:, node]

        if t > node_time_array[node+1]:
            new_dt = t - node_time_array[node+1]

            # if t goes beyond the current node, first of all reset the state to the new one
            node += 1
            x_res[:, i] = np.hstack((p[:, node], v[:, node]))
            p_res[:, i] = x_resi[0:p.shape[0]]
            v_res[:, i] = x_resi[p.shape[0]:]
            u_res[:, i] = u[:, node]

            # then, if the dt is big enough, recompute by using the new input starting from the state at the node
            if new_dt >= 1e-6:
                x_resi = F_integrator(x0=x_res[:, i], p=u[:, node], time=new_dt)['xf'].toarray().flatten()

                x_res[:, i] = x_resi
                p_res[:, i] = x_resi[0:p.shape[0]]
                v_res[:, i] = x_resi[p.shape[0]:]
                u_res[:, i] = u[:, node]


    x_resf = np.hstack((p[:, -1], v[:, -1]))
    x_res[:, -1] = x_resf
    p_res[:, -1] = x_resf[0:p.shape[0]]
    v_res[:, -1] = x_resf[p.shape[0]:]

    return p_res, v_res, u_res


def resampler_old(state_vec, input_vec, nodes_dt, desired_dt, dae, f_int=None):

    # convert to np if not np already
    states = np.array(state_vec)
    inputs = np.array(input_vec)

    state_dim = states.shape[0]
    input_dim = inputs.shape[0]
    n_nodes = states.shape[1]

    # construct array of times for each node (nodes could be of different time lenght)
    node_time_array = np.zeros([n_nodes])
    if hasattr(nodes_dt, "__iter__"):
        # if a list of times is passed, construct from this list (used when variable time node)
        for i in range(1, n_nodes):
            node_time_array[i] = node_time_array[i - 1] + nodes_dt[i - 1]
    else:
        # if a number is passed, construct from this number (used when constant time node)
        for i in range(1, n_nodes):
            node_time_array[i] = node_time_array[i - 1] + nodes_dt


    # number of nodes in resampled trajectory
    n_nodes_res = int(round(node_time_array[-1] / desired_dt)) + 1

    # state_abst = cs.SX.sym('state_abst', state_dim)
    # input_abst = cs.SX.sym('input_abst', input_dim)

    # L = 1
    # dae = {'x': state_abst, 'p': input_abst, 'ode': state_dot, 'quad': L}
    if f_int is None:
        F_integrator = integrators.RK4(dae, cs.SX)
    else:
        F_integrator = f_int

    # initialize resapmpled trajectories
    state_res = np.zeros([state_dim, n_nodes_res]) # state: number of resampled nodes
    input_res = np.zeros([input_dim, n_nodes_res - 1]) # input: number of resampled nodes - 1

    state_res[:, 0] = states[:, 0]
    input_res[:, 0] = inputs[:, 0]

    t = 0.
    i = 0
    node = 0
    while i < input_res.shape[1] - 1:
        # integrate the state using the input at the desired node
        state_res_i = F_integrator(state_res[:, i], inputs[:, node], desired_dt)[0].toarray().flatten()

        t += desired_dt
        i += 1

        state_res[:, i] = state_res_i
        input_res[:, i] = inputs[:, node]

        # this is required if the current t goes beyond the current node time.
        # I get new_dt, the exceeding time (t-node_time_array[node+1]
        if t > node_time_array[node + 1]:
            new_dt = t - node_time_array[node + 1]
            node += 1

            state_res[:, i] = states[:, node]

            if new_dt >= 1e-6:
                # I set the new_dt as the integrator time
                # integrate from the node i just exceed with the relative input for the exceeding time
                state_res_i = F_integrator(states[:, node], inputs[:, node], new_dt)[0].toarray().flatten()
                state_res[:, i] = state_res_i
                input_res[:, i] = inputs[:, node]

    # the last node of the resampled trajectory has the same value as the original trajectory
    state_res[:, -1] = states[:, -1]

    return state_res

def resampler(state_vec, input_vec, nodes_dt, desired_dt, dae, f_int=None):

    # convert to np if not np already
    states = np.array(state_vec)
    inputs = np.array(input_vec)

    state_dim = states.shape[0]
    input_dim = inputs.shape[0]
    n_nodes = states.shape[1]

    # construct array of times for each node (nodes could be of different time length)
    node_time_array = np.zeros([n_nodes])
    if hasattr(nodes_dt, "__iter__"):
        # if a list of times is passed, construct from this list (used when variable time node)
        for i in range(1, n_nodes):
            node_time_array[i] = node_time_array[i - 1] + nodes_dt[i - 1]
    else:
        # if a number is passed, construct from this number (used when constant time node)
        for i in range(1, n_nodes):
            node_time_array[i] = node_time_array[i - 1] + nodes_dt

    # number of nodes in resampled trajectory
    n_nodes_res = int(round(node_time_array[-1] / desired_dt)) + 1

    if f_int is None:
        F_integrator = integrators.RK4(dae, cs.SX)
    else:
        F_integrator = f_int

    # initialize resampled trajectories
    state_res = np.zeros([state_dim, n_nodes_res])  # state: number of resampled nodes
    input_res = np.zeros([input_dim, n_nodes_res - 1])  # input: number of resampled nodes - 1

    # the first node of the resampled trajectory has the same value as the original trajectory
    state_res[:, 0] = states[:, 0]
    # input_res[:, 0] = inputs[:, 0]

    t = 0.
    node = 0

    # first state/input is the initial state/input
    for i in range(1, input_res.shape[1] + 1):

        # advance t with desired dt
        t += desired_dt

        # use previous state and constant input to compute following state
        state_prev = state_res[:, i-1]
        input_prev = inputs[:, node]
        dt = desired_dt

        # if t exceed current node, recompute from new node
        while t > node_time_array[node + 1] and t < node_time_array[-1]:

            # compute exceeding time
            new_dt = t - node_time_array[node + 1]
            # forward node
            node += 1

            # integrate from the node just exceed with the relative input for the exceeding time
            state_prev = states[:, node]
            input_prev = inputs[:, node]
            dt = new_dt

        # integrate the state using the input at the desired node
        state_int = F_integrator(state_prev, input_prev, dt)[0].toarray().flatten()

        state_res[:, i] = state_int
        # input_res[:, i] = inputs[:, node]

        # if last resampled value is beyond or last original node, use last original value
        if t >= node_time_array[-1]:
            state_res[:, i] = states[:, -1]
            break

    return state_res

if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)

    tf = 1.0
    nodes_dt = 0.5
    n_nodes = int(tf / nodes_dt) + 1
    new_nodes_dt = 0.005
    print(f"n_time: {nodes_dt} (n. nodes {n_nodes})---> {new_nodes_dt} (n. nodes {int(tf / new_nodes_dt) + 1})")

    p = np.ones([2, n_nodes])
    v = np.ones([2, n_nodes])

    p[0, 0] = 0.5 * p[0, 0]
    p[1, 0] = -0.5 * p[1, 0]
    v[0, 0] = 0.5 * v[0, 0]
    v[1, 0] = 0   * v[1, 0]

    p[0, -1] = 1 * p[0, -1]
    p[1, -1] = -1 * p[1, -1]
    v[0, -1] = 0 * v[0, -1]
    v[1, -1] = 0 * v[1, -1]



    print(p)
    print(v)

    inputs = 0.1 * np.ones([2, n_nodes-1]) # input

    print(inputs.shape)
    states = cs.vertcat(p, v)


    # nodes_dt = [0.01, 0.02, 0.01, 0.01, 0.02, 0.03, 0.02, 0.01, 0.01, 0.01]
    # nodes_dt = [0.01, 0.02, 0.01, 0.01, 0.02, 0.03, 0.02, 0.01, 0.01, 0.01]
    state_res = resampler(states, inputs, nodes_dt, new_nodes_dt)

    print('state_res.shape', state_res.shape)
    print(state_res[0,:])
    # ===============================================================
    print(' ==================== other method ========================')
    state = cs.SX.sym('state', 4)
    input = cs.SX.sym('input', 2)
    state_dot = cs.vertcat(state[2:], input)

    a = inputs
    L = 1
    dae = {'x': state, 'p': input, 'ode': state_dot, 'quad': L}
    p_res, v_res, a_res = second_order_resample_integrator(p, v, a, nodes_dt, new_nodes_dt, dae)

    print('p_res.shape', p_res.shape)
    print('a_res.shape', a_res.shape)

    print(p_res[0, :])

