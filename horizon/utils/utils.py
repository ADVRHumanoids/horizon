import casadi as cs
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
import numpy as np

def jac(dict, var_string_list, function_string_list):
    """
    Args:
        dict: dictionary which maps variables and functions, eg. {'x': x, 'u': u, 'f': f}
        var_string_list: list of variables in dict, eg. ['x', 'u']
        function_string_list: list of functions in dict, eg. ['f']
    Returns:
        F: casadi Function for evaluation
        jac: dictionary with expression of derivatives

    NOTE: check /tests/jac_test.py for example of usage for Jacobian and Hessian computation
    """
    f = {}
    for function in function_string_list:
        f[function] = dict[function]

    vars_dict = {}
    X = []
    for var in var_string_list:
        vars_dict[var] = dict[var]
        X.append(dict[var])

    jac_list = []
    jac_id_list = []
    for function_key in f:
        for var in var_string_list:
            id = "D" + function_key + 'D' + var
            jac_id_list.append(id)
            jac_list.append(cs.jacobian(f[function_key], vars_dict[var]))

    jac_map = {}
    i = 0
    for jac_id in jac_id_list:
        jac_map[jac_id] = jac_list[i]
        i += 1

    F = cs.Function('jacobian', X, jac_list, var_string_list, jac_id_list)

    return F, jac_map


# def skew(q):
#     """
#     Create skew matrix from vector part of quaternion
#     Args:
#         q: vector part of quaternion [qx, qy, qz]
#
#     Returns:
#         S = skew symmetric matrix built using q
#     """
#     S = cs.SX.zeros(3, 3)
#     S[0, 1] = -q[2]; S[0, 2] = q[1]
#     S[1, 0] = q[2];  S[1, 2] = -q[0]
#     S[2, 0] = -q[1]; S[2, 1] = q[0]
#     return S

def quaterion_product(q, p):
    """
    Computes quaternion product between two quaternions q and p
    Args:
        q: quaternion
        p: quaternion

    Returns:
        quaternion product q x p
    """
    q0 = q[3]
    p0 = p[3]

    return [q0*p[0:3] + p0*q[0:3] + cs.mtimes(cs.skew(q[0:3]), p[0:3]), q0*p0 - cs.mtimes(q[0:3].T, p[0:3])]

def toRot(q):
    """
    Compute rotation matrix associated to given quaternion q
    Args:
        q: quaternion (q[0] = qw  q[1] = qx  q[2] = qy  q[3] = qz

    Returns:
        R: rotation matrix

    """

    R = cs.SX.zeros(3, 3)
    qi = q[1]; qj = q[2]; qk = q[3]; qr = q[0]
    R[0, 0] = 1. - 2. * (qj * qj + qk * qk)
    R[0, 1] = 2. * (qi * qj - qk * qr)
    R[0, 2] = 2. * (qi * qk + qj * qr)
    R[1, 0] = 2. * (qi * qj + qk * qr)
    R[1, 1] = 1. - 2. * (qi * qi + qk * qk)
    R[1, 2] = 2. * (qj * qk - qi * qr)
    R[2, 0] = 2. * (qi * qk - qj * qr)
    R[2, 1] = 2. * (qj * qk + qi * qr)
    R[2, 2] = 1. - 2. * (qi * qi + qj * qj)

    return R

def angular_velocities(q1, q2, dt):
    return (2 / dt) * np.array([
        q1[3]*q2[0] - q1[1]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1],
        q1[3]*q2[1] + q1[1]*q2[2] - q1[1]*q2[3] - q1[2]*q2[0],
        q1[3]*q2[2] - q1[1]*q2[1] + q1[1]*q2[0] - q1[2]*q2[3]])

def matrix_to_quaternion(matrix):
    # Ensure the matrix is a valid rotation matrix
    assert np.isclose(np.linalg.det(matrix), 1.0), "Input matrix is not a valid rotation matrix"

    # Extract the components of the rotation matrix
    r11, r12, r13 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    r21, r22, r23 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    r31, r32, r33 = matrix[2, 0], matrix[2, 1], matrix[2, 2]

    # Calculate the trace of the matrix
    trace = r11 + r22 + r33

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r32 - r23) * s
        y = (r13 - r31) * s
        z = (r21 - r12) * s
    elif r11 > r22 and r11 > r33:
        s = 2.0 * np.sqrt(1.0 + r11 - r22 - r33)
        w = (r32 - r23) / s
        x = 0.25 * s
        y = (r12 + r21) / s
        z = (r13 + r31) / s
    elif r22 > r33:
        s = 2.0 * np.sqrt(1.0 + r22 - r11 - r33)
        w = (r13 - r31) / s
        x = (r12 + r21) / s
        y = 0.25 * s
        z = (r23 + r32) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r33 - r11 - r22)
        w = (r21 - r12) / s
        x = (r13 + r31) / s
        y = (r23 + r32) / s
        z = 0.25 * s

    # Return the quaternion in the x, y, z, w order as a numpy array
    return np.array([x, y, z, w])

def rotationMatrixToQuaterion(R):
    """
    Compute quaternion from rotation matrix
    Args:
        R: rotation matrix

    Returns:
        q: quaternion

    """
    q = np.zeros(4)
    q[3] = 0.5 * cs.sqrt(1. + R[0, 0] + R[1, 1] + R[2, 2])
    q[0] = 0.5 * (R[2, 1] - R[1, 2]) / (4. * q[3])
    q[1] = 0.5 * (R[0, 2] - R[2, 0]) / (4. * q[3])
    q[2] = 0.5 * (R[1, 0] - R[0, 1]) / (4. * q[3])
    return q

def quat_to_rot(quat):
    """
    Covert a quaternion into a full three-dimensional rotation matrix
    [method specific for casadi function]

    Input
    :param quat: A 4 element array representing the quaternion (im(quat), re(quat))

    Output
    :return: A Casadi 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q1 = quat[0]
    q2 = quat[1]
    q3 = quat[2]
    q0 = quat[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    r0 = cs.horzcat(r00, r01, r02)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    r1 = cs.horzcat(r10, r11, r12)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    r2 = cs.horzcat(r20, r21, r22)

    # 3x3 rotation matrix
    rot_matrix = cs.vertcat(r0, r1, r2)

    return rot_matrix


def double_integrator_with_floating_base(q, qdot, qddot, base_velocity_reference_frame = cas_kin_dyn.CasadiKinDyn.LOCAL):
    """
    Construct the floating-base dynamic model:
                x = [q, qdot]
                xdot = [qdot, qddot]
    using quaternion dynamics: quatdot = quat x [omega, 0]
    NOTE: this implementation consider floating-base position and orientation expressed in GLOBAL (world) coordinates while
    if base_velocity_reference_frame = cas_kin_dyn.CasadiKinDyn.LOCAL
        linear and angular velocities expressed in LOCAL (base_link) coordinates.
    else if base_velocity_reference_frame = cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        linear and angular velocities expressed in WORLD coordinates.
    Args:
        q_sx: joint space coordinates: q = [x y z px py pz pw qj], where p is a quaternion
        qdot_sx: joint space velocities: ndot = [vx vy vz wx wy wz qdotj]
        qddot_sx: joint space acceleration: nddot = [ax ay ax wdotx wdoty wdotz qddotj]

    Returns:
        xdot: derivative of the state xdot = [qdot, qddot]
    """
    if base_velocity_reference_frame != cas_kin_dyn.CasadiKinDyn.LOCAL and base_velocity_reference_frame != cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED:
        raise Exception(f'base_velocity_reference_frame can be only LOCAL or LOCAL_WORLD_ALIGNED!')

    q_sx = cs.SX.sym('q_sx', q.shape[0])
    qdot_sx = cs.SX.sym('ndot_sx', qdot.shape[0])
    qddot_sx = cs.SX.sym('nddot_sx', qddot.shape[0])


    qw = cs.SX.zeros(4,1)
    qw[0:3] = 0.5 * qdot_sx[3:6]

    if base_velocity_reference_frame == cas_kin_dyn.CasadiKinDyn.LOCAL:
        if q_sx.shape[1] == 1:
            quaterniondot = quaterion_product(q_sx[3:7], qw)
        else:
            quaterniondot = quaterion_product(q_sx[3:7, :], qw)
    else:
        if q_sx.shape[1] == 1:
            quaterniondot = quaterion_product(qw, q_sx[3:7])
        else:
            quaterniondot = quaterion_product(qw, q_sx[3:7, :])

    R = toRot([0., 0., 0., 1.])
    if base_velocity_reference_frame == cas_kin_dyn.CasadiKinDyn.LOCAL:
        R = toRot(q_sx[3:7])
    # x = cs.vertcat(q_sx, qdot_sx)

    if qdot_sx.shape[1] == 1:
        first = cs.mtimes(R, qdot_sx[0:3])
    else:
        first = cs.mtimes(R, qdot_sx[0:3, :])

    if qdot_sx.shape[1] == 1:
        third = qdot_sx[6:qdot_sx.shape[0]]
    else:
        third = qdot_sx[6:qdot_sx.shape[0], :]

    xdot = cs.vertcat(first, cs.vertcat(*quaterniondot), third, qddot_sx)

    fun_sx = cs.Function('double_integrator_with_floating_base', [q_sx, qdot_sx, qddot_sx], [xdot])

    xdot = fun_sx(q, qdot, qddot)

    return xdot

def model_isdkosdkpoadpkas(x, u, kd, degree=2):

    if degree != 2:
        raise NotImplementedError('Only degree 2 is implemented')

    q = x[0:kd.nq()]
    v = x[kd.nq():]
    a = u[0:kd.nv()]

    if isinstance(x, cs.SX):
        casadi_type = cs.SX
    elif isinstance(x, cs.MX):
        casadi_type = cs.MX

    dt = casadi_type.sym('dt', 1)

    q[3:7] /= cs.norm_2(q[3:7]) 
    q[10:14] /= cs.norm_2(q[10:14]) 
    q[17:21] /= cs.norm_2(q[17:21]) 

    vnext = v + a*dt 
    vmean = (v + vnext)/2.
    qnext = kd.integrate()(q, vmean*dt)
    xnext = cs.vertcat(qnext, vnext)

    quad = casadi_type.zeros(1)

    return cs.Function('F_MI', [x, u, dt], [xnext, quad], ['x', 'u', 'dt'], ['f', 'q'])

def double_integrator(q, v, a, kd=None):
    
    if kd is None:
        xdot = cs.vertcat(v, a)
        return xdot

    qdot_fn = kd.qdot()

    return cs.vertcat(qdot_fn(q, v), a)


def double_integrator_jerk(q, v, a, j, fdot, kd=None):
    if kd is None:
        xdot = cs.vertcat(v, a, j)
        return xdot

    qdot_fn = kd.qdot()

    return cs.vertcat(qdot_fn(q, v), a, j, *fdot)

def single_integrator(q, v, kd=None):

    if kd is None:
        xdot = v
        return xdot

    qdot_fn = kd.qdot()

    return qdot_fn(q, v)


def barrier(x):
    return cs.if_else(x > 0, 0, x)

def barrier1(x):
    return cs.if_else(x < 0, 0, x)

def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([x, y, z, w])
