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
        q: quaternion

    Returns:
        R: rotation matrix

    """

    R = cs.SX.zeros(3, 3)
    qi = q[0]; qj = q[1]; qk = q[2]; qr = q[3]
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


def barrier(x):
    return cs.if_else(x > 0, 0, x)

