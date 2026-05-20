#!/usr/bin/env python3

from horizon import problem
from horizon.utils import utils
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
import numpy as np
import casadi as cs
import os

def main():

    path_to_tests = os.path.dirname(os.path.realpath(__file__))

    # load urdf
    urdffile = os.path.join(path_to_tests, '..', 'examples', 'urdf', 'spot.urdf')
    urdf = open(urdffile, 'r').read()
    kd = cas_kin_dyn.CasadiKinDyn(urdf)
    
    integr = kd.integrate()
    print(integr)
    
    q = np.random.uniform(size=(19, 5))
    v = np.random.uniform(size=(18, 5))
    
    qnext = integr(q.T, v.T)
    
    print(qnext.shape)
    
    exit()
    
    contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']

    # parameters
    n_c = 4
    n_q = kd.nq()
    n_v = kd.nv()
    n_f = 3
    dt = 0.05

    # define dynamics
    prb = problem.Problem(30, casadi_type=cs.MX, abstract_casadi_type=cs.MX)
    q = prb.createStateVariable('q', n_q)
    q_dot = prb.createStateVariable('q_dot', n_v)
    q_ddot = prb.createInputVariable('q_ddot', n_v)
    # f_list = [prb.createInputVariable(f'force_{i}', n_f) for i in contacts_name]
    x_next = utils.double_integrator_discrete_time(q, q_dot, q_ddot, dt, kd)
    prb.setDt(dt)
    prb.setDynamics(x_next, discrete_time=True)
    
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                       0.0, 0.9, -1.5238505,
                       0.0, 0.9, -1.5202315,
                       0.0, 0.9, -1.5300265,
                       0.0, 0.9, -1.5253125])
    
    vzero = np.random.uniform(size=q_dot.shape[0])*0
    vzero[3:6] = 0
    
    x0 = np.hstack([q0 ,vzero])
    
    fn_integrate: cs.Function = prb.f_int
    
    print(fn_integrate)
    
    
    jac = fn_integrate.factory('integrate_jac', ['x', 'u'], ['jac:f:x', 'jac:f:u'])
    
    print(jac(cs.vertcat(q, q_dot), q_ddot)[0])
    
    print(jac(x0, vzero)[0][3,3])
    

    
main()