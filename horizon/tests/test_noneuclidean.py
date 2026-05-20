from horizon.problem import Problem
from horizon.solvers import solver
import casadi as cs
import numpy as np



def Rsum(x: cs.SX, dx: cs.SX):
    R = x.reshape((3,3))    
    om = dx
    # rodriguez: I + sin(th)*K + (1 - cos(th))*K^2 = exp(S)
    th = cs.if_else(cs.norm_2(om) > 0, cs.norm_2(om), 0.)  # avoid division by zero
    k = om / th
    K = cs.skew(k)
    S = cs.skew(om)
        
    DeltaR_big = np.eye(3) + cs.sin(th)*K + (1 - cs.cos(th))*(K@K)
    DeltaR_sm = np.eye(3)*(1 - th*2/2.) + S + S@S/2.  # taylor
        
    return cs.if_else(th < 0.0001, R @ DeltaR_sm, R @ DeltaR_big).reshape((9, 1))

def Rdiff(x1: cs.SX, x2: cs.SX):
    R1 = x1.reshape((3,3))
    R2 = x2.reshape((3,3))
    R12 = R1 @ R2.T
    
    # tr(R) = 1 + 2*cos(th), -1 <= tr(R) <= 3
    # pathological case A: th = pi -> tr(R) = -1
    # pathological case B: th = 0  -> tr(R) = 3
    tr = cs.trace(R12)
    th = cs.acos((tr - 1)/2.)
    
    S_B = 0.5*(R12 - R12.T)
    S_ok = th/cs.sin(th)*S_B
    S = cs.if_else(tr > 2.999, S_B, S_ok)
    
    return cs.vertcat(S[2, 1], S[0, 2], S[1, 0])
    
Reye = np.eye(3)

N = 10
pb = Problem(N=N)
R = pb.createStateVariable('R', 9,) # vsum=Rsum, vdiff=Rdiff, vneutral=Reye.flatten())
om = pb.createStateVariable('omega', 3)
acc = pb.createInputVariable('acc', 3)

Rmat = R.reshape((3, 3))
Rtgt = Rsum(Reye, np.array([0.1, 0.2, 0.3]))


dt = 1./N

x_int = cs.vertcat(
    Rsum(R, om*dt + 0.5*acc*dt**2).reshape((9, 1)),
    om + acc*dt
)

pb.setDt(dt)

pb.setDynamics(x_int, discrete_time=True)


# dx = cs.SX.sym('dx', 3)
# y0 = pb.f_int(R, om, dt)
# y = pb.f_int(pb.xsum(R, dx), cs.SX.zeros(3), dt)
# y = pb.xdiff(y, y0)

# F = cs.Function('F', [R, om, dx], [y], ['x', 'u', 'dx'], ['y'])
# dF = F.factory('F_jac', ['x', 'u', 'dx'], ['jac:y:dx', 'jac:y:u'])

# dF = dF(Reye.flatten(), np.zeros(3), np.zeros(3))
# print(dF)

# exit()

# xsumjac = pb.xsum.jacobian()
# print(xsumjac)
# print(xsumjac(Reye.flatten(), np.zeros(3), Reye))


# exit()

pb.createIntermediateResidual('u_reg', om)
pb.createIntermediateResidual('a_reg', acc)
pb.createIntermediateResidual('x_reg', 1e-5*Rdiff(R, Reye))

R.setInitialGuess(Reye.flatten())
pb.setInitialState(np.hstack([Reye.flatten(), np.zeros(3)]))

R1 = Rsum(Reye, np.array([0.1, 0.2, 0.3]))
R2 = Rsum(Reye, -np.array([0.3, 0.1, 0.2]))
pb.createConstraint('wp1', Rdiff(R, R1), nodes=5)
pb.createConstraint('wp2', Rdiff(R, R2), nodes=N)
pb.createFinalConstraint('om_final', om)

solv = solver.Solver.make_solver('ilqr', pb, {'ilqr.verbose': False, 'ilqr.log': False, 'ilqr.enable_line_search': True, 'ilqr.max_iter': 10, 'ilqr.use_kkt_solver': False})

# # HACK state bounds
# xlb = np.full(shape=(3, N+1), fill_value=-cs.inf)
# xub = np.full(shape=(3, N+1), fill_value=cs.inf)
# xlb[:, 0] = 0
# xub[:, 0] = 0
# solv.ilqr.setStateBounds(xlb, xub, pb.xneutral)

try:
    solv.set_iteration_callback()
except:
    pass

solv.solve()

solv.print_timings()

solution = solv.getSolutionDict()

print(solution['omega'])
print(solution['acc'])

from matplotlib import pyplot as plt

# plot omega and acc
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(solution['omega'].T)
plt.title('Omega')
plt.subplot(2, 1, 2)
plt.plot(solution['acc'].T)
plt.title('Acceleration')
plt.tight_layout()
plt.show()

