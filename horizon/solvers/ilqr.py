try:
    from horizon.solvers.pyilqr import IterativeLQR
except ImportError:
    print('failed to import pyilqr extension; did you compile it?')
    exit(1)

from horizon.variables import Parameter
from horizon.solvers import Solver
from horizon.problem import Problem
from horizon.functions import Function, Constraint
from typing import Dict, List
from horizon.transcriptions import integrators
import casadi as cs
import numpy as np
from matplotlib import pyplot as plt

class SolverILQR(Solver):
    
    def __init__(self, 
                 prb: Problem, 
                 dt: float, 
                 opts: Dict = None) -> None:

        filtered_opts = None 
        if opts is not None:
            filtered_opts = {k: opts[k] for k in opts.keys() if k.startswith('ilqr.')}

        # init base class
        super().__init__(prb, dt, filtered_opts)

        # save max iter if any
        self.max_iter = self.opts.get('ilqr.max_iter', 100)
        
        # num shooting interval
        self.N = prb.getNNodes() - 1  

        # get integrator and compute discrete dynamics in the form (x, u, p) -> f
        integrator_name = self.opts.get('ilqr.integrator', 'RK4')
        dae = {'ode': self.xdot, 'x': self.x, 'p': self.u, 'quad': 0}

        # handle parametric time
        integrator_opt = {}
        if isinstance(dt, float):
            integrator_opt['tf'] = dt 
        elif isinstance(dt, Parameter):
            pass
        else:
            raise TypeError('ilqr supports only float and Parameter dt')

        self.int = integrators.__dict__[integrator_name](dae, integrator_opt)

        # handle possible parametric time 
        if self.int.n_in() == 3:
            time = cs.SX.sym('time', 1)
            x_int = self.int(self.x, self.u, time)[0]
        elif self.int.n_in() == 2:
            time = cs.SX.sym('time', 0)
            x_int = self.int(self.x, self.u)[0]
        else:
            raise IndexError('integrated dynamics should either have 2 or 3 inputs')


        self.dyn = cs.Function('f', 
                               {'x': self.x, 'u': self.u, 'p': time, 'f': x_int},
                               ['x', 'u', 'p'], ['f']
                               )

        # create ilqr solver
        self.ilqr = IterativeLQR(self.dyn, self.N, self.opts)

        
        self._set_constraint()
        self._set_cost()
        xlb, xub = self.prb.getState().getBounds(node=None)
        ulb, uub = self.prb.getInput().getBounds(node=None)

        print(xlb.shape, ulb.shape, xub.shape, uub.shape)

        self.ilqr.setStateBounds(xlb.reshape((self.nx, self.N+1)), xub.reshape((self.nx, self.N+1)))
        self.ilqr.setInputBounds(ulb.reshape((self.nu, self.N)), uub.reshape((self.nu, self.N)))

        # set a default iteration callback
        self.plot_iter = False
        self.xax = None 
        self.uax = None

        # empty solution dict
        self.solution_dict = dict()

    def save(self):
        data = self.prb.save()
        data['solver'] = dict()
        data['solver']['name'] = 'ilqr'
        data['solver']['opt'] = self.opts
        data['dynamics'] = self.dyn.serialize()
        return data

    
    def set_iteration_callback(self, cb=None):
        if cb is None:
            self.ilqr.setIterationCallback(self._iter_callback)
        else:
            self.ilqr.setIterationCallback(cb)


    def configure_rti(self) -> bool:
        self.opts['max_iter'] = 1
    
    def solve(self):
        
        # set initial state
        x0 = self.prb.getInitialState()
        xinit = self.prb.getState().getInitialGuess()
        uinit = self.prb.getInput().getInitialGuess()
        xinit[:, 0] = x0.flatten()

        # update initial guess
        self.ilqr.setStateInitialGuess(xinit)
        self.ilqr.setInputInitialGuess(uinit)
        self.ilqr.setIterationCallback(self._iter_callback)
        
        # update parameters
        self._set_param_values(container=self.prb.function_container.getCost())
        self._set_param_values(container=self.prb.function_container.getCnstr())
        if isinstance(self.dt, Parameter):
            # note: dt is a N+1 length vector, we ignore the last entry
            self.ilqr.setParameterValue('__dynamics__', self.dt.getValues()[0:-1])

        ret = self.ilqr.solve(self.max_iter)

        # get solution
        self.x_opt = self.ilqr.getStateTrajectory()
        self.u_opt = self.ilqr.getInputTrajectory()

        # populate solution dict
        for var in self.prb.getState().var_list:
            vname = var.getName()
            off, dim = self.prb.getState().getVarIndex(vname)
            self.solution_dict[vname] = self.x_opt[off:off+dim, :]
            
        for var in self.prb.getInput().var_list:
            vname = var.getName()
            off, dim = self.prb.getInput().getVarIndex(vname)
            self.solution_dict[vname] = self.u_opt[off:off+dim, :]

        return ret
    
    def getSolutionDict(self):
        return self.solution_dict

    def print_timings(self):

        prof_info = self.ilqr.getProfilingInfo()

        if len(prof_info.timings) == 0:
            return
        
        print('\ntimings (inner):')
        for k, v in prof_info.timings.items():
            if '_inner' not in k:
                continue
            print(f'{k[:-6]:30}{np.sum(v)/self.N/1000} ms')

        print('\ntimings (iter):')
        for k, v in prof_info.timings.items():
            if '_inner' in k:
                continue
            print(f'{k:30}{np.sum(v)/self.N/1000} ms')
    
    
    def _set_cost(self):
        
        self._set_fun(container=self.prb.function_container.getCost(),
                set_to_ilqr=self.ilqr.setIntermediateCost, 
                outname='l')

    
    def _set_constraint(self):

        self._set_fun(container=self.prb.function_container.getCnstr(),
                set_to_ilqr=self.ilqr.setIntermediateConstraint,
                outname='h')
    
    def _set_bounds_k(self, k):

        outname = 'h'
        set_to_ilqr = self.ilqr.setIntermediateConstraint
        pnull = cs.SX.sym('pnull', 0)
        
        # state
        xlb, xub = self.prb.getState().getBounds(node=k)
        eq_indices = np.array(np.nonzero(xlb == xub)).flatten().tolist()
        if k > 0 and len(eq_indices) > 0:  # note: skip initial condition
            l = cs.Function(f'xc_{k}', [self.x, self.u, pnull], [self.x[eq_indices] - xlb[eq_indices]], 
                            ['x', 'u', 'p'], [outname])
            set_to_ilqr([k], l)

        # input
        if k < self.N:
            ulb, uub = self.prb.getInput().getBounds(node=k)
            eq_indices = np.array(np.nonzero(ulb == uub)).flatten().tolist()
            if len(eq_indices) > 0:
                l = cs.Function(f'uc_{k}', [self.x, self.u, pnull], [self.u[eq_indices] - ulb[eq_indices]], 
                                ['x', 'u', 'p'], [outname])
                set_to_ilqr([k], l)


    def _set_fun(self, container, set_to_ilqr, outname):

        # check fn in container    
        for fname, f in container.items():
            
            # give a type to f
            f: Function = f

            # get input variables for this function
            input_list = f.getVariables()
            param_list = f.getParameters()
            p = cs.vertcat(*param_list)

            # save function value
            value = f.getFunction()(*input_list, *param_list)

            # active nodes
            nodes = f.getNodes()

            tgt_values = list()

            # in the case of constraints, check bound values
            if isinstance(f, Constraint):
                lb, ub = Constraint.getBounds(f)
                if np.any(lb != ub):
                    raise ValueError(f'[ilqr] constraint {fname} not an equality constraint')
                tgt_values = np.hsplit(lb, len(nodes))

            # wrap function
            l = cs.Function(fname, [self.x, self.u, p], [value], 
                                ['x', 'u', 'p'], [outname])

            # set it to solver
            if isinstance(f, Constraint):
                set_to_ilqr(nodes, l, tgt_values)
            else:
                set_to_ilqr(nodes, l)
        
    
    def _set_param_values(self, container):

        for fname, f in container.items():
            
            # give a type to f
            f: Function = f

            # get input variables for this function
            param_list = f.getParameters()

            if len(param_list) == 0:
                continue

            value_list = list()
            for p in param_list:
                v = p.getValues()
                value_list.append(v)
            
            p_value = np.vstack(value_list)
            
            self.ilqr.setParameterValue(fname, p_value)
    
    def _iter_callback(self, fpres):
        if not fpres.accepted:
            return
        fmt = ' <#09.3e'
        fmtf = ' <#04.2f'
        star = '*' if fpres.accepted else ' '
        print(f'{star}\
alpha={fpres.alpha:{fmtf}}  \
reg={fpres.hxx_reg:{fmt}}  \
merit={fpres.merit:{fmt}}  \
dm={fpres.merit_der:{fmt}}  \
mu_f={fpres.mu_f:{fmt}}  \
mu_c={fpres.mu_c:{fmt}}  \
cost={fpres.cost:{fmt}}  \
delta_u={fpres.step_length:{fmt}}  \
constr={fpres.constraint_violation:{fmt}}  \
gap={fpres.defect_norm:{fmt}}')

        if self.plot_iter and fpres.accepted:

            if self.xax is None:
                _, (self.xax, self.uax) = plt.subplots(1, 2)
            
            plt.sca(self.xax)
            plt.cla()
            plt.plot(fpres.xtrj.T)
            plt.grid()
            plt.title(f'State trajectory (iter {fpres.iter})')
            plt.xlabel('Node [-]')
            plt.ylabel('State')
            plt.legend([f'x{i}' for i in range(self.nx)])

            plt.sca(self.uax)
            plt.cla()
            plt.plot(fpres.utrj.T)
            plt.grid()
            plt.title(f'Input trajectory (iter {fpres.iter})')
            plt.xlabel('Node [-]')
            plt.ylabel('Input')
            plt.legend([f'u{i}' for i in range(self.nu)])
            plt.draw()
            print("Press a button!")
            plt.waitforbuttonpress()


                    
                
                


            

############# TESTING STUFF TO BE REMOVED #######################
if __name__ == '__main__':

    from matplotlib import pyplot as plt

    # create problem
    N = 100
    dt = 0.03
    prb = Problem(N)

    # create variables
    p = prb.createStateVariable('p', 2)
    theta = prb.createStateVariable('theta', 1)
    v = prb.createInputVariable('v', 1)
    omega = prb.createInputVariable('omega', 1)

    # define dynamics 
    x = prb.getState().getVars()
    u = prb.getInput().getVars()
    xdot = cs.vertcat(v*cs.cos(theta), 
                    v*cs.sin(theta),
                    omega)
    prb.setDynamics(xdot)

    # Cost function
    x_tgt = np.array([1, 0, 0])
    prb.createIntermediateCost("reg", 1e-6*cs.sumsqr(u))
    prb.createFinalConstraint("gothere", x - x_tgt)

    # initial state
    x0 = np.array([0, 0, np.pi/2])
    prb.setInitialState(x0=x0)

    # TEST ILQR
    sol = SolverILQR(prb, dt)
    sol.solve()
    sol.print_timings()

    plt.plot(sol.ilqr.getStateTrajectory().T, '-')
    plt.show()
