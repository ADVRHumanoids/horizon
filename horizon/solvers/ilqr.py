try:
    from horizon.solvers.pyilqr import IterativeLQR
except ImportError as e:
    print(f'failed to import pyilqr extension: {e}')
    exit(1)

from horizon.variables import Parameter
from horizon.solvers import Solver
from horizon.problem import Problem
from horizon.functions import Function, Constraint, Residual, RecedingResidual
from typing import Dict, List
from horizon.transcriptions import integrators
import casadi as cs
import numpy as np
from matplotlib import pyplot as plt

class SolverILQR(Solver):
    
    def __init__(self, 
                 prb: Problem,
                 opts: Dict = None) -> None:

        filtered_opts = None 
        if opts is not None:
            filtered_opts = {k: opts[k] for k in opts.keys() if k.startswith('ilqr.')}

        # init base class
        super().__init__(prb, filtered_opts)

        # get type of abstract variables in horizon problem (SX or MX)
        abstract_casadi_type = self.prb.default_abstract_casadi_type

        # save max iter if any
        self.max_iter = self.opts.get('ilqr.max_iter', 100)
        
        # num shooting interval
        self.N = prb.getNNodes() - 1  

        # # get integrator and compute discrete dynamics in the form (x, u, p) -> f
        # integrator_name = self.opts.get('ilqr.integrator', 'RK4')
        # dae = {'ode': self.xdot, 'x': self.x, 'p': self.u, 'quad': 0}

        # # handle parametric time
        # integrator_opt = {}

        # self.int = integrators.__dict__[integrator_name](dae, integrator_opt, self.prb.default_casadi_type)
        # if isinstance(self.dt, float):
        #     # integrator_opt['tf'] = self.dt
        #     x_int = self.int(self.x, self.u, self.dt)[0]
        #     dt_name = 'dt'
        #     time = abstract_casadi_type.sym(dt_name, 0)

        # elif isinstance(self.dt, Parameter):
        #     time = abstract_casadi_type.sym(self.dt.getName(), 1)
        #     x_int = self.int(self.x, self.u, time)[0]
        #     dt_name = self.dt.getName()
        #     pass
        # else:
        #     raise TypeError('ilqr supports only float and Parameter dt')

        # all_params = self.prb.getParameters()
        # depend_params = {}
        # for pname, p in all_params.items():
        #     if cs.depends_on(x_int, p):
        #         depend_params[pname] = p
        

        # self.dyn = cs.Function('f', 
        #                        {'x': self.x, 'u': self.u, dt_name: time, 'f': x_int, **depend_params},
        #                        ['x', 'u', dt_name] + list(depend_params.keys()), ['f']
        #                        )

        # create ilqr solver
        self.ilqr = IterativeLQR(self.prb.getIntegrator(), self.N, self.opts)

        # should we use GN approx for residuals?
        self.use_gn = self.opts.get('ilqr.enable_gn', False)

        # set constraints, costs, bounds
        self._set_constraint()
        self._set_cost()
        self._set_bounds()
        

        # set a default iteration callback
        self.plot_iter = False
        self.xax = None 
        self.uax = None
        self.dax = None
        self.hax = None

        # empty solution dict
        self.solution_dict = dict()

        # print iteration statistics
        self.set_iteration_callback()

    def save(self):
        data = self.prb.save()
        data['solver'] = dict()
        if isinstance(self.dt, float):
            data['solver']['dt'] = self.dt
        data['solver']['name'] = 'ilqr'
        data['solver']['opt'] = self.opts
        data['dynamics'] = self.dyn.serialize()
        return data

    
    def set_iteration_callback(self, cb=None):
        if cb is None:
            self.ilqr.setIterationCallback(self._iter_callback)
        else:
            print('setting custom iteration callback')
            self.ilqr.setIterationCallback(cb)


    def configure_rti(self) -> bool:
        self.opts['max_iter'] = 1
    
    def solve(self):
        
        # set initial state
        x0 = self.prb.getInitialState()
        xinit = self.prb.getState().getInitialGuess()
        uinit = self.prb.getInput().getInitialGuess()

        # update initial guess
        self.ilqr.setStateInitialGuess(xinit)
        self.ilqr.setInputInitialGuess(uinit)
        # self.ilqr.setIterationCallback(self._iter_callback)
        
        # update parameters
        self._set_param_values()

        # update bounds
        self._set_bounds()

        # update nodes
        self._update_nodes()

        # solve
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

        self.solution_dict['x_opt'] = self.x_opt
        self.solution_dict['u_opt'] = self.u_opt

        return ret
    
    def getSolutionDict(self):
        return self.solution_dict

    def getDt(self):
        self.dt_solution = np.zeros(self.prb.getNNodes() - 1)
        if isinstance(self.dt, float):
            for node_n in range(self.prb.getNNodes() - 1):
                self.dt_solution[node_n] = self.dt
        elif isinstance(self.dt, Parameter):
            for node_n in range(self.prb.getNNodes() - 1):
                self.dt_solution[node_n] = self.dt.getValues(node_n)

        return self.dt_solution

    def getSolutionState(self):
        return self.solution_dict['x_opt']

    def getSolutionInput(self):
        return self.solution_dict['u_opt']

    def print_timings(self):

        prof_info = self.ilqr.getProfilingInfo()

        # max string length
        keys = prof_info.timings.keys()
        max_len = max([len(k) for k in keys])

        if len(prof_info.timings) == 0:
            return
        
        print('\ntimings (inner):')
        for k, v in prof_info.timings.items():
            if '_inner' not in k:
                continue
            print(f'{k[:-6]:{max_len}}{np.mean(v)} us')

        print('\ntimings (iter):')
        for k, v in prof_info.timings.items():
            if '_inner' in k:
                continue
            print(f'{k:30}{np.mean(v)*1e-3} ms')

    def _update_nodes(self):

        for fname, f in self.prb.function_container.getCost().items():
            self.ilqr.setIndices(fname, f.getNodes())

        for fname, f in self.prb.function_container.getCnstr().items():
            self.ilqr.setIndices(fname, f.getNodes())

        self.ilqr.updateIndices()
    
    
    def _set_cost(self):
        
        self._set_fun(container=self.prb.function_container.getCost(),
                set_to_ilqr=self.ilqr.setIntermediateCost, 
                outname='l')
    
    def _set_constraint(self):

        self._set_fun(container=self.prb.function_container.getCnstr(),
                set_to_ilqr=self.ilqr.setIntermediateConstraint,
                outname='h')

    def _set_bounds(self):

        xlb, xub = self.prb.getState().getBounds(node=None)
        ulb, uub = self.prb.getInput().getBounds(node=None)
        self.ilqr.setStateBounds(xlb, xub)
        self.ilqr.setInputBounds(ulb, uub)

    def _set_fun(self, container, set_to_ilqr, outname):

        # check fn in container    
        for fname, f in container.items():
            
            # give a type to f
            f: Function = f

            # get input variables for this function
            input_list = f.getVariables()
            param_list = f.getParameters()

            # fn value
            value = f.getFunction()(*input_list, *param_list)

            # set to ilqr fn could change if this is a residual
            # and we're in gn mode
            set_to_ilqr_actual = set_to_ilqr
            outname_actual = outname

            # save function value
            if isinstance(f, (Residual, RecedingResidual)) and self.use_gn:
                outname_actual = 'res'
                set_to_ilqr_actual = self.ilqr.setIntermediateResidual
                print('got residual')
            elif isinstance(f, (Residual, RecedingResidual)) and not self.use_gn:
                value = cs.sumsqr(value)
                print('got residual disables')
                
            # wrap function
            l = cs.Function(fname, 
                            [self.x, self.u] + param_list, [value], 
                            ['x', 'u'] + [p.getName() for p in param_list], 
                            [outname_actual]
                            )


            set_to_ilqr_actual(f.getNodes(), l)
        
    
    def _set_param_values(self):


        # apparently ilqr creates the parameters with dimensions: par_dim x n+1
        # generally speaking, this is not necessarily true
        params = self.prb.var_container.getParList()

        for p in params:
            # todo small hack:
            #  the parameters inside the ilqr are defined on ALL nodes
            #  horizon allows to define parameters only on desired nodes
            #  so i'm masking the parameter values with a matrix of nan of N+1 dimension
            p_vals_temp = p.getValues()
            p_vals = np.empty((p.getDim(), self.N+1))
            p_vals[:] = np.nan
            p_vals[:, p.getNodes()] = p_vals_temp
            self.ilqr.setParameterValue(p.getName(), p_vals)

        self.ilqr.setParameterValue('dt', np.full((1, self.N+1), self.prb.getDt()))

    
    def _iter_callback(self, fpres):

        if not fpres.accepted:
            return
            print('-', end='', flush=True)
        else:
            print('*', end='', flush=True)
            
        fpres.print()

        if self.plot_iter:

            if self.dax is None:
                _, (self.dax, self.hax) = plt.subplots(2)
                self.dax.set_yscale('log')
                self.hax.set_yscale('log')
                
                plt.sca(self.dax)
                self.dline = plt.plot(np.linalg.norm(fpres.defect_values, axis=1))[0]
                plt.grid()
                plt.title(f'Dynamics gaps (iter {fpres.iter})')
                plt.xlabel('Node [-]')
                plt.ylabel('Gap')
                plt.legend([f'd{i}' for i in range(self.nx)])
                
                plt.sca(self.hax)
                self.hline = plt.plot(fpres.constraint_values)[0]
                plt.grid()
                plt.title(f'Constraint violation (iter {fpres.iter})')
                plt.xlabel('Node [-]')
                plt.ylabel('Constraint 1-norm')
                
                plt.ion()
                plt.show()

            
            self.dline.set_ydata(np.linalg.norm(fpres.defect_values, axis=1))
            self.hline.set_ydata(fpres.constraint_values)
            self.dax.relim()
            self.hax.relim()
            self.dax.autoscale_view()
            self.hax.autoscale_view()
            plt.pause(0.01)
            plt.waitforbuttonpress()

