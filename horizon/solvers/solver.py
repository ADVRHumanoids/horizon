import typing
import casadi as cs
from horizon.problem import Problem
from horizon.variables import AbstractVariable, Variable, Parameter, SingleVariable, SingleParameter
from typing import Dict, Iterable, List
from abc import ABC, abstractmethod
import numpy as np

class Solver(ABC):

    """
    Solver is an abstract interface for a generic solver
    that aims to find a solution of an instance of
    horizon.Problem.

    The typical usage requires to (i) given an horizon.Problem,
    call the factory make_solver() to retrieve a solver concrete instance, 
    (ii) to call solve() on it, and (iii) to retrieve the solution as
    solver.x_opt, and solver.u_opt.
    """

    @classmethod
    def make_solver(cls, 
                    type: str,
                    prb: Problem,
                    opts: Dict = None) -> 'Solver':
        """
        Construct a solver from its type name

        Args:
            type (str): a string indicating the solver type (e.g., blocksqp, ipopt, ilqr)
            prb (Problem): the horizon's problem instance to be solved
            dt (float): the discretization step, None if not needed
            opts (Dict, optional): A solver-dependent Dict of options. Defaults to None.
        """

        if type == 'blocksqp':
            from . import blocksqp
            ret = blocksqp.BlockSqpSolver(prb, opts)
        elif type == 'ipopt':
            from . import ipopt
            ret = ipopt.IpoptSolver(prb, opts)
        elif type == 'ilqr':
            from . import ilqr
            ret = ilqr.SolverILQR(prb, opts)
        elif type == 'gnsqp':
            from . import sqp
            qp_solver = 'osqp'
            if opts is not None:
                if 'gnsqp.qp_solver' in opts:
                    qp_solver = opts['gnsqp.qp_solver']
                    del opts['gnsqp.qp_solver']
            ret = sqp.GNSQPSolver(prb, opts, qp_solver)
        else:
            raise KeyError(f'unsupperted solver type "{type}"')

        ret.type = type
        return ret


    def __init__(self, 
                 prb: Problem,
                 opts: Dict = None) -> None:
        """
        Construct a solver

        Args:
            prb (Problem): the horizon's problem instance to be solved
            dt (float): the discretization step, None if not needed
            opts (Dict, optional): A solver-dependent Dict of options. Defaults to None.
        """
        
        # handle inputs
        if opts is None:
            opts = dict()

        self.opts = opts
        self.prb = prb

        self.var_solution: Dict[str:np.array] = dict()
        self.cnstr_solution: Dict[str:np.array] = dict()
        self.dt_solution: np.array = None

        # save state and control
        self.x = prb.getState().getVars()
        self.u = prb.getInput().getVars()
        self.dt = prb.getDt()

        # derived classes should at least provide the optimal state trajectory, 
        # and input trajectory
        self.nx = self.x.size1()
        self.nu = self.u.size1()
        self.x_opt = np.zeros(shape=(self.nx, prb.getNNodes()))
        self.u_opt = np.zeros(shape=(self.nu, prb.getNNodes()-1))

        # setup real-time iteration options
        self.rti = opts.get('realtime_iteration', False)

        if self.rti:
            self.configure_rti()
            del self.opts['realtime_iteration']
        

    def _getVarList(self, type):
        var_list = list()
        for var in self.prb.var_container.getVarList(offset=False):
            if type == 'ub':
                retriever = var.getUpperBounds()
            elif type == 'lb':
                retriever = var.getLowerBounds()
            elif type == 'ig':
                retriever = var.getInitialGuess()
            else:
                raise ValueError(f"value '{type}' not allowed.")

            var_list.append(retriever.flatten(order='F'))

        v = cs.vertcat(*var_list)
        return v

    def _getParList(self):
        par_list = list()
        for par in self.prb.var_container.getParList(offset=False):
            pval = par.getValues().flatten(order='F')
            par_list.append(pval)

        p = cs.vertcat(*par_list)
        return p

    def _getFunList(self, type):
        fun_list = list()
        for fun in self.prb.function_container.getCnstr().values():
            if type == 'ub':
                retriever = fun.getUpperBounds()
            elif type == 'lb':
                retriever = fun.getLowerBounds()
            else:
                raise ValueError(f"value '{type}' not allowed.")

            fun_list.append(retriever.flatten(order='F'))

        f = cs.vertcat(*fun_list)
        return f

    def _createCnsrtLambDict(self, solution):

        lambd_cnsrt_dict = dict()
        pos = 0
        for name, fun in self.prb.function_container.getCnstr().items(): # iterate through each symbolic constraint
            
            #extracting and reshaping the constraint values associated to name
            lam_g_vals = solution['lam_g'][pos:pos + fun.getDim() * len(fun.getNodes())]
            lam_g_vals_mat = np.reshape(lam_g_vals, (fun.getDim(), len(fun.getNodes())), order='F')

            lambd_cnsrt_dict[name + "_lambd"] = lam_g_vals_mat

            pos = pos + fun.getDim() * len(fun.getNodes()) 
        
        return lambd_cnsrt_dict

    def _createCnsrtSolDict(self, solution):

        fun_sol_dict = dict()
        pos = 0
        for name, fun in self.prb.function_container.getCnstr().items():
            val_sol = solution['g'][pos:pos + fun.getDim() * len(fun.getNodes())]
            fun_sol_matrix = np.reshape(val_sol, (fun.getDim(), len(fun.getNodes())), order='F')
            fun_sol_dict[name] = fun_sol_matrix
            pos = pos + fun.getDim() * len(fun.getNodes())

        return fun_sol_dict

    def _createVarSolDict(self, solution):
        pos = 0
        var_sol_dict = dict()
        for var in self.prb.var_container.getVarList(offset=False):
            val_sol = solution['x'][pos: pos + var.shape[0] * len(var.getNodes())]
            # this is to divide in rows each dim of the var
            val_sol_matrix = np.reshape(val_sol, (var.shape[0], len(var.getNodes())), order='F')
            var_sol_dict[var.getName()] = val_sol_matrix
            pos = pos + var.shape[0] * len(var.getNodes())

        return var_sol_dict

    def _createVarSolAsInOut(self, solution):

        input_vars = [v.getName() for v in self.prb.getInput().var_list]
        state_vars = [v.getName() for v in self.prb.getState().var_list]
        pos = 0
        for name, var in self.prb.var_container.getVar().items():
            val_sol = solution['x'][pos: pos + var.shape[0] * len(var.getNodes())]
            val_sol_matrix = np.reshape(val_sol, (var.shape[0], len(var.getNodes())), order='F')
            if name in state_vars:
                off, _ = self.prb.getState().getVarIndex(name)
                self.x_opt[off:off + var.shape[0], :] = val_sol_matrix
            elif name in input_vars:
                off, _ = self.prb.getInput().getVarIndex(name)
                self.u_opt[off:off + var.shape[0], :] = val_sol_matrix
            else:
                pass
            pos = pos + var.shape[0] * len(var.getNodes())

    def _createDtSol(self):

        # build dt_solution as an array
        self.dt_solution = np.zeros(self.prb.getNNodes() - 1)
        dt = self.prb.getDt()

        if not self.var_solution:
            raise ValueError('Solution container is empty.')
        # todo make this better
        # fill dt_solution

        # if it is a variable, its corresponding solution must be retrieved.
        # if it is a singleVariable or a singleParameter?
        # if dt is directly an optimization variable, that's ok, I get it from the var_solution
        # if dt is a function of some other optimization variables, get all of them and compute the optimized dt
        #   I do this by using a Function to wrap everything
        if isinstance(dt, cs.SX) and not isinstance(dt, (Variable, SingleVariable, Parameter, SingleParameter)):
            var_depend = list()
            for var in self.prb.getVariables().values():
                if cs.depends_on(dt, var):
                    var_depend.append(var)

            single_var_flag = False
            # check type of variable
            if all([isinstance(var, Variable) for var in var_depend]):
                pass
            elif all([isinstance(var, SingleVariable) for var in var_depend]):
                single_var_flag = True
            else:
                raise NotImplementedError('Yet to be done.')

            # create a function with all the variable dt depends on, and return dt
            temp_dt = cs.Function('temp_dt', var_depend, [dt])

            # fill the self.dt_solution with all the dt values
            node_n_out = 0
            for node_n in range(self.prb.getNNodes()-1):
                node_n_in = node_n

                # if the variable is a SingleVariable, fill dt_solution with the only value of the variable (node_n_out = 0)
                if not single_var_flag:
                    node_n_out = node_n

                self.dt_solution[node_n_in] = temp_dt(*[self.var_solution[var.getName()] for var in var_depend])[node_n_out]

        # fill the self.dt_solution with all the dt values
        elif isinstance(dt, Variable):
            for node_n in range(self.prb.getNNodes()-1):
                # from matrix to array
                sol_dt_array = self.var_solution[dt.getName()].flatten()
                self.dt_solution[node_n] = sol_dt_array[node_n]

        # fill the self.dt_solution with the same dt solution
        elif isinstance(dt, SingleVariable):
            for node_n in range(self.prb.getNNodes()-1):
                self.dt_solution[node_n] = self.var_solution[dt.getName()]

        # if dt is a value, set it to each element of dt_solution
        elif isinstance(dt, (Parameter, SingleParameter)):
            for node_n in range(self.prb.getNNodes()-1):
                # get only the nodes where the dt is selected
                # here dt at node 0 is not defined
                self.dt_solution[node_n] = dt.getValues(node_n)
        # if dt is a value, set it to each element of dt_solution
        elif isinstance(dt, (float, int)):
            for node_n in range(self.prb.getNNodes() - 1):
                self.dt_solution[node_n] = dt
        # if dt is a list, get each dt separately
        elif isinstance(dt, List):
            for node_n in range(self.prb.getNNodes()-1):
                dt_val = dt[node_n]
                if isinstance(dt_val, SingleVariable):
                    self.dt_solution[node_n] = self.var_solution[dt_val.getName()]
                elif isinstance(dt_val, Variable):
                    current_node = dt_val.getNodes().index(node_n)
                    sol_var = self.var_solution[dt_val.getName()].flatten()[current_node]
                    self.dt_solution[node_n] = sol_var
                elif isinstance(dt_val, (Parameter, SingleParameter)):
                    self.dt_solution[node_n] = dt_val.getValues(node_n)
                else:
                    self.dt_solution[node_n] = dt[node_n]
        else:
            raise ValueError(f'dt of type: {type(dt)} is not supported.')

    @abstractmethod
    def solve(self) -> bool:
        """
        Solve the horizon's problem

        Returns:
            bool: success flag
        """
        pass

    def getSolutionDict(self):
        """
        Get the horizon's solution

        Returns:
            dict: dictionary of variable optimized
        """
        pass

    def getDt(self):
        """
        Get the horizon's dt as a vector of dt for each node.

        Returns:
            array: array of dt values for each node
        """
        pass

    def getSolutionState(self):
        pass

    def getSolutionInput(self):
        pass

    def configure_rti(self) -> bool:
        """
        Take appropriate actions to make the solver suitable for
        a RTI scheme

        Returns:
            bool: success flag
        """