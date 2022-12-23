import copy

import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn
import casadi as cs
from horizon import misc_function as misc
from horizon.problem import Problem
from horizon.utils import utils, kin_dyn
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers.solver import Solver
from horizon.rhc.tasks.cartesianTask import CartesianTask
from horizon.ros import replay_trajectory
from horizon.rhc.taskInterface import TaskInterface
import rospy
import os
import subprocess
import itertools
import time
import logging

#todo could be closer to the way I did the PhaseManager in c++
class bcolors:

    CPURPLE = '\033[95m'
    CBLUE0 = '\033[94m'
    CCYAN0 = '\033[96m'
    CGREEN0 = '\033[92m'
    CYELLOW0 = '\033[93m'
    CRED0 = '\033[91m'

    CEND = '\33[0m'
    CBOLD = '\33[1m'
    CITALIC = '\33[3m'
    CUNDERLINE = '\33[4m'
    CBLINK = '\33[5m'
    CBLINK2 = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK = '\33[30m'
    CRED = '\33[31m'
    CGREEN = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE = '\33[36m'
    CWHITE = '\33[37m'

    CBLACKBG = '\33[40m'
    CREDBG = '\33[41m'
    CGREENBG = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG = '\33[46m'
    CWHITEBG = '\33[47m'

    CGREY = '\33[90m'
    CRED2 = '\33[91m'
    CGREEN2 = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2 = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2 = '\33[96m'
    CWHITE2 = '\33[97m'

    CGREYBG = '\33[100m'
    CREDBG2 = '\33[101m'
    CGREENBG2 = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2 = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2 = '\33[106m'
    CWHITEBG2 = '\33[107m'


class Phase:
    def __init__(self, phase_name, n_nodes_phase):
        self.name = phase_name
        self.n_nodes = n_nodes_phase

        self.constraints = dict()
        self.costs = dict()

        self.vars = dict()
        self.vars_node = dict()
        self.var_bounds = dict()

        self.pars = dict()
        self.pars_node = dict()
        self.par_values = dict()

    def addConstraint(self, constraint, nodes=None):
        active_nodes = range(self.n_nodes) if nodes is None else nodes
        self.constraints[constraint] = active_nodes

    def addCost(self, cost, nodes=None):
        active_nodes = range(self.n_nodes) if nodes is None else nodes
        self.costs[cost] = active_nodes

    def addVariableBounds(self, var, lower_bounds, upper_bounds, nodes=None):
        # todo: this is not very nice, find another way? would be nice to copy the structure of addCost and addConstraint
        active_nodes = range(self.n_nodes) if nodes is None else nodes
        self.vars[var.getName()] = var
        self.vars_node[var.getName()] = active_nodes
        self.var_bounds[var.getName()] = (lower_bounds, upper_bounds)

    def addParameterValues(self, par, values, nodes=None):
        active_nodes = range(self.n_nodes) if nodes is None else nodes
        self.pars[par.getName()] = par
        self.pars_node[par.getName()] = active_nodes
        self.par_values[par.getName()] = values

    def setDuration(self):
        pass


class PhaseToken(Phase):
    def __init__(self, *args, logger=None):
        self.__dict__ = args[0].__dict__.copy()

        if not logger:
            self.debug_mode = False
            self.logger = None
        else:
            self.logger = logger
            self.debug_mode = self.logger.isEnabledFor(logging.DEBUG)

        self.id = '1337c0d3'
        # self.active_nodes = np.zeros(n_nodes_phase).astype(int)
        # self.active_nodes = [0] * n_nodes_phase
        # self.active_nodes = set()
        self.active_nodes = list()
        self.constraints_in_horizon = dict.fromkeys(self.constraints.keys(), set())
        self.costs_in_horizon = dict.fromkeys(self.costs.keys(), set())
        self.vars_in_horizon = dict.fromkeys(self.vars.keys(), set())
        self.pars_in_horizon = dict.fromkeys(self.pars.keys(), set())

    def function_update(self, initial_node, input_container, output_container):

        for item, nodes in input_container.items():
            phase_nodes = [node for node in nodes if node in self.active_nodes]
            if item in output_container:
                if phase_nodes != output_container[item]:
                    if phase_nodes:
                        # tic = time.time()
                        # todo this is wrong if constraint nodes are not the full phase nodes
                        horizon_nodes = range(initial_node, initial_node + len(phase_nodes))
                        feas_nodes = horizon_nodes
                        # todo: computing feas_node is VERY computationally expensive
                        # feas_nodes = [node for node in horizon_nodes if
                        #               node in misc.getNodesFromBinary(item._getFeasNodes())]
                        # print(f'         compute horizon nodes: {time.time() - tic}')
                    else:
                        feas_nodes = []
                    output_container[item] = feas_nodes

                    if self.debug_mode:
                        self.logger.debug(
                            f'{bcolors.CCYAN0}   --->  {item.getName()}. Nodes to add: {list(feas_nodes)}{bcolors.CEND}')
                else:
                    del output_container[item]

    def variable_update(self, initial_node, input_node_container, input_item_container, output_container):

        for item, nodes in input_node_container.items():
            phase_nodes = [node for node in nodes if node in self.active_nodes]
            if item in output_container:
                if phase_nodes != output_container[item]:
                    if phase_nodes:
                        horizon_nodes = range(initial_node, initial_node + len(phase_nodes))
                        feas_nodes = horizon_nodes
                        # todo: computing feas_node is VERY computationally expensive
                        # feas_nodes = [node for node in horizon_nodes if
                        #               node in misc.getNodesFromBinary(input_item_container[item]._nodes_array)]
                        # print(f'         compute horizon nodes: {time.time() - tic}')
                    else:
                        feas_nodes = []
                    output_container[item] = feas_nodes

                    if self.debug_mode:
                        self.logger.debug(
                            f'{bcolors.CCYAN0}   --->  {input_item_container[item].getName()}. Nodes to add: {list(feas_nodes)}{bcolors.CEND}')
                else:
                    del output_container[item]
    def update(self, initial_node):

        '''
        update phase "anonymous" nodes appending the right nodes to the horizon
        '''

        if self.debug_mode:
            self.logger.debug(f'{bcolors.CCYAN0}Phase "{self.name}" starting at node: {initial_node}{bcolors.CEND}')
            self.logger.debug(f'{bcolors.CCYAN0}Phase is active on nodes: {self.active_nodes}{bcolors.CEND}')

        # initial_tic = time.time()
        # given the active nodes of a phase:
        #   for each constraint, get the active nodes and compute the horizon nodes it is active node
        # tic = time.time()
        self.function_update(initial_node, self.constraints, self.constraints_in_horizon)
        # print(f' --> updating constraints: {time.time() - tic}')

        # tic = time.time()
        self.function_update(initial_node, self.costs, self.costs_in_horizon)
        # print(f' --> updating costs: {time.time() - tic}')

        # tic = time.time()
        self.variable_update(initial_node, self.vars_node, self.vars, self.vars_in_horizon)
        # print(f' --> updating vars: {time.time() - tic}')

        # tic = time.time()
        self.variable_update(initial_node, self.pars_node, self.pars, self.pars_in_horizon)
        # print(f' --> updating pars: {time.time() - tic}')

        # print(f'updated one phase: {time.time() - initial_tic}')

    def reset(self):
        self.active_nodes = list()
        self.constraints_in_horizon = self.constraints_in_horizon.fromkeys(self.constraints_in_horizon, set())
        self.costs_in_horizon = self.costs_in_horizon.fromkeys(self.costs_in_horizon, set())
        self.vars_in_horizon = self.vars_in_horizon.fromkeys(self.vars_in_horizon, set())
        self.pars_in_horizon = self.pars_in_horizon.fromkeys(self.pars_in_horizon, set())


"""
1. add a phase starting from a specific node
2. add a pattern that repeats (how to stop?)
3. 
"""


class HorizonManager:
    """
    this class is required to manage multiple phases update.
    If multiple phases use the same constraint, the constraint must be updated with the nodes from all the phases.
    (doing setNodes will reset all the nodes, so I need to keep track of the nodes (coming possibly from different phases) for the same constraint)
    """

    def __init__(self, logger=None):

        if not logger:
            self.debug_mode = False
            self.logger = None
        else:
            self.logger = logger
            self.debug_mode = self.logger.isEnabledFor(logging.DEBUG)

        # self.phases = list()
        self.constraints = dict()
        self.costs = dict()

        self.vars = dict()  # dict of variableUpdaters
        self.pars = dict()  # dict of parameterUpdaters

    class VariableUpdater:
        def __init__(self, variable, logger=None):

            if not logger:
                self.debug_mode = False
                self.logger = None
            else:
                self.logger = logger
                self.debug_mode = self.logger.isEnabledFor(logging.DEBUG)

            self.name = variable.getName()
            self.var = variable

            self.var_nodes = list(self.var.getNodes())
            self.active_nodes = list()
            self.horizon_nodes = list()
            self.lower_bounds = -np.inf * np.ones((self.var.getDim(), len(self.var_nodes)))
            self.upper_bounds = np.inf * np.ones((self.var.getDim(), len(self.var_nodes)))

        def update(self, nodes, bounds, phase_nodes):
            # add new nodes and bounds to var if var already has them
            self.active_nodes.extend(phase_nodes)
            self.horizon_nodes.extend(nodes)

            if nodes:
                # only fill the required nodes
                self.lower_bounds[:, nodes] = bounds[0][:, phase_nodes]
                self.upper_bounds[:, nodes] = bounds[1][:, phase_nodes]
                # if self.debug_mode:
                    # self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD} '
                    #                   f'updated variable {self.var.getName()} at nodes: {self.horizon_nodes}'
                    #                   f'{bcolors.CEND}')

            else:
                self.horizon_nodes = []
                # if self.debug_mode:
                # self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD} '
                #                   f'updated variable {self.var.getName()} at all nodes.'
                #                   f'{bcolors.CEND}')
        def set_horizon_nodes(self):

            self.var.setBounds(self.lower_bounds, self.upper_bounds)
            # else:
            #     if self.horizon_nodes:
            #         self.var.setBounds(self.lower_bounds[:, self.horizon_nodes],
            #                            self.upper_bounds[:, self.horizon_nodes],
            #                            self.horizon_nodes)
            #     else:
            #         self.var.setBounds(self.lower_bounds,
            #                            self.upper_bounds)
            #
            if self.debug_mode:
                self.logger.debug(f"{bcolors.CCYAN0}{bcolors.CUNDERLINE}{bcolors.CBOLD}"
                                  f"updated variable's bounds {self.var.getName()}."
                                  f" Relevant nodes: {self.horizon_nodes}"
                                  f"{bcolors.CEND}")
            # def add_phase(self, phase):

    class ParameterUpdater:
        def __init__(self, parameter, logger=None):

            if not logger:
                self.debug_mode = False
                self.logger = None
            else:
                self.logger = logger
                self.debug_mode = self.logger.isEnabledFor(logging.DEBUG)

            self.name = parameter.getName()
            self.par = parameter

            self.active_nodes = list()
            self.horizon_nodes = list()
            self.values = np.zeros((self.par.getDim(), len(self.par.getNodes())))

        def update(self, nodes, values, phase_nodes):

            self.active_nodes.extend(phase_nodes)
            self.horizon_nodes.extend(nodes)

            if nodes:
                self.values[:, nodes] = values[:, phase_nodes]
                # nope self.par.assign(self.values[:, self.horizon_nodes], self.horizon_nodes)

                # if self.debug_mode:
                #     self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD} updated parameters {self.par.getName()} at nodes: {self.horizon_nodes}{bcolors.CEND}')
            else:
                self.horizon_nodes = []
                # nope self.par.assign(self.values)
                # if self.debug_mode:
                    # self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD} updated parameters {self.par.getName()} at all nodes.{bcolors.CEND}')

        def set_horizon_nodes(self):

            # todo: for now the values are assigned at all nodes (given that I keep track of all the added values)
            # nope self.par.assign(self.values[:, self.horizon_nodes], self.horizon_nodes)
            self.par.assign(self.values)
            #
            if self.debug_mode:
                self.logger.debug(f"{bcolors.CCYAN0}{bcolors.CUNDERLINE}{bcolors.CBOLD}"
                                  f"updated parameter's values {self.par.getName()}."
                                  f" Relevant nodes: {self.horizon_nodes}"
                                  f"{bcolors.CEND}")
    # def add_phase(self, phase):
    #
    #     self.phases.append(phase)
    #     self.update_phase(phase)

    def update_function(self, container, fun, nodes):
        # tic = time.time()
        if fun not in container:
            container[fun] = set(nodes)
            # nope fun.setNodes(list(nodes))
            # if self.debug_mode:
            #     self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD}'f'Adding nodes to '
            #                       f'{bcolors.CUNDERLINE}new{bcolors.CEND}'
            #                       f'{bcolors.CCYAN0}{bcolors.CBOLD} function {fun.getName()}: {fun.getNodes()}{bcolors.CEND}')
        else:
            if set(nodes) != container[fun]:
                # todo: there is a small inefficiency: resetting all the nodes even if just only a part are added
                [container[fun].add(node) for node in nodes]
                # nope fun.setNodes(list(container[fun]))
                # if self.debug_mode:
                #     self.logger.debug(f'{bcolors.CCYAN0}{bcolors.CBOLD} '
                #                       f'Adding nodes to function {fun.getName()}: {fun.getNodes()}{bcolors.CEND}')

        # print(f'        updated fun: {time.time() - tic}')
    def update_variable(self, var, nodes, bounds, phase_nodes):
        var_name = var.getName()
        if var_name not in self.vars:
            self.vars[var_name] = self.VariableUpdater(var, self.logger)

        for var_item in self.vars.values():
            var_item.update(nodes, bounds[var_name], phase_nodes)  # update var with variable updater

    def update_parameter(self, par, nodes, values, phase_nodes):
        par_name = par.getName()
        if par_name not in self.pars:
            self.pars[par_name] = self.ParameterUpdater(par, self.logger)

        for par_item in self.pars.values():
            par_item.update(nodes, values[par_name], phase_nodes)  # update var with parameter updater

    def update_constraint(self, constraint, nodes):
        self.update_function(self.constraints, constraint, nodes)

    def update_cost(self, cost, nodes):
        self.update_function(self.costs, cost, nodes)

    def update_phase(self, phase):
        # print(f'{bcolors.CBLUE2} updating phase: {phase.name}{bcolors.CEND}')
        initial_tic = time.time()

        # todo why functions does not require reset?
        # todo add bounds to constraints
        # tic = time.time()
        # [self.update_constraint(constraint, nodes) for constraint, nodes in phase.constraints_in_horizon.items()]
        for constraint, nodes in phase.constraints_in_horizon.items():
            self.update_constraint(constraint, nodes)
        # print('      --> constraints:', time.time() - tic)
        # tic = time.time()
        # [self.update_cost(cost, nodes) for cost, nodes in phase.costs_in_horizon.items()]
        for cost, nodes in phase.costs_in_horizon.items():
            self.update_cost(cost, nodes)
        # print('      --> costs:', time.time() - tic)

        # tic = time.time()
        # [self.update_variable(phase.vars[var], nodes, phase.var_bounds, phase.active_nodes) for var, nodes in phase.vars_in_horizon.items()]
        for var, nodes in phase.vars_in_horizon.items():
            self.update_variable(phase.vars[var], nodes, phase.var_bounds, phase.active_nodes)
        # print('      --> vars:', time.time() - tic)

        # tic = time.time()
        # [self.update_parameter(phase.pars[par], nodes, phase.par_values, phase.active_nodes) for par, nodes in phase.pars_in_horizon.items()]
        for par, nodes in phase.pars_in_horizon.items():
            self.update_parameter(phase.pars[par], nodes, phase.par_values, phase.active_nodes)
        # print('      --> pars:', time.time() - tic)

        if self.debug_mode:
            print(f'{bcolors.CITALIC}{bcolors.CYELLOW0} -> update single phase {time.time() - initial_tic}{bcolors.CEND}')

    def reset(self):

        self.constraints = dict()
        self.costs = dict()
        self.vars = dict()
        self.pars = dict()

    def set_horizon_nodes(self):
        # todo what about bounds in constraints?
        # todo incorporate these two
        for constraint, nodes in self.constraints.items():
            constraint.setNodes(list(nodes))
            if self.debug_mode:
                self.logger.debug(f'{bcolors.CBLUE}{bcolors.CUNDERLINE}{bcolors.CBOLD}'
                                  f'updated function {constraint.getName()}: {constraint.getNodes()}'
                                  f'{bcolors.CEND}')

        for cost, nodes in self.costs.items():
            cost.setNodes(list(nodes))
            if self.debug_mode:
                self.logger.debug(f'{bcolors.CBLUE}{bcolors.CUNDERLINE}{bcolors.CBOLD}'
                                  f'updated function {cost.getName()}: {cost.getNodes()}'
                                  f'{bcolors.CEND}')

        for var_item in self.vars.values():
            var_item.set_horizon_nodes()

        for par_item in self.pars.values():
            par_item.set_horizon_nodes()

# def add_flatten_lists(the_lists):
#     result = []
#     for _list in the_lists:
#         result += _list
#     return result
#

class SinglePhaseManager:
    """
    set of actions which involves combinations of constraints and bounds
    """

    def __init__(self, nodes, name=None, logger=None):

        if not logger:
            self.debug_mode = False
            self.logger = None
        else:
            self.logger = logger
            self.debug_mode = self.logger.isEnabledFor(logging.DEBUG)

        # prb: Problem, urdf, kindyn, contacts_map, default_foot_z
        self.name = name
        # self.prb = problem
        self.registered_phases = dict()  # container of all the registered phases
        self.n_tot = nodes  # self.prb.getNNodes() + 1

        self.horizon_manager = HorizonManager(self.logger)

        self.phases = list()
        self.active_phases = list()  # list of all the active phases
        self.activated_nodes = list()
        self.horizon_nodes = np.nan * np.ones(self.n_tot)

        # empty nodes in horizon --> at the very beginning, all
        self.trailing_empty_nodes = self.n_tot

        # self.default_action = Phase('default', 1)
        # self.registerPhase(self.default_action)

    def registerPhase(self, p):

        self.registered_phases[p.name] = p

    def getRegisteredPhase(self, p=None):

        if p is None:
            return list(self.registered_phases.items())
        else:
            return self.registered_phases[p] if p in self.registered_phases else None

    def addPhase(self, phases, pos=None):

        if self.debug_mode:
            self.logger.debug(
                f'{bcolors.CVIOLET} =========== Happening in timeline: {self.name} =========== {bcolors.CEND}')

        # stupid checks
        if isinstance(phases, list):
            for phase in phases:
                assert isinstance(phase, Phase)
        else:
            assert isinstance(phases, Phase)
            phases = [phases]

        if self.debug_mode:
            self.logger.debug(
                f'{bcolors.CRED}{bcolors.CBOLD} Adding phases: {[phase.name for phase in phases]} at position: {pos}{bcolors.CEND}')

        # print(f'{bcolors.FAIL} Current phases:')
        # for phase in self.phases:
        #     print(f'{bcolors.FAIL}    - {phase.name}: {phase.active_nodes}')

        # generate a phase token from each phase commanded by the user
        phases_to_add = [PhaseToken(phase) for phase in phases]
        # append or insert, depending on the user
        if pos is None:
            self.phases.extend(phases_to_add)
        else:
            # if pos is beyond the horizon (outside of the active_phases), skip useless computation:
            # if pos <= len(self.active_phases):
            # recompute empty nodes in horizon, clear active nodes of all the phases AFTER the one inserted
            # insert phase + everything AFTER it back
            self.trailing_empty_nodes = self.n_tot - sum(len(s.active_nodes) for s in self.phases[:pos])
            # if I'm removing phases from the current stack, i need to remove current nodes from the phases
            # todo make this faster?
            [elem.reset() for elem in self.phases[pos:]]
            self.horizon_manager.reset()
            [self.horizon_manager.update_phase(phase) for phase in self.phases[:pos]]
            # todo this reset is wrong!
            # new active phases
            self.active_phases = [phase for phase in self.phases if phase.active_nodes]

            self.phases[pos:pos] = phases_to_add
            phases_to_add.extend(self.phases[pos + 1:])

        # print(f'{bcolors.CYELLOW} updating phases:')
        # for phase in phases_to_add:
        #     print(f'{bcolors.CYELLOW}    - {phase.name}{bcolors.CEND}')

        if self.trailing_empty_nodes > 0:
            # set active node for each added phase
            current_phase = 0
            for phase in phases_to_add:
                current_phase += 1
                self.pos_in_horizon = self.n_tot - self.trailing_empty_nodes
                self.trailing_empty_nodes -= phase.n_nodes
                len_phase = phase.n_nodes
                if self.trailing_empty_nodes <= 0:
                    remaining_nodes = len_phase + self.trailing_empty_nodes
                    phase.active_nodes.extend(range(remaining_nodes))
                    self.active_phases.append(phase)
                    phase.update(self.pos_in_horizon)
                    self.trailing_empty_nodes = 0
                    break
                self.active_phases.append(phase)
                phase.active_nodes.extend(range(len_phase))
                phase.update(self.pos_in_horizon)

            # update only if phase_to_add is inside horizon ( --> :current_phase)
            [self.horizon_manager.update_phase(phase) for phase in phases_to_add[:current_phase]]
            self.horizon_manager.set_horizon_nodes()

        # print(f'{bcolors.CRED} Updated phases:')
        # for phase in self.phases:
        #     print(f'{bcolors.CRED}    - {phase.name}: {phase.active_nodes}{bcolors.CEND}')

    def getActivePhase(self, phase=None):

        if phase is None:
            return self.active_phases.copy()
        else:
            raise NotImplementedError
            # return self.active_phases.[phase]

    def getHorizonNodes(self):

        assert self.horizon_nodes.shape[0] == self.n_tot
        return self.horizon_nodes

    def _shift_phases(self):
        initial_tic = time.time()
        # tic = time.time()

        if self.debug_mode:
            self.logger.debug(
                f'{bcolors.CVIOLET} =========== Happening in timeline: {self.name} =========== {bcolors.CEND}')
        # if phases is empty, skip everything
        if self.phases:
            # update active_phases with all the phases that have active nodes
            self.active_phases = [phase for phase in self.phases if phase.active_nodes]

            # print('adding phase to active phase:', time.time() - tic)

            # tic = time.time()

            last_active_phase = self.active_phases[-1]
            # if 'last active node' of 'last active phase' is the last of the phase, add new phase, otherwise continue to fill the phase
            if last_active_phase.n_nodes - 1 in last_active_phase.active_nodes:
                n_active_phases = len(self.active_phases)
                # add new active phase from list of added phases
                if n_active_phases < len(self.phases):
                    elem = self.phases[n_active_phases].active_nodes[-1] + 1 if self.phases[n_active_phases].active_nodes else 0
                    self.phases[n_active_phases].active_nodes.append(elem)
                    self.active_phases.append(self.phases[n_active_phases])
            else:
                # add new node to last active phase
                elem = last_active_phase.active_nodes[-1] + 1 if last_active_phase.active_nodes else 0
                last_active_phase.active_nodes.append(elem)

            # print('update nodes of phases:', time.time() - tic)
            # tic = time.time()

            # remove first element in phases
            self.active_phases[0].active_nodes = self.active_phases[0].active_nodes[1:]

            # print('remove first element in phases:', time.time() - tic)

            # tic = time.time()
            # burn depleted phases
            if not self.phases[0].active_nodes:
                if self.debug_mode:
                    self.logger.debug(f'{bcolors.CYELLOW}Removing depleted phase: {self.phases[0].name}{bcolors.CEND}')
                self.phases = self.phases[1:]
            # print('burn depleted phases:', time.time() - tic)

            # tic = time.time()
            self.horizon_manager.reset()
            # print('reset horizon_manager:', time.time() - tic)

            # tic = time.time()
            i = 0
            for phase in self.active_phases:
                phase.update(i)
                i += len(phase.active_nodes)
            # print('update nodes for each phase:', time.time() - tic)

            # tic = time.time()
            [self.horizon_manager.update_phase(phase) for phase in self.active_phases]
            self.horizon_manager.set_horizon_nodes()
            # print('update phases all together:', time.time() - tic)

        self.trailing_empty_nodes = self.n_tot - sum(len(s.active_nodes) for s in self.active_phases)

        # print('active phases: ')
        # for phase in self.active_phases:
        #     print(phase.name, phase.active_nodes)

        # print("free nodes:", self.trailing_empty_nodes)
        # print('one shift:', time.time() - initial_tic)
        # for phase in self.active_phases:
        #     phase.reset()

class PhaseManager:
    def __init__(self, nodes, opts=None):

        self.logger = None
        self.debug_mode = False

        if opts is not None and 'logging_level' in opts:
            self.logger = logging.getLogger('logger')
            self.logger.setLevel(level=opts['logging_level'])
            self.debug_mode = self.logger.isEnabledFor(logging.DEBUG)

        self.nodes = nodes
        self.timelines = []
        self.n_timelines = 0

    def addTimeline(self, name=None):

        new_timeline = SinglePhaseManager(self.nodes, name, self.logger)
        self.timelines.append(new_timeline)
        self.n_timelines += 1

        return new_timeline

    def registerPhase(self, p, timeline):
        self.timelines[timeline].registerPhase(p)

    def addPhase(self, phase, pos=None, timeline=0):
        self.timelines[timeline].addPhase(phase, pos)

    def _shift_phases(self):

        for timeline in self.timelines:
            # tic = time.time()
            timeline._shift_phases()
            # print('timeline cycle:', time.time() - tic)


if __name__ == '__main__':

    n_nodes = 12
    prb = Problem(n_nodes, receding=True, casadi_type=cs.SX)
    x = prb.createStateVariable('x', 2)
    y = prb.createStateVariable('y', 2)
    u = prb.createInputVariable('u', 2)
    # z = prb.createVariable('u', 2, nodes=[3, 4, 5])

    # z.getImpl([2, 3, 4])
    # exit()
    pm = PhaseManager(n_nodes)
    pm.addTimeline('0')

    # cnsrt4 = prb.createConstraint('constraint_4', x * z, nodes=[3, 4])
    cnsrt1 = prb.createIntermediateConstraint('constraint_1', x - u, [])
    cnsrt2 = prb.createConstraint('constraint_2', x - y, [])
    cnsrt3 = prb.createConstraint('constraint_3', 3 * x, [])
    cost1 = prb.createIntermediateResidual('cost_1', x + u, [])

    in_p = Phase('initial', 5)
    st_p = Phase('stance', 5)

    in_p.addConstraint(cnsrt1)
    in_p.addVariableBounds(y, np.array([[0, 0]] * 5).T, np.array([[0, 0]] * 5).T)
    st_p.addConstraint(cnsrt2)

    pm.registerPhase(in_p, 0)
    #
    print(pm.timelines)
    pm.addPhase(in_p, timeline=0)
    pm.addPhase(in_p, timeline=0)

    exit()
    pm.addPhase(in_p, timeline=0)
    pm.addPhase(st_p, timeline=0)
    # pm.addPhase(in_p)
    #
    for j in range(10):
        print(f'--------- shifting {j} -----------')
        tic = time.time()
        pm._shift_phases()
        print('elapsed time in shifting:', time.time() - tic)

    exit()

    n_nodes = 11
    prb = Problem(n_nodes, receding=True, casadi_type=cs.SX)
    x = prb.createStateVariable('x', 2)
    y = prb.createStateVariable('y', 2)
    u = prb.createInputVariable('u', 2)
    # z = prb.createVariable('u', 2, nodes=[3, 4, 5])

    # z.getImpl([2, 3, 4])
    # exit()
    pm = PhaseManager(n_nodes + 1)
    pm.addTimeline('0')
    pm.addTimeline('1')

    # cnsrt4 = prb.createConstraint('constraint_4', x * z, nodes=[3, 4])
    cnsrt1 = prb.createIntermediateConstraint('constraint_1', x - u, [])
    cnsrt2 = prb.createConstraint('constraint_2', x - y, [])
    cnsrt3 = prb.createConstraint('constraint_3', 3 * x, [])
    cost1 = prb.createIntermediateResidual('cost_1', x + u, [])

    in_p = Phase('initial', 5)
    st_p = Phase('stance', 6)
    fl_p = Phase('flight', 2)

    # in_p.addConstraint(cnsrt4)
    in_p.addConstraint(cnsrt1, nodes=range(0, 2))
    # in_p.addConstraint(cnsrt2)
    # in_p.addCost(cost1, nodes=[3, 4])
    fl_p.addConstraint(cnsrt2)
    st_p.addConstraint(cnsrt3)

    pm.registerPhase(in_p, timeline=0)
    pm.registerPhase(st_p, timeline=1)
    pm.registerPhase(fl_p, timeline=0)

    tic = time.time()
    for i in range(1):
        pm.addPhase(in_p, timeline=0)
        pm.addPhase(st_p, timeline=1)
        pm.addPhase(fl_p, timeline=0)
    print(f'elapsed_time: {time.time() - tic}')

    # for phase in pm.active_phases:
    #     print(f'{phase.name, phase.constraints}')
    # for c_name, c_item in prb.getConstraints().items():
    #     print(f'{c_name}: {c_item.getNodes()}')

    # print('========= all added phases ===========')
    # for phase in pm.phases:
    #     print(phase.name, phase.active_nodes)

    for j in range(30):
        print('--------- shifting -----------')
        tic = time.time()
        pm._shift_phases()
        print('elapsed time in shifting:', time.time() - tic)
        if j == 3:
            print('-------- adding phase -----------')
            pm.addPhase(in_p, 1)

        if j == 20:
            print('-------- adding phase -----------')
            pm.addPhase(st_p, 3)

        # for phase in pm.phases:
        #     print(f'{phase.name}:, {phase.active_nodes}:')

    # for phase in pm.active_phases:
    #     print(f'{phase.name, phase.constraints}')
    # for c_name, c_item in prb.getConstraints().items():
    #     print(f'{c_name}: {c_item.getNodes()}')
