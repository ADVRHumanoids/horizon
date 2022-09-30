import numpy as np

from horizon import problem
from horizon.variables import Variable, SingleVariable, Parameter, SingleParameter
from horizon.utils import utils, kin_dyn, resampler_trajectory, mat_storer
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.ros.replay_trajectory import *
from horizon.solvers import solver
import horizon.variables as sv
import horizon.functions as fn
import horizon.misc_function as misc
import matplotlib.pyplot as plt
from horizon.solvers import Solver
from itertools import groupby
from operator import itemgetter
from typing import List


# todo only work with a 'dt' specified as Variable. In fact, if dt is:
#  - a list: changing the number of nodes with the refiner means inject new nodes between two old nodes. This requires
#    the possibility to add locally nodes, which is not yet implemented. This is important because, if nodes are injected in a given interval,
#    all the variables and the constraints defined in that interval needs to be expanded.
#  - constant: all the constraint and cost functions depending on a constant dt needs to be:
#     detected (so it cannot be constant, but maybe parameter) and then modified depending on the injected nodes.
def group_elements(vec):
    ranges_vec = list()
    for k, g in groupby(enumerate(vec), lambda x: x[0] - x[1]):
        group = (map(itemgetter(1), g))
        group = list(map(int, group))
        # all elements:
        ranges_vec.append(group)
        # ranges:
        # ranges_vec.append((group[0], group[-1]))

    return ranges_vec

def resample(solution, prb, dt_res, contacts_name):

    n_nodes = prb.getNNodes()

    prev_f_list = [prev_solution[f'force_{c_name}'] for c_name in contacts_name]

    solution_res = dict()
    u_res = resampler_trajectory.resample_input(
        solution['u_opt'],
        solution['dt'].flatten(),
        dt_res)

    x_res = resampler_trajectory.resampler(
        solution['x_opt'],
        solution['u_opt'],
        solution['dt'].flatten(),
        dt_res,
        dae=None,
        f_int=prb.getIntegrator())

    for s in prb.getState():
        sname = s.getName()
        off, dim = prb.getState().getVarIndex(sname)
        solution_res[f'{sname}_res'] = x_res[off:off + dim, :]

    for s in prb.getInput():
        sname = s.getName()
        off, dim = prb.getInput().getVarIndex(sname)
        solution_res[f'{sname}_res'] = u_res[off:off + dim, :]

    solution_res['dt_res'] = dt_res
    solution_res['x_opt_res'] = x_res
    solution_res['u_opt_res'] = u_res

    fmap = dict()
    for frame, wrench in contact_map.items():
        fmap[frame] = solution[f'{wrench.getName()}']

    fmap_res = dict()
    for frame, wrench in contact_map.items():
        fmap_res[frame] = solution_res[f'{wrench.getName()}_res']

    f_res_list = list()
    for f in prev_f_list:
        f_res_list.append(resampler_trajectory.resample_input(f, solution['dt'].flatten(), dt_res))

    tau = solution['inverse_dynamics']
    tau_res = np.zeros([tau.shape[0], u_res.shape[1]])

    id = kin_dyn.InverseDynamics(kindyn, fmap_res.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

    for i in range(tau.shape[1]):
        fmap_i = dict()
        for frame, wrench in fmap.items():
            fmap_i[frame] = wrench[:, i]
        tau_i = id.call(solution['q'][:, i], solution['q_dot'][:, i], solution['q_ddot'][:, i], fmap_i)
        tau[:, i] = tau_i.toarray().flatten()

    for i in range(tau_res.shape[1]):
        fmap_res_i = dict()
        for frame, wrench in fmap_res.items():
            fmap_res_i[frame] = wrench[:, i]
        tau_res_i = id.call(solution_res['q_res'][:, i], solution_res['q_dot_res'][:, i],
                            solution_res['q_ddot_res'][:, i],
                            fmap_res_i)
        tau_res[:, i] = tau_res_i.toarray().flatten()

    solution_res['tau'] = tau
    solution_res['tau_res'] = tau_res

    num_samples = tau_res.shape[1]

    nodes_vec = np.zeros([n_nodes])
    for i in range(1, n_nodes):
        nodes_vec[i] = nodes_vec[i - 1] + solution['dt'].flatten()[i - 1]

    nodes_vec_res = np.zeros([num_samples + 1])
    for i in range(1, num_samples + 1):
        nodes_vec_res[i] = nodes_vec_res[i - 1] + dt_res

    return solution_res, nodes_vec, nodes_vec_res, num_samples

class Refiner:
    def __init__(self, prb: problem.Problem, node_map, new_dt_vec, solver):

        # self.solver = solver
        # self.prev_solution = self.solver.getSolutionDict()
        self.prev_solution = solver
        self.prb = prb
        self.old_n_nodes = self.prb.getNNodes()

        self.new_dt_vec = new_dt_vec
        # # new time array.
        # #    Dimension: new n of nodes
        # #    Elements: cumulative dt
        # self.new_nodes_vec = new_nodes_vec
        #
        # prev_dt = solver['dt'].flatten()  # todo marcio
        # # prev_dt = self.solver.getDt()
        #
        # self.nodes_vec = self.get_node_time(prev_dt)
        # # get number of new node
        # self.new_n_nodes = self.new_nodes_vec.shape[0]
        # # get array of new dt
        # self.new_dt_vec = np.diff(self.new_nodes_vec)
        #
        # # get new indeices and old indices in the new array of times
        # old_values = np.in1d(self.new_nodes_vec, self.nodes_vec)
        # self.new_indices = np.arange(len(self.new_nodes_vec))[~old_values]
        # self.base_indices = np.arange(len(self.new_nodes_vec))[old_values]
        #
        # # map from base_indices to expanded indices: [0, 1, 2, 3, 4] ---> [0, 1, [2new, 3new, 4new], 2, 3, 4]
        # self.old_to_new = dict(zip(range(self.old_n_nodes + 1), self.base_indices))
        # self.new_to_old = {v: k for k, v in self.old_to_new.items()}
        # # group elements (collapses lists to ranges)
        # group_base_indices = self.group_elements(self.base_indices)
        # group_new_indices = self.group_elements(self.new_indices)
        #
        # # each first indices in ranges_base_indices
        # first_indices = [item[-1] for item in group_base_indices[:-1]]
        # indices_to_expand = [self.new_to_old[elem] for elem in first_indices]



        # ============= fake elem and expansion ===============
        # supp_num = 2
        # supp_tot = 0
        # self.elem_and_expansion = list()
        # for node in range(self.old_n_nodes - 1):
        #     next_node = supp_tot + 1
        #     supp_nodes = list(range(next_node, next_node + supp_num))
        #     self.elem_and_expansion.append((node, supp_nodes))
        #     supp_tot += supp_num + 1
        #
        # self.new_indices = list()
        # self.base_indices = list()
        #
        # supp_tot = 0
        # for node in range(self.old_n_nodes):
        #     self.base_indices.append(node + supp_tot)
        #     supp_tot = [supp_tot + len(elem[1]) for elem in self.elem_and_expansion if node in elem[0]]
        #
        # print(self.base_indices)
        # exit()
        # self.new_indices = np.array(self.new_indices)
        # self.base_indices = np.array(self.base_indices)
        #
        # self.old_to_new = dict(zip(range(self.old_n_nodes + 1), self.base_indices))
        # self.new_to_old = {v: k for k, v in self.old_to_new.items()}
        #
        # self.new_n_nodes = (self.old_n_nodes - 1) * (supp_num + 1) + 1
        # ============================================
        self.node_map = node_map

        # map old to new
        self.old_to_new = dict()
        self.base_indices = list()
        self.new_indices = list()
        self.elem_and_expansion = list()
        n_supplementary_nodes = 0
        for old_node in range(old_n_nodes):
            converted_node = old_node + n_supplementary_nodes
            self.old_to_new[old_node] = converted_node
            self.base_indices.append(converted_node)
            if old_node in self.node_map.keys():

                new_node = converted_node + 1
                list_new_nodes = list(range(new_node, new_node+self.node_map[old_node]))
                self.new_indices.extend(list_new_nodes)
                self.elem_and_expansion.append((old_node, list_new_nodes))
                # if in map, add node + supplementary nodes
                n_supplementary_nodes += self.node_map[old_node]

        self.new_to_old = {v: k for k, v in old_to_new.items()}


        # last new node + 1 is the number of total new nodes
        self.new_n_nodes = old_to_new[old_n_nodes - 1] + 1

        print('node_map: ', self.node_map)

    def get_node_time(self, dt):
        # get cumulative list of times
        nodes_vec = np.zeros([self.old_n_nodes])
        for i in range(1, self.old_n_nodes):
            nodes_vec[i] = nodes_vec[i - 1] + dt[i - 1]

        return nodes_vec

    # def expand_nodes(self, vec_to_expand):
    #
    #     # fill new_samples_nodes with corresponding new_nodes (convert old nodes to new nodes)
    #     new_nodes_vec = [self.old_to_new[elem] for elem in vec_to_expand]
    #
    #     # search for the nodes couples and expand them: return elements detected in self.elem_and_expansion
    #     elem_and_expansion_masked = self.find_nodes_to_inject(vec_to_expand)
    #
    #     # add nodes of elem_and_expansion to new_nodes_vec
    #     for elem in elem_and_expansion_masked:
    #         new_nodes_vec.extend(elem[1])
    #
    #     new_nodes_vec.sort()
    #     return new_nodes_vec

    # def find_nodes_to_inject(self, vec_to_expand):
    #     # the rationale behind this. Given 'augmented nodes':
    #     # if augmented nodes in between two nodes: inject
    #     # if augmented nodes after the last nodes: inject
    #
    #     # search for the nodes couples and expand if necessary
    #     recipe_vec = self.elem_and_expansion
    #
    #     recipe_vec_masked = list()
    #     # expand couples of nodes
    #     # for each element in vector, check if couples to expand are present
    #     for i in range(len(vec_to_expand)):
    #         for j in range(len(recipe_vec)):
    #             if vec_to_expand[i] == recipe_vec[j][0]:
    #                 # if last element:
    #                 if i == len(vec_to_expand) - 1:
    #                     print(f'last element detected: {vec_to_expand[i]}: injecting {recipe_vec[j][1]}')
    #                     recipe_vec_masked.append(recipe_vec[j])
    #
    #                 elif vec_to_expand[i + 1] == vec_to_expand[i] + 1:
    #                     print(
    #                         f'couple detected: {vec_to_expand[i], vec_to_expand[i] + 1}: injecting {recipe_vec[j][1]}')
    #                     recipe_vec_masked.append(recipe_vec[j])
    #
    #     return recipe_vec_masked

    def addVarBounds(self, item, var_lb, var_ub):

        # todo: THIS IS UGLY, should be redone
        for node in item.getNodes():
            # set bounds for old nodes
            if node in self.base_indices:
                old_node_index = self.new_to_old[node]
                item.setBounds(var_lb[:, old_node_index], var_ub[:, old_node_index], node)
            elif node in self.new_indices:
                node_elem = [elem[0] for elem in self.elem_and_expansion if node in elem[1]]
                item.setBounds(var_lb[:, node_elem], var_ub[:, node_elem], nodes=node)

    def addConstraintBounds(self, item, var_lb, var_ub):

        # todo: THIS IS UGLY, should be redone
        for node in item.getNodes():
            # set bounds for old nodes
            if node in self.base_indices:
                old_node_index = self.new_to_old[node]
                # size of old bounds is size of feasible nodes of constraint function
                old_feas_nodes = self.old_feas_cnsrt_nodes[item.getName()]
                pos_old_node = misc.convertNodestoPos(old_node_index, old_feas_nodes)
                item.setBounds(var_lb[:, pos_old_node], var_ub[:, pos_old_node], node)
            elif node in self.new_indices:
                node_elem = [elem[0] for elem in self.elem_and_expansion if node in elem[1]]
                # set bounds for injected nodes (using the bounds of old nodes)
                old_feas_nodes = self.old_feas_cnsrt_nodes[item.getName()]
                pos_old_node = misc.convertNodestoPos(node_elem, old_feas_nodes)
                item.setBounds(var_lb[:, pos_old_node], var_ub[:, pos_old_node], nodes=node)


    def addInitialBounds(self, item, var_ig):

        # todo: THIS IS UGLY, should be redone
        for node in item.getNodes():
            # set bounds for old nodes
            if node in self.base_indices:
                old_node_index = self.new_to_old[node]
                item.setInitialGuess(var_ig[:, old_node_index], node)
            elif node in self.new_indices:
                node_elem = [elem[0] for elem in self.elem_and_expansion if node in elem[1]]
                item.setInitialGuess(var_ig[:, node_elem], nodes=node)

    def resetProblem(self):
        # get every node, bounds for old functions and variables
        self.old_var_bounds = dict()
        self.old_var_nodes = dict()
        for name, var in self.prb.getVariables().items():
            self.old_var_bounds[name] = deepcopy(var.getBounds())
            self.old_var_nodes[name] = var.getNodes().copy()

        self.old_cnrst_bounds = dict()
        self.old_cnsrt_nodes = dict()
        self.old_feas_cnsrt_nodes = dict()
        for name, cnsrt in self.prb.getConstraints().items():
            # cnsrt_lb = -np.inf * np.ones([cnsrt.getDim(), self.old_n_nodes])
            # cnsrt_ub = np.inf * np.ones([cnsrt.getDim(), self.old_n_nodes])
            # cnsrt_lb[:, cnsrt.getNodes()] = cnsrt.getLowerBounds()
            # cnsrt_ub[:, cnsrt.getNodes()] = cnsrt.getUpperBounds()
            self.old_cnrst_bounds[name] = deepcopy(cnsrt.getBounds())
            self.old_cnsrt_nodes[name] = cnsrt.getNodes().copy()
            self.old_feas_cnsrt_nodes[name] = cnsrt._getFeasNodes().copy()

        self.old_cost_nodes = dict()
        for name, cost in self.prb.getCosts().items():
            self.old_cost_nodes[name] = cost.getNodes().copy()

        # modify the nodes in the problem, which modifies variables nodes
        self.prb.modifyNodes(self.node_map)
        exit()
        # set bounds of modified variables
        for var_name, var_item in self.prb.getVariables().items():
            var_lb, var_ub = self.old_var_bounds[var_name]
            self.addVarBounds(var_item, var_lb, var_ub)

        # set bounds of modified variables
        # this is not entirely correct, as the initialization should be some kind of interpolation
        for var_name, var_item in self.prb.getVariables().items():
            var_ig = self.prev_solution[var_name]
            self.addInitialBounds(var_item, var_ig)

        plot_bounds = False
        plot_ig = False
        if plot_bounds:
            from horizon.variables import InputVariable, RecedingInputVariable
            for name, var in self.prb.getVariables().items():

                old_var_lb, old_var_ub = self.old_var_bounds[name]
                var_lb, var_ub = var.getBounds()

                old_nodes_vec_vis = self.nodes_vec
                nodes_vec_vis = self.new_nodes_vec
                if isinstance(var, (RecedingInputVariable, InputVariable)):
                    old_nodes_vec_vis = self.nodes_vec[:-1]
                    nodes_vec_vis = self.new_nodes_vec[:-1]

                plt.figure()
                plt.title(f'lower bounds: {name}')
                for dim in range(var_lb.shape[0]):
                    plt.scatter(old_nodes_vec_vis, old_var_lb[dim, :], color='red')
                    plt.scatter(nodes_vec_vis, var_lb[dim, :], edgecolors='blue', facecolor='none')

                plt.figure()
                plt.title(f'upper bounds: {name}')
                for dim in range(var_lb.shape[0]):
                    plt.scatter(old_nodes_vec_vis, old_var_ub[dim, :], color='orange')
                    plt.scatter(nodes_vec_vis, var_ub[dim, :], edgecolors='green', facecolor='none')

            plt.show()

        if plot_ig:
            from horizon.variables import RecedingInputVariable, InputVariable
            for name, var in self.prb.getVariables().items():
                for dim in range(self.prev_solution[name].shape[0]):
                    nodes_vec_vis = self.nodes_vec
                    if isinstance(var, (RecedingInputVariable, InputVariable)):
                        nodes_vec_vis = self.nodes_vec[:-1]

                    plt.scatter(nodes_vec_vis, self.prev_solution[name][dim, :], color='red')

                var_to_print = var.getInitialGuess()

                for dim in range(var_to_print.shape[0]):
                    nodes_vec_vis = self.new_nodes_vec
                    if isinstance(var, (RecedingInputVariable, InputVariable)):
                        nodes_vec_vis = self.new_nodes_vec[:-1]
                    plt.scatter(nodes_vec_vis, var_to_print[dim, :], edgecolors='blue', facecolor='none')

                plt.show()
        # check for dt (if it is a symbolic variable, transform it to a parameter)
        # if combination of state/input variable but NOT a variable itself:
        #       i don't know
        # if value:
        #       ok no prob
        # if single variable (one for each node):
        #       remove variable and add parameter
        # if single parameter (one for each node):
        #       keep the parameter
        # if a mixed array of values/variable/parameters:
        #       for each variable, remove the variable and add the parameter
        # if a variable defined only on certain nodes
        #       ... dunno, I have to change the logic a bit

        # todo check if nodes of variable is correct
        # converting variable dt into parameter, if possible

        old_dt = self.prb.getDt()

        # todo right now, if dt is constant (or a list with even one constant) the refiner will throw. This is for many reasons:
        # - dt has to change (since some nodes will be introduced somewhere)
        # - constraints or costs that uses dt need to change. If dt was a constant, it wouldn't be possible.
        # if not isinstance(old_dt, (Variable, Parameter)) and not all(isinstance(elem, (Variable, Parameter)) for elem in old_dt):
        if isinstance(old_dt, List):
            error_description = 'A constant value for the dt is not supported. Since dt has to change, it is required a dt of type Variable or Parameter.'
            raise NotImplementedError(error_description)
            # if old_dt is a list, get all the variables of the list.
            # substitute them with parameters.
            # variable_to_substitute = list()
            # for elem in old_dt:
            #     if isinstance(elem, (Variable, SingleVariable)) and elem not in variable_to_substitute:
            #         variable_to_substitute.append(elem)

            # for var in variable_to_substitute:
            #     self.prb.toParameter(var.getName())
        else:
            if isinstance(old_dt, (Variable, SingleVariable)):
                self.prb.toParameter(old_dt.getName())


        # self.expandDt()

    def expandDt(self):

        # if dt is a list, expand the list to match the new number of nodes
        dt = self.prb.getDt()
        if isinstance(dt, List):

            old_n = range(self.prb.getNNodes() - 1)
            elem_and_expansion_masked = self.find_nodes_to_inject(old_n)

            for expansion in reversed(elem_and_expansion_masked):
                dt[expansion[0]:expansion[0]] = len(expansion[1]) * [dt[expansion[0]]]

    def resetFunctions(self):
        # set constraints


        for cnsrt_name, cnsrt_item in self.prb.getConstraints().items():
            old_nodes = cnsrt_item.getNodes().copy()
            print(cnsrt_name)
            print('old nodes:', old_nodes)
            cnsrt_lb, cnsrt_ub = self.old_cnrst_bounds[cnsrt_name]
            self.addConstraintBounds(cnsrt_item, cnsrt_lb, cnsrt_ub)
            print('new nodes:', cnsrt_item.getNodes())
            if not (old_nodes == cnsrt_item.getNodes()).all():
                raise ValueError('updating the constraint bounds destroyed something')

        # for name, cnsrt in self.prb.getConstraints().items():
        #     print(f'========================== constraint {name} =========================================')
        #     old_n = self.old_cnsrt_nodes[name]
        #     old_lb, old_ub = self.old_cnrst_bounds[name]
        #
        #     # if constraint depends on dt, what to do?
        #     # if it is a variable, it is ok. Can be changed and recognized easily.
        #     # What if it is a constant?
        #     # I have to change that constant value to the new value (old dt to new dt).
        #
        #     # a possible thing is that i "mark" it, so that I can find it around.
        #     # Otherwise it would be impossible to understand which constraint depends on dt?
        #
        #     # manage bounds
        #     # old nodes: set their old bounds
        #     for node in cnsrt.getNodes():
        #         if node in self.base_indices:
        #             old_node_index = old_n.index(self.new_to_old[node])
        #             print(f'setting bounds at old nodes: {node}')
        #             cnsrt.setBounds(old_lb[:, old_node_index], old_ub[:, old_node_index], node)
        #
        #     elem_and_expansion_masked = self.find_nodes_to_inject(old_n)
        #
        #     # injected nodes: set the bounds of the previous node
        #     for elem in elem_and_expansion_masked:
        #         # setting bounds using old bound index!
        #         # careful: retrieved old bounds corresponds to the nodes where the constraint is defined.
        #         # If the constraint is defined over [20, 21, 22, 23], and I want to set the bounds in node 23 (= elem[0]), I need to use the index 4 of old_bound_index
        #         print(f'setting bounds at injected nodes: {elem[1]}')
        #         old_bound_index = old_n.index(elem[0])
        #         cnsrt.setBounds(old_lb[:, old_bound_index], old_ub[:, old_bound_index], nodes=elem[1])

        plot_bounds = False
        if plot_bounds:
            for name, cnsrt in self.prb.getConstraints().items():
                # if name != 'rh_foot_fc_after_lift':
                #     continue

                old_lb, old_ub = self.old_cnrst_bounds[name]
                lb, ub = cnsrt.getBounds()

                old_n = self.old_cnsrt_nodes[name]
                cnsrt_nodes = cnsrt.getNodes()

                old_feas_nodes = self.old_feas_cnsrt_nodes[name]
                pos_old_node = misc.convertNodestoPos(old_n, old_feas_nodes)

                pos_new_node = misc.convertNodestoPos(cnsrt_nodes, cnsrt._getFeasNodes())

                old_t = self.nodes_vec[self.old_cnsrt_nodes[name]]
                t = self.new_nodes_vec[cnsrt_nodes]

                plt.figure()
                plt.title(f'lower bounds: {name}')
                for dim in range(lb.shape[0]):
                    plt.scatter(old_t, old_lb[dim, pos_old_node], color='red')
                    plt.scatter(t, lb[dim, pos_new_node], edgecolors='blue', facecolor='none')

                plt.figure()
                plt.title(f'upper bounds: {name}')
                for dim in range(lb.shape[0]):
                    plt.scatter(old_t, old_ub[dim, pos_old_node], color='orange')
                    plt.scatter(t, ub[dim, pos_new_node], edgecolors='green', facecolor='none')

            plt.show()

    # def resetVarBounds(self):
    #
    #     plot_bounds = False
    #
    #     # manage bounds
    #     for name_var, var in self.prb.getVariables().items():
    #         var_nodes_old = self.old_var_nodes[name_var]
    #         var_lb, var_ub = self.old_var_bounds[name_var]
    #
    #         for node in var.getNodes():
    #             # set bounds for old nodes
    #             if node in self.base_indices:
    #                 old_node_index = var_nodes_old.index(self.new_to_old[node])
    #                 var.setBounds(var_lb[:, old_node_index], var_ub[:, old_node_index], node)
    #
    #         elem_and_expansion_masked = self.find_nodes_to_inject(var_nodes_old)
    #
    #         # set bounds for injected nodes (using the bounds of old nodes)
    #         for elem in elem_and_expansion_masked:
    #             print(elem[1])
    #             var.setBounds(var_lb[:, elem[0]], var_ub[:, elem[0]], nodes=elem[1])
    #
    #     if plot_bounds:
    #         for name, var in self.prb.getVariables().items():
    #
    #             old_var_lb, old_var_ub = self.old_var_bounds[name]
    #             var_lb, var_ub = var.getBounds()
    #
    #             from horizon.variables import InputVariable
    #             old_nodes_vec_vis = self.nodes_vec
    #             nodes_vec_vis = self.new_nodes_vec
    #             if isinstance(var, InputVariable):
    #                 old_nodes_vec_vis = self.nodes_vec[:-1]
    #                 nodes_vec_vis = self.new_nodes_vec[:-1]
    #
    #             plt.figure()
    #             plt.title(f'lower bounds: {name}')
    #             for dim in range(var_lb.shape[0]):
    #                 plt.scatter(old_nodes_vec_vis, old_var_lb[dim, :], color='red')
    #                 plt.scatter(nodes_vec_vis, var_lb[dim, :], edgecolors='blue', facecolor='none')
    #
    #                 plt.figure()
    #                 plt.title(f'upper bounds: {name}')
    #             for dim in range(var_lb.shape[0]):
    #                 plt.scatter(old_nodes_vec_vis, old_var_ub[dim, :], color='orange')
    #                 plt.scatter(nodes_vec_vis, var_ub[dim, :], edgecolors='green', facecolor='none')
    #
    #         plt.show()

    # def resetInitialGuess(self):
    #
    #     plot_ig = False
    #
        # variables
        # for name_var, var in self.prb.getVariables().items():
        #
        #     var_nodes_old = self.old_var_nodes[name_var]
        #     print(f'============================ var {name_var} =======================================')
        #     for node in var.getNodes():
        #         if node in self.base_indices:
        #             old_node_index = var_nodes_old.index(self.new_to_old[node])
        #             var.setInitialGuess(self.prev_solution[name_var][:, old_node_index], node)
        #
        #         if node in self.new_indices:
        #             for elem in self.elem_and_expansion:
        #                 if node in elem[1]:
        #                     prev_node = elem[0]
        #
        #             print(f'node {node} initialized with old node {prev_node}.')
        #             var.setInitialGuess(np.zeros([var.shape[0], var.shape[1]]), node)
        #             var.setInitialGuess(self.prev_solution[name_var][:, prev_node], node)
        #
        # if plot_ig:
        #     for name, var in self.prb.getVariables().items():
        #         for dim in range(self.prev_solution[name].shape[0]):
        #             from horizon.variables import InputVariable
        #             nodes_vec_vis = self.nodes_vec
        #             if isinstance(var, InputVariable):
        #                 nodes_vec_vis = self.nodes_vec[:-1]
        #
        #             plt.scatter(nodes_vec_vis, self.prev_solution[name][dim, :], color='red')
        #
        #         var_to_print = var.getInitialGuess()
        #
        #         for dim in range(var_to_print.shape[0]):
        #             nodes_vec_vis = self.new_nodes_vec
        #             if isinstance(var, InputVariable):
        #                 nodes_vec_vis = self.new_nodes_vec[:-1]
        #             plt.scatter(nodes_vec_vis, var_to_print[dim, :], edgecolors='blue', facecolor='none')
        #
        #         plt.show()

    def solveProblem(self):

        # =============
        # SOLVE PROBLEM
        # =============

        for name, var in self.prb.getVariables().items():
            print(f'variable {name}: {var} active on nodes: {var.getNodes()}')
            # print(f'lb: {var.getLowerBounds()}')
            # print(f'ub: {var.getUpperBounds()}')

        for name, par in self.prb.getParameters().items():
            print(f'parameter {name}: {par} active on nodes: {par.getNodes()}')

        for name, cnsrt in self.prb.getConstraints().items():
            print(f'constraint {name}: {cnsrt} active on nodes: {cnsrt.getNodes()}')
            # print(f'lb: {cnsrt.getLowerBounds()}')
            # print(f'ub: {cnsrt.getUpperBounds()}')

        for name, cost in self.prb.getCosts().items():
            print(f'cost {name}: {cost} active on nodes: {cost.getNodes()}')

        opts = {'ipopt.tol': 0.001,
                'ipopt.constr_viol_tol': 0.001,
                'ipopt.max_iter': 2000,
                'ipopt.linear_solver': 'ma57'}

        # parametric time
        param_dt = self.prb.getDt()
        if isinstance(param_dt, List):
            raise NotImplementedError('dt of type List is yet to implement')
            # for i in range(len(param_dt)):
            #     if isinstance(param_dt[i], Parameter):
            #         param_dt[i].assign(self.new_dt_vec[i], nodes=i)
            #     if isinstance(param_dt[i], SingleParameter):
            #         param_dt[i].assign(self.new_dt_vec[i])
            #
            # self.prb.setDt(param_dt)
        elif isinstance(param_dt, Parameter):
            for i in range(len(self.new_dt_vec)):
                param_dt.assign(self.new_dt_vec[i], nodes=i)

        self.sol = Solver.make_solver('ipopt', self.prb, opts)
        self.sol.solve()

    def getSolver(self):
        return self.sol

    def getSolution(self):

        # add check for the solving of the problem
        sol_var = self.sol.getSolutionDict()
        sol_cnsrt = self.sol.getConstraintSolutionDict()
        sol_dt = self.sol.getDt()

        # hplt = plotter.PlotterHorizon(self.prb, sol_var)
        # hplt.plotVariables(show_bounds=True, legend=False)
        # hplt.plotFunctions(show_bounds=True)
        # hplt.plotFunction('inverse_dynamics', show_bounds=True, legend=True, dim=range(6))
        # plt.show()

        return sol_var, sol_cnsrt, sol_dt

    def addProximalCosts(self):

        # todo work on this:
        #    what to do with costs?
        # one strategy is adding a "regularization" term
        proximal_cost_state = 1e5

        self.prb.removeCostFunction('min_q_dot')
        # self.prb.removeCostFunction('min_qdot')
        self.prb.removeCostFunction('min_qddot')

        # minimize states
        for state_var in self.prb.getState().getVars(abstr=True):
            for node in range(self.new_n_nodes):
                if node in self.base_indices:
                    old_sol = self.prev_solution[state_var.getName()]
                    old_n = self.new_to_old[node]
                    print(f'Creating proximal cost for variable {state_var.getName()} at node {node}')
                    self.prb.createCost(f"{state_var.getName()}_proximal_{node}",
                                        proximal_cost_state * cs.sumsqr(state_var - old_sol[:, old_n]), nodes=node)
                if node in self.new_indices:
                    print(f'Proximal cost of {state_var.getName()} not created for node {node}: required a value')
                    # prb.createCostFunction(f"q_close_to_res_node_{node}", 1e5 * cs.sumsqr(q - q_res[:, zip_indices_new[node]]), nodes=node)

        proximal_cost_input = 1
        # minimize inputs
        for input_var in self.prb.getInput().getVars(abstr=True):
            for node in range(self.new_n_nodes - 1):
                if not isinstance(input_var, (Parameter, SingleParameter)):
                    if node in self.base_indices:
                        old_sol = self.prev_solution[input_var.getName()]
                        old_n = self.new_to_old[node]
                        self.prb.createCost(f"minimize_{input_var.getName()}_node_{node}",
                                            proximal_cost_input * cs.sumsqr(input_var - old_sol[:, old_n]), nodes=node)
                    if node in self.new_indices:
                        print(
                            f'Proximal cost of {input_var.getName()} created for node {node}: without any value, it is just minimized w.r.t zero')
                        self.prb.createCost(f"minimize_{input_var.getName()}_node_{node}",
                                            proximal_cost_input * cs.sumsqr(input_var), nodes=node)

    def getAugmentedProblem(self):

        # todo add checks for the building of the problem
        return self.prb


if __name__ == '__main__':

    # =========================================
    transcription_method = 'multiple_shooting'  # direct_collocation # multiple_shooting
    transcription_opts = dict(integrator='RK4')

    # rospack = rospkg.RosPack()
    # rospack.get_path('spot_urdf')
    urdffile = '../examples/urdf/spot.urdf'
    urdf = open(urdffile, 'r').read()
    kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

    # joint names
    joint_names = kindyn.joint_names()
    if 'universe' in joint_names: joint_names.remove('universe')
    if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')

    n_nodes = 50

    node_start_step = 20
    node_end_step = 40
    node_peak = 30
    jump_height = 0.2

    n_c = 4
    n_q = kindyn.nq()
    n_v = kindyn.nv()
    n_f = 3

    # SET PROBLEM STATE AND INPUT VARIABLES
    prb = problem.Problem(n_nodes, receding=True)
    q = prb.createStateVariable('q', n_q)
    q_dot = prb.createStateVariable('q_dot', n_v)
    q_ddot = prb.createInputVariable('q_ddot', n_v)

    contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
    f_list = [prb.createInputVariable(f'force_{i}', n_f) for i in contacts_name]

    # SET CONTACTS MAP
    contact_map = dict(zip(contacts_name, f_list))

    load_initial_guess = True
    # import initial guess if present
    if load_initial_guess:
        ms = mat_storer.matStorer('../playground/mesh_refiner/spot_jump.mat')
        prev_solution = ms.load()
        q_ig = prev_solution['q']
        q_dot_ig = prev_solution['q_dot']
        q_ddot_ig = prev_solution['q_ddot']
        f_ig_list = [prev_solution[f.getName()] for f in f_list]
        for i in range(n_c):
            [f.setInitialGuess(f_ig) for f, f_ig in zip(f_list, f_ig_list)]

        dt_ig = prev_solution['dt']

    # SET DYNAMICS
    dt = prb.createInputVariable("dt", 1)  # variable dt as input
    # dt = 0.01
    # Computing dynamics
    x_dot = utils.double_integrator_with_floating_base(q, q_dot, q_ddot)
    prb.setDynamics(x_dot)
    prb.setDt(dt)

    # SET BOUNDS
    # q bounds
    q_min = [-10., -10., -10., -1., -1., -1., -1.]  # floating base
    q_min.extend(kindyn.q_min()[7:])
    q_min = np.array(q_min)

    q_max = [10., 10., 10., 1., 1., 1., 1.]  # floating base
    q_max.extend(kindyn.q_max()[7:])
    q_max = np.array(q_max)

    q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                       0.0, 0.9, -1.5238505,
                       0.0, 0.9, -1.5202315,
                       0.0, 0.9, -1.5300265,
                       0.0, 0.9, -1.5253125])

    # q_dot bounds
    q_dot_lim = 100. * np.ones(n_v)
    # q_ddot bounds
    q_ddot_lim = 100. * np.ones(n_v)
    # f bounds
    f_lim = 10000. * np.ones(n_f)

    dt_min = 0.01  # [s]
    dt_max = 0.1  # [s]

    # set bounds and of q
    q.setBounds(q_min, q_max)
    q.setBounds(q_init, q_init, 0)
    # set bounds of q_dot
    q_dot_init = np.zeros(n_v)
    q_dot.setBounds(-q_dot_lim, q_dot_lim)
    q_dot.setBounds(q_dot_init, q_dot_init, 0)
    # set bounds of q_ddot
    q_ddot.setBounds(-q_ddot_lim, q_ddot_lim)
    # set bounds of f
    # for f in f_list:
    #     f.setBounds(-f_lim, f_lim)

    f_min = [-10000., -10000., -10.]
    f_max = [10000., 10000., 10000.]
    for f in f_list:
        f.setBounds(f_min, f_max)
    # set bounds of dt
    if isinstance(dt, cs.SX):
        dt.setBounds(dt_min, dt_max)

    # SET INITIAL GUESS
    if load_initial_guess:
        for node in range(q_ig.shape[1]):
            q.setInitialGuess(q_ig[:, node], node)

        for node in range(q_dot_ig.shape[1]):
            q_dot.setInitialGuess(q_dot_ig[:, node], node)

        for node in range(q_ddot_ig.shape[1]):
            q_ddot.setInitialGuess(q_ddot_ig[:, node], node)

        for f, f_ig in zip(f_list, f_ig_list):
            for node in range(f_ig.shape[1]):
                f.setInitialGuess(f_ig[:, node], node)

        if isinstance(dt, cs.SX):
            for node in range(dt_ig.shape[1]):
                dt.setInitialGuess(dt_ig[:, node], node)

    else:
        q.setInitialGuess(q_init)
        if isinstance(dt, cs.SX):
            dt.setInitialGuess(dt_min)

    # SET TRANSCRIPTION METHOD
    th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

    # SET INVERSE DYNAMICS CONSTRAINTS
    tau_lim = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
                        1000., 1000., 1000.,  # Contact 1
                        1000., 1000., 1000.,  # Contact 2
                        1000., 1000., 1000.,  # Contact 3
                        1000., 1000., 1000.])  # Contact 4

    tau = kin_dyn.InverseDynamics(kindyn, contact_map.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(q,
                                                                                                                 q_dot,
                                                                                                                 q_ddot,
                                                                                                                 contact_map)
    prb.createIntermediateConstraint("inverse_dynamics", tau, bounds=dict(lb=-tau_lim, ub=tau_lim))

    # SET FINAL VELOCITY CONSTRAINT
    prb.createFinalConstraint('final_velocity', q_dot)

    # SET CONTACT POSITION CONSTRAINTS
    active_leg = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']

    mu = 1
    R = np.identity(3, dtype=float)  # environment rotation wrt inertial frame

    fb_during_jump = np.array([q_init[0], q_init[1], q_init[2] + jump_height, 0.0, 0.0, 0.0, 1.0])
    q_final = q_init

    for frame, f in contact_map.items():
        # 2. velocity of each end effector must be zero
        FK = kindyn.fk(frame)
        p = FK(q=q)['ee_pos']
        p_start = FK(q=q_init)['ee_pos']
        p_goal = p_start + [0., 0., jump_height]
        DFK = kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        v = DFK(q=q, qdot=q_dot)['ee_vel_linear']
        DDFK = kindyn.frameAcceleration(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
        a = DDFK(q=q, qdot=q_dot)['ee_acc_linear']

        prb.createConstraint(f"{frame}_vel_before_lift", v, nodes=range(0, node_start_step))
        c = prb.createConstraint(f"{frame}_vel_after_lift", v, nodes=range(node_end_step, n_nodes + 1))

        # friction cones must be satisfied
        fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(f, mu, R)
        prb.createIntermediateConstraint(f"{frame}_fc_before_lift", fc, nodes=range(0, node_start_step),
                                         bounds=dict(lb=fc_lb, ub=fc_ub))
        cfc = prb.createIntermediateConstraint(f"{frame}_fc_after_lift", fc, nodes=range(node_end_step, n_nodes),
                                               bounds=dict(lb=fc_lb, ub=fc_ub))

        prb.createConstraint(f"{frame}_no_force_during_lift", f, nodes=range(node_start_step, node_end_step))

        prb.createConstraint(f"start_{frame}_leg", p - p_start, nodes=node_start_step)
        prb.createConstraint(f"lift_{frame}_leg", p - p_goal, nodes=node_peak)
        prb.createConstraint(f"land_{frame}_leg", p - p_start, nodes=node_end_step)

    # SET COST FUNCTIONS
    # prb.createCostFunction(f"jump_fb", 10000 * cs.sumsqr(q[2] - fb_during_jump[2]), nodes=node_start_step)
    prb.createCost("min_q_dot", 1. * cs.sumsqr(q_dot))
    prb.createFinalCost(f"final_nominal_pos", 1000 * cs.sumsqr(q - q_init))
    for f in f_list:
        prb.createIntermediateCost(f"min_{f.getName()}", 0.01 * cs.sumsqr(f))

    # prb.createIntermediateCost('min_dt', 100 * cs.sumsqr(dt))

    # =============
    # SOLVE PROBLEM
    # =============
    #
    opts = {'ipopt.tol': 0.001,
            'ipopt.constr_viol_tol': 0.001,
            'ipopt.max_iter': 2000,
            'ipopt.linear_solver': 'ma57'}

    solver = solver.Solver.make_solver('ipopt', prb, opts)

    ms = mat_storer.matStorer('refiner_data_100.mat')
    # ========================================== direct solve ==========================================================
    solver.solve()
    prev_solution = solver.getSolutionDict()
    prev_solution.update(solver.getConstraintSolutionDict())
    n_nodes = prb.getNNodes() - 1
    prev_dt = solver.getDt().flatten()

    # exit()
    # ========================================= from stored data =======================================================
    # prev_solution = ms.load()

    # ==================================================================================================================
    prev_q = prev_solution['q']
    prev_q_dot = prev_solution['q_dot']
    prev_q_ddot = prev_solution['q_ddot']
    prev_f_list = [prev_solution[f.getName()] for f in f_list]

    # RESAMPLE
    dt_res = 0.001
    prev_solution_res, nodes_vec, nodes_vec_res, num_samples = resample(prev_solution, prb, dt_res, contacts_name)
    #
    info_dict = dict(n_nodes=prb.getNNodes(), times=nodes_vec, times_res=nodes_vec_res, dt=prev_dt, dt_res=dt_res)
    ms.store({**prev_solution, **prev_solution_res, **info_dict})

    exit()
    contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
    prev_contact_map = dict(zip(contacts_name, prev_f_list))

    joint_names = kindyn.joint_names()
    if 'universe' in joint_names:
        joint_names.remove('universe')
    if 'floating_base_joint' in joint_names:
        joint_names.remove('floating_base_joint')

    tau_res = prev_solution['tau_res']

    nodes_vec = prev_solution['times'][0]
    nodes_vec_res = prev_solution['times_res'][0]

    tau_sol_base = tau_res[:6, :]

    threshold = 5
    ## get index of values greater than a given threshold for each dimension of the vector, and remove all the duplicate values (given by the fact that there are more dimensions)
    indices_exceed = np.unique(np.argwhere(np.abs(tau_sol_base) > threshold)[:, 1])
    # these indices correspond to some nodes
    values_exceed = nodes_vec_res[indices_exceed]

    ## search for duplicates and remove them, both in indices_exceed and values_exceed
    indices_duplicates = np.where(np.in1d(values_exceed, nodes_vec))
    value_duplicates = values_exceed[indices_duplicates]

    values_exceed = np.delete(values_exceed, np.where(np.in1d(values_exceed, value_duplicates)))
    indices_exceed = np.delete(indices_exceed, indices_duplicates)

    ## base vector nodes augmented with new nodes + sort
    nodes_vec_augmented = np.concatenate((nodes_vec, values_exceed))
    nodes_vec_augmented.sort(kind='mergesort')

    plot_tau_base = False
    if plot_tau_base:
        plt.figure()
        for dim in range(6):
            plt.plot(nodes_vec_res[:-1], np.array(tau_res[dim, :]))
        for dim in range(6):
            plt.scatter(nodes_vec[:-1], np.array(tau[dim, :]))
        plt.title('tau on base')

        plt.hlines([threshold], nodes_vec[0], nodes_vec[-1], linestyles='dashed', colors='k', linewidth=0.4)
        plt.hlines([-threshold], nodes_vec[0], nodes_vec[-1], linestyles='dashed', colors='k', linewidth=0.4)

    plt.show()

    # ===================================================================================================================


    # get number of new node
    old_n_nodes = prb.getNNodes()
    new_n_nodes = nodes_vec_augmented.shape[0]
    new_dt_vec = np.diff(nodes_vec_augmented)

    # get new indices and old indices in the new array of times
    old_values = np.in1d(nodes_vec_augmented, nodes_vec)
    new_indices = np.arange(len(nodes_vec_augmented))[~old_values]
    base_indices = np.arange(len(nodes_vec_augmented))[old_values]

    # map from base_indices to expanded indices: [0, 1, 2, 3, 4] ---> [0, 1, [2new, 3new, 4new], 2, 3, 4]
    old_to_new = dict(zip(range(old_n_nodes + 1), base_indices))
    new_to_old = {v: k for k, v in old_to_new.items()}
    # group elements (collapses lists to ranges)
    group_base_indices = group_elements(base_indices)
    group_new_indices = group_elements(new_indices)

    # each first indices in ranges_base_indices
    first_indices = [item[-1] for item in group_base_indices[:-1]]
    indices_to_expand = [new_to_old[elem] for elem in first_indices]

    elem_and_expansion = list(zip(indices_to_expand, group_new_indices))
    node_map = {node: len(expanded_nodes) for node, expanded_nodes in elem_and_expansion}

    # todo: better to reason with nodes, as the old nodes are required to stay there
    ref = Refiner(prb, node_map, new_dt_vec, prev_solution)


    plot_nodes = False
    if plot_nodes:
        plt.figure()
        # nodes old
        plt.scatter(nodes_vec_augmented, np.zeros([nodes_vec_augmented.shape[0]]), edgecolors='red', facecolor='none')
        plt.scatter(nodes_vec, np.zeros([nodes_vec.shape[0]]), edgecolors='blue', facecolor='none')
        plt.show()

    # ======================================================================================================================
    ref.resetProblem()
    ref.resetFunctions()
    ref.addProximalCosts()
    ref.solveProblem()
    solver = ref.getSolver()
    new_prb = ref.getAugmentedProblem()

    solution = solver.getSolutionDict()
    solution['dt'] = solver.getDt()
    solution.update(solver.getConstraintSolutionDict())
    dt_res = 0.001
    solution_res, nodes_vec, nodes_vec_res, num_samples = resample(solution, new_prb, dt_res)

    ms = mat_storer.matStorer(f'refiner_data_after.mat')
    info_dict = dict(n_nodes=new_prb.getNNodes(), times=nodes_vec, times_res=nodes_vec_res, dt=solution['dt'], dt_res=dt_res)
    ms.store({**solution, **solution_res, **info_dict})

# refine_solution = True
# if refine_solution:
#     from horizon.utils.refiner import Refiner
#
#     prev_solution = solution
#     num_samples = q_res.shape[1]
#     cumulative_dt = np.zeros([n_nodes + 1])
#     for i in range(1, n_nodes + 1):
#         cumulative_dt[i] = cumulative_dt[i - 1] + dt_sol[i - 1]
#
#     cumulative_dt_res = np.zeros([num_samples + 1])
#     for i in range(1, num_samples + 1):
#         cumulative_dt_res[i] = cumulative_dt_res[i - 1] + dt_res
#
#     tau_sol_base = tau_res[:6, :]
#
#     threshold = 10
#     ## get index of values greater than a given threshold for each dimension of the vector, and remove all the duplicate values (given by the fact that there are more dimensions)
#     indices_exceed = np.unique(np.argwhere(np.abs(tau_sol_base) > threshold)[:, 1])
#     # these indices corresponds to some nodes ..
#     values_exceed = cumulative_dt_res[indices_exceed]
#
#     ## search for duplicates and remove them, both in indices_exceed and values_exceed
#     indices_duplicates = np.where(np.in1d(values_exceed, cumulative_dt))
#     value_duplicates = values_exceed[indices_duplicates]
#
#     values_exceed = np.delete(values_exceed, np.where(np.in1d(values_exceed, value_duplicates)))
#     indices_exceed = np.delete(indices_exceed, indices_duplicates)
#
#     ## base vector nodes augmented with new nodes + sort
#     cumulative_dt_augmented = np.concatenate((cumulative_dt, values_exceed))
#     cumulative_dt_augmented.sort(kind='mergesort')
#
#     ref = Refiner(prb, cumulative_dt_augmented, solv)
#
#     plot_nodes = True
#     if plot_nodes:
#         plt.figure()
#         # nodes old
#         plt.scatter(cumulative_dt_augmented, np.zeros([cumulative_dt_augmented.shape[0]]), edgecolors='red', facecolor='none')
#         plt.scatter(cumulative_dt, np.zeros([cumulative_dt.shape[0]]), edgecolors='blue', facecolor='none')
#         plt.show()
#
#     # ======================================================================================================================
#     ref.resetProblem()
#     ref.resetFunctions()
#     ref.resetVarBounds()
#     ref.resetInitialGuess()
#     ref.addProximalCosts()
#     ref.solveProblem()
#     sol_var, sol_cnsrt, sol_dt = ref.getSolution()
#
#     new_prb = ref.getAugmentedProblem()
#
#     from utils import mat_storer
#
#     ms = mat_storer.matStorer(f'trial_old.mat')
#     sol_cnsrt_dict = dict()
#     for name, item in prb.getConstraints().items():
#         lb, ub = item.getBounds()
#         lb_mat = np.reshape(lb, (item.getDim(), len(item.getNodes())), order='F')
#         ub_mat = np.reshape(ub, (item.getDim(), len(item.getNodes())), order='F')
#         sol_cnsrt_dict[name] = dict(val=solution_constraints[name], lb=lb_mat, ub=ub_mat, nodes=item.getNodes())
#
#     info_dict = dict(n_nodes=prb.getNNodes(), times=cumulative_dt, dt=dt_sol)
#     ms.store({**solv.getSolutionDict(), **sol_cnsrt_dict, **info_dict})
#
#
#     ms = mat_storer.matStorer(f'trial.mat')
#     sol_cnsrt_dict = dict()
#     for name, item in new_prb.getConstraints().items():
#         lb, ub = item.getBounds()
#         lb_mat = np.reshape(lb, (item.getDim(), len(item.getNodes())), order='F')
#         ub_mat = np.reshape(ub, (item.getDim(), len(item.getNodes())), order='F')
#         sol_cnsrt_dict[name] = dict(val=sol_cnsrt[name], lb=lb_mat, ub=ub_mat, nodes=item.getNodes())
#
#     info_dict = dict(n_nodes=new_prb.getNNodes(), times=cumulative_dt_augmented, dt=sol_dt)
#     ms.store({**sol_var, **sol_cnsrt_dict, **info_dict})


# def findExceedingValues(vec, threshold):
#     indices_exceed = np.unique(np.argwhere(np.abs(vec) > threshold)[:, 1])
#     # these indices corresponds to some nodes ..
#     values_exceed = cumulative_dt_res[indices_exceed]
#
#     ## search for duplicates and remove them, both in indices_exceed and values_exceed
#     indices_duplicates = np.where(np.in1d(values_exceed, cumulative_dt))
#     value_duplicates = values_exceed[indices_duplicates]
#
#     values_exceed = np.delete(values_exceed, np.where(np.in1d(values_exceed, value_duplicates)))
#     indices_exceed = np.delete(indices_exceed, indices_duplicates)
#
#     ## base vector nodes augmented with new nodes + sort
#     cumulative_dt_augmented = np.concatenate((cumulative_dt, values_exceed))
#     cumulative_dt_augmented.sort(kind='mergesort')
#
#     return cumulative_dt_augmented
