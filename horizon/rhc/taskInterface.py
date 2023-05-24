import casadi as cs
from urllib3 import Retry

from horizon.problem import Problem
from horizon.utils import utils, kin_dyn, mat_storer, resampler_trajectory

from casadi_kin_dyn import pycasadi_kin_dyn
from horizon.rhc.tasks.cartesianTask import CartesianTask
from horizon.rhc.tasks.contactTask import ContactTask
from horizon.rhc.tasks.interactionTask import InteractionTask, SurfaceContact, VertexContact
from horizon.rhc.tasks.rollingTask import RollingTask
from horizon.rhc.model_description import FullModelInverseDynamics, SingleRigidBodyDynamicsModel
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.rhc.tasks.posturalTask import PosturalTask
from horizon.rhc.tasks.limitsTask import JointLimitsTask
from horizon.rhc.tasks.regularizationTask import RegularizationTask
from horizon.transcriptions import integrators
from typing import List, Dict, Union, Tuple
import numpy as np
from horizon.rhc import task_factory, plugin_handler, solver_interface
from horizon.rhc.yaml_handler import YamlParser
from horizon.solvers.solver import Solver
from horizon.ros.replay_trajectory import replay_trajectory
import logging
import time


class TaskInterface:
    def __init__(self,
                 prb,
                 model):

        # get the model
        self.prb = prb
        self.model = model
        
        
        # here I register the the default tasks
        # todo: should I do it here?
        # uniform the names of these tasks
        task_factory.register('Cartesian', CartesianTask)
        task_factory.register('Contact', ContactTask)
        task_factory.register('Wrench', SurfaceContact)
        task_factory.register('VertexForce', VertexContact)
        task_factory.register('Postural', PosturalTask)
        task_factory.register('JointLimits', JointLimitsTask)
        task_factory.register('Regularization', RegularizationTask)
        task_factory.register('Rolling', RollingTask)


        # self.a0 = np.zeros(self.nv)

        # task list
        self.task_list = []
        

    def finalize(self):
        """
        to be called after all variables have been created
        """
        self.model.setDynamics()
        self._create_solver()

    
    def bootstrap(self):
        t = time.time()
        self.solver_bs.solve()
        elapsed = time.time() - t
        print(f'bootstrap solved in {elapsed} s')
        try:
            self.solver_rti.print_timings()
        except:
            pass
        self.solution = self.solver_bs.getSolutionDict()

    
    def rti(self):
                
        t = time.time()
        check = self.solver_rti.solve()
        elapsed = time.time() - t
        print(f'rti solved in {elapsed} s')

        self.solution = self.solver_rti.getSolutionDict()

        return check

    def resample(self, dt_res, dae=None, nodes=None, resample_tau=True):

        if nodes is None:
            nodes = list(range(self.prb.getNNodes() + 1))

        if dae is None:
            integrator = self.prb.getIntegrator()
        else:
            integrator = integrators.EULER(dae)

        u_res = resampler_trajectory.resample_input(
            self.solution['u_opt'][:, [index for index in nodes if index < self.prb.getNNodes() - 1]],
            self.prb.getDt(),
            dt_res)

        x_res = resampler_trajectory.resampler(
            self.solution['x_opt'][:, [index for index in nodes if index < self.prb.getNNodes()]],
            self.solution['u_opt'][:, [index for index in nodes if index < self.prb.getNNodes() - 1]],
            self.prb.getDt(),
            dt_res,
            dae=None,
            f_int=integrator)

        self.solution['dt_res'] = dt_res
        self.solution['x_opt_res'] = x_res
        self.solution['u_opt_res'] = u_res

        for s in self.prb.getState():
            sname = s.getName()
            off, dim = self.prb.getState().getVarIndex(sname)
            self.solution[f'{sname}_res'] = x_res[off:off + dim, :]

        for s in self.prb.getInput():
            sname = s.getName()
            off, dim = self.prb.getInput().getVarIndex(sname)
            self.solution[f'{sname}_res'] = u_res[off:off + dim, :]

        # new fmap with resampled forces
        if self.model.fmap:

            fmap = dict()
            for frame, wrench in self.model.fmap.items():
                fmap[frame] = self.solution[f'{wrench.getName()}']

            fmap_res = dict()
            for frame, wrench in self.model.fmap.items():
                fmap_res[frame] = self.solution[f'{wrench.getName()}_res']

            # get tau resampled
            if resample_tau:
                tau = np.zeros([self.model.tau.shape[0], self.prb.getNNodes() - 1])
                tau_res = np.zeros([self.model.tau.shape[0], u_res.shape[1]])

                id = kin_dyn.InverseDynamics(self.model.kd, fmap_res.keys(), self.model.kd_frame)

                # id_fn = kin_dyn.InverseDynamics(self.kd, self.fmap.keys(), self.kd_frame)
                # self.tau = id_fn.call(self.q, self.v, self.a, self.fmap)
                # self.prb.createIntermediateConstraint('dynamics', self.tau[:6])

                # todo: this is horrible. id.call should take matrices, I should not iter over each node

                for i in range(tau.shape[1]):
                    fmap_i = dict()
                    for frame, wrench in fmap.items():
                        fmap_i[frame] = wrench[:, i]
                    tau_i = id.call(self.solution['q'][:, i], self.solution['v'][:, i], self.solution['a'][:, i],
                                    fmap_i)
                    tau[:, i] = tau_i.toarray().flatten()

                for i in range(tau_res.shape[1]):
                    fmap_res_i = dict()
                    for frame, wrench in fmap_res.items():
                        fmap_res_i[frame] = wrench[:, i]
                    tau_res_i = id.call(self.solution['q_res'][:, i], self.solution['v_res'][:, i],
                                        self.solution['a_res'][:, i], fmap_res_i)
                    tau_res[:, i] = tau_res_i.toarray().flatten()

                self.solution['tau'] = tau
                self.solution['tau_res'] = tau_res


    def save_solution(self, filename):
        import copy
        ms = mat_storer.matStorer(filename)
        joint_names = self.model.joint_names.copy()
        sol_to_save = copy.deepcopy(self.solution)

        # todo: duplicate code in replay_trajectory
        # add all the fixed joint values (for now, do it for q_res)
        ns = np.shape(sol_to_save['q_res'])[1]

        for fixed_joint, fixed_val in self.model.fixed_joint_map.items():
            # append fixed joints name to joints name list
            joint_names.append(fixed_joint)
            # expand fixed value along the nodes and append the row to the q_sol
            fixed_val_array = np.full([1, ns], fixed_val)
            sol_to_save['q_res'] = np.vstack((sol_to_save['q_res'], fixed_val_array))

        sol_to_save['joint_names'] = joint_names
        sol_to_save['dt'] = self.prb.getDt()
        ms.store(sol_to_save)


    def load_solution(self, filename):
        ms = mat_storer.matStorer(filename)
        ig = ms.load()
        self.load_initial_guess(from_dict=ig)

    
    def load_initial_guess(self, from_dict=None):
        if from_dict is None:
            from_dict = self.solution

        x_opt = from_dict['x_opt']
        u_opt = from_dict['u_opt']
        
        self.prb.getState().setInitialGuess(x_opt)
        self.prb.getInput().setInitialGuess(u_opt)
        self.prb.setInitialState(x0=x_opt[:, 0])

    def replay_trajectory(self, trajectory_markers=[], trajectory_markers_opts={}):

        # single replay
        joint_names = self.model.kd.joint_names()
        q_sol = self.solution['q']
        q_sol_minimal = np.zeros([q_sol.shape[0], self.prb.getNNodes()])

        # if q is not minimal (continuous joints are present) make it minimal
        for col in range(q_sol.shape[1]):
            q_sol_minimal[:, col] = self.model.kd.getMinimalQ(q_sol[:, col])

        frame_force_mapping = {cname: self.solution[f.getName()] for cname, f in self.model.fmap.items()}

        repl = replay_trajectory(self.prb.getDt(),
                                 joint_names,
                                 q_sol_minimal,
                                 frame_force_mapping,
                                 self.model.kd_frame,
                                 self.model.kd,
                                 fixed_joint_map=self.model.fixed_joint_map,
                                 trajectory_markers=trajectory_markers,
                                 trajectory_markers_opts=trajectory_markers_opts)
        repl.sleep(1.)
        repl.replay(is_floating_base=True, base_link='pelvis')

    # a possible method could read from yaml and create the task list
    def setTaskFromYaml(self, yaml_config):

        self.task_desrc_list, self.non_active_task, self.solver_options = YamlParser.load(yaml_config)
        self.setSolverOptions(self.solver_options)

        # todo: this should be updated everytime a task is added
        for task_descr in self.task_desrc_list:

            if 'weight' in task_descr and isinstance(task_descr['weight'], dict):
                weight_dict = task_descr['weight']
                if 'position' in weight_dict:
                    weight_dict['q'] = weight_dict.pop('position')
                if 'velocity' in weight_dict:
                    weight_dict['v'] = weight_dict.pop('velocity')
                if 'acceleration' in weight_dict:
                    weight_dict['a'] = weight_dict.pop('acceleration')

                # todo this is wrong: if new forces are added, this is not adding them into the Task
                if 'force' in weight_dict:
                    weight_force = weight_dict.pop('force')
                    for f in self.model.fmap.values():
                        weight_dict[f.getName()] = weight_force

            self.setTaskFromDict(task_descr)


    def setTaskFromDict(self, task_description):
        # todo if task is dict... ducktyping

        task = self.generateTaskFromDict(task_description)
        self.setTask(task)
        return task

    def generateTaskFromDict(self, task_description):

        # todo this should probably go in each single task definition --> i don't have the info from the ti then
        shortcuts = {
            'nodes': {'final': self.prb.getNNodes() - 1, 'all': list(range(self.prb.getNNodes()))},
            # todo: how to choose the value to substitute depending on the item? (indices of q: self.model.nq, indices of f: self.f.size ...)
            # 'indices': {'floating_base': range(7), 'joints': range(7, self.model.nq + 1)}
        }
        task_descr_resolved = YamlParser.resolve(task_description, shortcuts)

        task_description_with_subtasks = self._handle_subtask(task_descr_resolved)
        task_specific = self.generateTaskContext(task_description_with_subtasks)

        task = task_factory.create(task_specific)

        # todo: check if here it is necessary
        # self.setTask(task)

        return task

    def generateTaskContext(self, task_description):
        '''
        add specific context to task depending on its type
        '''

        task_description_mod = task_description.copy()
        # automatically provided info:

        # add generic context
        task_description_mod['prb'] = self.prb
        task_description_mod['kin_dyn'] = self.model.kd
        task_description_mod['model'] = self.model

        # todo horrible to do it here
        # add specific context
        if task_description_mod['type'] == 'Postural':
            task_description_mod['postural_ref'] = self.model.q0

        if task_description_mod['type'] == 'TorqueLimits':
            task_description_mod['var'] = self.model.tau

        return task_description_mod

    def _handle_subtask(self, task_description):

        # transform description of subtask (dict) into an instance of the task and pass it to the parent task
        task_description_copy = task_description.copy()
        # check for subtasks:
        subtasks = dict()
        if 'subtask' in task_description_copy:
            subtask_description_list = task_description_copy.pop(
                'subtask') if 'subtask' in task_description_copy else []


            # inherit from parent:
            for subtask_description in subtask_description_list:

                # TODO: wrong way to handle YAML subtasks
                if isinstance(subtask_description, str):
                    for task in self.non_active_task:
                        if task['name'] == subtask_description:
                            subtask_description = task
                            break
                # child inherit from parent the values, if not present
                # parent define the context for the child: child can override it
                
                # todo: better handling of parameter propagation
                # for key, value in task_description_copy.items():
                #     if key not in subtask_description and key != 'subtask':
                #         subtask_description[key] = value

                s_t = self.getTask(subtask_description['name'])
                
                if s_t is None:
                    s_t = self.generateTaskFromDict(subtask_description)

                subtasks[s_t.name] = s_t
                task_description_copy.update({'subtask': subtasks})

        return task_description_copy

    def setTask(self, task):
        # check if task is of registered_type # todo what about plugins?
        assert isinstance(task, tuple(task_factory.get_registered_tasks()))
        self.task_list.append(task)

    def getTask(self, task_name):
        # return the task with name task_name
        for task in self.task_list:
            if task.name == task_name:
                return task
        return None

    def getTaskByClass(self, task_class):
        list_1 = [t for t in self.task_list if isinstance(t, task_class)]
        return list_1

    def loadPlugins(self, plugins):
        plugin_handler.load_plugins(plugins)

    def getTasksType(self, task_type=None):
        return task_factory.get_registered_tasks(task_type)

    # todo
    def setTaskOptions(self):
        pass

    def setSolverOptions(self, solver_options):
        solver_type = solver_options.pop('type')
        is_receding = solver_options.pop('receding', False)

        self.si = solver_interface.SolverInterface(solver_type, is_receding, solver_options)

    def _create_solver(self):

        if self.si.type != 'ilqr':
            # todo get options from yaml
            th = Transcriptor.make_method('multiple_shooting', self.prb)

        # todo if receding is true ....
        self.solver_bs = Solver.make_solver(self.si.type, self.prb, self.si.opts)
        
        try:
            self.solver_bs.set_iteration_callback()
        except:
            pass

        scoped_opts_rti = self.si.opts.copy()
        scoped_opts_rti['ilqr.enable_line_search'] = False
        scoped_opts_rti['ilqr.max_iter'] = 1
        self.solver_rti = Solver.make_solver(self.si.type, self.prb, scoped_opts_rti)

        return self.solver_bs, self.solver_rti
