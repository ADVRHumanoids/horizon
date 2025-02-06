from control_cluster_bridge.utilities.shared_data.rhc_data import RhcStatus
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcInternal
import numpy as np
from EigenIPC.PyEigenIPC import VLevel
from horizon.rhc.taskInterface import TaskInterface

class SharedMemoryInterface:
    def __init__(self, ti: TaskInterface, namespace, n_contacts, n_dofs, joint_names, n_nodes):

        self.__namespace = namespace
        self.__n_contacts = n_contacts
        self.__controller_index = 0
        self.__n_dofs = n_dofs
        self.__controller_side_jnt_names = joint_names
        self.__n_nodes = n_nodes
        self.__verbose = False

        self.__dtype = np.float32

        self.__ti = ti

        cost_data = self._get_cost_data()
        constr_data = self._get_constr_data()

        self.__rhc_costs = dict()
        self.__rhc_constr = dict()

        config = RhcInternal.Config(is_server=True,
                                enable_q= True,
                                enable_v=True,
                                enable_a=True,
                                enable_a_dot=False,
                                enable_f=True,
                                enable_f_dot=False,
                                enable_eff=False,
                                cost_names=cost_data[0],
                                cost_dims=cost_data[1],
                                constr_names=constr_data[0],
                                constr_dims=constr_data[1],
                                )
        self.rhc_internal = RhcInternal(config=config,
                                namespace=self.__namespace,
                                rhc_index = self.__controller_index,
                                n_contacts=self.__n_contacts,
                                n_jnts=self.__n_dofs,
                                jnt_names=self.__controller_side_jnt_names,
                                n_nodes=self.__n_nodes,
                                verbose = self.__verbose,
                                vlevel=VLevel.V3,
                                force_reconnection=True,
                                safe=True)

        self.rhc_internal.run()

    def update_db_data(self, cost_values, constraint_values):

        self.__rhc_costs.update(cost_values) #self.__ti.solver_rti.getCostsValues()
        self.__rhc_constr.update(constraint_values) #self.__ti.solver_rti.getConstraintsValues()

    def _get_cost_data(self):

        cost_dict = self.__ti.getSolverRti().getCostsValues()
        cost_names = list(cost_dict.keys())
        cost_dims = [1] * len(cost_names)  # costs are always scalar
        return cost_names, cost_dims

    def _get_constr_data(self):

        constr_dict = self.__ti.getSolverRti().getConstraintsValues()
        constr_names = list(constr_dict.keys())
        constr_dims = [-1] * len(constr_names)
        i = 0
        for constr in constr_dict:
            constr_val = constr_dict[constr]
            constr_shape = constr_val.shape
            constr_dims[i] = constr_shape[0]
            i += 1
        return constr_names, constr_dims

    def _get_q_from_sol(self):
        return self.__ti.solution['q'].astype(self.__dtype)

    def _get_v_from_sol(self):
        return self.__ti.solution['v'].astype(self.__dtype)

    def _get_a_from_sol(self):
        return self.__ti.solution['a'].astype(self.__dtype)

    def _get_f_from_sol(self):
        # to be overridden by child class
        contact_names = self.__ti.model.getContacts()  # we use controller-side names
        try:
            data = []
            for key in contact_names:
                contact_f = self.__ti.solution["f_" + key].astype(self.__dtype)
                np.nan_to_num(contact_f, nan=1e6, posinf=1e6, neginf=-1e6, copy=False)
                np.clip(a=contact_f, a_max=1e6, a_min=-1e6, out=contact_f)
                data.append(contact_f)
            return np.concatenate(data, axis=0)
        except:
            return None

    def _get_f_dot_from_sol(self):
        # to be overridden by child class
        return None

    def _get_eff_from_sol(self):
        # to be overridden by child class
        return None

    def _get_cost_from_sol(self,
                           cost_name: str):
        return self.__rhc_costs[cost_name]

    def _get_constr_from_sol(self,
                             constr_name: str):
        return self.__rhc_constr[constr_name]

    def update(self):
        # data which is not enabled in the config is not actually
        # written so overhead is minimal for non-enabled data
        self.rhc_internal.write_q(data=self._get_q_from_sol(),
                                  retry=True)
        self.rhc_internal.write_v(data=self._get_v_from_sol(),
                                  retry=True)
        self.rhc_internal.write_a(data=self._get_a_from_sol(),
                                  retry=True)
        self.rhc_internal.write_f(data=self._get_f_from_sol(),
                                  retry=True)
        self.rhc_internal.write_f_dot(data=self._get_f_dot_from_sol(),
                                      retry=True)
        self.rhc_internal.write_eff(data=self._get_eff_from_sol(),
                                    retry=True)
        for cost_idx in range(self.rhc_internal.config.n_costs):
            # iterate over all costs and update all values
            cost_name = self.rhc_internal.config.cost_names[cost_idx]
            self.rhc_internal.write_cost(data=self._get_cost_from_sol(cost_name=cost_name),
                                         cost_name=cost_name,
                                         retry=True)
        for constr_idx in range(self.rhc_internal.config.n_constr):
            # iterate over all constraints and update all values
            constr_name = self.rhc_internal.config.constr_names[constr_idx]
            self.rhc_internal.write_constr(data=self._get_constr_from_sol(constr_name=constr_name),
                                           constr_name=constr_name,
                                           retry=True)