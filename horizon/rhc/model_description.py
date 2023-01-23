
from horizon.problem import Problem
from horizon.utils import utils, kin_dyn
from typing import Tuple, Union
import casadi as cs
import numpy as np
from casadi_kin_dyn import pycasadi_kin_dyn
import urdf_parser_py.urdf as upp

np.set_printoptions(precision=3, suppress=True)

class FullModelInverseDynamics:
    
    def __init__(self, problem, kd, q_init, params=0, base_init=None, floating_base=True, fixed_joint_map=None, **kwargs):
        # todo: adding contact dict

        if fixed_joint_map is None:
            fixed_joint_map = {}

        self.prb: Problem = problem
        self.kd = kd
        self.kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        self.fixed_joint_map = fixed_joint_map

        # number of dof
        self.nq = self.kd.nq()
        self.nv = self.kd.nv()

        # manage starting position
        # initial guess (also initial condition and nominal pose)
        self.q0 = self.kd.mapToQ(q_init)

        if floating_base is True:
            self.q0[:7] = base_init
            self.joint_names = self.kd.joint_names()[2:]
        else:
            self.joint_names = self.kd.joint_names()[1:]

        self.v0 = np.zeros(self.nv)


        self.kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED

        self.nq = self.kd.nq()
        self.nv = self.kd.nv()

        # custom choices
        # todo this is ugly
        self.q = self.prb.createStateVariable('q', self.nq)
        self.v = self.prb.createStateVariable('v', self.nv)
        self.a = self.prb.createInputVariable('a', self.nv)

        # velocity limits
        vel_lims = kd.velocityLimits()
        self.v.setBounds(-vel_lims, vel_lims)

        # parse contacts
        self.fmap = dict()
        self.cmap = dict()

        self.params = params

    def fk(self, frame) -> Tuple[Union[cs.SX, cs.MX]]:
        """
        returns the tuple (ee_pos, ee_rot), evaluated
        at the symbolic state variable q
        """
        fk_fn = self.kd.fk(frame)
        return fk_fn(self.q)


    def setContactFrame(self, contact_frame, contact_type, contact_params=dict()):

        # todo add more guards
        if contact_frame in self.fmap.keys():
            raise Exception(f'{contact_frame} frame is already a contact')

        if contact_type == 'surface':
            return self._make_surface_contact(contact_frame, contact_params)
        elif contact_type == 'vertex':
            return self._make_vertex_contact(contact_frame, contact_params) 
        elif contact_type == 'point':
            return self._make_point_contact(contact_frame, contact_params) 

        raise ValueError(f'{contact_type} is not a valid contact type')

    def _make_surface_contact(self, contact_frame, contact_params):
        # create input (todo: support degree > 0)
        wrench = self.prb.createInputVariable('f_' + contact_frame, dim=6)
        self.fmap[contact_frame] = wrench
        self.cmap[contact_frame] = [wrench]
        return wrench

    def _make_point_contact(self, contact_frame, contact_params):
        # create input (todo: support degree > 0)
        force = self.prb.createInputVariable('f_' + contact_frame, dim=3)
        self.fmap[contact_frame] = force
        self.cmap[contact_frame] = [force]
        return force

    def _make_vertex_contact(self, contact_frame, contact_params):

        # todo WARNING: contact_params['vertex_frames'] must be a list!!!!!!!!!!!!!!!!!

        vertex_frames = contact_params['vertex_frames']  # todo improve error

        # create inputs (todo: support degree > 0)
        vertex_forces = [self.prb.createInputVariable('f_' + vf, dim=3) for vf in vertex_frames]

        # save vertices
        for frame, force in zip(vertex_frames, vertex_forces):
            self.fmap[frame] = force

        self.cmap[contact_frame] = vertex_forces

        return vertex_forces

    def setDynamics(self):
        # todo refactor this floating base stuff

        self.xdot = utils.double_integrator(self.q, self.v, self.a, self.kd)
        self.prb.setDynamics(self.xdot)

        # underactuation constraints
        if self.fmap:
            id_fn = kin_dyn.InverseDynamics(self.kd, self.fmap.keys(), self.kd_frame)
            self.tau = id_fn.call(self.q, self.v, self.a, self.fmap, params=self.params)
            self.dynamic_constraint = self.prb.createIntermediateConstraint('dynamics', self.tau[:6])

            # torque limits
            # torque_lims = self.kd.effortLimits()
            # self.dynamic_constraint.setBounds([0., 0., 0., 0., 0., 0.] + (-1*torque_lims[6:]).tolist(), [0., 0., 0., 0., 0., 0.] + torque_lims[6:].tolist())

        else:
            id_fn = kin_dyn.InverseDynamics(self.kd)
            self.tau = id_fn.call(self.q, self.v, self.a, params=self.params)
            self.dynamic_constraint = self.prb.createIntermediateConstraint('dynamics', self.tau)

    def getContacts(self):
        return list(self.cmap.keys())

    # def getInput(self):
    #     return self.a
    #
    # def getState(self):
    #     return



class SingleRigidBodyDynamicsModel:
        #  problem, kd, q_init, base_init, floating_base=True):
    def __init__(self, problem, kd, q_init, base_init, **kwargs):
        
        self.prb: Problem = problem
        self.kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        self.kd_real = kd

        # compute q0 from real robot
        q0_real = self.kd_real.mapToQ(q_init)
        q0_real[:7] = base_init

        # srbd generation
        srbd_robot = upp.URDF()
        srbd_robot.name = 'srbd'

        # add world link
        world_link = upp.Link('world')
        srbd_robot.add_link(world_link)

        # todo sync origin, link name, ...
        srbd_inertia = 3.*np.eye(3)

        self._make_floating_link(srbd_robot, 
                        link_name='base_link', 
                        parent_name='world', 
                        mass=self.kd_real.mass(), 
                        inertia=srbd_inertia)

        # parse contacts
        self.fmap = dict()
        self.cmap = dict()
        contact_dict = kwargs['contact_dict']
        for cframe, cparams in contact_dict.items():
            
            ctype = cparams['type']

            self._make_floating_link(srbd_robot, 
                        link_name=cframe, 
                        parent_name='base_link', 
                        mass=0, 
                        inertia=np.zeros((3, 3)))
            
            if ctype == 'surface':
                pass
            elif ctype == 'vertex':
                vertex_frames = cparams['vertex_frames']
                pos_0, rot_0 = self.kd_real.fk(cframe)(q0_real)
                for vf in vertex_frames:
                    pos_v, _ = self.kd_real.fk(vf)(q0_real)
                    origin = (rot_0.T @ (pos_v - pos_0)).full().flatten()
                    self._add_frame(
                        srbd_robot,
                        vf,
                        cframe,
                        origin
                    )
            elif ctype == 'point':
                pass
        
        # create srbd urdf
        urdf_srbd = srbd_robot.to_xml_string()
        self.kd_srbd = pycasadi_kin_dyn.CasadiKinDyn(urdf_srbd)
        self.kd = self.kd_srbd
        self.joint_names = self.kd_srbd.joint_names()

        # create state and input
        self.nq = self.kd_srbd.nq()
        self.nv = self.kd_srbd.nv()
        self.v0 = np.zeros(self.nv)

        # kinodynamic model?
        self.use_kinodynamic = kwargs.get('use_kinodynamic', False)

        self.q = self.prb.createStateVariable('q', self.nq)
        self.v = self.prb.createStateVariable('v', self.nv)

        if self.use_kinodynamic:
            # note: base acceleration computation is postponed to setDynamics.
            # when we'll know the forces
            self.aj = self.prb.createInputVariable('aj', self.nv - 6)
        else:
            self.a = self.prb.createInputVariable('a', self.nv)

        

        _, base_rot_0 = self.kd_real.fk('base_link')(q0_real)  # todo: why ?
        base_pos_0, _, _ = self.kd_real.centerOfMass()(q0_real, 0, 0)
        self.q0 = self.kd_srbd.mapToQ({})
        self.q0[:3] = base_pos_0.full().flatten()
        self.q0[3:7] = utils.rotationMatrixToQuaterion(base_rot_0)

        q0_idx = 7
        for jn in self.kd_srbd.joint_names()[2:]:
            distal_link_name = jn[:-6]  # remove trailing '_joint'
            pos_0, rot_0 = self.kd_real.fk(distal_link_name)(q0_real)
            rel_pos = base_rot_0.T @ ( pos_0 - base_pos_0 )
            rel_rot = base_rot_0.T @ rot_0
            
            self.q0[q0_idx:q0_idx+3] = rel_pos.full().flatten()
            self.q0[q0_idx+3:q0_idx+7] = utils.rotationMatrixToQuaterion(rel_rot)
            q0_idx += 7


    def _make_floating_link(self, srbd_robot, link_name, parent_name, mass, inertia):
        world_joint = upp.Joint(name=f'{link_name}_joint', 
                                parent=parent_name,
                                child=link_name,
                                joint_type='floating',
                                origin=upp.Pose(xyz=[0, 0, 0.0]))
        
        srbd_robot.add_joint(world_joint)
        srbd_robot.add_link(
            self._make_box_link(link_name, 
                                mass=mass,
                                inertia=inertia  # todo compute inertia
                                )
                        )
    
    def _make_box_link(self, name, mass, inertia, oxyz=[0, 0, 0], visual=None, center=False):
        
        ixx = inertia[0, 0]
        iyy = inertia[1, 1]
        izz = inertia[2, 2]
        ixy = inertia[0, 1]
        ixz = inertia[0, 2]
        iyz = inertia[1, 2]

        if mass > 0:
            # compute size from inertia and mass
            # ixx = 1./12.*mass*(b**2 + c**2)
            # iyy = 1./12.*mass*(a**2 + c**2)
            # izz = 1./12.*mass*(a**2 + b**2)
            

            idiag = np.array([ixx, iyy, izz])
            
            A = np.array([[0, 1, 1],
                        [1, 0, 1], 
                        [1, 1, 0]])
            
            size = np.sqrt(np.linalg.inv(A) @ idiag*12.0/mass)
        else:
            size = [0.4, 0.2, 0.05]

        # create link

        link = upp.Link(name=name)
        geo = upp.Box(size=size)
        pose = upp.Pose(xyz=list(oxyz))
        
        if not center:
            pose.xyz[2] += size[2]/2.0

        ine = upp.Inertia(ixx=ixx, iyy=iyy, izz=izz, ixy=ixy, ixz=ixz, iyz=iyz)
        
        link.collision = upp.Collision(geometry=geo, origin=pose)

        if visual:
            link.visual = visual
        else:
            link.visual = upp.Visual(geometry=geo, origin=pose)

        link.visual.material = upp.Material(name='dfl_color', color=upp.Color([0.8, 0.8, 0.8, 1]))

        link.inertial = upp.Inertial(mass=mass, inertia=ine, origin=pose)
        
        return link

    
    def _add_frame(self, srbd_robot, name, parent_name, oxyz, visual=None):

        link = upp.Link(name=name)
        geo = upp.Sphere(radius=0.02)
        pose = upp.Pose()
        ine = upp.Inertia()
        
        if visual:
            link.visual = visual
        else:
            link.visual = upp.Visual(geometry=geo, origin=pose)

        link.visual.material = upp.Material(name='dfl_color', color=upp.Color([0.8, 0.8, 0.8, 1]))

        link.inertial = upp.Inertial(mass=0, inertia=ine, origin=pose)
        
        joint = upp.Joint(name=f'{name}_joint', 
                          parent=parent_name,
                          child=name,
                          joint_type='fixed',
                          origin=upp.Pose(xyz=oxyz))

        srbd_robot.add_joint(joint)
        srbd_robot.add_link(link)
        


    def fk(self, frame) -> Tuple[Union[cs.SX, cs.MX]]:
        """
        returns the tuple (ee_pos, ee_rot), evaluated
        at the symbolic state variable q
        """
        fk_fn = self.kd_srbd.fk(frame)
        return fk_fn(self.q)
    
    
    def setContactFrame(self, contact_frame, contact_type, contact_params=dict()):

        # todo add more guards
        if contact_frame in self.fmap.keys():
            raise Exception(f'{contact_frame} frame is already a contact')

        if contact_type == 'surface':
            return self._make_surface_contact(contact_frame, contact_params)
        elif contact_type == 'vertex':
            return self._make_vertex_contact(contact_frame, contact_params) 
        elif contact_type == 'point':
            return self._make_point_contact(contact_frame, contact_params) 

        raise ValueError(f'{contact_type} is not a valid contact type')

    
    def _make_surface_contact(self, contact_frame, contact_params):
        # create input (todo: support degree > 0)
        wrench = self.prb.createInputVariable('f_' + contact_frame, dim=6)
        self.fmap[contact_frame] = wrench
        self.cmap[contact_frame] = [wrench]
        return wrench

    
    def _make_point_contact(self, contact_frame, contact_params):
        # create input (todo: support degree > 0)
        force = self.prb.createInputVariable('f_' + contact_frame, dim=3)
        self.fmap[contact_frame] = force
        self.cmap[contact_frame] = [force]
        return force

    def _make_vertex_contact(self, contact_frame, contact_params):
        
        vertex_frames = contact_params['vertex_frames']  # todo improve error

        # create inputs (todo: support degree > 0)
        vertex_forces = [self.prb.createInputVariable('f_' + vf, dim=3) for vf in vertex_frames]

        # save vertices
        for frame, force in zip(vertex_frames, vertex_forces):
            self.fmap[frame] = force

        self.cmap[contact_frame] = vertex_forces

        # do we need to reconstruct the total wrench?
        return vertex_forces

    def setDynamics(self):

        xdot = utils.double_integrator(self.q, self.v, self.a, self.kd_srbd)

        self.prb.setDynamics(xdot)

        # underactuation constraints
        if self.fmap:
            id_fn = kin_dyn.InverseDynamics(self.kd, self.fmap.keys(), self.kd_frame)
            self.tau = id_fn.call(self.q, self.v, self.a, self.fmap)
            self.prb.createIntermediateConstraint('dynamics', self.tau[:6])


    def getContacts(self):
        return self.cmap.keys()


