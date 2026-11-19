"""Native robot computations for Horizon Problem expressions."""
import casadi as cs
import numpy as np

from horizon.native import NativeFunction
from horizon.solvers import pyilqr


def _key(value):
    if isinstance(value, np.ndarray):
        return (value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, dict):
        return tuple((name, _key(item)) for name, item in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_key(item) for item in value)
    if hasattr(value, "frame") and hasattr(value, "dimension"):
        return (value.frame, value.dimension)
    return value


class NativeRobot:
    """Direct Pinocchio functions used as ordinary numeric/MX operations.

    Costs, constraints, bounds and dynamics belong to Problem. The robot keeps
    native expression owners while building graphs; Problem then retains them.
    """
    def __init__(self, urdf, actuation_matrix=None):
        self.raw = pyilqr.PinocchioModel(str(urdf))
        if actuation_matrix is None:
            first = self.raw.joint_names[0] if self.raw.joint_names else None
            floating = first is not None and self.raw.joint_nq(first) == 7 and self.raw.joint_nv(first) == 6
            actuation_matrix = np.eye(self.nv)[:, 6 if floating else 0:]
        self.actuation_matrix = np.asarray(actuation_matrix, dtype=float).copy()
        if self.actuation_matrix.ndim != 2 or self.actuation_matrix.shape[0] != self.nv or not np.all(np.isfinite(self.actuation_matrix)):
            raise ValueError("actuation_matrix must be finite with nv rows")
        self.na = self.actuation_matrix.shape[1]
        self._functions = {}

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def _operation(self, name, arguments, setup=(), options=None):
        options = options or {}
        key = (name, _key(setup), _key(options))
        function = self._functions.get(key)
        if function is None:
            function = NativeFunction(getattr(self.raw, name)(*setup, **options),
                                      configuration_space=self.configuration_space, nq=self.nq)
            self._functions[key] = function
        return function(*arguments)

    def integrate(self, q, displacement):
        return self._operation("integrate", (q, displacement))

    def step(self, q, v, dt):
        return self.integrate(q, dt*v)

    def difference(self, first, second):
        return self._operation("difference", (first, second))

    def framePose(self, frame, q):
        return self._operation("framePose", (q,), (frame,))

    def framePoseError(self, frame, q, target):
        return self._operation("framePoseError", (q, target), (frame,))

    def framePositions(self, frames, q):
        frames = tuple(frames)
        value = self._operation("framePositions", (q,), (frames,))
        return cs.reshape(value, len(frames), 3).T if isinstance(value, cs.MX) else value.reshape(3, len(frames))

    def fk(self, frame, q):
        return self._operation("fk", (q,), (frame,))

    def jacobian(self, frame, q, reference=pyilqr.LOCAL_WORLD_ALIGNED):
        value = self._operation("jacobian", (q,), (frame, reference))
        return cs.reshape(value, 6, self.nv) if isinstance(value, cs.MX) else value.reshape(6, self.nv, order="F")

    def frameVelocity(self, frame, q, v, reference=pyilqr.LOCAL_WORLD_ALIGNED):
        return self._operation("frameVelocity", (q, v), (frame, reference))

    def frameAcceleration(self, frame, q, v, a, reference=pyilqr.LOCAL_WORLD_ALIGNED):
        return self._operation("frameAcceleration", (q, v, a), (frame, reference))

    def frameSpatialAcceleration(self, frame, q, v, a, reference=pyilqr.LOCAL_WORLD_ALIGNED):
        return self._operation("frameSpatialAcceleration", (q, v, a), (frame, reference))

    def rnea(self, q, v, a, gravity=None):
        arguments = (q, v, a) if gravity is None else (q, v, a, gravity)
        return self._operation("rnea", arguments, (gravity is not None,))

    def contactDynamics(self, q, v, tau, contacts, actuation_matrix=None):
        setup = (tuple(contacts),)
        options = {"actuation_matrix": self.actuation_matrix if actuation_matrix is None else np.asarray(actuation_matrix)}
        value = self._operation("contactDynamics", (q, v, tau), setup, options)
        return value[:self.nv], value[self.nv:]

    def forwardDynamics(self, q, v, tau, actuation_matrix=None):
        matrix = self.actuation_matrix if actuation_matrix is None else np.asarray(actuation_matrix)
        return self._operation("aba", (q, v, matrix@tau))

    def inverseDynamics(self, q, v, a, forces=None, contacts=()):
        arguments = (q, v, a) if forces is None else (q, v, a, forces)
        return self._operation("inverseDynamics", arguments, (tuple(contacts),))

    def sphereCenters(self, q, spheres):
        return self._operation("sphereCenters", (q,), (tuple(spheres),))


def _vector_operation(name):
    def operation(self, *arguments):
        return self._operation(name, arguments)
    operation.__name__ = name
    return operation


for _name in ("qdot", "aba", "gravity", "nonlinearEffects",
              "kineticEnergy", "potentialEnergy", "centerOfMass", "comPosition",
              "comVelocity", "comAcceleration", "centroidalMomentum",
              "computeCentroidalDynamics", "centroidalDynamics"):
    setattr(NativeRobot, _name, _vector_operation(_name))


def _matrix_operation(name, rows):
    def operation(self, q):
        value = self._operation(name, (q,))
        nrows = self.nv if rows is None else rows
        return cs.reshape(value, nrows, self.nv) if isinstance(value, cs.MX) else value.reshape(nrows, self.nv, order="F")
    return operation


for _name, _rows in (("massMatrix", None), ("crba", None), ("centroidalMap", 6), ("ccrba", 6)):
    setattr(NativeRobot, _name, _matrix_operation(_name, _rows))
