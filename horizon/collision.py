"""Collision geometry helpers; values and clearance Jacobians run in C++.

SDF construction is a setup operation. Robot/contact/tool kinematics remains
independent of these collision-only sphere descriptions.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import casadi as cs
import numpy as np

from horizon.native import NativeFunction
from horizon.solvers import pyilqr


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite xyz vector")
    return result


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _rotation(value):
    value = np.eye(3) if value is None else np.asarray(value, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)) or not np.allclose(value.T@value, np.eye(3), atol=1e-9) or not np.isclose(np.linalg.det(value), 1.):
        raise ValueError("rotation must be a proper orthonormal 3x3 matrix")
    return value


@dataclass
class Sphere:
    center: object
    radius: float
    def _prepare(self):
        center, radius = _vector(self.center, "center"), _positive(self.radius, "radius")
        return lambda points: np.linalg.norm(points-center, axis=1)-radius


@dataclass
class Box:
    center: object
    size: object
    rotation: object = None
    def _prepare(self):
        center, size, rotation = _vector(self.center, "center"), _vector(self.size, "size"), _rotation(self.rotation)
        if np.any(size <= 0):
            raise ValueError("box sizes must be positive")
        def distance(points):
            delta = np.abs((points-center)@rotation)-size/2
            return np.linalg.norm(np.maximum(delta, 0), axis=1)+np.minimum(np.max(delta, axis=1), 0)
        return distance


@dataclass
class Capsule:
    start: object
    end: object
    radius: float
    def _prepare(self):
        start, end, radius = _vector(self.start, "start"), _vector(self.end, "end"), _positive(self.radius, "radius")
        direction = end-start
        if direction@direction < 1e-24:
            raise ValueError("capsule endpoints must differ")
        def distance(points):
            t = np.clip((points-start)@direction/(direction@direction), 0, 1)
            return np.linalg.norm(points-start-t[:, None]*direction, axis=1)-radius
        return distance


@dataclass
class Cylinder:
    center: object
    radius: float
    length: float
    rotation: object = None
    def _prepare(self):
        center, rotation = _vector(self.center, "center"), _rotation(self.rotation)
        radius, half = _positive(self.radius, "radius"), _positive(self.length, "length")/2
        def distance(points):
            local = (points-center)@rotation
            delta = np.column_stack((np.linalg.norm(local[:, :2], axis=1)-radius, np.abs(local[:, 2])-half))
            return np.minimum(np.max(delta, axis=1), 0)+np.linalg.norm(np.maximum(delta, 0), axis=1)
        return distance


@dataclass
class Plane:
    point: object
    normal: object
    def _prepare(self):
        point, normal = _vector(self.point, "point"), _vector(self.normal, "normal")
        if np.linalg.norm(normal) < 1e-12:
            raise ValueError("plane normal must be nonzero")
        normal = normal/np.linalg.norm(normal)
        return lambda points: (points-point)@normal


@dataclass
class Mesh:
    path: object
    pose: object = None
    scale: object = 1.
    def _prepare(self):
        import trimesh
        pose = np.eye(4) if self.pose is None else np.asarray(self.pose, dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)) or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError("mesh pose must be a finite rigid 4x4 transform")
        _rotation(pose[:3, :3])
        scale = np.broadcast_to(np.asarray(self.scale, dtype=float), (3,))
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("mesh scale must be positive")
        mesh = trimesh.load(str(self.path), force="mesh", process=True)
        mesh.apply_scale(scale)
        mesh.apply_transform(pose)
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError("signed mesh SDF requires a watertight, consistently wound mesh")
        # Trimesh uses positive inside; Horizon uses positive outside.
        return lambda points: -trimesh.proximity.signed_distance(mesh, points)


def build_sdf(objects, *, bounds, voxel_size, chunk_size=65536, max_voxels=16_000_000):
    """Sample the signed-distance union of primitives and watertight meshes.

    Returns an immutable C++ SdfGrid. Queries and gradients during optimization
    are C++ trilinear interpolation. bounds must contain all query points.
    Plane solids occupy their negative half-space. Cylinder local axis is z.
    Mesh setup requires trimesh and rtree; primitives require only NumPy.
    """
    lower, upper = (_vector(value, "SDF bounds") for value in bounds)
    voxel_size = _positive(voxel_size, "voxel size")
    if np.any(upper <= lower) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("SDF bounds must increase and chunk_size must be positive")
    objects = list(objects)
    if not objects:
        raise ValueError("an SDF needs at least one primitive or mesh")
    dimensions = np.ceil((upper-lower)/voxel_size).astype(np.int64)+1
    count = int(np.prod(dimensions, dtype=object))
    if count > max_voxels or np.any(dimensions > np.iinfo(np.int32).max):
        raise ValueError("SDF exceeds max_voxels; adjust bounds, voxel_size or max_voxels")
    distance_functions = [obj._prepare() for obj in objects]
    values = np.empty(count, dtype=np.float32)
    nx, ny, _ = dimensions
    for start in range(0, count, chunk_size):
        indices = np.arange(start, min(start+chunk_size, count), dtype=np.int64)
        points = lower+voxel_size*np.column_stack((indices % nx, indices//nx % ny, indices//(nx*ny)))
        distances = np.full(len(indices), np.inf)
        for function in distance_functions:
            distances = np.minimum(distances, function(points))
        if not np.all(np.isfinite(distances)):
            raise ValueError("SDF construction produced nonfinite distances")
        values[start:start+len(indices)] = distances
    return pyilqr.SdfGrid(dimensions.astype(np.int32), lower, voxel_size, values)


class CollisionSpheres:
    """Robot collision spheres: (link name, link-local center, radius)."""
    def __init__(self, spheres):
        self.spheres = tuple((str(link), _vector(center, "sphere center"), _positive(radius, "sphere radius"))
                             for link, center, radius in spheres)
        if not self.spheres or any(not sphere[0] for sphere in self.spheres):
            raise ValueError("collision geometry must contain named link spheres")
        self.native = tuple(pyilqr.LinkSphere(*sphere) for sphere in self.spheres)

    @classmethod
    def from_urdf(cls, path):
        spheres = []
        for link in ET.parse(path).getroot().findall("link"):
            for collision in link.findall("collision"):
                sphere = collision.find("./geometry/sphere")
                if sphere is None:
                    raise ValueError(f"robot collision geometry must be spherical: {link.get('name')}")
                origin = collision.find("origin")
                xyz = "0 0 0" if origin is None else origin.get("xyz", "0 0 0")
                spheres.append((link.get("name"), [float(value) for value in xyz.split()], float(sphere.get("radius"))))
        return cls(spheres)


def _policy(disabled_pairs):
    if isinstance(disabled_pairs, (str, Path)):
        disabled_pairs = json.loads(Path(disabled_pairs).read_text())["disabled_pairs"]
    result = set()
    for pair in disabled_pairs:
        first, second = (pair["link1"], pair["link2"]) if isinstance(pair, dict) else pair
        if not isinstance(first, str) or not isinstance(second, str) or first == second:
            raise ValueError("disabled pairs must name two distinct links")
        result.add(tuple(sorted((first, second))))
    return result


def _clearance(robot, q, geometry, pairs, sdf, key):
    function = robot._functions.get(key)
    if function is None:
        function = NativeFunction(robot.raw.sphereClearance(geometry.native, pairs, sdf),
                                  configuration_space=robot.configuration_space, nq=robot.nq)
        function._geometry, function._environment = geometry, sdf
        robot._functions[key] = function
    return function(q)


def self_collision_pairs(geometry, *, disabled_pairs=()):
    """Return enabled cross-link sphere pairs for scalar or batched collision costs."""
    disabled = _policy(disabled_pairs)
    return [(i, j) for i, first in enumerate(geometry.spheres)
            for j in range(i+1, len(geometry.spheres))
            if first[0] != geometry.spheres[j][0]
            and tuple(sorted((first[0], geometry.spheres[j][0]))) not in disabled]


def self_collision(robot, q, geometry, *, disabled_pairs=()):
    """Signed sphere clearances for every enabled cross-link self pair.

    No margin, weight or penalty is applied. Use Problem constraint bounds.
    Missing policies fail; an omitted policy enables all cross-link pairs.
    """
    disabled = _policy(disabled_pairs)
    pairs = self_collision_pairs(geometry, disabled_pairs=disabled)
    if not pairs:
        return cs.MX.zeros(0, 1) if isinstance(q, cs.MX) else np.zeros(0)
    return _clearance(robot, q, geometry, pairs, None,
                      ("self_collision", id(geometry), tuple(sorted(disabled))))


def environment_collision(robot, q, geometry, sdf):
    """Signed sphere-to-SDF clearances, one per robot sphere, evaluated in C++."""
    return _clearance(robot, q, geometry, [], sdf, ("environment_collision", id(geometry), id(sdf)))
