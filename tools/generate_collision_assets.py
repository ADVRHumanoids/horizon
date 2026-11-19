#!/usr/bin/env python3
"""Generate a sphere URDF and MoveIt disabled-collision JSON from xacro."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def resolve_meshes(root, resource_root, temporary):
    """Bake URDF mesh scale and collision origin into link-local meshes for FOAM."""
    from ament_index_python.packages import get_package_share_directory
    from foam.utility import load_mesh_file
    from trimesh.transformations import euler_matrix

    for index, collision in enumerate(root.findall('./link/collision')):
        origin = collision.find('origin')
        if origin is None:
            origin = ET.SubElement(collision, 'origin')
        origin.attrib.setdefault('xyz', '0 0 0')
        origin.attrib.setdefault('rpy', '0 0 0')
        mesh = collision.find('./geometry/mesh')
        if mesh is None:
            continue
        filename = mesh.get('filename', '')
        if filename.startswith('package://'):
            package, separator, relative = filename[len('package://'):].partition('/')
            if not separator:
                raise ValueError(f'invalid package mesh URI: {filename}')
            path = Path(get_package_share_directory(package)) / relative
        else:
            path = Path(filename)
            if not path.is_absolute():
                path = resource_root / path
        if not path.is_file():
            raise FileNotFoundError(f'collision mesh does not exist: {path}')
        scale = [float(x) for x in mesh.get('scale', '1 1 1').split()]
        xyz = [float(x) for x in origin.get('xyz').split()]
        rpy = [float(x) for x in origin.get('rpy').split()]
        if any(len(values) != 3 for values in (scale, xyz, rpy)):
            raise ValueError(f'invalid collision mesh transform: {filename}')
        transformed = load_mesh_file(path).copy()
        transformed.apply_scale(scale)
        transform = euler_matrix(*rpy, axes='sxyz')
        transform[:3, 3] = xyz
        transformed.apply_transform(transform)
        destination = temporary / f'collision-mesh-{index}.stl'
        transformed.export(destination)
        mesh.set('filename', str(destination))
        mesh.set('scale', '1 1 1')
        origin.set('xyz', '0 0 0')
        origin.set('rpy', '0 0 0')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sphere-urdf', required=True, type=Path)
    parser.add_argument('--collision-json', required=True, type=Path)
    parser.add_argument('--collision-generator', default='moveit_compute_default_collisions')
    parser.add_argument('--trials', type=int, default=75000)
    parser.add_argument('--resource-root', type=Path, default=Path.cwd(),
                        help='base directory for relative mesh filenames')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--branch', type=int, default=8)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--foam-script', type=Path, help='path to FOAM generate_sphere_urdf.py')
    parser.add_argument('xacro_command', nargs=argparse.REMAINDER,
                        help='xacro command and arguments, preceded by --')
    args = parser.parse_args()
    command = args.xacro_command
    if command and command[0] == '--':
        command = command[1:]
    if not command or args.trials < 1 or args.threads < 1 or args.branch < 1 or args.depth < 0:
        parser.error('provide a xacro command and positive trials, threads and branch')
    if args.sphere_urdf.resolve() == args.collision_json.resolve():
        parser.error('output paths must be different')

    spec = importlib.util.find_spec('foam')
    if args.foam_script:
        foam_script = args.foam_script.resolve()
    elif spec and spec.origin:
        foam_script = Path(spec.origin).resolve().parent.parent / 'scripts/generate_sphere_urdf.py'
    else:
        raise RuntimeError('FOAM is not installed; provide --foam-script and install its Python dependencies')
    if not foam_script.is_file():
        raise FileNotFoundError(f'FOAM generator not found: {foam_script}')

    with tempfile.TemporaryDirectory(prefix='horizon-collision-assets-') as temp:
        temp = Path(temp)
        xml = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
        root = ET.fromstring(xml)
        if root.tag != 'robot':
            raise ValueError('xacro output is not a robot URDF')
        collision_counts = {link.attrib['name']: len(link.findall('collision'))
                            for link in root.findall('link') if link.findall('collision')}
        resolve_meshes(root, args.resource_root.resolve(), temp)
        # Preserve existing spheres exactly, including nonzero collision origins.
        # FOAM's primitive path does not apply those transforms to spheres.
        existing_spheres = {}
        for link in root.findall('link'):
            for collision in list(link.findall('collision')):
                if collision.find('./geometry/sphere') is not None:
                    existing_spheres.setdefault(link.attrib['name'], []).append(copy.deepcopy(collision))
                    link.remove(collision)
        prepared = temp / 'input.urdf'
        ET.ElementTree(root).write(prepared, encoding='utf-8', xml_declaration=True)
        sphere_file = temp / 'spheres.urdf'
        subprocess.run([sys.executable, str(foam_script), str(prepared),
                        '--output', str(sphere_file), '--database', str(temp / 'foam.json'),
                        '--threads', str(args.threads), '--branch', str(args.branch),
                        '--depth', str(args.depth), '--volume_heuristic_ratio', '4'], check=True)
        # FOAM records its temporary input path on the robot; do not ship it.
        sphere_tree = ET.parse(sphere_file)
        sphere_tree.getroot().attrib.pop('path', None)
        for link in sphere_tree.findall('./link'):
            for collision in existing_spheres.get(link.attrib['name'], []):
                link.append(collision)
        sphere_tree.write(sphere_file, encoding='utf-8', xml_declaration=True)
        collisions = sphere_tree.findall('./link/collision')
        if not collisions or any(item.find('./geometry/sphere') is None for item in collisions):
            raise ValueError('FOAM did not produce exclusively sphere collision geometry')
        converted_counts = {link.attrib['name']: len(link.findall('collision'))
                            for link in sphere_tree.findall('./link')}
        missing = [name for name, count in collision_counts.items()
                   if converted_counts.get(name, 0) < count]
        if missing:
            raise ValueError('FOAM omitted collision geometry on: ' + ', '.join(missing))
        json_file = temp / 'disabled_collisions.json'
        subprocess.run([args.collision_generator, '--urdf_path', str(sphere_file),
                        '--json_path', str(json_file), '--num_trials', str(args.trials)], check=True)
        data = json.loads(json_file.read_text())
        if data.get('format') != 'horizon.disabled_collision_pairs.v1' or not isinstance(data.get('disabled_pairs'), list):
            raise ValueError('collision generator produced an invalid policy')
        for source, destination in ((sphere_file, args.sphere_urdf), (json_file, args.collision_json)):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        print(f'Generated {len(collisions)} spheres in {args.sphere_urdf} and '
              f'{len(data["disabled_pairs"])} disabled pairs in {args.collision_json}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, ET.ParseError) as error:
        print(f'collision asset generation failed: {error}', file=sys.stderr)
        sys.exit(1)
