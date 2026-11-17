"""Optional post-solve Viser and ROS2/RViz replay for the Kyon example.

No visualization packages are imported by the headless profiling workflow.
Both viewers use the exported trajectory and collision geometry, in world axes.
"""
import json
import os
import signal
from pathlib import Path
import shutil
import subprocess
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np


def resolve_mesh(filename, urdf):
    if filename.startswith("package://"):
        package, relative = filename[len("package://"):].split("/", 1)
        for directory in urdf.parents:
            candidate = directory / relative if directory.name == package else directory / package / relative
            if candidate.is_file():
                return candidate.resolve()
        try:
            from ament_index_python.packages import get_package_share_directory
            candidate = Path(get_package_share_directory(package)) / relative
            if candidate.is_file():
                return candidate.resolve()
        except (ImportError, LookupError):
            pass
        raise ValueError(f"Cannot resolve mesh {filename}; make the URDF package available")
    path = Path(filename.removeprefix("file://"))
    if not path.is_absolute():
        path = urdf.parent / path
    if not path.is_file():
        raise ValueError(f"Mesh does not exist: {path}")
    return path.resolve()


def visual_urdf(urdf, uri=False):
    """Pelvis-rooted visual model; the free base is published/moved separately."""
    tree = ET.parse(urdf); root = tree.getroot()
    floating = [j for j in root.findall("joint") if j.get("type") == "floating"]
    if len(floating) != 1 or floating[0].find("child").get("link") != "pelvis":
        raise ValueError("This example expects Kyon's single world-to-pelvis floating joint")
    parent = floating[0].find("parent").get("link"); root.remove(floating[0])
    for link in root.findall("link"):
        if link.get("name") == parent:
            root.remove(link)
    omitted = set()
    for link in root.findall("link"):
        for kind in ("visual", "collision"):
            for element in list(link.findall(kind)):
                mesh = element.find("geometry/mesh")
                if mesh is None:
                    continue
                filename = mesh.get("filename")
                try:
                    path = resolve_mesh(filename, urdf)
                except ValueError:
                    if filename.startswith("package://kyon_urdf/"):
                        raise
                    link.remove(element); omitted.add(filename)
                    continue
                mesh.set("filename", path.as_uri() if uri else str(path))
    for filename in sorted(omitted):
        warnings.warn(f"Optional sensor mesh unavailable; omitted from replay: {filename}")
    return ET.tostring(root, encoding="unicode")


def load_replays(args, manifest):
    return {c["mode"]: {key: value.copy() for key, value in np.load(args.output / c["mode"] / "replay.npz").items()}
            for c in manifest["cases"]}


def interpolate(replay, seconds, model):
    import pinocchio as pin
    t = replay["time"]; k = int(np.clip(np.searchsorted(t, seconds, side="right") - 1, 0, len(t)-2))
    fraction = np.clip((seconds-t[k])/(t[k+1]-t[k]), 0, 1)
    return pin.interpolate(model, replay["q"][k], replay["q"][k+1], float(fraction)), k


def viser_view(args, manifest, replays):
    import pinocchio as pin
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf
    from horizon.examples.kyon_collision import sphere_model
    model = pin.buildModelFromUrdf(str(args.urdf)); data = model.createData()
    geometry = sphere_model()
    urdf_path = args.output / "visual_robot.urdf"; urdf_path.write_text(visual_urdf(args.urdf))
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    try:
        server.scene.set_up_direction("+z")
        sample = next(iter(replays.values())); duration = float(sample["time"][-1])
        profile = manifest["cases"][0]["profile"]
        server.scene.add_grid("/ground", width=3, height=3, cell_size=.1, section_size=.5,
                              position=(0, 0, float(sample["feet"][:, 2].min())))
        server.scene.add_icosphere("/obstacle", radius=profile["obstacle_radius"], color=(225, 68, 62),
                                   opacity=.65, position=tuple(profile["obstacle_center"]))
        server.scene.add_icosphere("/target", radius=.015, color=(56, 132, 227), position=tuple(profile["target"]))
        server.scene.add_point_cloud("/fixed_feet", sample["feet"], (60, 125, 220), point_size=.035, precision="float32")
        play = server.gui.add_checkbox("Play", initial_value=True)
        clock = server.gui.add_slider("Time [s]", min=0, max=duration, step=duration/manifest["nodes"], initial_value=0)
        speed = server.gui.add_slider("Playback speed", min=min(.1, args.speed), max=max(2, args.speed), step=.1, initial_value=args.speed)
        selection = server.gui.add_dropdown("Robot", options=["Both", *[f"Collision {m}" for m in replays]], initial_value="Both")
        spheres_visible = server.gui.add_checkbox("Collision spheres", initial_value=args.show_spheres)
        @clock.on_update
        def seek(event):
            if event.client is not None:
                play.value = False
        server.gui.add_markdown("**Green:** collision on · **Orange:** collision off\n\nBlue target and fixed feet; red SDF obstacle. Profiling finished before replay.")
        handles = {}; sphere_handles = {}
        colors = {"on": (0, 145, 124), "off": (225, 132, 52)}
        for mode, replay in replays.items():
            root = server.scene.add_frame(f"/{mode}/robot", show_axes=False)
            visual = yourdfpy.URDF.load(urdf_path, load_meshes=True, build_scene_graph=True)
            robot = ViserUrdf(server, visual, root_node_name=f"/{mode}/robot", mesh_color_override=(*colors[mode], .72))
            path = replay["tool"]; server.scene.add_line_segments(f"/{mode}/path", np.stack((path[:-1], path[1:]), axis=1),
                                                                  colors[mode], line_width=4)
            sphere_handles[mode] = [server.scene.add_icosphere(f"/{mode}/sphere_{i}", radius=s[2], color=colors[mode],
                                                               opacity=.18, subdivisions=1, visible=args.show_spheres)
                                    for i, s in enumerate(geometry)]
            handles[mode] = (root, robot)
        @server.on_client_connect
        def connected(client):
            client.camera.position = (2.2, 2.0, 1.3)
            client.camera.look_at = (0, 0, -.15)
            client.camera.up_direction = (0, 0, 1)
        begin = previous = time.monotonic()
        print(f"Viser: http://127.0.0.1:{args.port} · Ctrl-C to stop", flush=True)
        while not args.visualization_seconds or time.monotonic()-begin < args.visualization_seconds:
            now = time.monotonic()
            if play.value:
                clock.value = (clock.value + (now-previous)*speed.value) % duration
            previous = now
            for mode, replay in replays.items():
                q, _ = interpolate(replay, clock.value, model)
                pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
                root, robot = handles[mode]; pose = data.oMf[model.getFrameId("pelvis")]
                root.position = pose.translation; root.wxyz = pin.Quaternion(pose.rotation).coeffs()[[3, 0, 1, 2]]
                root.visible = selection.value in ("Both", f"Collision {mode}")
                robot.update_cfg(np.array([q[model.joints[model.getJointId(name)].idx_q] for name in robot.get_actuated_joint_names()]))
                for i, (name, offset, _) in enumerate(geometry):
                    handle = sphere_handles[mode][i]
                    handle.visible = spheres_visible.value and root.visible
                    if handle.visible:
                        handle.position = data.oMf[model.getFrameId(name)].act(offset)
            time.sleep(1/30)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def rviz_configuration(output, modes, ground_z):
    import yaml
    displays = [{"Class": "rviz_default_plugins/Grid", "Name": "Grid", "Enabled": True, "Cell Size": .2,
                 "Offset": {"X": 0, "Y": 0, "Z": float(ground_z)}}]
    for mode in modes:
        displays.append({"Class": "rviz_default_plugins/RobotModel", "Name": f"Collision {mode}", "Enabled": True,
                         "Description Source": "Topic", "Description Topic": {"Value": f"/kyon_collision_{mode}/robot_description"},
                         "TF Prefix": mode, "Alpha": .7, "Visual Enabled": True, "Collision Enabled": False})
    displays.append({"Class": "rviz_default_plugins/MarkerArray", "Name": "Scene and paths", "Enabled": True,
                     "Marker Topic": {"Value": "/kyon_collision/markers"}})
    config = {"Panels": [{"Class": "rviz_common/Displays", "Name": "Displays"},
                         {"Class": "rviz_common/Views", "Name": "Views"}],
              "Window Geometry": {"Width": 1280, "Height": 850},
              "Visualization Manager": {"Class": "", "Global Options": {"Fixed Frame": "world", "Background Color": "245; 247; 249"},
                                        "Displays": displays, "Views": {"Current": {"Class": "rviz_default_plugins/Orbit", "Distance": 2.8,
                                                                                  "Pitch": .4, "Yaw": .8, "Focal Point": {"X": 0, "Y": 0, "Z": -.2}}}}}
    path = output / "kyon_collision.rviz"; path.write_text(yaml.safe_dump(config, sort_keys=False)); return path


def ros2_view(args, manifest, replays):
    import pinocchio as pin
    import rclpy
    import yaml
    from geometry_msgs.msg import Point, TransformStamped
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker, MarkerArray
    from tf2_ros import TransformBroadcaster
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from horizon.examples.kyon_collision import sphere_model
    if not shutil.which("ros2"):
        raise ValueError("Source the ROS2 environment before using the ROS2 visualizer")
    description = visual_urdf(args.urdf, uri=True)
    model = pin.buildModelFromUrdf(str(args.urdf)); data = model.createData(); geometry = sphere_model()
    names = [name for i, name in enumerate(model.names) if i > 0 and model.joints[i].nq == 1]
    q_indices = [model.joints[model.getJointId(name)].idx_q for name in names]
    colors = {"on": (0., .57, .49), "off": (.88, .52, .20)}
    processes = []; owned = not rclpy.ok()
    if owned:
        rclpy.init(args=[])
    node = rclpy.create_node("kyon_collision_replay"); broadcaster = TransformBroadcaster(node)
    pubs = {m: node.create_publisher(JointState, f"/kyon_collision_{m}/joint_states", 10) for m in replays}
    marker_pub = node.create_publisher(MarkerArray, "/kyon_collision/markers", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    config = rviz_configuration(args.output, replays, next(iter(replays.values()))["feet"][:, 2].min())
    try:
        for mode in replays:
            params = args.output / mode / "robot_state_publisher.yaml"
            params.write_text(yaml.safe_dump({"/**": {"ros__parameters": {"robot_description": description, "frame_prefix": f"{mode}/"}}}))
            log = (args.output / mode / "robot_state_publisher.log").open("w")
            try:
                processes.append(subprocess.Popen(["ros2", "run", "robot_state_publisher", "robot_state_publisher", "--ros-args",
                                                    "-r", f"__ns:=/kyon_collision_{mode}", "--params-file", str(params)], stdout=log, stderr=subprocess.STDOUT,
                                                    start_new_session=True))
            finally:
                log.close()
        if args.rviz:
            if not shutil.which("rviz2"):
                raise ValueError("Install rviz2 or omit --rviz")
            processes.append(subprocess.Popen(["rviz2", "-d", str(config)], start_new_session=True))
        print(f"ROS2 replay: /kyon_collision_{{on,off}}/joint_states, /tf, /kyon_collision/markers\nRViz config: {config}", flush=True)
        profile = manifest["cases"][0]["profile"]; sample = next(iter(replays.values())); duration = sample["time"][-1]
        begin = time.monotonic()
        def marker(identifier, kind, position, scale, color, ns="scene"):
            m = Marker(); m.header.frame_id = "world"; m.header.stamp = node.get_clock().now().to_msg()
            m.ns = ns; m.id = identifier; m.type = kind; m.action = Marker.ADD
            m.pose.orientation.w = 1.; m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, position)
            m.scale.x, m.scale.y, m.scale.z = map(float, scale)
            m.color.r, m.color.g, m.color.b, m.color.a = map(float, color)
            return m
        while rclpy.ok() and (not args.visualization_seconds or time.monotonic()-begin < args.visualization_seconds):
            for p in processes:
                if p.poll() is not None:
                    raise RuntimeError("A visualization process exited; inspect robot_state_publisher logs / RViz output")
            seconds = ((time.monotonic()-begin)*args.speed) % duration; stamp = node.get_clock().now().to_msg()
            radius = profile["obstacle_radius"]
            markers = [marker(0, Marker.SPHERE, profile["obstacle_center"], (2*radius,)*3, (1., .2, .15, .55)),
                       marker(1, Marker.SPHERE, profile["target"], (.03,)*3, (.2, .5, .95, 1.)),
                       marker(2, Marker.CUBE, (0, 0, sample["feet"][:, 2].min()-.01), (3., 3., .01), (.6, .65, .7, .2))]
            for mode, replay in replays.items():
                q, _ = interpolate(replay, seconds, model)
                message = JointState(); message.header.stamp = stamp; message.name = names
                message.position = [float(q[i]) for i in q_indices]; pubs[mode].publish(message)
                pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
                pose = data.oMf[model.getFrameId("pelvis")]; transform = TransformStamped()
                transform.header.stamp = stamp; transform.header.frame_id = "world"; transform.child_frame_id = f"{mode}/pelvis"
                transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = map(float, pose.translation)
                quat = pin.Quaternion(pose.rotation).coeffs()
                transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = map(float, quat)
                broadcaster.sendTransform(transform)
                path = marker(0, Marker.LINE_STRIP, (0, 0, 0), (.008, 0, 0), (*colors[mode], 1.), ns=mode)
                path.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in replay["tool"]]; markers.append(path)
                if args.show_spheres:
                    for i, (name, offset, r) in enumerate(geometry):
                        markers.append(marker(i+1, Marker.SPHERE, data.oMf[model.getFrameId(name)].act(offset), (2*r,)*3,
                                              (*colors[mode], .2), ns=mode))
            feet = marker(3, Marker.SPHERE_LIST, (0, 0, 0), (.03,)*3, (.2, .5, .95, 1.))
            feet.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in sample["feet"]]; markers.append(feet)
            marker_pub.publish(MarkerArray(markers=markers)); rclpy.spin_once(node, timeout_sec=0)
            time.sleep(1/30)
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for p in processes:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
        node.destroy_node()
        if owned:
            rclpy.shutdown()


def visualize(args, manifest):
    replays = load_replays(args, manifest)
    if args.visualizer == "viser":
        viser_view(args, manifest, replays)
    else:
        ros2_view(args, manifest, replays)
