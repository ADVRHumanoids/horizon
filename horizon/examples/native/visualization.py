"""Optional trajectory viewers shared by the native walkthrough.

The numerical examples do not import Viser, yourdfpy or matplotlib until the
user requests visualization. Robot configurations remain in Horizon's native
ordering; joint names map them to the viewer's URDF ordering.
"""
from pathlib import Path
import copy
import socket
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np


def visualize_drive(trajectory, dt, options, *, target=None):
    """Plot the differential-drive path and heading over time."""
    import matplotlib.pyplot as plt

    states = np.asarray(trajectory)
    time_axis = np.arange(states.shape[1]) * dt
    fig, (path, heading) = plt.subplots(1, 2, figsize=(9, 4))
    path.plot(states[0], states[1], "-o", markersize=2, label="trajectory")
    path.scatter([states[0, 0], states[0, -1]], [states[1, 0], states[1, -1]],
                 c=["tab:green", "tab:red"], label="start / finish")
    if target is not None:
        path.scatter([target[0]], [target[1]], marker="x", color="tab:orange", label="target")
    path.set(xlabel="x [m]", ylabel="y [m]", title="Differential-drive path")
    path.axis("equal")
    path.grid(True)
    path.legend()
    heading.plot(time_axis, states[2])
    heading.set(xlabel="Time [s]", ylabel="Heading [rad]", title="Orientation")
    heading.grid(True)
    fig.tight_layout()
    if options.visualization_seconds:
        plt.show(block=False)
        time.sleep(options.visualization_seconds)
        plt.close(fig)
    else:
        plt.show()


def _viewer_urdf(path, directory):
    """Prepare a visual URDF, using collision spheres when meshes are missing.

    ViserUrdf handles scalar joints, while a floating base is moved separately
    with its native frame pose. Mesh paths are made absolute before writing the
    temporary URDF so package:// resources work independently of the cwd.
    """
    from horizon.examples.kyon_collision_viewer import resolve_mesh

    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    floating = [joint for joint in root.findall("joint") if joint.get("type") == "floating"]
    if len(floating) > 1:
        raise ValueError("Viser replay supports at most one floating base")
    base_frame = None
    if floating:
        joint = floating[0]
        base_frame = joint.find("child").get("link")
        parent = joint.find("parent").get("link")
        root.remove(joint)
        for link in root.findall("link"):
            if link.get("name") == parent:
                root.remove(link)
                break

    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is not None:
                try:
                    mesh.set("filename", str(resolve_mesh(mesh.get("filename"), path)))
                except (ValueError, OSError):
                    link.remove(visual)
        # The bundled whole-body URDF includes sphere approximations even when
        # external visual meshes are unavailable on the user's machine.
        if not link.findall("visual"):
            for collision in link.findall("collision"):
                if collision.find("geometry/sphere") is not None:
                    visual = ET.SubElement(link, "visual")
                    for child in collision:
                        if child.tag in ("origin", "geometry"):
                            visual.append(copy.deepcopy(child))
        for collision in list(link.findall("collision")):
            link.remove(collision)

    prepared = Path(directory) / "viewer.urdf"
    ET.ElementTree(root).write(prepared, encoding="unicode")
    return prepared, base_frame


def visualize_robot(robot, urdf, trajectory, dt, options, *, target=None,
                    collision_spheres=None, obstacle=None, mesh=None):
    """Replay a solved robot trajectory in Viser until interrupted.

    ``obstacle`` is (center, radius); ``mesh`` is an STL path in world frame.
    A finite ``--visualization-seconds`` is useful for unattended runs.
    """
    try:
        import viser
        import yourdfpy
        from viser.extras import ViserUrdf
    except ImportError as error:
        raise RuntimeError("Robot visualization requires 'viser' and 'yourdfpy'.") from error

    states = np.asarray(trajectory)
    if states.ndim != 2 or states.shape[0] != robot.nq or states.shape[1] < 2:
        raise ValueError("Expected a configuration trajectory with shape (nq, nodes + 1)")
    with tempfile.TemporaryDirectory(prefix="horizon-viewer-") as directory:
        prepared, base_frame = _viewer_urdf(urdf, directory)
        port = options.viewer_port
        if port == 0:
            # Viser reports port 0 literally, so choose a free loopback port first.
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
        server = viser.ViserServer(host="127.0.0.1", port=port)
        try:
            server.scene.set_up_direction("+z")
            contacts = None
            ground_height = -.12
            if base_frame is not None:
                contacts = robot.framePositions([f"contact_{i}" for i in range(1, 5)], states[:, 0])
                ground_height = float(np.min(contacts[2])) - .01
            server.scene.add_grid("/ground", width=4, height=4, cell_size=.2,
                                  section_size=1, position=(0, 0, ground_height))
            if contacts is not None:
                for i, point in enumerate(contacts.T):
                    server.scene.add_icosphere(f"/contact/{i}", radius=.025,
                                               color=(40, 150, 220), position=tuple(point))
            root = server.scene.add_frame("/robot", show_axes=False)
            visual = yourdfpy.URDF.load(prepared, load_meshes=True, build_scene_graph=True)
            viewer = ViserUrdf(server, visual, root_node_name="/robot")
            joint_indices = [robot.joint_iq(name) for name in viewer.get_actuated_joint_names()]

            # Show the tool's knot positions as a path independently of playback.
            tool = "tool" if base_frame is None else "dagana_1_base"
            points = np.array([robot.framePositions([tool], q).reshape(3) for q in states.T])
            server.scene.add_line_segments("/tool_path", np.stack((points[:-1], points[1:]), axis=1),
                                           (40, 150, 220), line_width=3)
            server.gui.add_markdown("Blue: tool path and initial contacts · Orange: target · Red: obstacle")

            if target is not None:
                server.scene.add_icosphere("/target", radius=.035, color=(240, 100, 35),
                                           position=tuple(np.asarray(target).reshape(3)))
            if obstacle is not None:
                center, radius = obstacle
                server.scene.add_icosphere("/obstacle", radius=radius, color=(220, 55, 60),
                                           opacity=.6, position=tuple(center))
            if mesh is not None:
                import trimesh
                obstacle_mesh = trimesh.load(str(mesh), force="mesh")
                obstacle_mesh.visual.face_colors = (220, 55, 60, 150)
                server.scene.add_mesh_trimesh("/obstacle_mesh", obstacle_mesh)

            spheres = []
            if collision_spheres is not None:
                for i, (_, _, radius) in enumerate(collision_spheres.spheres):
                    spheres.append(server.scene.add_icosphere(
                        f"/collision/{i}", radius=radius, color=(55, 160, 220), opacity=.2,
                        subdivisions=1, visible=len(collision_spheres.spheres) <= 2))
                show_spheres = server.gui.add_checkbox("Collision spheres", initial_value=len(spheres) <= 2)
            else:
                show_spheres = None

            duration = dt * (states.shape[1] - 1)
            play = server.gui.add_checkbox("Play", initial_value=True)
            clock = server.gui.add_slider("Time [s]", min=0, max=duration, step=dt / 10, initial_value=0)
            @clock.on_update
            def seek(event):
                if event.client is not None:
                    play.value = False

            @server.on_client_connect
            def camera(client):
                client.camera.position = (2.8, 2.2, 2.0)
                client.camera.look_at = (0.7, 0, 0) if base_frame is None else (0, 0, 0)
                client.camera.up_direction = (0, 0, 1)

            start = previous = time.monotonic()
            print(f"Viser: http://127.0.0.1:{server.get_port()} (Ctrl-C to stop)", flush=True)
            while not options.visualization_seconds or time.monotonic() - start < options.visualization_seconds:
                now = time.monotonic()
                if play.value:
                    clock.value = (clock.value + now - previous) % duration
                previous = now
                # Display a knot exactly: no interpolation across quaternion coordinates.
                k = min(round(clock.value / dt), states.shape[1] - 1)
                q = states[:, k]
                if base_frame is not None:
                    pose = np.asarray(robot.framePose(base_frame, q)).reshape(7)
                    root.position = pose[:3]
                    root.wxyz = pose[[6, 3, 4, 5]]
                viewer.update_cfg(q[joint_indices])
                if spheres:
                    centers = np.asarray(robot.sphereCenters(q, collision_spheres.native)).reshape(3, -1)
                    for i, handle in enumerate(spheres):
                        handle.visible = show_spheres.value
                        if handle.visible:
                            handle.position = centers[:, i]
                time.sleep(1 / 30)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                server.stop()
            except RuntimeError as error:
                # Viser can finish closing its event loop during stop().
                if str(error) != "Event loop is closed":
                    raise
