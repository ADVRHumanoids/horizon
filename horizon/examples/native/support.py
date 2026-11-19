"""Small shared setup and checks for the floating-base Kyon lessons."""
import argparse
from pathlib import Path

import numpy as np


def standing_configuration(robot, *, collision_free=False):
    q = robot.configuration_space.neutral()
    def set_joint(name, value):
        q[robot.joint_iq(name)] = value
    for leg in range(1, 5):
        sign = 1 if leg % 2 else -1
        set_joint(f"hip_pitch_{leg}", -.7 * sign)
        set_joint(f"knee_pitch_{leg}", 1.4 * sign)
        set_joint(f"hip_roll_{leg}", .15 if leg in (1, 4) else -.15)
    for arm in (1, 2):
        sign = 1 if arm == 1 else -1
        for joint, value in (("shoulder_pitch", 1.3), ("elbow_pitch", 2.5), ("wrist_pitch", .45)):
            set_joint(f"{joint}_{arm}", value * sign)
    if collision_free:
        for name, value in {"shoulder_yaw_1": .8, "shoulder_pitch_1": 1., "elbow_pitch_1": 2.,
                            "wrist_pitch_1": 1.2, "wrist_yaw_1": .5, "shoulder_yaw_2": 1.,
                            "shoulder_pitch_2": -1.15, "elbow_pitch_2": -1.35,
                            "wrist_pitch_2": -.65, "wrist_yaw_2": -.75}.items():
            set_joint(name, value)
    return q


def motion_error(robot, frames, trajectory, initial, target):
    positions = np.array([robot.framePositions(frames, q).T for q in trajectory.T])
    error = np.linalg.norm(positions[-1, 4] - target)
    drift = np.linalg.norm(positions[:, :4] - initial[:4], axis=-1).max()
    return error, drift


def contact_equilibrium(robot, q, contacts, selection):
    """Find one static combination of joint torques and point-contact forces."""
    jacobian = np.vstack([robot.jacobian(frame, q)[:3] for frame in contacts])
    return np.linalg.lstsq(np.column_stack((selection, jacobian.T)), robot.gravity(q), rcond=None)[0]


def parse_options(nodes=30, iterations=50, dt=0.1, *, arm=False, default_urdf=None,
                  action_choices=None, default_action=None):
    """Parse the options shared by every lesson, with lesson-specific defaults."""
    root = Path(__file__).resolve().parents[2] / "cpp"
    parser = argparse.ArgumentParser(description=__import__("__main__").__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-N", "--nodes", type=int, default=nodes, help="number of shooting intervals")
    parser.add_argument("--threads", type=int, default=1, help="iLQR worker threads")
    parser.add_argument("--max-iterations", type=int, default=iterations, help="solver iteration limit")
    parser.add_argument("--dt", type=float, default=dt, help="interval duration in seconds")
    default_model = default_urdf or root / ("examples/data/planar_arm.urdf" if arm else "tests/data/kyon_full_spheres.urdf")
    parser.add_argument("--urdf", type=Path, default=default_model, help="robot model override")
    parser.add_argument("--policy", type=Path, default=root / "tests/data/kyon_full_disabled_collisions.json", help="disabled self-collision pairs override")
    parser.add_argument("--sdf", type=Path, default=root / "examples/data/obstacle.vdb", help="OpenVDB field override")
    parser.add_argument("--globalization", choices=("legacy", "lm_merit", "filter_sqp"), default="legacy", help="iLQR globalization strategy (lesson 11)")
    parser.add_argument("--quiet", action="store_true", help="suppress iteration reports")
    parser.add_argument("--visualize", action="store_true", help="show the solved trajectory")
    parser.add_argument("--mesh", type=Path, help="mesh to display for a custom environment SDF")
    parser.add_argument("--viewer-port", type=int, default=8080, help="Viser server port (0 chooses an available port)")
    if action_choices is not None:
        parser.add_argument("--action", choices=action_choices, default=default_action)
    args = parser.parse_args()
    if (args.nodes < 2 or args.threads < 0 or args.max_iterations < 1 or not 0 <= args.viewer_port <= 65535
            or (args.dt is not None and (not np.isfinite(args.dt) or args.dt <= 0))):
        parser.error("nodes >= 2, threads >= 0, max-iterations >= 1, dt > 0, a valid port, and visualization-seconds >= 0 are required")
    if args.dt is None:
        args.dt = 2. / args.nodes
    return args


def configure_logging(solver, options):
    """The Python iLQR wrapper reports iterations unless quiet is requested."""
    if options.quiet:
        solver.set_iteration_callback(lambda _: True)
