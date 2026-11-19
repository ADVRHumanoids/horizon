"""Robot presets, CLI and validation for the native Pinocchio example."""
import argparse
import os
from pathlib import Path
import numpy as np
from horizon.solvers import pyilqr


def initial_configuration(urdf, robot):
    import pinocchio as pin
    # Python Pinocchio is used only to prepare the initial configuration.
    model = pin.buildModelFromUrdf(str(urdf))
    q0 = robot.configuration_space.neutral()
    for leg in range(1, 5):
        sign = 1 if leg % 2 else -1
        for joint, value in ((f"hip_pitch_{leg}", -.7*sign),
                             (f"knee_pitch_{leg}", 1.4*sign),
                             (f"hip_roll_{leg}", .15 if leg in (1, 4) else -.15)):
            q0[model.joints[model.getJointId(joint)].idx_q] = value
    for arm in (1, 2):
        sign = 1 if arm == 1 else -1
        for joint, value in ((f"shoulder_pitch_{arm}", 1.3*sign),
                             (f"elbow_pitch_{arm}", 2.5*sign),
                             (f"wrist_pitch_{arm}", .45*sign)):
            q0[model.joints[model.getJointId(joint)].idx_q] = value

    return q0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=os.environ.get("KYON_URDF"))
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--solver", choices=("ilqr", "ipopt"), default="ilqr")
    parser.add_argument("--map-threads", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not args.urdf or not args.urdf.is_file() or args.N < 4 or args.threads < 0 or args.map_threads < 1:
        parser.error("Provide an existing --urdf (or KYON_URDF), N >= 4, and threads >= 0")
    if not hasattr(pyilqr, "PinocchioModel"):
        parser.error("Rebuild pyilqr with HORIZON_WITH_PINOCCHIO=ON")
    return args


def frame_rows(values, count):
    """Convert native SoA frame coordinates to one xyz row per frame."""
    return np.asarray(values).reshape(3, count).T


def symbolic_frame_columns(values, count):
    """Convert the same layout to symbolic xyz columns."""
    import casadi as cs
    return cs.reshape(values, count, 3).T


def validate_motion(solver, positions):
    solution = solver.getSolutionDict()
    points = np.array([frame_rows(positions.evaluate(q), 5) for q in solution["q"].T])
    drift = np.linalg.norm(points[:, :4] - points[0, :4], axis=-1).max()
    target = solver.prb.getParameters("tool_target").getValues()[:, -1]
    error = np.linalg.norm(points[-1, 4] - target)
    return error, drift


def solver_options(solver_type, N, threads, ipopt_options):
    if solver_type == "ipopt":
        options = {"ipopt.max_iter": 200, "ipopt.tol": 1e-11, "ipopt.print_level": 0,
               "ipopt.mu_strategy": "adaptive", "ipopt.mu_min": 1e-13, "print_time": False}
        options.update(ipopt_options or {})
        return options
    return {"ilqr.max_iter": 80, "ilqr.n_threads": threads,
            "ilqr.enable_gn": True, "ilqr.merit_der_threshold": 1e-12,
            "ilqr.huu_reg": .001/N, "ilqr.constr_decomp_type": "svd",
            "ilqr.constraint_violation_threshold": 1e-7,
            "ilqr.defect_norm_threshold": 1e-7}
