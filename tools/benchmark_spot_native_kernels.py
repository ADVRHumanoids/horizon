"""Compare native and CasADi Spot kernels through the same Horizon interface."""
import argparse
from pathlib import Path
import statistics
import time

import casadi as cs
import numpy as np
import pinocchio as pin
from casadi_kin_dyn import pycasadi_kin_dyn as ck

from horizon.solvers import pyilqr
from horizon.utils.kin_dyn import InverseDynamics


def timings(function, samples, iterations):
    for _ in range(20):
        function()
    measurements = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        measurements.append((time.perf_counter_ns() - start) / iterations / 1000)
    return statistics.median(measurements), min(measurements), max(measurements)


def input_timings(function, inputs, samples, iterations):
    for index in range(64):
        function(inputs[index % len(inputs)])
    measurements = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        for index in range(iterations):
            function(inputs[index % len(inputs)])
        measurements.append((time.perf_counter_ns() - start) / iterations / 1000)
    return statistics.median(measurements), min(measurements), max(measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if min(args.samples, args.iterations) < 1:
        parser.error("sample and iteration counts must be positive")
    urdf = Path(__file__).resolve().parents[1] / "horizon/examples/urdf/spot.urdf"
    robot = pyilqr.PinocchioModel(str(urdf))
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    kindyn = ck.CasadiKinDyn(urdf.read_text())
    frames = ("lf_foot", "rf_foot", "lh_foot", "rh_foot")
    random = np.random.default_rng(42)
    q = pin.integrate(model, robot.configuration_space.neutral(),
                      random.normal(0, .3, robot.nv))
    v = random.normal(0, .3, robot.nv)
    a = random.normal(0, .3, robot.nv)
    forces = random.normal(0, 40, 3 * len(frames))
    configuration_directions = np.empty((robot.nq, robot.nv))
    for column in range(robot.nv):
        direction = np.zeros(robot.nv)
        direction[column] = 1e-6
        configuration_directions[:, column] = (
            pin.integrate(model, q, direction) - pin.integrate(model, q, -direction)
        ) / (2e-6)

    velocity_input = cs.SX.sym("velocity_input", robot.nq + robot.nv)
    velocity_result = kindyn.frameVelocity(
        frames[0], ck.CasadiKinDyn.LOCAL_WORLD_ALIGNED)(
            q=velocity_input[:robot.nq], qdot=velocity_input[robot.nq:])
    velocity_expression = cs.vertcat(velocity_result["ee_vel_linear"],
                                     velocity_result["ee_vel_angular"])
    contact_input = cs.SX.sym("contact_input", robot.nq + 2 * robot.nv + len(forces))
    contact_model = InverseDynamics(kindyn, frames, ck.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
    force_offset = robot.nq + 2 * robot.nv
    contact_expression = contact_model.call(
        contact_input[:robot.nq], contact_input[robot.nq:robot.nq + robot.nv],
        contact_input[robot.nq + robot.nv:force_offset],
        {frame: contact_input[force_offset + 3 * index:force_offset + 3 * index + 3]
         for index, frame in enumerate(frames)})
    state = cs.SX.sym("x", robot.nq + robot.nv)
    control = cs.SX.sym("u", robot.nv + len(forces))
    legacy_input = cs.vertcat(state, control)
    legacy_velocity = cs.substitute(velocity_expression[:3], velocity_input,
                                    state)
    legacy_torque = cs.substitute(contact_expression[:6], contact_input,
                                  legacy_input)
    factory_cases = {}
    for name, expression in (("foot_velocity", legacy_velocity),
                             ("contact_inverse_dynamics", legacy_torque)):
        value_function = cs.Function(name + "_legacy", [state, control], [expression],
                                     ["x", "u"], ["h"])
        derivative_function = value_function.factory(
            name + "_legacy_jac", ["x", "u"], ["jac:h:x", "jac:h:u"])
        factory_cases[name] = (value_function, derivative_function)
    cases = (
        ("foot_velocity", robot.frameVelocity(frames[0], pyilqr.LOCAL_WORLD_ALIGNED),
         pyilqr.CasadiFunction(cs.Function("foot_velocity", [velocity_input], [velocity_expression]), False),
         pyilqr.CasadiFunction(cs.Function("foot_velocity_selected", [velocity_input], [velocity_expression[:3]]), False),
         np.r_[q, v], 3),
        ("contact_inverse_dynamics", robot.inverseDynamics(list(frames)),
         pyilqr.CasadiFunction(cs.Function("contact_inverse_dynamics", [contact_input], [contact_expression]), False),
         pyilqr.CasadiFunction(cs.Function("contact_inverse_selected", [contact_input], [contact_expression[:6]]), False),
         np.r_[q, v, a, forces], 6),
    )
    print(f"samples={args.samples} iterations={args.iterations}; times in microseconds per call")
    print(f"nq={robot.nq} nv={robot.nv}; pyilqr={pyilqr.__file__}")
    for name, native, legacy, selected, inputs, rows in cases:
        native_value, native_jac = native.valueAndJacobian(inputs)
        legacy_value, legacy_jac = legacy.valueAndJacobian(inputs)
        selected_value, selected_jac = selected.valueAndJacobian(inputs)
        native_tangent = np.column_stack((native_jac[:, :robot.nq] @ configuration_directions,
                                          native_jac[:, robot.nq:]))
        legacy_tangent = np.column_stack((legacy_jac[:, :robot.nq] @ configuration_directions,
                                          legacy_jac[:, robot.nq:]))
        value_error = np.max(np.abs(native_value - legacy_value))
        jacobian_error = np.max(np.abs(native_tangent - legacy_tangent))
        np.testing.assert_allclose(native_value, legacy_value, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(native_tangent, legacy_tangent, atol=2e-7, rtol=2e-7)
        np.testing.assert_allclose(native_value[:rows], selected_value, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(native_tangent[:rows, :robot.nv],
                                   selected_jac[:, :robot.nq] @ configuration_directions,
                                   atol=2e-7, rtol=2e-7)
        np.testing.assert_allclose(native_tangent[:rows, robot.nv:],
                                   selected_jac[:, robot.nq:], atol=2e-7, rtol=2e-7)
        value_function, derivative_function = factory_cases[name]
        state_values = cs.DM(np.r_[q, v])
        control_values = cs.DM(np.r_[a, forces])
        factory_value = np.asarray(value_function(state_values, control_values)).ravel()
        factory_state_jac, factory_control_jac = derivative_function(state_values, control_values)
        factory_jac = np.column_stack((np.asarray(factory_state_jac),
                           np.asarray(factory_control_jac)))
        np.testing.assert_allclose(factory_value, native_value[:rows], atol=1e-9, rtol=1e-9)
        relevant_jac = factory_jac[:, :len(inputs)]
        np.testing.assert_allclose(relevant_jac[:, :robot.nq] @ configuration_directions,
                       native_tangent[:rows, :robot.nv], atol=2e-7, rtol=2e-7)
        np.testing.assert_allclose(relevant_jac[:, robot.nq:],
                       native_tangent[:rows, robot.nv:], atol=2e-7, rtol=2e-7)
        print(f"\n{name}: max value error={value_error:.3g}, tangent Jacobian error={jacobian_error:.3g}")
        varied_inputs = [inputs.copy() for _ in range(16)]
        for index, value in enumerate(varied_inputs):
            value[robot.nq] += index * .001
        for input_label, input_values in (("same", [inputs]), ("varying", varied_inputs)):
            for operation in ("evaluate", "jacobian", "valueAndJacobian"):
                for label, function in (("native_full", native), ("casadi_full", legacy),
                                        ("casadi_selected", selected)):
                    result = input_timings(getattr(function, operation), input_values,
                                           args.samples, args.iterations)
                    print(f"  {input_label:7} {operation:18} {label:15} "
                          f"median={result[0]:8.3f} range=[{result[1]:.3f}, {result[2]:.3f}]")
        print("  Legacy factory calls below use CasADi's Python binding, not Horizon's C++ wrapper:")
        for label, function in (("value", value_function), ("jacobian", derivative_function)):
            result = timings(lambda: function(state_values, control_values), args.samples, args.iterations)
            print(f"    {label:8} median={result[0]:8.3f} range=[{result[1]:.3f}, {result[2]:.3f}]")
    zero = np.zeros(robot.nv)
    frame = model.getFrameId(frames[0])
    raw_cases = (
        ("pin_fk_velocity", lambda: pin.forwardKinematics(model, data, q, v)),
        ("pin_fk_derivatives", lambda: pin.computeForwardKinematicsDerivatives(model, data, q, v, zero)),
        ("pin_frame_velocity_derivatives_local", lambda: pin.getFrameVelocityDerivatives(model, data, frame, pin.LOCAL)),
        ("pin_rnea", lambda: pin.rnea(model, data, q, v, a)),
        ("pin_rnea_derivatives", lambda: pin.computeRNEADerivatives(model, data, q, v, a)),
    )
    print("\nPinocchio primitive timings include Python binding and output conversion overhead:")
    for name, function in raw_cases:
        result = timings(function, args.samples, args.iterations)
        print(f"  {name:36} median={result[0]:8.3f} range=[{result[1]:.3f}, {result[2]:.3f}]")


if __name__ == "__main__":
    main()