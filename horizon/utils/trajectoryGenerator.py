from scipy.interpolate import BPoly, CubicSpline
import numpy as np


class TrajectoryGenerator:

    def __init__(self):
        pass

    def bell_trj(self, tau):
        return 64. * tau ** 3 * (1 - tau) ** 3

    def sin_trj(self, tau):
        return np.sin(tau * np.pi)

    def bezier_trj(self, tau):
        P0 = 1
        P1 = 3
        P2 = 2

        fun = (1 - tau) ** 2 * P0 + 2 * (1 - tau) * tau * P1 + tau * 2 * P2
        return fun

    def compute_polynomial_trajectory(self, k_start, nodes, nodes_duration, p_start, p_goal, clearance, dim=None):
        if dim is None:
            dim = [0, 1, 2]

        traj_array = np.zeros(len(nodes))
        start = p_start[dim]
        goal = p_goal[dim]

        index = 0
        for k in nodes:
            tau = (k - k_start) / nodes_duration
            trj = self.bezier_trj(tau) * clearance
            trj += (1 - tau) * start + tau * goal
            traj_array[index] = trj
            index = index + 1

        return np.array(traj_array)

    def from_derivatives(self, nodes, p_start, p_goal, clearance, derivatives=None, second_der=None, third_der=None):
        if derivatives is None:
            derivatives = [None] * nodes
        if second_der is None:
            second_der = [None] * nodes
        if third_der is None:
            third_der = [None] * nodes

        cxi = [0, 0.5, 1]

        if p_start >= p_goal:
            cyi = [p_start, p_start + clearance, p_goal]
        else:
            cyi = [p_start, p_goal + clearance, p_goal]

        xcurve = np.linspace(0, 1, nodes)

        yder = []
        for i, (d1, d2, d3) in enumerate(zip(derivatives, second_der, third_der)):
            constraints = [cyi[i]]
            if d1 is not None:
                constraints.append(d1)
            if d2 is not None:
                constraints.append(d2)
            if d3 is not None:
                constraints.append(d3)
            yder.append(constraints)

        bpoly = BPoly.from_derivatives(cxi, yder)
        y_bpoly = bpoly(xcurve)

        return y_bpoly

    def derivative_of_trajectory(self, nodes, p_start, p_goal, clearance, derivatives=None, second_der=None, third_der=None):
        if derivatives is None:
            derivatives = [None] * nodes
        if second_der is None:
            second_der = [None] * nodes
        if third_der is None:
            third_der = [None] * nodes

        cxi = [0, 0.5, 1]

        if p_start >= p_goal:
            cyi = [p_start, p_start + clearance, p_goal]
        else:
            cyi = [p_start, p_goal + clearance, p_goal]

        xcurve = np.linspace(0, 1, nodes)

        yder = []
        for i, (d1, d2, d3) in enumerate(zip(derivatives, second_der, third_der)):
            constraints = [cyi[i]]
            if d1 is not None:
                constraints.append(d1)
            if d2 is not None:
                constraints.append(d2)
            if d3 is not None:
                constraints.append(d3)
            yder.append(constraints)

        bpoly = BPoly.from_derivatives(cxi, yder)
        # Compute the first derivative of the BPoly object
        bpoly_derivative = bpoly.derivative()

        # Evaluate the derivative at the given points
        y_bpoly_derivative = bpoly_derivative(xcurve)

        return y_bpoly_derivative

    def second_derivative_of_trajectory(self, nodes, p_start, p_goal, clearance, derivatives=None, second_der=None, third_der=None):
        """
        Compute the second derivative of the trajectory.

        Parameters:
        - nodes: int, number of points in the trajectory
        - p_start: float, start position
        - p_goal: float, goal position
        - clearance: float, distance above or below the straight line
        - derivatives: list of first derivative constraints or `None`
        - second_der: list of second derivative constraints or `None`
        - third_der: list of third derivative constraints or `None`

        Returns:
        - y_bpoly_second_derivative: np.ndarray, second derivative values
        """
        if derivatives is None:
            derivatives = [None] * nodes
        if second_der is None:
            second_der = [None] * nodes
        if third_der is None:
            third_der = [None] * nodes

        cxi = [0, 0.5, 1]

        # Adjust clearance for the middle point
        if p_start >= p_goal:
            cyi = [p_start, p_start + clearance, p_goal]
        else:
            cyi = [p_start, p_goal + clearance, p_goal]

        xcurve = np.linspace(0, 1, nodes)

        yder = []
        for i, (d1, d2, d3) in enumerate(zip(derivatives, second_der, third_der)):
            constraints = [cyi[i]]
            if d1 is not None:
                constraints.append(d1)
            if d2 is not None:
                constraints.append(d2)
            if d3 is not None:
                constraints.append(d3)
            yder.append(constraints)

        # Create the BPoly object from derivatives
        bpoly = BPoly.from_derivatives(cxi, yder)
        # Compute the second derivative of the BPoly object
        bpoly_second_derivative = bpoly.derivative().derivative()

        # Evaluate the second derivative at the given points
        y_bpoly_second_derivative = bpoly_second_derivative(xcurve)

        return y_bpoly_second_derivative

    def third_derivative_of_trajectory(self, nodes, p_start, p_goal, clearance, derivatives=None, second_der=None, third_der=None):
        """
        Compute the third derivative of the trajectory.

        Parameters:
        - nodes: int, number of points in the trajectory
        - p_start: float, start position
        - p_goal: float, goal position
        - clearance: float, distance above or below the straight line
        - derivatives: list of first derivative constraints or `None`
        - second_der: list of second derivative constraints or `None`
        - third_der: list of third derivative constraints or `None`

        Returns:
        - y_bpoly_third_derivative: np.ndarray, third derivative values
        """
        if derivatives is None:
            derivatives = [None] * nodes
        if second_der is None:
            second_der = [None] * nodes
        if third_der is None:
            third_der = [None] * nodes

        cxi = [0, 0.5, 1]

        # Adjust clearance for the middle point
        if p_start >= p_goal:
            cyi = [p_start, p_start + clearance, p_goal]
        else:
            cyi = [p_start, p_goal + clearance, p_goal]

        xcurve = np.linspace(0, 1, nodes)

        yder = []
        for i, (d1, d2, d3) in enumerate(zip(derivatives, second_der, third_der)):
            constraints = [cyi[i]]
            if d1 is not None:
                constraints.append(d1)
            if d2 is not None:
                constraints.append(d2)
            if d3 is not None:
                constraints.append(d3)
            yder.append(constraints)

        # Create the BPoly object from derivatives
        bpoly = BPoly.from_derivatives(cxi, yder)
        # Compute the third derivative of the BPoly object
        bpoly_third_derivative = bpoly.derivative().derivative().derivative()

        # Evaluate the third derivative at the given points
        y_bpoly_third_derivative = bpoly_third_derivative(xcurve)

        return y_bpoly_third_derivative


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    tg = TrajectoryGenerator()

    n_samples = 100

    # Specify derivatives for testing
    der = [None, 0, 0]  # First derivatives at key points
    second_der = [None, None, 0]  # Second derivatives at key points
    third_der = [None, None, 0]  # Third derivatives at key points

    # Compute the trajectory
    z_trj = tg.from_derivatives(
        n_samples,
        0,
        0,
        0.12,
        derivatives=der,
        second_der=second_der,
        third_der=third_der
    )

    # First derivative of the trajectory
    z_trj_derivative = tg.derivative_of_trajectory(
        n_samples,
        0,
        0,
        0.12,
        derivatives=der,
        second_der=second_der,
        third_der=third_der
    )

    # Compute the second derivative of the trajectory
    z_trj_second_derivative = tg.second_derivative_of_trajectory(
        n_samples,
        0,
        0,
        0.12,
        derivatives=der,
        second_der=second_der,
        third_der=third_der
    )

    # Compute the third derivative of the trajectory
    z_trj_third_derivative = tg.third_derivative_of_trajectory(
        n_samples,
        0,
        0,
        0.12,
        derivatives=der,
        second_der=second_der,
        third_der=third_der
    )

    axis = np.linspace(0, 1, num=z_trj.shape[0])

    # Plot the trajectory, its first, second, and third derivatives
    plt.figure(figsize=(18, 8))

    plt.subplot(2, 2, 1)
    plt.plot(axis, z_trj, label="Trajectory")
    plt.title("Trajectory")
    plt.xlabel("Time (normalized)")
    plt.ylabel("Position")
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(axis, z_trj_derivative, label="First Derivative", linestyle="--")
    plt.title("First Derivative of Trajectory")
    plt.xlabel("Time (normalized)")
    plt.ylabel("Velocity")
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(axis, z_trj_second_derivative, label="Second Derivative", linestyle="-.")
    plt.title("Second Derivative of Trajectory")
    plt.xlabel("Time (normalized)")
    plt.ylabel("Acceleration")
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(axis, z_trj_third_derivative, label="Third Derivative", linestyle=":")
    plt.title("Third Derivative of Trajectory")
    plt.xlabel("Time (normalized)")
    plt.ylabel("Jerk")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()
