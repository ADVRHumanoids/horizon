import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import splprep


from numpy import linspace, sin, pi
from scipy.interpolate import BPoly, CubicSpline

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

        # todo check dimension of parameter before assigning it

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

    def from_derivatives(self, nodes, p_start, p_goal, clearance, derivatives=None, second_der=None):
        if derivatives is None:
            derivatives = [None] * len(p_start)
        
        if second_der is None:
            second_der = [None] * len(p_start)

        cxi = [0, 0.5, 1]
        cyi = [p_start, p_goal + clearance, p_goal]

        xcurve = np.linspace(0, 1, nodes)

        yder = []
        for i, (d1, d2) in enumerate(zip(derivatives, second_der)):
            constraints = [cyi[i]]
            if d1 is not None:
                constraints.append(d1)
            if d2 is not None:
                constraints.append(d2)
            yder.append(constraints)

        bpoly = BPoly.from_derivatives(cxi, yder)
        y_bpoly = bpoly(xcurve)

        return y_bpoly

    def derivative_of_trajectory(self, nodes, p_start, p_goal, clearance, derivatives=None, second_der=None):
        if derivatives is None:
            derivatives = [None] * len(p_start)
        
        if second_der is None:
            second_der = [None] * len(p_start)

        cxi = [0, 0.5, 1]
        cyi = [p_start, p_goal + clearance, p_goal]

        xcurve = np.linspace(0, 1, nodes)

        yder = []
        for i, (d1, d2) in enumerate(zip(derivatives, second_der)):
            constraints = [cyi[i]]
            if d1 is not None:
                constraints.append(d1)
            if d2 is not None:
                constraints.append(d2)
            yder.append(constraints)

        bpoly = BPoly.from_derivatives(cxi, yder)
        
        # Compute the first derivative of the BPoly object
        bpoly_derivative = bpoly.derivative()

        # Evaluate the derivative at the given points
        y_bpoly_derivative = bpoly_derivative(xcurve)

        return y_bpoly_derivative
    
    def second_derivative_of_trajectory(self, nodes, p_start, p_goal, clearance, derivatives=None, second_der=None):
        if derivatives is None:
            derivatives = [None] * len(p_start)
        
        if second_der is None:
            second_der = [None] * len(p_start)

        cxi = [0, 0.5, 1]
        cyi = [p_start, p_goal + clearance, p_goal]

        xcurve = np.linspace(0, 1, nodes)

        yder = []
        for i, (d1, d2) in enumerate(zip(derivatives, second_der)):
            constraints = [cyi[i]]
            if d1 is not None:
                constraints.append(d1)
            if d2 is not None:
                constraints.append(d2)
            yder.append(constraints)

        bpoly = BPoly.from_derivatives(cxi, yder)
        
        # Compute the second derivative of the BPoly object
        bpoly_second_derivative = bpoly.derivative().derivative()

        # Evaluate the second derivative at the given points
        y_bpoly_second_derivative = bpoly_second_derivative(xcurve)

        return y_bpoly_second_derivative
    
if __name__ == '__main__':

    tg = TrajectoryGenerator()

    n_samples = 100

    der= [None, 0, 0]
    second_der=[None, 0, 0]
    
    # Original trajectory with first and second derivative constraints
    z_trj = tg.from_derivatives(
        n_samples, 
        -1, 
        -1, 
        1, 
        derivatives=der,
        second_der=second_der
    )

    # First derivative of the trajectory with first and second derivative constraints
    z_trj_derivative = tg.derivative_of_trajectory(
        n_samples, 
        -1, 
        -1, 
        1, 
        derivatives=der,
        second_der=second_der
    )

    # Second derivative of the trajectory with first and second derivative constraints
    z_trj_second_derivative = tg.second_derivative_of_trajectory(
        n_samples, 
        -1, 
        -1, 
        1, 
        derivatives=der,
        second_der=second_der
    )

    axis = np.linspace(0, 1, num=z_trj.shape[0])

    # Plotting the trajectory, its first derivative, and its second derivative
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(axis, z_trj, label="Trajectory")
    plt.title("Trajectory")
    plt.xlabel("Time (normalized)")
    plt.ylabel("Position")
    plt.grid()
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(axis, z_trj_derivative, label="Trajectory Derivative", linestyle="--")
    plt.title("Derivative of Trajectory")
    plt.xlabel("Time (normalized)")
    plt.ylabel("Velocity")
    plt.grid()
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(axis, z_trj_second_derivative, label="Trajectory Second Derivative", linestyle="-.")
    plt.title("Second Derivative of Trajectory")
    plt.xlabel("Time (normalized)")
    plt.ylabel("Acceleration")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

