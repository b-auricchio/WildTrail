import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import sys
sys.path.append('../')

def plot_covariance_ellipse(x, P, fc='g', alpha=0.2, std=1, ax=None):
    """Plot covariance ellipse at point x with covariance matrix P.
    """
    P = P[:2,:2]
    U, s, _ = np.linalg.svd(P)
    orientation = np.arctan2(U[1, 0], U[0, 0])
    width, height = 2 * std * np.sqrt(s)
    ell = Ellipse(xy=x[:2],
                  width=width, height=height,
                  angle=np.rad2deg(orientation),
                  facecolor=fc, alpha=alpha)
    ell.set_clip_box(ax.bbox)
    ax.add_artist(ell)

import matplotlib.animation as animation

class Animator:
    def __init__(self, logger, constraints=None):
        self.track, self.drone_state, self.kf_state, self.kf_cov, self.predictions, self.ctrl = logger.to_numpy()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))
        self.constraints = constraints

        self.dt = logger.dt

    def plot(self, t):
        # plot track
        self.ax.clear()
        self.ax.plot(self.track[:t+1, 0], self.track[:t+1, 1], 'grey', linestyle="-", label='Target Track', alpha=0.5, linewidth=1)

        # plot position of the drone
        self.ax.plot(self.drone_state[:t+1, 0], self.drone_state[:t+1, 1], 'k--', label='Drone State')
        self.ax.plot(self.drone_state[t, 0], self.drone_state[t, 1], 'kx', label='Current Drone State', markersize=6, linewidth=5)

        # plot position of the animal
        self.ax.plot(self.kf_state[:t+1, 0], self.kf_state[:t+1, 1], 'b-', label='Kalman Filter State', linewidth=1)
        self.ax.plot(self.kf_state[t, 0], self.kf_state[t, 1], 'bo', label='Current Kalman Filter State')
        self.ax.grid(True)

        # plot current prediction
        self.ax.plot(self.predictions[t+1].T[:, 0], self.predictions[t+1].T[:, 1], 'r')

        # axis equal
        self.ax.axis('equal')

        for i in range(t):
            if i % 10 == 0:
                plot_covariance_ellipse(self.kf_state[i][0:2], self.kf_cov[i][0:2,0:2], fc='g', alpha=0.2, std=3, ax=self.ax)

        if self.constraints is not None:
            for constraint in self.constraints:
                constraint.plot(self.ax)

        # set labels
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.set_title(f'Time: {t*self.dt:.1f} s')
        self.ax.grid(True)

    def animate(self, filename):
        ani = animation.FuncAnimation(self.fig, self.plot, frames=len(self.drone_state)-3, blit=False)
        ani.save(filename, fps=10)

