import numpy as np
import casadi as ca
from matplotlib.patches import Ellipse

class CylinderConstraint:
    # apply cylinder constraint around a point
    def __init__(self, r, x, y):
        self.r, self.x, self.y = r, x, y
        self.not_in_cylinder = lambda x, y: (x - self.x)**2 + (y - self.y)**2 >= self.r**2

    def apply(self, opti, X):
        N = X.shape[1] - 1
        for k in range(N):
            current_x, current_y = X[0,k+1], X[1,k+1]
            opti.subject_to(self.not_in_cylinder(current_x, current_y))

    def plot(self, ax):
        """Plot covariance ellipse at point x with covariance matrix P.
        """
        ell = Ellipse(xy=[self.x, self.y], width=2*self.r, height=2*self.r, facecolor='r', alpha=0.5)
        ell.set_clip_box(ax.bbox)
        ax.add_artist(ell)