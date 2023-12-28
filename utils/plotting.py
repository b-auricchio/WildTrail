import matplotlib.pyplot as plt
from filterpy.stats import plot_covariance
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


def animate(ax, t, logger, track, z2pos, N, show_cov=True, show_meas=True):
    plt.gca()
    track = np.array(track)[:t]
    drone_pos = logger.drone_pos[:t]
    # drone_vel = logger.drone_vel[t-1]
    x = logger.x[:t]
    z = logger.z[:t]
    z_pos = np.array([z2pos(pos, m) for m, pos in zip(z, drone_pos)])[:t]
    covs = logger.cov[:t]

    plt.cla()

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    ax.plot(track[:,0], track[:,1], 0, 'k--', label='track')
    ax.plot(x[:,0], x[:,1], 0, 'b-', label='estimate')
    ax.plot(drone_pos[:,0], drone_pos[:,1], drone_pos[:,2], label='drone', linestyle='dotted', color='k')

    if show_meas:
        ax.plot(z_pos[:,0], z_pos[:,1], 0, label='measurement', color='r', marker='.', markersize=1, linestyle='none')

    ax.plot(x[-1,0], x[-1,1], 0, 'bo', label='x_pos', markersize=5)
    ax.plot(drone_pos[-1,0], drone_pos[-1,1], drone_pos[-1,2], 'g+', label='drone_pos', markersize=10)

    # v = Arrow3D([drone_pos[-1,0], drone_vel[0]], [drone_pos[-1,1], drone_vel[1]], 
    #             [drone_pos[-1,2], drone_vel[2]], mutation_scale=20, 
    #             lw=3, arrowstyle="-|>", color="r")
    # ax.add_artist(v)

    if show_cov:
        for i in range(len(x)):
                if i % (N/20) == 0:
                    plot_covariance(x[i][0:2], covs[i][0:2,0:2], fc='g', alpha=0.2, std=1, ax=ax)

    ax.set_zlim(bottom=0)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)