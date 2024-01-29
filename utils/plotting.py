import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def plot_covariance(x, P, fc='g', alpha=0.2, std=1, ax=None):
    """Plot covariance ellipse at point x with covariance matrix P.
    """
    if ax is None:
        ax = plt.gca()

    P = P[:2,:2]
    U, s, _ = np.linalg.svd(P)
    orientation = np.arctan2(U[1, 0], U[0, 0])
    width, height = 2 * std * np.sqrt(s)
    ell = Ellipse(xy=x[:2],
                  width=width, height=height,
                  angle=np.rad2deg(orientation),
                  facecolor=fc, alpha=alpha)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(alpha)
    ax.add_artist(ell)
    return ell


def plot_logger(ax, t, logger, track, z2pos, N, show_cov=True, show_meas=True):
    plt.gca()
    track = np.array(track)[:t]
    drone_pos = logger.drone_pos[:t]
    # drone_vel = logger.drone_vel[t-1]
    x = logger.x[:t]
    z = logger.z[:t]
    z_pos = np.array([z2pos(pos, m) for m, pos in zip(z, drone_pos)])[:t]
    covs = logger.cov[:t]

    plt.cla()
    plt.gcf().canvas.mpl_connect('key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    if show_meas:
        ax.plot(z_pos[:,0], z_pos[:,1], label='measurements', color='r', marker='.', markersize=1, linestyle='none')

    ax.plot(track[:,0], track[:,1],  'k--', label='target track')
    ax.plot(x[:,0], x[:,1],  'b-', label='kalman filter')
    ax.plot(x[-1,0], x[-1,1], 'bo', label='kalman_pos', markersize=6)
    ax.plot(x[-1,0], x[-1,1], 'ko', label='true_pos', markersize=3, markeredgewidth=2, alpha=0.5)

    ax.plot(drone_pos[:,0], drone_pos[:,1], label='drone track', linestyle='dotted', color='k')
    ax.plot(drone_pos[-1,0], drone_pos[-1,1], 'g+', label='drone pos', markersize=10)

    for i in range(t):
                if i % (N/20) == 0:
                    ax.scatter(x[i][0], x[i][1], marker='s', color='b', s=20)

    for i in range(t):
                if i % (N/20) == 0:
                    ax.scatter(x[i][0], x[i][1], marker='s', color='b', s=20)

    if show_cov:
        for i in range(t):
                if i % (N/20) == 0:
                    plot_covariance(x[i][0:2], covs[i][0:2,0:2], fc='g', alpha=0.2, std=1, ax=ax)
    
def plot_stats(logger, track, N):
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.title('Std. deviation of state variables')
    plt.xlim([0, N])
    plt.plot(np.sqrt(logger.cov[:,0,0]), 'r-', label='$\sigma_{x}$')

    plt.subplot(4, 1, 2)
    plt.title('Drone altitude')
    plt.ylabel('z [m]')
    plt.xlim([0, N])
    plt.plot(logger.drone_pos[:,2], 'g--')

    plt.subplot(4, 1, 3)
    plt.title('Positional error')
    plt.ylabel('error [m]')
    track = np.array(track)
    error = np.array([np.linalg.norm(x-t) for x, t in zip(logger.x[:,:2], track[:,:2])])
    plt.plot(error, 'r-')

    plt.subplot(4, 1, 2)
    plt.title('Range to target')
    plt.ylabel('range [m]')

    plt.xlim([0, N])
    plt.plot(logger.range, 'r-')


    plt.show()