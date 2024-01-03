import matplotlib.pyplot as plt
from filterpy.stats import plot_covariance
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


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

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    # ax.set_zlabel('z [m]')

    if show_meas:
        ax.plot(z_pos[:,0], z_pos[:,1], label='measurements', color='r', marker='.', markersize=1, linestyle='none')

    ax.plot(track[:,0], track[:,1],  'k--', label='target track')
    ax.plot(x[:,0], x[:,1],  'b-', label='kalman filter')
    ax.plot(drone_pos[:,0], drone_pos[:,1], label='drone track', linestyle='dotted', color='k')

    # ax.plot(x[-1,0], x[-1,1], 'bo', label='x_pos', markersize=5)
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
    
    ax.legend()

    # ax.set_zlim(bottom=0)


def plot(logger, track, z2pos, N, show_cov=True, show_meas=True):
      
    fig = plt.figure(figsize=(12,6))

    fig.subplots_adjust(left=0.07, bottom=0.08, right=0.95, top=0.95, wspace=0.15, hspace=0.35)
    plt.gcf().canvas.mpl_connect('key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    ax_kalman = fig.add_subplot(1, 2, 1)
    plot_logger(ax_kalman, N, logger, track, z2pos, N, show_cov=show_cov, show_meas=show_meas)
    ax_kalman.set_title('Kalman filter')

    ax_cov = fig.add_subplot(3, 2, 2)
    ax_cov.set_title('Std. deviation of state variables')
    ax_cov.set_xlim([0, N])
    ax_cov.plot(np.sqrt(logger.cov[:,0,0]), 'r-', label='$\sigma_{x}$')
    ax_cov.plot(np.sqrt(logger.cov[:,1,1]), 'b-', label='$\sigma_{y}$')
    ax_cov.plot(np.sqrt(logger.cov[:,2,2]), 'g-', label='$\sigma_{a}$')
    ax_cov.plot(np.sqrt(logger.cov[:,3,3]), 'm-', label='$\sigma_{v}$')

    ax_cov.legend()

    ax_alt = fig.add_subplot(3, 2, 4)
    ax_alt.set_title('Drone altitude')
    ax_alt.set_ylabel('z [m]')

    ax_alt.set_xlim([0, N])
    ax_alt.plot(logger.drone_pos[:,2], 'g--')

    ax_error = fig.add_subplot(3, 2, 6)
    ax_error.set_title('Positional error')
    ax_error.set_ylabel('error [m]')
    ax_error.set_xlabel('time step')
    track = np.array(track)
    error = np.array([np.linalg.norm(x-t) for x, t in zip(logger.x[:,:2], track[:,:2])])
    ax_error.plot(error, 'r-')


    plt.show()