import numpy as np
import matplotlib.pyplot as plt

from models.drone import get_jacobian_H
from models.target import get_jacobian_F
from models.target import Target, get_jacobian_F
from models.drone import get_jacobian_H, Drone, pos2z, z2pos
from filters.kalman import ExtendedKalmanFilter
from utils.plotting import plot_logger, plot_stats, plot_paths
from models.path_planning import PathPlanner
import time
import imageio.v2 as io
start_time = time.time()

# np.random.seed(933)

save_gif = True
dt = 1 # time step
N = 100 # number of time steps

# generate drone object
drone = Drone(np.array([-100,-100, 80, 0, 0, 0])) #[x,y,z, vx, vy, vz]'

# generate target object
x0 = np.array([0, 0, np.random.uniform(0,2*np.pi), 1*np.random.uniform(1,2)]) # initial state [x, y, alpha, velocity]'
target = Target(x0) # x,y,a = 0, v = 1

# get track
Q_model = np.diag([0., 0., 0.2, 0.2])**2 # process noise [x, y, alpha, velocity]'
track = [target.update_states(dt, Q_model) for _ in range(N)]

# generate kalman filter object
R = np.diag([0.1, 0.1])**2 # measurement noise
P0 = np.eye(4)*100 # initial covariance
Q = Q_model*3 # kalman filter process noise 3x larger than actual process noise

kf = ExtendedKalmanFilter(x0, P0, Q, R, pos2z, get_jacobian_H, target.transition, get_jacobian_F)

def transition_function(state, u, dt):
    """Returns the next state given the current state, control input, and time step"""
    state = state.copy()
    state[:3] += u[:3]*dt

    return state

# generate path planner object
pp = PathPlanner(dt, drone.transition, v_bounds=[3, 0.5], alt_bounds=[75, 120], r1_weight=1.,  r2_weight=0.5)

node_tree = []

t = 0
while t < N:
    u_min, nodes = pp.get_best_input(1, 2, 2, drone.state, kf)
    print(nodes[0]) # print best node dataset

    for u in u_min:
        node_tree.append([node.get_state_path() for node in nodes])

        drone.state = drone.transition(drone.state, u, dt)
        
        # predict / update kalman filter
        z = drone.sense(track[t], R.diagonal())
        kf.predict(dt)
        kf.update(z, drone.get_pos())

        print(f't: {t}')

        t += 1

print("--- %s seconds ---" % (time.time() - start_time))

kf.logger.to_numpy()


# ###### ANIMATION ######
fig, ax1 = plt.subplots(figsize=(7,5))

images = []
if save_gif:
    print('Creating gif...')
    with io.get_writer('animation.gif', mode='I') as writer:
        for t in range(1, len(track)):
            plot_logger(ax1, t, kf.logger, track, z2pos, N, show_cov=True, show_meas=True, node_tree=node_tree)
            plt.savefig('./temp.png', dpi=100)
            writer.append_data(io.imread('./temp.png'))

for t in range(1, len(track)):
    plot_logger(ax1, t, kf.logger, track, z2pos, N, show_cov=True, show_meas=True, node_tree=node_tree)
    plt.pause(0.0001)

plt.show()

plot_stats(kf.logger, track, z2pos, N)
plt.show()