import numpy as np
import matplotlib.pyplot as plt

from models.drone import get_jacobian_H
from models.target import get_jacobian_F
from models.target import Target, get_jacobian_F
from models.drone import get_jacobian_H, Drone, pos2z, z2pos
from filters.kalman import ExtendedKalmanFilter
from utils.plotting import plot_logger
from models.path_planning import PathPlanner, evalue_trace
import time
import os
import imageio.v2 as io

start_time = time.time()

np.random.seed(35)

save_gif = False
dt = 1 # time step
N = 100 # number of time steps

num_targets = 1

# generate drone object
drone = Drone(np.array([-10, 0, 80, 0, 0, 0]), drag=1e-1) #[x,y,z, vx, vy, vz]'

# generate target object
Q_model = np.diag([0., 0., 0.1, 0.1])**2 # process noise [x, y, alpha, velocity]'
R = np.diag([0.1, 0.1])**2 # measurement noise
P0 = np.eye(4)*100 # initial covariance
Q = Q_model # kalman filter process noise 3x larger than actual process noise

kfs = []
for i in range(num_targets):
    # x0 = np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0*np.random.uniform(0, 2*np.pi), 3]) # initial state [x, y, alpha, velocity]'
    
    x0 = np.array([0, 0, 0, 3]) # initial state [x, y, alpha, velocity]'
    target = Target(x0) # x,y,a = 0, v = 1

    # get track
    track = [target.update_states(dt, Q_model) for _ in range(N)]

    # generate kalman filter object
    kf = ExtendedKalmanFilter(x0, P0, Q, R, pos2z, get_jacobian_H, target.transition, get_jacobian_F, track, target)
    kfs.append(kf)

# generate path planner object
pp = PathPlanner(dt, drone.transition, v_bounds=[2, 0.2], alt_bounds=[75, 120], r1_weight=1.,  r2_weight=0)

node_tree = []
current_idx = 0
value_threshold = 100
t = 0
while t < N:
    traces = [evalue_trace(kf.P) for kf in kfs]
    print(traces)

    # iterate between targets until uncertainty reaches threshold:
    current_kf = kfs[current_idx]
    if traces[current_idx] < value_threshold:
        current_idx = np.argmax(traces)

    # get best input(s)
    u_min, nodes = pp.get_best_input(1, 12, 12, drone.state, current_kf)

    for u in u_min:
        node_tree.append([node.get_state_path() for node in nodes])

        drone.state = drone.transition(drone.state, u, dt)
        
        # predict / update kalman filter
        for kf in kfs:
            z = drone.sense(kf.track[t], R.diagonal())
            kf.predict(dt)
            kf.update(z, drone.state)
        print(f't: {t}')

        t += 1

print("--- %s seconds ---" % (time.time() - start_time))

for kf in kfs:
    kf.logger.to_numpy()


# ###### ANIMATION ######
images = []
fig, ax1 = plt.subplots(figsize=(7,2))

if save_gif:
    print('Creating gif...')
    with io.get_writer('animation.gif', mode='I') as writer:
        for t in range(1, len(track), 2):
            plt.cla()
            for kf in kfs:
                plot_logger(ax1, t, kf.logger, kf.track, z2pos, N, show_cov=True, show_meas=True, node_tree=node_tree, clear=False)
            plt.savefig('./temp.png', dpi=100)
            writer.append_data(io.imread('./temp.png'))
    os.remove('./temp.png')
else:
    for kf in kfs:
        plot_logger(ax1, N-1, kf.logger, kf.track, z2pos, N, show_cov=True, show_meas=True, node_tree=node_tree, clear=False)
        plt.savefig('./sim.svg', dpi=100)

        
# save data to csv
for i, kf in enumerate(kfs):
    kf.logger.to_csv(f'./data/kf{i}.csv', track)

# plt.show()