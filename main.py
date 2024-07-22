import numpy as np
import matplotlib.pyplot as plt

from models.path_planning import MPC
from models.target import Target
from models.drone import Drone, pos2z, z2pos
from models.jacobians import GetJacobianH, GetJacobianF
from filters.kalman import ExtendedKalmanFilter
from utils.plotting import plot
import time
import os
import imageio.v2 as io

start_time = time.time()  # start timer
save_gif = True

dt = 1  # time step
N = 100  # number of time steps
num_targets = 1  # number of animals to track

# generate drone object
drone = Drone(np.array([-10, 0, 80, 0, 0, 0]), drag=1e-1)   #[x,y,z, vx, vy, vz]'

# generate target object
Q_model = np.diag([0., 0., 0.4, 0.1])**2  # process noise [x, y, alpha, velocity]'
R = np.diag([0.11, 0.15])**2  # measurement noise
P0 = 100*np.eye(4)  # initial covariance
Q = Q_model  # kalman filter process noise 3x larger than actual process noise

x0 = np.array([0, 0, 0, 6])  # initial state [x, y, alpha, velocity]'
target = Target(x0)  # x,y,a = 0, v = 1

# get track
track = [target.update_states(dt, Q_model) for _ in range(N)]

# generate kalman filter object
kf = ExtendedKalmanFilter(x0, P0, Q, R, pos2z, GetJacobianH.numpy, target.transition, GetJacobianF.numpy, track, target)
mpc = MPC(10, dt, drone.transition_casadi, kf, GetJacobianH.casadi, GetJacobianF.casadi)

t = 0
error = 100
costs = []
for t in range(N-1):
    print(f'---------------\nt: {t}')

    # predict / update kalman filter
    z = drone.sense(kf.track[t], R.diagonal())
    kf.predict(dt)

    kf.update(z, drone.state)

    u_min, cost = mpc(drone.state)
    
    print(f'cost: {cost}')
    costs.append(cost)
    drone.state = drone.transition(drone.state, u_min, dt)
    error = (np.linalg.norm(kf.x[:2] - kf.track[t+1][:2]))

    print(f'error: {error}')


print("--- %s seconds ---" % (time.time() - start_time))  # end timer

kf.logger.to_numpy()


###### Plotting ######
plt.rcParams['text.usetex'] = True

images = []
fig, ax = plt.subplots(figsize=(7,4))

if save_gif:
    print('Creating gif...')
    with io.get_writer('animation.gif', mode='I') as writer:
        for t in range(1, len(track), 2):
            plt.cla()
            plot(ax, t, kf.logger, kf.track, z2pos, N, show_cov=True, show_meas=True, node_tree=None, clear=False, constraints=[])

            plt.savefig('./temp.png', dpi=100)
            writer.append_data(io.imread('./temp.png'))
    os.remove('./temp.png')
else:
    plot(ax, N-1, kf.logger, kf.track, z2pos, N, show_cov=True, show_meas=True, node_tree=None, clear=False, constraints=[])
    plt.savefig('./sim.pdf', dpi=100)

# save data to csv
kf.logger.to_csv(f'./data/data.csv', track, costs)

plt.show()