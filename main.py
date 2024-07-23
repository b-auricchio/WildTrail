import numpy as np
import matplotlib.pyplot as plt

from models.mpc import MPC
from models.target import Target
from models.drone import Drone, DroneTransition
from models.jacobians import GetJacobianH, GetJacobianF
from filters.kalman import ExtendedKalmanFilter
from utils.logger import Logger
import time
from utils.animation import Animator


start_time = time.time()  # start timer
save_gif = True

dt = 1  # time step
N = 100  # number of time in the simulation

# generate drone object
drone = Drone(np.array([-10, -10, 80, 0, 0, 0]))   #[x,y,z, vx, vy, vz]'

# generate target object
Q_model = np.diag([0., 0., 0.2, 0.1])**2  # process noise [x, y, alpha, velocity]'
R = np.diag([0.15, 0.15])**2  # measurement noise
P0 = 100*np.eye(4)  # initial covariance
Q = Q_model  # kalman filter process noise 3x larger than actual process noise

animal_state0 = np.array([0, 0, 0, 8])  # initial state [x, y, alpha, velocity]'
target = Target(animal_state0)  # x,y,a = 0, v = 1

# get animal's track
track = [target.update_states(dt, Q_model) for _ in range(N)]

# generate kalman filter object
kf = ExtendedKalmanFilter(animal_state0, P0, Q, R, drone.pos2z, GetJacobianH.numpy, target.transition, GetJacobianF.numpy, track, target)
mpc = MPC(10, dt, DroneTransition.casadi, kf, GetJacobianH.casadi, GetJacobianF.casadi, r=0.2)
logger = Logger(dt)

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
    drone.state = DroneTransition.numpy(drone.state, u_min, dt)
    error = (np.linalg.norm(kf.x[:2] - kf.track[t+1][:2]))

    print(f'error: {error}')

    # store data in logger
    logger.track.append(kf.track[t])
    logger.drone_state.append(drone.state)
    logger.kf_state.append(kf.x)
    logger.kf_cov.append(kf.P)

    X, _ = mpc.get_predictions()
    logger.predictions.append(X)
    logger.ctrl.append(u_min)

print("--- %s seconds ---" % (time.time() - start_time))  # end timer

###### Plotting ######
plt.rcParams['text.usetex'] = True

ani = Animator(logger)
ani.animate('animation.gif')