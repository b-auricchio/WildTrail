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
from utils.constraints import CylinderConstraint
from tqdm import tqdm

"""
TO DO:
- Add better way to interrogate the data
- Add tqdm to show progress
"""

# set random seed
np.random.seed(42)

start_time = time.time()  # start timer
save_gif = True

dt = 1  # time step
N = 80  # number of time in the simulation

# generate drone object
drone = Drone(np.array([-10, -10, 80, 0, 0, 0]))   #[x,y,z, vx, vy, vz]'

# generate target object
Q_model = np.diag([0., 0., 0.2, 0.1])**2  # process noise [x, y, alpha, velocity]'
R = np.diag([0.15, 0.15])**2  # measurement noise
P0 = 100*np.eye(4)  # initial covariance
Q = Q_model  # kalman filter process noise 3x larger than actual process noise

animal_state0 = np.array([0, 0, 0, 8])  # initial state [x, y, alpha, velocity]'
target = Target(animal_state0)  # x,y,a = 0, v = 1

constraints = [CylinderConstraint(50, 100, 0),
               CylinderConstraint(35, 400, 10),
               CylinderConstraint(60, 400, -100),
               CylinderConstraint(35, 200, 0)]  # constraints

constraints = None

# get animal's track
track = [target.update_states(dt, Q_model) for _ in range(N)]

# generate kalman filter object
kf = ExtendedKalmanFilter(animal_state0, P0, Q, R, drone.pos2z, GetJacobianH.numpy, target.transition, GetJacobianF.numpy, track, target)
mpc = MPC(10, dt, DroneTransition.casadi, kf, GetJacobianH.casadi, GetJacobianF.casadi, constraints=constraints)
logger = Logger(dt)

t = 0
pbar = tqdm(range(N-1))
for t in pbar:
    # predict / update kalman filter
    z = drone.sense(kf.track[t], R.diagonal())
    kf.predict(dt)

    kf.update(z, drone.state)

    u_min, cost = mpc(drone.state)
    drone.state = DroneTransition.numpy(drone.state, u_min, dt)
    error = (np.linalg.norm(kf.x[:2] - kf.track[t+1][:2]))

    pbar.set_description(f"Error: {error:.2f}, Cost: {cost:.2f}")

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

ani = Animator(logger, constraints)
ani.animate('animation.gif')