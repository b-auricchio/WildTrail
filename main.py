import numpy as np
import matplotlib.pyplot as plt

from models.target import Target, get_jacobian_F
from models.drone import get_jacobian_H, Drone, pos2z, z2pos
from filters.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter
from utils.plotting import plot

np.random.seed(35)

N = 100 # number of time steps
Q_model = np.diag([0, 0, 0.3, 0.6])**2 # process noise
x0 = np.array([10, 10, 0, 0.1]) # initial state [x, y, alpha, velocity]'

R = np.diag([0.1, 0.1])**2 # measurement noise
P0 = np.eye(4)*10 # initial covariance

# get track 
target = Target(x0) # x,y,a = 0, v = 1
dt = 1
track = [target.update_states(dt, Q_model) for _ in range(N)]

# get sensor measurements
drone = Drone(np.array([0, 0, 10]), np.array([0, 0, 0.1])) # pos, vel

Q = Q_model*2 # kalman filter process noise 3x larger than actual process noise
abk = (1e-3, 2, 1)
kf = ExtendedKalmanFilter(x0, P0, Q_model, R)

for t in track:
    z = drone.sense(t, R.diagonal())
    kf.predict(target.transition, get_jacobian_F ,dt)
    kf.update(z, drone.get_pos(), pos2z, get_jacobian_H)
    drone.update(dt)

kf.logger.to_numpy()

###### ANIMATION ######
plot(kf.logger, track, z2pos, N, show_cov=True, show_meas=True)