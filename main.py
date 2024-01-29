from models.drone import get_jacobian_H
from models.target import get_jacobian_F
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from models.target import Target, get_jacobian_F
from models.drone import get_jacobian_H, Drone, pos2z, z2pos
from filters.kalman import ExtendedKalmanFilter
from utils.plotting import plot_logger
from models.path_planning import DiscreteInputPathPlanner

np.random.seed(10)

dt = 1
N = 300 # number of time steps

# generate drone object 
drone = Drone(np.array([-100,-100, 20])) #[x,y,z]'

# generate target object
x0 = np.array([0, 0, np.random.uniform(0,2*np.pi), np.random.uniform(1,2)]) # initial state [x, y, alpha, velocity]'
target = Target(x0) # x,y,a = 0, v = 1

# get track
Q_model = np.diag([0, 0, 0.2, 0.2])**2 # process noise [x, y, alpha, velocity]'
track = [target.update_states(dt, Q_model) for _ in range(N)]

# generate kalman filter object
R = np.diag([0.1, 0.1])**2 # measurement noise
P0 = np.eye(4)*100 # initial covariance
Q = Q_model*3 # kalman filter process noise 3x larger than actual process noise
kf = ExtendedKalmanFilter(x0, P0, Q, R, pos2z, get_jacobian_H, target.transition, get_jacobian_F)

# generate path planner object
pp = DiscreteInputPathPlanner()
us = []

for t in track:
    umin, pos = pp.get_next_position(kf, drone, dt, 1) # get optimal control input
    us.append(umin)
    drone.set_pos(pos)

    # predict / update kalman filter
    z = drone.sense(t, R.diagonal())
    kf.predict(dt)
    kf.update(z, drone.get_pos())

kf.logger.to_numpy()

###### ANIMATION ######
for t in range(1, len(track)):
    plot_logger(plt.gca(), t, kf.logger, track, z2pos, N, show_cov=True, show_meas=True)
    plt.pause(0.01)
plt.show()