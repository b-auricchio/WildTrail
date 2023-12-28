import numpy as np
import matplotlib.pyplot as plt

from models.target import Target, get_jacobian_F
from models.drone import get_jacobian_H, Drone, pos2z, z2pos
from filters.kalman import ExtendedKalmanFilter
from utils.plotting import animate

# np.random.seed(9430)

N = 100 # number of time steps
Q_model = np.diag([0, 0, 0.3, 0.6])**2 # process noise
x0 = np.array([0, 0, 0, 0.1]) # initial state [x, y, alpha, velocity]'

R = np.diag([0.1, 0.1])**2 # measurement noise
P0 = np.eye(4)*10 # initial covariance

# get track 
target = Target(x0) # x,y,a = 0, v = 1
dt = 1
track = [target.update_states(dt, Q_model) for _ in range(N)]

# get sensor measurements
drone = Drone(np.array([0, 0, 10]), np.array([1, 0, 0.5]))

Q = Q_model # kalman filter process noise 3x larger than actual process noise
kf = ExtendedKalmanFilter(x0, P0, Q_model, R)

for t in track:
    # drone.velocity+= np.random.multivariate_normal([0,0,0], np.diag([0.1, 0.1, 0.1]))
    z = drone.sense(t, R.diagonal())
    kf.predict(target.transition, get_jacobian_F, dt)
    kf.update(z, drone.get_pos(), pos2z, get_jacobian_H)
    drone.update(dt)

kf.logger.to_numpy()

###### ANIMATION ######

t_steps = N

fig = plt.figure(figsize=(6,6))

fig.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1, wspace=0.32, hspace=0.1)
plt.gcf().canvas.mpl_connect('key_release_event',
    lambda event: [exit(0) if event.key == 'escape' else None])


ax = fig.add_subplot(projection='3d', proj_type='ortho') 
ax.view_init(elev=90, azim=-90)


for k in range(1, t_steps, 1):
    animate(ax, k, kf.logger, track, z2pos, N)
    plt.pause(0.01)

plt.show()
# plot residuals
xs = kf.logger.x

track = np.array(track)
error = np.array([np.linalg.norm(x-t) for x, t in zip(xs[:,:2], track[:,:2])])
range = np.array(kf.logger.range)

plt.figure()
plt.plot(range, error)

plt.ylabel('error')
plt.xlabel('range')

plt.show()  
