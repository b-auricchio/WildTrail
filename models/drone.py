import numpy as np

"""
drone.py 
====================================
Handles drone functionality for the target tracking problem

"""

def z2pos(drone_pos, z): # z = [pitch, bearing]
    """Returns the target position given the drone's position and a measurement vector"""
    range = drone_pos[2] / np.tan(z[0])
    x = drone_pos[0] + range * np.cos(z[1])
    y = drone_pos[1] + range * np.sin(z[1])

    return np.array([x, y])

def pos2z(drone_pos, state): # target_pos = [x, y]
    """Returns the measurement vector given the drone's position and the target position"""
    range = np.sqrt((state[0] - drone_pos[0])**2 + (state[1] - drone_pos[1])**2)
    
    bearing = np.arctan2(state[1] - drone_pos[1], state[0] - drone_pos[0])
    pitch = np.arcsin(drone_pos[2]/(range**2 + drone_pos[2]**2)**0.5)

    # handle case when drone is directly above target
    if np.isnan(pitch):
        pitch = np.pi/2

    return np.array([pitch, bearing])

class Drone:
    """Drone class for the target tracking problem"""
    def __init__(self, x0, drag): # x0 = [x, y, alpha, velocity]'
        """Initialises the target at a given initial state"""
        self.state = x0
        self.drag = drag

    def sense(self, target_pos, noise): # target_pos = [x, y]
        """Sense the target and return a non-linear noisy measurement vector [pitch, bearing]^T"""
        z = pos2z(self.state[:3], target_pos) # get measurement vector
        pitch, bearing = z[0], z[1]

        dz = [np.random.normal(0, noise[0]), np.random.normal(0, noise[1])] # generate noise vector
        
        pitch += dz[0] # add noise to pitch
        bearing += dz[1] # add noise to bearing

        return np.array([pitch, bearing]).T # return non-linear measurement vector
    
    def get_pos(self):
        """Returns the drone's position"""
        return self.state[:3]
    
    def transition(self, x, u, dt):
        """Applies the transition function to the state"""

        A = np.array([[1, 0, 0, dt, 0, 0],
                      [0, 1, 0, 0, dt, 0],
                      [0, 0, 1, 0, 0, dt],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])

        # control input matrix

        B = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [dt, 0, 0],
                      [0, dt, 0],
                      [0, 0, dt]])

        x = A@x + B@u - self.drag*np.array([0, 0, 0, x[3], x[4], x[5]])

        return x

def get_jacobian_H(state, drone_pos): # drone_pos = [x, y, z], prev_pos = [x, y]
    """Calculate the jacobian of the measurement function at the previous target position and drone position"""
    x, y = state[:2]
    x_d, y_d, z_d = drone_pos

    denom1 = np.sqrt((x-x_d)**2 + (y-y_d)**2) * (z_d**2+(x-x_d)**2 + (y-y_d)**2)
    denom2 = (x-x_d)**2 + (y-y_d)**2

    e = [-z_d*(x-x_d)/denom1, -z_d*(y-y_d)/denom1, (y_d-y)/denom2, (x-x_d)/denom2]

    for i in range(len(e)):
        if np.isnan(e[i]):
            e[i] = 0
    
    jacobian = np.array([[e[0], e[1], 0, 0], [e[2], e[3], 0, 0]])

    return jacobian
    