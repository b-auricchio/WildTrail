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
    pitch = np.arctan2(drone_pos[2], range)

    return np.array([pitch, bearing])

class Drone:
    """Drone class for the target tracking problem"""
    def __init__(self, x0): # x0 = [x, y, alpha, velocity]'
        """Initialises the target at a given initial state"""
        self.state = x0


    def sense(self, target_pos, noise): # target_pos = [x, y]
        x, y, z = self.state
        """Sense the target and return a non-linear noisy measurement vector [pitch, bearing]^T"""
        z = pos2z([x, y, z], target_pos) # get measurement vector
        pitch, bearing = z[0], z[1]

        pitch += np.random.normal(0, noise[0]) # add noise to range
        bearing += np.random.normal(0, noise[1]) # add noise to bearing

        return np.array([pitch, bearing]).T # return non-linear measurement vector
    
    def set_pos(self, pos):
        """Set the drone's position"""
        self.state[:3] = pos

    def get_pos(self):
        """Returns the current position of the drone"""
        return self.state[:3]
    
def get_jacobian_H(state, drone_pos): # drone_pos = [x, y, z], prev_pos = [x, y]
    """Calculate the jacobian of the measurement function at the previous target position and drone position"""
    x, y = state[:2]
    x_d, y_d, z_d = drone_pos

    # eliminate division by zero
    if np.abs(x-x_d) < 1e-2:
        x += 1e-2
    
    if np.abs(y-y_d) < 1e-2:
        y += 1e-2


    denom1 = np.sqrt((x-x_d)**2 + (y-y_d)**2) * (z_d**2+(x-x_d)**2 + (y-y_d)**2)
    denom2 = (x-x_d)**2 + (y-y_d)**2

    return np.array([[-z_d*(x-x_d)/denom1, -z_d*(y-y_d)/denom1, 0, 0], [(y_d-y)/denom2, (x-x_d)/denom2, 0, 0]])
    