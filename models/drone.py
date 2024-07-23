import numpy as np
import casadi as ca

class Drone:
    """Drone class for the target tracking problem"""
    def __init__(self, state0): # x0 = [x, y, z, vx, vy, vz]
        """Initialises the drone at a given initial state"""
        self.state = state0

    @staticmethod
    def z2pos(drone_pos, z): # z = [pitch, bearing]
        """Returns the target position given the drone's position and a measurement vector"""
        range = drone_pos[2] / np.tan(z[0])
        x = drone_pos[0] + range * np.cos(z[1])
        y = drone_pos[1] + range * np.sin(z[1])

        return np.array([x, y])

    @staticmethod
    def pos2z(drone_pos, target_state): # target_pos = [x, y]
        """Returns the measurement vector given the drone's position and the target position"""
        range = np.sqrt((target_state[0] - drone_pos[0])**2 + (target_state[1] - drone_pos[1])**2)
        
        bearing = np.arctan2(target_state[1] - drone_pos[1], target_state[0] - drone_pos[0])
        pitch = np.arcsin(drone_pos[2]/(range**2 + drone_pos[2]**2)**0.5)

        # handle case when drone is directly above target
        if np.isnan(pitch):
            pitch = np.pi/2

        return np.array([pitch, bearing])

    def sense(self, target_state, noise): # target_pos = [x, y]
        """Sense the target and return a non-linear noisy measurement vector [pitch, bearing]^T"""
        target_pos = target_state[:2] # get target position
        z = self.pos2z(self.state[:3], target_pos) # get measurement vector
        pitch, bearing = z[0], z[1]

        dz = [np.random.normal(0, noise[0]), np.random.normal(0, noise[1])] # generate noise vector
        
        pitch += dz[0] # add noise to pitch
        bearing += dz[1] # add noise to bearing

        return np.array([pitch, bearing]).T # return non-linear measurement vector
    
    def get_pos(self):
        """Returns the drone's position"""
        return self.state[:3]
    

class DroneTransition:
    @staticmethod
    def numpy(state, u, dt):
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

        state = A@state + B@u - 0.1*np.array([0, 0, 0, state[3]**2, state[4]**2, state[5]**2])

        return state
    
    @staticmethod
    def casadi(state, u, dt):
        """Applies the transition function to the state
        
        returns casadi.MX object
        """

        A = ca.MX(np.array([[1, 0, 0, dt, 0, 0],
                            [0, 1, 0, 0, dt, 0],
                            [0, 0, 1, 0, 0, dt],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]]))

        # control input matrix

        B = ca.MX(np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [dt, 0, 0],
                            [0, dt, 0],
                            [0, 0, dt]]))
        
        zero = ca.MX(0)
        d = ca.vertcat(zero, zero, zero, state[3]**2, state[4]**2, state[5]**2)

        state = A@state + B@u - 0.1*d

        return state

    
