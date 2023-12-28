
import numpy as np

class Target:
    """Target class for the target tracking problem"""
    def __init__(self, x0): # x0 = [x, y, alpha, velocity]'
        """Initialises the target at a given initial state"""
        self.state = x0

    def update_states(self, dt, Q_model):
        """Moves the target with a given speed and noise"""
        process_noise = Q_model.diagonal() # get variances from Q covariance matrix

        state = self.transition(self.state, dt)

        state[0] += np.random.normal(0, process_noise[0]) # add noise to x
        state[1] += np.random.normal(0, process_noise[1])
        state[2] += np.random.normal(0, process_noise[2])
        state[3] += np.random.normal(0, process_noise[3])

        self.state = state
        return self.state
    
    def transition(self, state, dt):
        """Applies the transition function to the state"""
        x, y, a, v = state[0], state[1], state[2], state[3]

        x += v * dt * np.cos(a)
        y += v * dt * np.sin(a)

        state = np.array([x, y, a, v]).T
        return state

    def get_pos(self):
        """Returns the current position of the target"""
        return np.array([self.state[0], self.state[1]])
    
    def update_velocity(self, v):
        """Updates the velocity of the target"""
        self.state[3] = v

    def update_angle(self, a):
        """Updates the angle of the target"""
        self.state[2] = a

def get_jacobian_F(state, dt):
    """Calculate the symbolic jacobian of the transition function at the previous target position"""
    a, v = state[2], state[3]
    jF = np.eye(4)
    jF[0,2] = -dt*v*np.sin(a)
    jF[0,3] = dt*np.cos(a)
    jF[1,2] = dt*v*np.cos(a)
    jF[1,3] = dt*np.sin(a)
    return jF