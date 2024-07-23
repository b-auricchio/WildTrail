import numpy as np
import casadi as ca

class Target:
    def __init__(self, state0): # state_0 = [x, y, alpha, velocity]'
        """Initialises the target at a given initial state"""
        self.state = state0

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
    
    @staticmethod
    def transition(state, dt):
        """Applies the transition function to the state"""
        x, y, a, v = state[0], state[1], state[2], state[3]

        x += v * dt * np.cos(a)
        y += v * dt * np.sin(a)

        state = np.array([x, y, a, v]).T
        return state

    def get_pos(self):
        """Returns the current position of the target"""
        return np.array([self.state[0], self.state[1]])