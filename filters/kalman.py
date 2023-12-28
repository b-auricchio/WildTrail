import numpy as np
import matplotlib as mpl
from utils.logger import KalmanLogger

class ExtendedKalmanFilter:
    """Extended Kalman Filter implementation"""
    def __init__(self, x0, P0, Q, R):
        """Initialises the EKF with the given parameters. h is the non-linear measurement function.
        x0: initial state
        P0: initial covariance
        Q: process noise covariance
        R: measurement noise covariance
        """
        self.logger = KalmanLogger()
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R
        self.x_post = x0
        self._I = np.eye(len(x0))
        self.y = 1e6

    def predict(self, f, jFfun, dt):
        """Predicts the next state of the system
        f: transition function
        jFfun: jacobian of the transition function
        dt: time step
        """
        x = self.x
        jF = jFfun(x, dt)
        self.x = f(x, dt) # posterior -> prior prediction step
        self.P = jF @ self.P @ jF.T + self.Q # prior covariance prediction step

    def update(self, z, drone_pos, h, jHfun):
        """Updates the state of the system with the given measurement
        z: measurement
        drone_pos: position of the drone
        h: measurement function
        jHfun: jacobian of the measurement function
        """

        jH = jHfun(self.x_post, drone_pos) # get jacobian H
        self.y = z - h(drone_pos, self.x) # pass position to measurement function

        S = jH @ self.P @ jH.T + self.R # transformation of covariance to measurement space + measurement noise
        K = self.P @ jH.T @ np.linalg.inv(S) # kalman gain

        self.x = self.x + K @ self.y # prior -> posterior update step 

        I_KH = self._I - np.dot(K, jH)
        self.P = np.dot(I_KH, self.P).dot(I_KH.T) + np.dot(K, self.R).dot(K.T) # posterior covariance update step - joseph form

        self.x_post = self.x # store posterior state for next iteration

        # print(f'y: {self.y}')
        self.logger.cov.append(self.P)
        self.logger.x.append(self.x)
        self.logger.z.append(z)
        self.logger.y.append(np.linalg.norm(self.y))
        self.logger.drone_pos.append(drone_pos)
        self.logger.range.append(np.linalg.norm(self.x[:2] - drone_pos[:2]))

    def get_state(self):
        """Returns the current state of the system"""
        return self.x

    def get_covariance(self):
        """Returns the current covariance of the system"""
        return self.P