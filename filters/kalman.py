import numpy as np

def normalise_angle(angle): # normalise angle to be between pi and -pi
    return (angle + np.pi) % (2 * np.pi) - np.pi

class ExtendedKalmanFilter:
    """Extended Kalman Filter implementation"""
    def __init__(self, x0, P0, Q, R, hx, jHx, fx, jFx, track, target):
        """Initialises the EKF with the given parameters. h is the non-linear measurement function.
        x0: initial state
        P0: initial covariance
        Q: process noise covariance
        R: measurement noise covariance
        """
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R
        self.x_post = x0
        self._I = np.eye(len(x0))
        self.y = 1e6
        self.hx = hx
        self.jHx = jHx
        self.fx = fx
        self.jFx = jFx

        self.track = track
        self.target = target

    def predict(self, dt):
        """Predicts the next state of the system
        f: transition function
        jFfun: jacobian of the transition function
        dt: time step
        """
        jF = self.jFx(self.x, dt)
        self.x = self.fx(self.x, dt) # posterior -> prior prediction step
        self.P = jF @ self.P @ jF.T + self.Q # prior covariance prediction step


    def update(self, z: object, drone_state: object):
        """Updates the state of the system with the given measurement
        z: measurement
        drone_pos: position of the drone
        h: measurement function
        jHfun: jacobian of the measurement function
        """

        drone_pos = drone_state[:3]

        h = self.hx(drone_pos, self.x_post) # get measurement
        jH = self.jHx(self.x_post, drone_pos) # get jacobian H

        self.y = z - h # pass position to measurement function
        self.y = np.array([normalise_angle(self.y[0]), normalise_angle(self.y[1])]) # normalise angles


        S = jH @ self.P @ jH.T + self.R # transformation of covariance to measurement space + measurement noise
        K = self.P @ jH.T @ np.linalg.inv(S) # kalman gain

        self.x = self.x + K @ self.y # prior -> posterior update step 

        I_KH = self._I - np.dot(K, jH)
        self.P = np.dot(I_KH, self.P).dot(I_KH.T) + np.dot(K, self.R).dot(K.T) # posterior covariance update step - joseph form

        self.x_post = self.x # store posterior state for next iteration

    def get_state(self):
        """Returns the current state of the system"""
        return self.x

    def get_covariance(self):
        """Returns the current covariance of the system"""
        return self.P
    
