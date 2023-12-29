import numpy as np
from utils.logger import KalmanLogger
from scipy.linalg import cholesky

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

    def predict(self, fx, jFx, dt):
        """Predicts the next state of the system
        f: transition function
        jFfun: jacobian of the transition function
        dt: time step
        """
        x = self.x
        jF = jFx(x, dt)
        self.x = fx(x, dt) # posterior -> prior prediction step
        self.P = jF @ self.P @ jF.T + self.Q # prior covariance prediction step

    def update(self, z, drone_pos, hx, jHx):
        """Updates the state of the system with the given measurement
        z: measurement
        drone_pos: position of the drone
        h: measurement function
        jHfun: jacobian of the measurement function
        """

        jH = jHx(self.x_post, drone_pos) # get jacobian H
        self.y = z - hx(drone_pos, self.x) # pass position to measurement function

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
    

class UnscentedKalmanFilter:
    """Extended Kalman Filter implementation"""
    def __init__(self, x0, P0, Q, R, abk):
        """Initialises the EKF with the given parameters. h is the non-linear measurement function.
        x0: initial state
        P0: initial covariance
        Q: process noise covariance
        R: measurement noise covariance
        """
        self.logger = KalmanLogger()
        self.x = x0
        self.P = P0
        self.Pp = P0 # prior covariance
        self.Q = Q
        self.R = R
        self.sigmas_f = None
        self._num_sigmas = 0

        self.n = len(x0)
        self._I = np.eye(self.n)
        self.y = 1e6
        self.Wc, self.Wm = self.__get_weights(self.n, abk)
        self.abk = abk

    def predict(self, fx, dt):
        x = self.x
        P = self.P

        sigmas = self.__get_sigma_points(x, P, self.abk) # get sigma points
        self._num_sigmas = len(sigmas)

        self.sigmas_f = np.zeros_like(sigmas)

        for i in range(self._num_sigmas):
            self.sigmas_f[i] = fx(sigmas[i], dt)

        self.x, self.Pp = self.__unscented_transform(self.sigmas_f, self.Q) # unscented transform

    def update(self, z, drone_pos, hx):

        sigmas_f = self.sigmas_f
        _dim_z = len(z)

        # get sigma points for measurement update
        sigmas_h = np.zeros((self._num_sigmas, _dim_z))
        for i in range(self._num_sigmas):
            sigmas_h[i] = hx(drone_pos, sigmas_f[i])

        # unscented transform
        zp, Pz = self.__unscented_transform(sigmas_h, self.R)

        Pxz = np.zeros((self.n, _dim_z))

        for i in range(self._num_sigmas):
            Pxz += self.Wc[i] * np.outer(sigmas_f[i] - self.x, sigmas_h[i] - zp)

        K = Pxz @ np.linalg.inv(Pz)

        self.x = self.x + K @ (z - zp)
        self.P = self.Pp - K @ Pz @ K.T

        # print(f'y: {self.y}')
        self.logger.cov.append(self.P)
        self.logger.x.append(self.x)
        self.logger.z.append(z)
        self.logger.y.append(np.linalg.norm(self.y))
        self.logger.drone_pos.append(drone_pos)
        self.logger.range.append(np.linalg.norm(self.x[:2] - drone_pos[:2]))

    def __get_weights(self, n, abk):
        alpha, beta, kappa = abk
        lambda_ = alpha**2 * (n + kappa) - n
        Wc = np.full(2*n+1, 1/(2*(n+lambda_)))
        Wm = np.full(2*n+1, 1/(2*(n+lambda_)))
        Wm[0] = lambda_/(n+lambda_)
        Wc[0] = lambda_/(n+lambda_) + (1 - alpha**2 + beta)

        return Wc, Wm

    def __get_sigma_points(self, x, P, abk):
        n = self.n
        
        # get sigma points
        alpha, _, kappa = abk
        lambda_ = alpha**2 * (n + kappa) - n
        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = x

        u = cholesky((n+lambda_)*P) # equivalent to sqrt((n+lambda_)*P)

        for i in range(n):
            sigmas[i+1] = x + u[i]
            sigmas[n+i+1] = x - u[i]

        return sigmas

    def __unscented_transform(self, sigmas_f, Q):
        x = np.dot(self.Wm, sigmas_f)
        kmax, n = sigmas_f.shape
        P = np.zeros((n, n))

        for k in range(kmax):
            y = sigmas_f[k] - x
            P += self.Wc[k] * np.outer(y, y)
        
        P += Q
        return x, P

    def get_state(self):
        """Returns the current state of the system"""
        return self.x

    def get_covariance(self):
        """Returns the current covariance of the system"""
        return self.P