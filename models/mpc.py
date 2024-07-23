import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

class MPC:
    def __init__(self, N:int, dt:float, drone_transition:callable, kf:object, jHx:callable, jFx:callable, r=1, constraints=None):
        """
        Model Predictive Controller class

        Parameters
        ----------
        N : int - Number of time steps
        dt : float - Time step
        drone_transition : callable - Transition function of the drone
        kf : object - Kalman filter object
        """
        self.warm_X = None
        self.warm_U = None

        self.dt = dt
        self.f = drone_transition
        self.N = N
        self.kf = kf
        self.r = r # control penalty

        self.jHx = jHx  
        self.jFx = jFx

        # create optimizer container and define its optimization variables
        self.opti = ca.Opti()
        self.X = self.opti.variable(6, N+1)
        self.U = self.opti.variable(3, N+1)

        self.x = self.opti.parameter(4,1) # kalman filter state
        self.P = self.opti.parameter(4,4) # kalman filter covariance


        # apply initial condition constraints
        self.init = self.opti.parameter(6,1)
        self.opti.subject_to(self.X[:,0] == self.init)

        # apply dynamics constraints with euler integrator
        for k in range(N):
            self.opti.subject_to(self.X[:,k+1] == drone_transition(self.X[:,k], self.U[:,k], dt))

        # apply state constraints
        for k in range(N+1):
            self.opti.subject_to(self.X[2,k] >= 75) # altitude constraint 
            self.opti.subject_to(self.X[2,k] <= 120) # altitude constraint
            self.opti.subject_to(self.X[3,k]**2 + self.X[4,k]**2 + self.X[5,k]**2 < 10**2) # velocity constraint

        # define input constraints
        self.opti.subject_to(self.opti.bounded(-5, self.U, 5)) # input constraints

        # apply custom constraints
        if constraints is not None:
            for constraint in constraints:
                constraint.apply(self.opti, self.X)
        
        self.opti.minimize(self.J(self, self.x, self.P, self.X, self.U, self.r))

        # tell the opti container we want to use IPOPT to optimize, and define settings for the solver
        opts = {
            'ipopt.print_level':0, 
            'print_time':0,
            'ipopt.tol': 1e-6
        } # silence!
        self.opti.solver('ipopt', opts)

    def __call__(self, drone_x0):
        """Solve the optimization problem"""
        self.opti.set_value(self.init, drone_x0)
        self.opti.set_value(self.x, self.kf.x)
        self.opti.set_value(self.P, self.kf.P)

        if self.warm_X is not None:
            self.opti.set_initial(self.X, self.warm_X)
            self.opti.set_initial(self.U, self.warm_U)

        # solve the optimization problem
        sol = self.opti.solve()
        u_sol = sol.value(self.U)


        # update warm start
        self.warm_X = np.concatenate((sol.value(self.X)[:,1:], sol.value(self.X)[:,-1].reshape(-1,1)), axis=1)
        self.warm_U = np.concatenate((sol.value(self.U)[:,1:], sol.value(self.U)[:,-1].reshape(-1,1)), axis=1)

        cost = sol.value(self.opti.f)
        return u_sol[:,0], cost


    def estimate_step_forward(self, x, P, drone_state, u): # output: x, P, u
        """Estimate the state and covariance of the kalman filter after dt seconds
        given the current state x, covariance P, control input u, and drone state transition function x_{t+1} = f(x_{t}, u, dt)
        
        returns `x`, `P`, `u`, `new_state`"""

        new_state = self.f(drone_state, u, self.dt)
        new_pos = new_state[:3]

        jFfun = self.jFx
        jHfun = self.jHx
        Q = self.kf.Q
        R = self.kf.R
        
        jF = jFfun(x, self.dt)
        jH = jHfun(x, new_pos)

        # step state forward
        new_x = jF@x

        # step covariance forward
        P_prior = jF@P@jF.T + Q
        S = jH@P_prior@jH.T + R
        K = P_prior@jH.T@ca.solve(S, ca.MX.eye(2))
        new_P = (ca.MX.eye(4) - K@jH)@P_prior

        return new_x, new_P

    @staticmethod
    def J(self, x0, P0, X, U, r):
            """Cost function"""
            x = x0
            P = P0

            cost = 0
            for k in range(self.N):
                x, P = self.estimate_step_forward(x, P, X[:,k], U[:,k]) 
                cost += 10 * ca.sqrt(P[0,0]**2 + P[0,1]**2 + P[1,1]**2)  # using frobenius norm of P
                cost += r * ca.dot(U[:,k], U[:,k])
                # add regularization term to reward smooth control inputs
                cost += 0.1 * ca.mtimes([(U[:, k] - U[:, k-1]).T, (U[:, k] - U[:, k-1])])
            return cost
    
    def get_predictions(self):
        """Get the predictions of the optimisation problem"""
        return self.opti.value(self.X), self.opti.value(self.U)