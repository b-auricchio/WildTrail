import numpy as np
import casadi as ca

class GetJacobianH:
    @staticmethod
    def numpy(state, drone_pos:np.array):
        """Calculate the jacobian of the measurement function at the previous target position and drone position"""
        x, y = state[:2]
        x_d = drone_pos[0]
        y_d = drone_pos[1]
        z_d = drone_pos[2]

        denom1 = np.sqrt((x-x_d)**2 + (y-y_d)**2) * (z_d**2+(x-x_d)**2 + (y-y_d)**2)
        denom2 = (x-x_d)**2 + (y-y_d)**2

        e = ([-z_d*(x-x_d)/denom1, -z_d*(y-y_d)/denom1, (y_d-y)/denom2, (x-x_d)/denom2])

        jacobian = np.array([[e[0], e[1], 0, 0], [e[2], e[3], 0, 0]])

        return jacobian
    
    @staticmethod
    def casadi(state, drone_pos:ca.MX): # drone_pos = [x, y, z], prev_pos = [x, y]
        """Calculate the jacobian of the measurement function at the previous target position and drone position"""
        x = state[0]
        y = state[1]
        x_d = drone_pos[0]
        y_d = drone_pos[1]
        z_d = drone_pos[2]

        denom1 = ca.sqrt((x-x_d)**2 + (y-y_d)**2) * (z_d**2+(x-x_d)**2 + (y-y_d)**2)
        denom2 = (x-x_d)**2 + (y-y_d)**2

        e = ([-z_d*(x-x_d)/denom1, -z_d*(y-y_d)/denom1, (y_d-y)/denom2, (x-x_d)/denom2])

        jacobian = ca.MX(2, 4)
        jacobian[0,0] = e[0]
        jacobian[0,1] = e[1]
        jacobian[1,0] = e[2]
        jacobian[1,1] = e[3]
        return jacobian
    
class GetJacobianF:
    @staticmethod
    def numpy(state, dt):
        """Calculate the symbolic jacobian of the transition function at the previous target position"""
        a, v = state[2], state[3]
        jF = np.eye(4)
        jF[0,2] = -dt*v*np.sin(a)
        jF[0,3] = dt*np.cos(a)
        jF[1,2] = dt*v*np.cos(a)
        jF[1,3] = dt*np.sin(a)
        return jF
    
    @staticmethod
    def casadi(state, dt):
        """Calculate the symbolic jacobian of the transition function at the previous target position"""
        a = state[2]
        v = state[3]
        jF = ca.MX.eye(4)
        jF[0,2] = -dt*v*ca.sin(a)
        jF[0,3] = dt*ca.cos(a)
        jF[1,2] = dt*v*ca.cos(a)
        jF[1,3] = dt*ca.sin(a)
        return jF