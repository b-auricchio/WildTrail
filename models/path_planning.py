import numpy as np

# Cost functions
def evalue_trace(P):
    """Calculate the trace of eigenvalues of the covariance matrix P - used as cost function"""
    evals, _ = np.linalg.eig(P)
    return np.sum(evals[:2])

def estimate_step_forward(x, drone_pos, P, jFfun, jHfun, Q, R, dt):
    """Estimate the state and covariance of the drone after dt seconds
    given the current state x, covariance P and control input u"""

    jF = jFfun(x, dt)
    jH = jHfun(x, drone_pos)

    # step state forward
    x = jF@x

    # step covariance forward
    P_prior = jF@P@jF.T + Q
    S = jH@P_prior@jH.T + R
    K = P_prior@jH.T@np.linalg.inv(S)
    P = (np.eye(4) - K@jH)@P_prior

    return x, P, drone_pos

class DiscreteInputPathPlanner:
    """Path planner with discrete set of control inputs"""
    def __init__(self, min_range=None, altitude_bounds= None):
        self.min_range = min_range
        self.altitude_bounds = altitude_bounds

            # u array
        self.uarray = np.array([[0, 0, 0], 
                    [0, 0, 1], [0, 0, -1], [0, 1, 0], 
                    [0, -1, 0], [1, 0, 0], [-1, 0, 0],
                    [0, 1, 1], [0, 1, -1], [1, 1, 0], 
                    [1, -1, 0], [1, 0, 1], [-1, 0, 1],
                    [0, 1, 1], [0, -1, -1], [-1, 1, 0], 
                    [1, -1, 0], [1, 0, -1], [-1, 0, 1]]) # possible control inputs
    
    def generate_tree(self,kf, drone, dt):
        """Generate a tree of possible states and covariances given a discrete set of control inputs"""
        tree = [(estimate_step_forward(kf.get_state(), drone.get_pos()+u, kf.get_covariance(), kf.jFx, kf.jHx, kf.Q, kf.R, dt)) for u in self.uarray]
        return tree
    
    def get_best_control_inputs(self, tree, num):
        """Get the best control input from the tree"""
        indices = np.argsort([evalue_trace(P) for _, P, _ in tree])[:num]
        return self.uarray[indices] # return the best num control inputs
    
    def get_next_position(self, kf, drone, dt, timesteps):
        """Get the next position of the drone"""
        tree = self.generate_tree(kf, drone, dt)
        umin = self.get_best_control_inputs(tree, 1)
        return umin, drone.get_pos() + umin
    


# def enforce_constraints(tree, min_range, altitude_bounds = [20, 150]):
#     """Enforce constraints on the control inputs"""
#     new_tree = []
#     for x, P, drone_pos in tree:
#         if np.linalg.norm(x[:2] - drone_pos[:2]) >= min_range and drone_pos[2] >= altitude_bounds[0] and drone_pos[2] <= altitude_bounds[1]:
#             new_tree.append([np.array(x), np.array(P), np.array(drone_pos)])
#         else:
#             print("------------------\nConstraint violated")
#             print("altitude: ", drone_pos[2])
#             print("range: ", np.linalg.norm(x[:2] - drone_pos[:2]))

#     if len(new_tree) == 0:
#         print("------------------\nAll constraints violated")
#         print("altitude: ", drone_pos[2])
#         print("ranges: ", [np.linalg.norm(x[:2] - drone_pos[:2]) for x, _, drone_pos in tree])
#         raise Exception("No control inputs satisfy constraints")

#     return new_tree