import numpy as np

# Cost functions
def evalue_trace(P):
    """Calculate the trace of eigenvalues of the covariance matrix P - used as cost function"""
    evals, _ = np.linalg.eig(P)
    return np.sum(evals[:2])

class Node:
    """Node class for RRT-like path planning"""
    def __init__(self, u, value, x, P, parent, position):
        self.u = u
        self.value = value
        self.x = x
        self.P = P
        self.position = position
        self.parent = parent

    def get_path(self):
        """Get the path from the root to this node"""
        path = [self.u]
        node = self
        while node.parent is not None:
            node = node.parent
            path.append(node.u)
        return path

    def __repr__(self):
        return f"u:{str(self.u)}, value: {self.value}, rank:{len(self.get_path())-1}, pos: {self.position}\n"

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
                    [0, -1, 1], [0, -1, -1], [-1, 1, 0], 
                    [-1, -1, 0], [1, 0, -1], [-1, 0, -1]]) # possible control inputs
        
    def estimate_step_forward(self, x, P, drone_pos, u, kf, dt):
        """Estimate the state and covariance of the drone after dt seconds
        given the current state x, covariance P and control input u"""

        drone_pos = drone_pos + u

        jFfun = kf.jFx
        jHfun = kf.jHx
        Q = kf.Q
        R = kf.R
        
        jF = jFfun(x, dt)
        jH = jHfun(x, drone_pos)

        # step state forward
        x = jF@x

        # step covariance forward
        P_prior = jF@P@jF.T + Q
        S = jH@P_prior@jH.T + R
        K = P_prior@jH.T@np.linalg.inv(S)
        P = (np.eye(4) - K@jH)@P_prior

        return x, P, u
    
    def get_best_nodes(self, num, kf, dt, parent_nodes=[None]):
        """Generate a tree of possible states and covariances given a discrete set of control inputs"""

        nodes = []
        for parent in parent_nodes:
            
            # generate tree of possible states, sort by cost function and return the num best nodes
            tree = [(self.estimate_step_forward(parent.x, parent.P, parent.position, u, kf, dt)) for u in self.uarray]

            for leaf in (tree):
                nodes.append(Node(leaf[2], evalue_trace(leaf[1]), leaf[0], leaf[1], parent, position=parent.position+leaf[2]))

        # sort nodes by cost function and take the num best nodes
        nodes = sorted(nodes, key=lambda x: x.value)
        nodes = nodes[:num]

        return nodes
    
    def generate_nodes(self, num, timesteps, drone_pos, kf, dt):
        """Generate a list of nodes given a discrete set of control inputs"""
        nodes = np.empty((timesteps, num), dtype=Node)
        parents = [Node([0, 0, 0], 0, kf.get_state(), kf.get_covariance(), None, drone_pos)]
        
        for t in range(timesteps):
            nodes[t] = self.get_best_nodes(num, kf, dt, parents)
            parents = nodes[t]

        return nodes
    
    def get_best_input(self, num, timesteps, drone_pos, kf, dt):
        nodes = self.generate_nodes(num, timesteps, drone_pos, kf, dt)
        """Get the best input from the list of nodes"""
        umin = nodes[-1, 0].get_path()[-2]
        best_node = nodes[-1, 0]

        return umin, best_node