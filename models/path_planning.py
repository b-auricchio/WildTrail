import numpy as np

# Cost functions
def evalue_trace(P):
    """Calculate the trace of eigenvalues of the covariance matrix P - used as cost function"""
    evals, _ = np.linalg.eig(P)
    return np.sum(evals[:2])

class Node:
    """Node class for RRT-like path planning"""
    def __init__(self, u, value, x, P, parent, state):
        self.u = u
        self.value = value
        self.x = x
        self.P = P
        self.state = state
        self.parent = parent

    def get_control_path(self):
        """Get the path from the root to this node"""
        path = [self.u]
        node = self
        while node.parent is not None:
            node = node.parent
            if node.u is not None:
                path.append(node.u)
            else: break

        return path
    
    def get_state_path(self):
        """Get the path from the root to this node"""
        path = [self.state]
        node = self

        while node.parent is not None:
            node = node.parent
            path.append(node.state)

        return np.array(path)

    def __repr__(self):
        return f"u:{str(self.u)}, value: {self.value}, rank:{len(self.get_control_path())-1}, pos: {self.state}\n"

class PathPlanner:
    """PathPlanner class for RRT-like path planning
    ----------------------------------------------
    Inputs:
        dt: time step
    
    Additional functions:
        drone_transition: function that takes in drone state and control input and returns the next drone state - x_{t+1} = f(x_{t}, u, dt)
                    
    """

    def __init__(self, dt, transition_function):
        self.dt = dt
        self.transition_function = transition_function

        self.uarray = np.array([[0, 0, 0], 
                    [0, 0, 1], [0, 0, -1], [0, 1, 0], 
                    [0, -1, 0], [1, 0, 0], [-1, 0, 0],
                    [0, 1, 1], [0, 1, -1], [1, 1, 0], 
                    [1, -1, 0], [1, 0, 1], [-1, 0, 1],
                    [0, -1, 1], [0, -1, -1], [-1, 1, 0], 
                    [-1, -1, 0], [1, 0, -1], [-1, 0, -1]]) # possible control inputs

        
    def estimate_step_forward(self, x, P, kf, drone_state, u): # output: x, P, u
        """Estimate the state and covariance of the drone after dt seconds
        given the current state x, covariance P, control input u, and drone state transition function x_{t+1} = f(x_{t}, u, dt)
        
        returns `x`, `P`, `u`, `new_state`"""

        new_state = self.transition_function(drone_state, u, self.dt)
        new_pos = new_state[:3]

        jFfun = kf.jFx
        jHfun = kf.jHx
        Q = kf.Q
        R = kf.R
        
        jF = jFfun(x, self.dt)
        jH = jHfun(x, new_pos)

        # step state forward
        x = jF@x

        # step covariance forward
        P_prior = jF@P@jF.T + Q
        S = jH@P_prior@jH.T + R
        K = P_prior@jH.T@np.linalg.inv(S)
        P = (np.eye(4) - K@jH)@P_prior

        return x, P, u, new_state
    
    def get_best_nodes(self, num, kf, parent_nodes=[None]):
        """Generate a tree of possible states and covariances given a discrete set of control inputs"""
        nodes = []
        for parent in parent_nodes:
            # generate tree of possible states, sort by cost function and return the num best nodes
            tree = [(self.estimate_step_forward(parent.x, parent.P, kf, parent.state, u)) for u in self.uarray]

            for leaf in (tree):
                v = evalue_trace(leaf[1])

                # If velocity is too high, penalize the node
                if np.linalg.norm(leaf[3][3:]) > 5:
                    v *= 100

                nodes.append(Node(leaf[2], v, leaf[0], leaf[1], parent, state=leaf[3]))

        # sort nodes by cost function and take the num best nodes
        nodes = sorted(nodes, key=lambda x: x.value)
        nodes = nodes[:num]

        return nodes
    
    def generate_nodes(self, num, timesteps, drone_state, kf):
        """Generate a list of nodes given a discrete set of control inputs"""
        
        nodes = np.empty((timesteps, num), dtype=Node)
        parents = [Node(None, None, kf.get_state(), kf.get_covariance(), None, drone_state)]
        
        for t in range(timesteps):
            nodes[t] = self.get_best_nodes(num, kf, parents)
            parents = nodes[t]

        return nodes
    
    def get_best_input(self, num, timesteps, drone_state, kf):
        nodes = self.generate_nodes(num, timesteps, drone_state, kf)
        """Get the best input from the list of nodes"""
        last_nodes = nodes[-1]
        umin = last_nodes[0].get_control_path()[-1]

        return umin, last_nodes
    
class RandomThrustPlanner(PathPlanner):
    def __init__(self, dt, transition_function):
        super().__init__(dt, transition_function)

    def random_thrust_vector(min_mag=0.5, max_mag=2):
        # generate random thrust vector
        mag = np.random.uniform(min_mag, max_mag)
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        x = np.sin(phi)*np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(phi)

        return mag*np.array([x, y, z])