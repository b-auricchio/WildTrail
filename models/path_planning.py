import numpy as np

def random_thrust_vector(min_mag=0.5, max_mag=2):
    # generate random thrust vector
    mag = np.random.uniform(min_mag, max_mag)
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, np.pi)
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)

    return mag*np.array([x, y, z])

# Cost functions
def evalue_trace(P):
    """Calculate the trace of eigenvalues of the covariance matrix P"""
    evals, _ = np.linalg.eig(P)
    return np.sum(evals[:2])

def energy(u_array):
    """Calculate the total energy from series of control inputs"""
    return np.sum(np.linalg.norm(u_array, axis=1))

def reward(P, u_array, weight_R1=1, weight_R2=0.5):
    """Calculate the reward for a give node"""
    return weight_R1 * evalue_trace(P) + weight_R2 * energy(u_array)

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
        if self.u is None:
            return []
        else:
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

    def __init__(self, dt, transition_function, v_bounds=5, alt_bounds=[75, 120], r1_weight=2,  r2_weight=0.5):
        self.dt = dt
        self.transition_function = transition_function

        self.v_bounds = v_bounds
        self.alt_bounds = alt_bounds
        self.r1_weight = r1_weight
        self.r2_weight = r2_weight

        uarray = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], 
                    [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0],
                    [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
                    [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
                    [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
                    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]) 

        norm = np.linalg.norm(uarray, axis=1, keepdims=True)
        self.uarray = np.vstack((uarray / norm, np.array([0,0,0])))

        
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
        # self.uarray = np.array([random_thrust_vector() for _ in range(30)])
        for parent in parent_nodes:
            # generate tree of possible states, sort by cost function and return the num best nodes
            tree = [(self.estimate_step_forward(parent.x, parent.P, kf, parent.state, u)) for u in self.uarray]

            for leaf in (tree):
                x = leaf[0]
                P = leaf[1]
                u = leaf[2]
                u_path = parent.get_control_path()
                u_path.append(u)

                state = leaf[3]
                v = reward(P, u_path, self.r1_weight, self.r2_weight)

                # Apply soft constraints to the control inputs
                if (np.linalg.norm(state[3:]) > self.v_bounds[0] or np.linalg.norm(state[3:]) < self.v_bounds[1]
                    or state[2] < self.alt_bounds[0] or state[2] > self.alt_bounds[1]):
                    v *= 5
                
                nodes.append(Node(u, v, x, P, parent, state=state))

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
    
    def get_best_input(self, num_inputs, num_nodes, timesteps, drone_state, kf):
        nodes = self.generate_nodes(num_nodes, timesteps, drone_state, kf)
        """Get the best input from the list of nodes"""
        last_nodes = nodes[-1]
        umin = last_nodes[0].get_control_path()[-1*num_inputs:]

        if len(umin) == 1:
            umin = [umin[0]]

        return umin, last_nodes
