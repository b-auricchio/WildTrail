import numpy as np

class Logger:
    def __init__(self, dt):
        """
        A class to log the data during the simulation.
        """
        self.track = []
        self.drone_state = []
        self.kf_state = []
        self.kf_cov = []
        self.predictions = []
        self.ctrl = []
        self.dt = dt

    def to_numpy(self):
        return np.array(self.track), np.array(self.drone_state), np.array(self.kf_state), np.array(self.kf_cov), np.array(self.predictions), np.array(self.ctrl)
    
    def __len__(self):
        return len(self.track)