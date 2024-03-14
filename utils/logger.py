import numpy as np  
import pandas as pd
import sys
sys.path.append('../')

from models.path_planning import evalue_trace

class KalmanLogger():
    def __init__(self) -> None:
        self.cov: list = []
        self.x: list = []
        self.z: list = []
        self.y: list = []
        self.drone_state: list = []
        self.drone_pos: list = []
        self.range: list = []

    def to_numpy(self) -> tuple:
        self.cov = np.array(self.cov)
        self.x = np.array(self.x)
        self.z = np.array(self.z)
        self.y = np.array(self.y)
        self.drone_state = np.array(self.drone_state)
        self.drone_pos = np.array(self.drone_pos)

        self.range = np.array(self.range)

    def to_csv(self, path: str, track) -> None:
        self.to_numpy()
        track = np.array(track)
        df = pd.DataFrame({'cost': [evalue_trace(cov) for cov in self.cov], 
                           'error' : [np.linalg.norm(x-t) for x, t in zip(self.x[:,:2], track[:,:2])],
                           'drone_alt': self.drone_state[:,2],
                           'drone_vel': np.linalg.norm(self.drone_state[:,3:], axis=1)})
        
        df.to_csv(path, index=False)

    def get_rms_error(self, track) -> float:
        self.to_numpy()
        track = np.array(track)
        errors = [np.linalg.norm(x-t) for x, t in zip(self.x[:,:2], track[:,:2])]
        return np.sqrt(sum(errors)**2) / len(errors)
        

    def __str__(self) -> str:
        return f'covs: {len(self.covs)}\nxs: {len(self.xs)}\nzs: {len(self.zs)}\nys: {len(self.ys)}\ndrone_pos: {len(self.drone_state)}'
    
    def __repr__(self) -> str:
        return f'covs: {len(self.covs)}\nxs: {len(self.xs)}\nzs: {len(self.zs)}\nys: {len(self.ys)}\ndrone_pos: {len(self.drone_state)}'

