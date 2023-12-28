import numpy as np
class KalmanLogger():
    def __init__(self) -> None:
        self.cov: list = []
        self.x: list = []
        self.z: list = []
        self.y: list = []
        self.drone_pos: list = []
        self.range: list = []

    def to_numpy(self) -> tuple:
        self.cov = np.array(self.cov)
        self.x = np.array(self.x)
        self.z = np.array(self.z)
        self.y = np.array(self.y)
        self.drone_pos = np.array(self.drone_pos)
        self.range = np.array(self.range)

    def __str__(self) -> str:
        return f'covs: {len(self.covs)}\nxs: {len(self.xs)}\nzs: {len(self.zs)}\nys: {len(self.ys)}\ndrone_pos: {len(self.drone_pos)}'
    
    def __repr__(self) -> str:
        return f'covs: {len(self.covs)}\nxs: {len(self.xs)}\nzs: {len(self.zs)}\nys: {len(self.ys)}\ndrone_pos: {len(self.drone_pos)}'

