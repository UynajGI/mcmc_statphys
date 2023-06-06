import numpy as np
import copy
from typing import Tuple
from scipy.spatial.distance import pdist, squareform
import pandas as pd

__all__ = ["NVT"]


def potential(r: float, r0: float, r1: float, epsilon: float,
              type: str) -> float:
    if type == "hard-core":
        if r < r0:
            return np.inf
        else:
            return 0
    elif type == "lennard-jones":
        return 4 * epsilon * ((r0 / r)**12 - (r0 / r)**6)
    elif type == "soft-core":
        if r < r0:
            return np.inf
        elif r < r1 and r >= r0:
            return -epsilon
        else:
            return 0


class NVT():

    def __init__(self,
                 N: int,
                 V: float,
                 r0: float,
                 r1: float,
                 epsilon: float,
                 poten="lennard-jones"):
        self.N = N
        self.L = N
        self.dim = 1
        self.V = V
        self.poten = poten
        self.r0 = r0
        self.r1 = r1
        self.epsilon = epsilon
        self._init_spin(type="nvt")
        self._get_total_energy()

    def __len__(self):
        return self.L

    def _init_spin(self, type="nvt", *args, **kwargs):
        self.spin = np.random.uniform(0, self.V**(1 / 3), size=(self.N, 3))
        self.distance = squareform(pdist(self.spin))
        self.type = type

    def _get_total_energy(self) -> float:
        energy = 0
        for i in range(self.L):
            for j in range(i + 1, self.L):
                energy += potential(self.distance[i, j],
                                    r0=self.r0,
                                    r1=self.r1,
                                    epsilon=self.epsilon,
                                    type=self.poten)
        self.energy = energy
        return energy

    def _get_per_energy(self) -> float:
        return self._get_total_energy() / self.N

    def _change_site_spin(self, index: Tuple[int, ...]):
        self.spin[index] = np.random.uniform(0, self.V**(1 / 3), size=(1, 3))
        self.distance = squareform(pdist(self.spin))

    def _change_delta_energy(self, index: Tuple[int, ...]):
        """Get the delta energy of the site"""
        old_energy = copy.deepcopy(self.energy)
        self._change_site_spin(index)
        new_energy = self._get_total_energy()
        detle_energy = new_energy - old_energy
        return detle_energy

    def set_spin(self, spin):
        self.spin = spin
        self._get_total_energy()

    def get_energy(self) -> float:
        return self.energy

    def _init_data(self):
        data: pd.DataFrame = pd.DataFrame(columns=[
            "uid",
            "iter",
            "T",
            "energy",
            "spin",
        ])
        data.set_index(["uid", "iter"], inplace=True)
        return data

    def _save_date(self, T, uid, data):
        if uid not in data.index.get_level_values("uid").values:
            data.loc[(uid, 1), :] = [
                T,
                self.model.energy,
                0,
            ]
            data.at[(uid, 1), "spin"] = copy.deepcopy(self.model.spin)
        else:
            iterplus = data.loc[uid].index.max() + 1
            data.loc[(uid, iterplus), :] = [
                T,
                self.model.energy,
                0,
            ]
            data.at[(uid, iterplus), "spin"] = copy.deepcopy(self.model.spin)
        return data
