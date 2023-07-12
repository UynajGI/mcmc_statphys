import numpy as np
import copy
from typing import Tuple
from scipy.spatial.distance import pdist, squareform
import pandas as pd

__all__ = ["NVT"]


def V_LJ(r0, epsilon, **kwargs):
    if "r" not in kwargs.keys():

        def func(r):
            return 4 * epsilon * ((r0 / r) ** 12 - (r0 / r) ** 6)

        return func
    else:
        r = kwargs["r"]
        return 4 * epsilon * ((r0 / r) ** 12 - (r0 / r) ** 6)


def V_hc(r0, **kwargs):
    if "r" not in kwargs.keys():

        def func(r):
            if r < r0:
                return 1e10
            else:
                return 0

        return func
    else:
        r = kwargs["r"]
        if r < r0:
            return 1e10
        else:
            return 0


def V_sc(r0, r1, epsilon, **kwargs):
    if "r" not in kwargs.keys():

        def func(r):
            if r < r0:
                return 1e10
            elif r < r1 and r >= r0:
                return -epsilon
            else:
                return 0

        return func
    else:
        r = kwargs["r"]
        if r < r0:
            return 1e10
        elif r < r1 and r >= r0:
            return -epsilon
        else:
            return 0


class NVT:
    def __init__(self, N: int, V: float, delta: float, potential=V_LJ(r0=0.1, epsilon=0.5)):
        self.N = N
        self.L = N
        self.dim = 1
        self.V = V
        self.delta = delta
        self.potential = potential
        self._init_spin(type="nvt")
        self._get_total_energy()

    def __len__(self):
        return self.L

    def _init_spin(self, type="nvt"):
        self.spin = np.random.uniform(0, self.V ** (1 / 3), size=(self.N, 3))
        self.distance = squareform(pdist(self.spin))
        self.type = type

    def _get_total_energy(self) -> float:
        energy = np.sum(self.potential(self.distance)) / 2
        self.energy = energy
        return energy

    def _get_per_energy(self) -> float:
        return self._get_total_energy() / self.N

    def _change_site_spin(self, index: Tuple[int, ...]):
        # 在 index 的各个方向增加 delta
        self.spin[index] += np.random.uniform(-self.delta, self.delta, size=(1, len(self.spin[index])))
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
        data: pd.DataFrame = pd.DataFrame(
            columns=[
                "uid",
                "iter",
                "T",
                "energy",
                "spin",
            ]
        )
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
