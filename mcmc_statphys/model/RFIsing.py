# -*- encoding: utf-8 -*-
'''
@File    :   Ising.py
@Time    :   2023/05/31 11:47:09
@Author  :   UynajGI
@Contact :   suquan12148@outlook.com
@License :   (MIT)Copyright 2023
'''

# here put the import lib
from typing import Any, Tuple
import numpy as np
from .Ising import Ising
import pandas as pd
import copy

__all__ = ["RFIsing"]


class RFIsing(Ising):

    def __init__(self,
                 L: int,
                 Jij: float = 1,
                 Hmean: float = 0,
                 Hsigma: float = 1,
                 Hform: str = "norm",
                 dim: int = 2,
                 *args: Any,
                 **kwargs: Any):
        H = self._init_H(Hmean=Hmean, Hsigma=Hsigma, Hform=Hform)
        super().__init__(L=L, Jij=Jij, H=H, dim=dim, *args, **kwargs)
        self._init_spin(type="rfising")
        self._get_total_energy()
        self._get_total_magnetization()
        self._max_energy()

    def _init_H(self, Hmean, Hsigma, Hform):
        if Hform == "norm":
            H = np.random.normal(Hmean, Hsigma, (self.L, ) * self.dim)
        elif Hform == "uniform":
            H = np.random.choice([-Hsigma, Hsigma], size=(self.L, ) * self.dim)
        else:
            raise ValueError("Invalid Hform")
        return H

    def _get_site_energy(self, index: Tuple[int, ...]) -> float:
        """Get the energy of the site / cn: 获取格点的能量

        Args:
            index (Tuple[int, ...]): The index of the site / cn: 格点的坐标

        Raises:
            ValueError: Invalid type of spin / cn: 无效的自旋类型

        Returns:
            float: The energy of the site / cn: 格点的能量
        """
        neighbors_spin = self._get_neighbor_spin(index)
        energy = 0
        for neighbor_spin in neighbors_spin:
            energy -= self.Jij * np.dot(self.spin[index], neighbor_spin)
        energy -= self.H[index] * self.spin[index]
        return energy

    def _init_data(self):
        data: pd.DataFrame = pd.DataFrame(columns=[
            "uid",
            "iter",
            "T",
            "H",
            "energy",
            "magnetization",
            "spin",
        ])
        data.set_index(["uid", "iter"], inplace=True)
        return data

    def _save_date(self, T, uid, data):
        if uid not in data.index.get_level_values("uid").values:
            data.loc[(uid, 1), :] = [
                T,
                0,
                self.energy,
                self.magnetization,
                0,
            ]
            data.at[(uid, 1), "spin"] = copy.deepcopy(self.spin)
            data.at[(uid, 1), "H"] = self.H
        else:
            iterplus = data.loc[uid].index.max() + 1
            data.loc[(uid, iterplus), :] = [
                T,
                0,
                self.energy,
                self.magnetization,
                0,
            ]
            data.at[(uid, iterplus), "spin"] = copy.deepcopy(self.spin)
            data.at[(uid, iterplus), "H"] = self.H
        return data
