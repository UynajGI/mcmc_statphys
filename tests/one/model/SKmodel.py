# -*- encoding: utf-8 -*-
'''
@File    :   SKmodel.py
@Time    :   2023/05/31 11:50:34
@Author  :   UynajGI
@Contact :   suquan12148@outlook.com
@License :   (MIT)Copyright 2023
'''

# here put the import lib
from typing import Tuple
import numpy as np
import copy

__all__ = ["SKmodel"]


class SKmodel():
    # SKModel is the Sherrington-Kirkpatrick model
    def __init__(self, N, Jmean=0, Jsigma=1, H=0, *args, **kwargs):
        self.Jmean = Jmean
        self.Jsigma = Jsigma
        self.N = N
        self.H = H
        self.L = self.N
        self.dim = 1
        self._init_spin(type="SK")
        self._init_Jij()
        self._get_total_energy()
        self._get_total_magnetization()
        self._max_energy()

    def _init_spin(self, type="SK"):
        self.spin = np.random.choice([-1, 1], size=(self.N, ))
        self.type = type

    def _init_Jij(self):
        self.Jij = np.random.normal(self.Jmean / self.N,
                                    self.Jsigma / np.sqrt(self.N),
                                    (self.N, self.N))
        self.Jij = (self.Jij + self.Jij.T) / 2
        np.fill_diagonal(self.Jij, 0)
        self.Jij = self.Jij.astype(np.float32)

    def _get_per_energy(self) -> float:
        """Get the per energy of the system / cn: 获取系统的单位能量

        Returns:
            float: The per energy of the system / cn: 系统的单位能量
        """
        return self._get_total_energy() / self.N

    def _get_total_energy(self) -> float:
        """Get the total energy of the system / cn: 获取系统的总能量

        Returns:
            float: The total energy of the system / cn: 系统的总能量
        """
        self.energy = -1 / 2 * np.dot(self.spin, np.dot(self.Jij, self.spin))
        self.energy -= np.sum(self.H * self.spin)
        return self.energy

    def _get_per_magnetization(self) -> float:
        """Get the per magnetization of the system / cn:

        Returns:
            float: The per magnetization of the system / cn: 系统的单位磁矩
        """
        return self._get_total_magnetization() / self.N

    def _get_total_magnetization(self) -> float:
        """Get the magnetization of the system / cn: 获取系统的总磁矩

        Returns:
            float: The magnetization of the system / cn: 系统的总磁矩
        """
        self.magnetization = np.sum(self.spin)
        return self.magnetization

    # 获取类的属性
    def _get_info(self) -> Tuple[np.ndarray, float, float]:
        """Get the info of the system / cn: 获取系统的信息

        Returns:
            Tuple[np.ndarray, float, float]: The info of the system / cn: 系统的信息
        """
        return self.spin, self.energy, self.magnetization

    def _get_per_info(self):
        return self.spin, self._get_per_energy(), self._get_per_magnetization()

    def _change_site_spin(self, index: Tuple[int, ...]):
        """Change the spin of the site / cn: 改变格点的自旋

        Args:
            index (Tuple[int, ...]): The index of the site / cn: 格点的坐标

        Raises:
            ValueError: Invalid type of spin / cn: 无效的自旋类型
        """
        self.spin[index] *= -1

    def _change_delta_energy(self, index: Tuple[int, ...]):
        """Get the delta energy of the site"""
        old_energy = copy.deepcopy(self.energy)
        self._change_site_spin(index)
        new_energy = self._get_total_energy()
        detle_energy = new_energy - old_energy
        self.magnetization += 2 * self.spin[index]
        return detle_energy

    def _max_energy(self):
        raw_spin = copy.deepcopy(self.spin)
        self.set_spin(np.ones_like(self.spin))
        self.maxenergy = copy.deepcopy(self.energy)
        self.set_spin(raw_spin)

    def set_spin(self, spin):
        self.spin = spin
        self._get_total_energy()
        self._get_total_magnetization()

    def get_energy(self) -> float:
        """Get the total energy of the system / cn: 获取系统的总能量

        Returns:
            float: The total energy of the system / cn: 系统的总能量
        """
        return self.energy

    def get_magnetization(self) -> float:
        """Get the magnetization of the system / cn: 获取系统的总磁矩

        Returns:
            float: The per magnetization of the system / cn: 系统的单位磁矩
        """
        return self.magntization
