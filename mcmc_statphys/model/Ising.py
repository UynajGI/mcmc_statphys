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
import copy

__all__ = ["Ising"]


class Ising(object):

    def __init__(self,
                 L: int,
                 Jij: float = 1,
                 H: float = 0,
                 dim: int = 2,
                 *args: Any,
                 **kwargs: Any):
        L = int(L)
        self.L = L
        self.dim = dim
        self.N = L**dim
        self.Jij = Jij
        self.H = H
        self._init_spin(type="ising")
        self._get_total_energy()
        self._get_total_magnetization()
        self._max_energy()

    def __len__(self):
        return self.L

    def __getitem__(self, index: Tuple[int, ...]):
        return self.spin[index]

    def _init_spin(self, type="ising", *args, **kwargs):
        """Initialize the spin of the system

        Args:
            type (str, optional): The type of the spin / cn: 自旋的类型 (Defaults \'ising\')
        """
        self.spin = np.random.choice([-1, 1], size=(self.L, ) * self.dim)
        self.spin = self.spin.astype(np.int8)
        self.type = type

    def _get_neighbor(self, index: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get the neighbor of the site / cn: 获取格点的邻居

        Args:
            index (Tuple[int, ...]): The index of the site / cn: 格点的坐标

        Returns:
            Tuple[int, ...]: The neighbor of the site / cn: 格点的邻居
        """
        neighbors = []
        for i in range(self.dim):
            for j in [-1, 1]:
                neighbors.append(index[:i] + ((index[i] + j) % self.L, ) +
                                 index[i + 1:])
        neighbors = list(set(neighbors))  # 去重
        return neighbors

    def _get_neighbor_spin(self, index: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get the spin of the neighbor of the site / cn: 获取格点的邻居的自旋

        Args:
            index (Tuple[int, ...]): The index of the site / cn: 格点的坐标

        Returns:
            Tuple[int, ...]: The spin of the neighbor of the site / cn: 格点的邻居的自旋
        """
        neighbors = self._get_neighbor(index)
        neighbors_spin = []
        for neighbor in neighbors:
            neighbors_spin.append(self.spin[neighbor])
        return neighbors_spin

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
        energy -= self.H * self.spin[index]
        return energy

    def _get_per_energy(self) -> float:
        """Get the per energy of the system / cn: 获取系统的单位能量

        Returns:
            float: The per energy of the system / cn: 系统的单位能量
        """
        energy = 0
        for index in np.ndindex(self.spin.shape):
            energy += self._get_site_energy(index)
        return energy / self.N / 2

    def _get_total_energy(self) -> float:
        """Get the total energy of the system / cn: 获取系统的总能量

        Returns:
            float: The total energy of the system / cn: 系统的总能量
        """
        energy = self._get_per_energy() * self.N
        self.energy = energy
        return energy

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
        self.magnetization = np.sum(self.spin, axis=tuple(range(self.dim)))
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
        old_site = self.spin[index]
        old_site_energy = self._get_site_energy(index)
        self._change_site_spin(index)
        new_site = self.spin[index]
        new_site_energy = self._get_site_energy(index)
        detle_energy = new_site_energy - old_site_energy
        self.energy += detle_energy
        self.magnetization += (new_site - old_site)
        return detle_energy

    def _max_energy(self):
        raw_spin = copy.deepcopy(self.spin)
        self.set_spin(np.ones_like(self.spin))
        self.maxenergy = copy.deepcopy(self.energy)
        self.set_spin(raw_spin)

    def set_spin(self, spin):
        # TODO[0.4.0]: 增加 spin 格式审查
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
