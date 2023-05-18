# -*- encoding: utf-8 -*-
"""
@File    :   IsingObj.py
@Time    :   2023/05/13 10:49:12
@Author  :   UynajGI
@Version :   beta0.0.1
@Contact :   betterWL@hotmail.com
@License :   (CC-BY-4.0)Copyright 2023
"""

from typing import Any, Tuple

# here put the import lib
import numpy as np


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

    def __len__(self):
        return self.L

    def __getitem__(self, index: Tuple[int, ...]):
        return self.spin[index]

    def _init_spin(self, type="ising", *args, **kwargs):
        """Initialize the spin of the system

        Args:
            type (str, optional): The type of the spin / cn: 自旋的类型 (Defaults \'ising\')

        Raises:
            ValueError: Invalid type of spin / cn: 无效的自旋类型
        """
        if type == "ising":
            self.spin = np.random.choice([-1, 1], size=(self.L, ) * self.dim)
            self.spin = self.spin.astype(np.int8)
        elif type == "heisenberg":
            self.spin = 2 * np.random.rand(self.L, self.L, self.L,
                                           self.dim) - 1
            self.spin = self.spin.astype(np.float32)
        elif type == "XY":
            self.spin = 2 * np.random.rand(self.L, self.L, self.dim) - 1
            self.spin = self.spin.astype(np.float32)
        elif type == "potts":
            p = kwargs.pop("p", 2)
            self.spin = np.random.choice(range(p), size=(self.L, ) * self.dim)
            self.spin = self.spin.astype(np.int8)
        else:
            raise ValueError("Invalid type of spin")
        self.tpye = type

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
        if self.tpye == "ising" or self.tpye == "heisenberg" or self.tpye == "XY":
            for neighbor_spin in neighbors_spin:
                energy -= self.Jij * np.dot(self.spin[index], neighbor_spin)
            if self.tpye == "ising":
                energy -= self.H * self.spin[index]
        elif self.tpye == "potts":
            for neighbor_spin in neighbors_spin:
                if self.spin[index] == neighbor_spin:
                    energy -= self.Jij
        else:
            raise ValueError("Invalid type of spin")
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
        return self.spin, self._get_total_energy(
        ), self._get_total_magnetization()

    def _get_per_info(self):
        return self.spin, self._get_per_energy(), self._get_per_magnetization()

    def _change_site_spin(self, index: Tuple[int, ...]):
        """Change the spin of the site / cn: 改变格点的自旋

        Args:
            index (Tuple[int, ...]): The index of the site / cn: 格点的坐标

        Raises:
            ValueError: Invalid type of spin / cn: 无效的自旋类型
        """
        if self.tpye == "ising":
            self.spin[index] *= -1
        elif self.tpye == "heisenberg" or self.tpye == "XY":
            self.spin[index] = 2 * np.random.rand(self.dim) - 1
        elif self.tpye == "potts":
            self.spin[index] = np.random.choice(range(self.p))
        else:
            raise ValueError("Invalid type of spin")

    def _change_delta_energy(self, index: Tuple[int, ...]):
        """Get the delta energy of the site"""
        old_site_energy = self._get_site_energy(index)
        self._change_site_spin(index)
        new_site_energy = self._get_site_energy(index)
        detle_energy = new_site_energy - old_site_energy
        return detle_energy

    def set_spin(self, spin):
        # TODO[0.3.0]: 增加 spin 格式审查
        self.spin = spin
        self._get_total_energy()
        self._get_total_magnetization()

    def get_energy(self) -> float:
        """Get the total energy of the system / cn: 获取系统的总能量

        Returns:
            float: The total energy of the system / cn: 系统的总能量
        """
        return self._get_total_energy()

    def get_magnetization(self) -> float:
        """Get the magnetization of the system / cn: 获取系统的总磁矩

        Returns:
            float: The per magnetization of the system / cn: 系统的单位磁矩
        """
        return self._get_total_magnetization()


class Heisenberg(Ising):

    def __init__(self, L, Jij=1, H=0, *args, **kwargs):
        super().__init__(L, Jij, H, dim=3, *args, **kwargs)
        self._init_spin(type="heisenberg")


class XY(Ising):

    def __init__(self, L, Jij=1, H=0, *args, **kwargs):
        super().__init__(L, Jij, H, dim=2, *args, **kwargs)
        self._init_spin(type="XY")


class Potts(Ising):

    def __init__(self, L, Jij=1, H=0, dim=2, p=3, *args, **kwargs):
        # p is the number of states of the system
        self.p = p
        super().__init__(L, Jij, H, dim, *args, **kwargs)
        self._init_spin(type="potts", p=p)
