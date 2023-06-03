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
import pandas as pd

__all__ = ["Ising"]


class Ising(object):
    """
    Ising
    =====

    The Ising model is a mathematical model for describing ferromagnetism, named by physicists Ernst Ising and Wilhelm Lenz.
    The model consists of a set of discrete variables representing the magnetic dipole moments of the atomic "spins", which can be in two states (+1 or -1).
    The spins are arranged in a diagram (usually a lattice) so that each spin can interact with its neighbors.
    The same neighboring spin has a lower energy than a different neighboring spin;
    the system tends to have the lowest energy, but heat perturbs this tendency, resulting in a different structural phase. The model can be used as a realistic simplified model to identify phase transitions.

    Definition of Ising Model
    ---------------

    Consider a set of lattice points, each of which has a set of neighboring lattice points (e.g. a graph) forming a lattice of one dimension.
    For each lattice point, there is a discrete variable, satisfying, that represents the spin of the point.
    A spin configuration is one that assigns a spin value to each lattice point.
    For any two neighboring lattice points, there is an interaction. In addition, a lattice point has an external magnetic field with which it interacts.
    The energy of a configuration is given by the Hamiltonian function

    H = -J \\sum_{\\langle i,j \\rangle} s_i s_j - h \\sum_i s_i

    The first summation is performed over adjacent spin pairs (each pair is counted only once), and the second summation is performed over all spins.

    Analytical and numerical methods for Ising models
    -------------------------

    The one-dimensional Ising model can be solved by Ising himself in his 1924 paper, and it has no phase transition.
    The two-dimensional square lattice Ising model is much more difficult and was not described analytically until 1944 by Lars Onsager.
    It is usually solved by transfer matrix methods, although some methods related to quantum field theory also exist.
    In the case of greater than four dimensions, the phase transition of the Ising model can be described by mean-field theory.
    In addition to analytical methods, the Ising model can also be solved numerically, for example by Monte Carlo simulations.
    This method can be used to generate spin configurations at different temperatures and to calculate relevant
    physical quantities such as magnetization strength, specific heat, magnetization rate, etc.

    Ref
    ---

    -  [1] `Ising model -
    Wikipedia <https://en.wikipedia.org/wiki/Ising_model>`__
    -  [2] `Shekaari, A., & Jafari, M. (2021). Theory and Simulation of the
    Ising Model. <http://arxiv.org/abs/2105.00841>`__
    -  [3] `Ising Model -
    Scholarpedia <http://www.scholarpedia.org/article/Ising_model>`__
    """

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
        return self.magnetization

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

    def _save_date(self, T, uid, data: pd.DataFrame):
        if uid not in data.index.get_level_values("uid").values:
            data.loc[(uid, 1), :] = [
                T,
                self.H,
                self.energy,
                self.magnetization,
                0,
            ]
            data.at[(uid, 1), "spin"] = copy.deepcopy(self.spin)
        else:
            iterplus = data.loc[uid].index.max() + 1
            data.loc[(uid, iterplus), :] = [
                T,
                self.H,
                self.energy,
                self.magnetization,
                0,
            ]
            data.at[(uid, iterplus), "spin"] = copy.deepcopy(self.spin)
        return data
