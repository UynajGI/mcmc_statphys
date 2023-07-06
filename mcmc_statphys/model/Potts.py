# -*- encoding: utf-8 -*-
"""
@File    :   Potts.py
@Time    :   2023/05/31 11:49:11
@Author  :   UynajGI
@Contact :   suquan12148@outlook.com
@License :   (MIT)Copyright 2023
"""

# here put the import lib
from typing import Tuple
import numpy as np
import copy
from .Ising import Ising

__all__ = ["Potts"]


class Potts(Ising):
    def __init__(self, L, Jij=1, H=0, dim=2, p=3, *args, **kwargs):
        # p is the number of states of the system
        self.p = p
        super().__init__(L, Jij, H, dim, *args, **kwargs)
        self._init_spin(type="potts", p=p)
        self._max_energy()

    def _init_spin(self, type="potts", p=3, *args, **kwargs):
        """Initialize the spin of the system

        Args:
            type (str, optional): The type of the spin / cn: 自旋的类型 (Defaults \'ising\')

        Raises:
            ValueError: Invalid type of spin / cn: 无效的自旋类型
        """
        self.spin = np.random.choice(range(p), size=(self.L,) * self.dim)
        self.spin = self.spin.astype(np.int8)
        self.type = type

    def _change_site_spin(self, index: Tuple[int, ...]):
        """Change the spin of the site / cn: 改变格点的自旋

        Args:
            index (Tuple[int, ...]): The index of the site / cn: 格点的坐标

        Raises:
            ValueError: Invalid type of spin / cn: 无效的自旋类型
        """
        self.spin[index] = np.random.choice(range(self.p))

    def _get_site_energy(self, index: Tuple[int, ...]) -> float:
        """Get the energy of the site / cn: 获取格点的能量

        Args:
            index (Tuple[int, ...]): The index of the site / cn: 格点的坐标

        Raises:
            ValueError: Invalid type of spin / cn: 无效的自旋类型

        Returns:
            float: The energy of the site / cn: 格点的能量
        """
        neighbors_spin = super()._get_neighbor_spin(index)
        energy = 0
        for neighbor_spin in neighbors_spin:
            if self.spin[index] == neighbor_spin:
                energy -= self.Jij
        return energy

    def _max_energy(self):
        raw_spin = copy.deepcopy(self.spin)
        max_spin = np.zeros_like(self.spin)
        max_spin[:, :] = self.p - 1
        self.set_spin(max_spin)
        self.maxenergy = copy.deepcopy(self.energy)
        self.set_spin(raw_spin)
