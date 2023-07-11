# -*- encoding: utf-8 -*-
"""
@File    :   XY.py
@Time    :   2023/05/31 11:48:37
@Author  :   UynajGI
@Contact :   suquan12148@outlook.com
@License :   (MIT)Copyright 2023
"""

# here put the import lib
from typing import Tuple
import numpy as np
import copy
from .Ising import Ising

__all__ = ["XY"]


class XY(Ising):
    def __init__(self, L, Jij=1, H=0):
        super().__init__(L, Jij, H, dim=2)
        self._init_spin(type="XY")

    def _init_spin(self, type="XY"):
        """Initialize the spin of the system

        Args:
            type (str, optional): The type of the spin / cn: 自旋的类型 (Defaults \'ising\')
        """
        self.spin = 2 * np.random.rand(self.L, self.L, self.dim) - 1
        self.spin = self.spin.astype(np.float32)
        self.type = type

    def _change_site_spin(self, index: Tuple[int, ...]):
        """Change the spin of the site / cn: 改变格点的自旋

        Args:
            index (Tuple[int, ...]): The index of the site / cn: 格点的坐标

        Raises:
            ValueError: Invalid type of spin / cn: 无效的自旋类型
        """
        self.spin[index] = 2 * np.random.rand(self.dim) - 1

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
            energy -= self.Jij * np.dot(self.spin[index], neighbor_spin)
            energy -= self.H * np.dot(self.spin[index], self.spin[index])
        return energy
