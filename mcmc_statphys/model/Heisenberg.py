# -*- encoding: utf-8 -*-
'''
@File    :   Heisenberg.py
@Time    :   2023/05/31 11:47:52
@Author  :   UynajGI
@Contact :   suquan12148@outlook.com
@License :   (MIT)Copyright 2023
'''

# here put the import lib
from typing import Tuple
import numpy as np
import copy
from .Ising import Ising

__all__ = ["Heisenberg"]


class Heisenberg(Ising):
    """
    Heisenberg model
    =================

    """

    def __init__(self, L, Jij=1, H=0, *args, **kwargs):
        super().__init__(L, Jij, H=0, dim=3, *args, **kwargs)
        self._init_spin(type="heisenberg")
        self._max_energy()

    def _init_spin(self, type="heisenberg", *args, **kwargs):
        """Initialize the spin of the system

        Args:
            type (str, optional): The type of the spin / cn: 自旋的类型 (Defaults \'ising\')
        """
        self.spin = 2 * np.random.rand(self.L, self.L, self.L, self.dim) - 1
        self.spin = self.spin.astype(np.float32)
        self.type = type

    def _change_site_spin(self, index: Tuple[int, ...]):
        """Change the spin of the site / cn: 改变格点的自旋

        Args:
            index (Tuple[int, ...]): The index of the site / cn: 格点的坐标
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

    def _max_energy(self):
        raw_spin = copy.deepcopy(self.spin)
        max_spin = np.zeros_like(self.spin)
        max_spin[:, :, :, 0] = 1
        max_spin[:, :, :, 1] = 0
        max_spin[:, :, :, 2] = 0
        self.set_spin(max_spin)
        self.maxenergy = copy.deepcopy(self.energy)
        self.set_spin(raw_spin)
