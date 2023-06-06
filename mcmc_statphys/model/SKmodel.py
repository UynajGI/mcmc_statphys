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
from .Ising import Ising

__all__ = ["SKmodel"]


class SKmodel(Ising):
    # SKModel is the Sherrington-Kirkpatrick model
    def __init__(self,
                 N,
                 Jmean=0,
                 Jsigma=1,
                 Jform='norm',
                 H=0,
                 *args,
                 **kwargs):
        super().__init__(L=N, Jij=1, H=H, dim=1)
        self.Jmean = Jmean
        self.Jsigma = Jsigma
        self.N = N
        self._init_spin(type="SK")
        self._init_Jij(Jform)
        self._get_total_energy()
        self._get_total_magnetization()
        self._max_energy()

    def _init_Jij(self, Jform):
        if Jform == 'norm':
            self.Jij = np.random.normal(self.Jmean / self.N,
                                        self.Jsigma / np.sqrt(self.N),
                                        (self.N, self.N))
            self.Jij = np.tril(self.Jij)
            self.Jij = self.Jij + self.Jij.T
            np.fill_diagonal(self.Jij, 0)
            self.Jij = self.Jij.astype(np.float32)
        elif Jform == 'uniform':
            self.Jij = np.random.choice([-self.Jsigma, self.Jsigma],
                                        size=(self.N, self.N))
            self.Jij = np.tril(self.Jij)
            self.Jij = self.Jij + self.Jij.T
            np.fill_diagonal(self.Jij, 0)
            self.Jij = self.Jij.astype(np.float32)

    def _get_total_energy(self) -> float:
        """Get the total energy of the system / cn: 获取系统的总能量

        Returns:
            float: The total energy of the system / cn: 系统的总能量
        """
        self.energy = -1 / 2 * np.dot(self.spin, np.dot(self.Jij, self.spin))
        self.energy -= np.sum(self.H * self.spin)
        return self.energy

    def _change_delta_energy(self, index: Tuple[int, ...]):
        """Get the delta energy of the site"""
        old_energy = copy.deepcopy(self.energy)
        self._change_site_spin(index)
        new_energy = self._get_total_energy()
        detle_energy = new_energy - old_energy
        self.magnetization += 2 * self.spin[index]
        return detle_energy
