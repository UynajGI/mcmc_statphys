# -*- encoding: utf-8 -*-
'''
@File    :   IsingObj.py
@Time    :   2023/05/13 10:49:12
@Author  :   UynajGI
@Version :   beta0.0.1
@Contact :   betterWL@hotmail.com
@License :   (CC-BY-4.0)Copyright 2023
'''

from typing import Any, Tuple

# here put the import lib
import numpy as np


class Ising(object):

    def __init__(self,
                 L: int,
                 Jij: float = 1,
                 H: float = 0,
                 dimension: int = 2,
                 *args: Any,
                 **kwargs: Any):
        L = int(L)
        self.L = L
        self.dimension = dimension
        self.N = L**dimension
        self.Jij = Jij
        self.H = H
        self._init_spin(type='ising')
        self._get_total_energy()
        self._get_total_magnetization()

    def __len__(self):
        return self.L

    def __getitem__(self, index: Tuple[int, ...]):
        
        return self.spin[index]

    def _init_spin(self, type='ising', *args, **kwargs):
        '''Initialize the spin of the system'''
        if type == 'ising':
            self.spin = np.random.choice([-1, 1],
                                         size=(self.L, ) * self.dimension)
            self.spin = self.spin.astype(np.int8)
        elif type == 'heisenberg':
            self.spin = 2 * np.random.rand(self.L, self.L, self.L,
                                           self.dimension) - 1
            self.spin = self.spin.astype(np.float32)
        elif type == 'XY':
            self.spin = 2 * np.random.rand(self.L, self.L, self.dimension) - 1
            self.spin = self.spin.astype(np.float32)
        elif type == 'potts':
            p = kwargs.pop('p', 2)
            self.spin = np.random.choice(range(p),
                                         size=(self.L, ) * self.dimension)
            self.spin = self.spin.astype(np.int8)
        else:
            raise ValueError('Invalid type of spin')
        self.tpye = type

    def _get_neighbor(self, index: Tuple[int, ...]):
        '''Get the neighbor of the site'''
        neighbors = []
        for i in range(self.dimension):
            for j in [-1, 1]:
                neighbors.append(index[:i] + ((index[i] + j) % self.L, ) +
                                 index[i + 1:])
        neighbors = list(set(neighbors))  # 去重
        return neighbors

    def _get_neighbor_spin(self, index: Tuple[int, ...]):
        '''Get the spin of the neighbor of the site'''
        neighbors = self._get_neighbor(index)
        neighbors_spin = []
        for neighbor in neighbors:
            neighbors_spin.append(self.spin[neighbor])
        return neighbors_spin

    def _get_site_energy(self, index: Tuple[int, ...]):
        '''Get the energy of the site'''
        neighbors_spin = self._get_neighbor_spin(index)
        energy = 0
        if self.tpye == 'ising' or self.tpye == 'heisenberg' or self.tpye == 'XY':
            for neighbor_spin in neighbors_spin:
                energy -= self.Jij * np.dot(self.spin[index], neighbor_spin)
            if self.tpye == 'ising':
                energy -= self.H * self.spin[index]
        elif self.tpye == 'potts':
            for neighbor_spin in neighbors_spin:
                if self.spin[index] == neighbor_spin:
                    energy -= self.Jij
        else:
            raise ValueError('Invalid type of spin')
        return energy

    def _get_per_energy(self):
        '''Get the per energy of the system'''
        energy = 0
        for index in np.ndindex(self.spin.shape):
            energy += self._get_site_energy(index)
        return energy / self.N / 2

    def _get_total_energy(self):
        '''Get the total energy of the system'''
        total_energy = self._get_per_energy() * self.N
        self.total_energy = total_energy
        return total_energy

    def _get_per_magnetization(self):
        '''Get the per magnetization of the system'''
        return self._get_total_magnetization() / self.N

    def _get_total_magnetization(self):
        '''Get the magnetization of the system'''
        self.total_magnetization = np.sum(self.spin,
                                          axis=tuple(range(self.dimension)))
        return self.total_magnetization

    # 获取类的属性
    def _get_info(self):
        return self.spin, self._get_total_energy(
        ), self._get_total_magnetization()

    def _get_per_info(self):
        return self.spin, self._get_per_energy(), self._get_per_magnetization()

    def _change_site_spin(self, index: Tuple[int, ...]):
        '''Change the spin of the site'''
        if self.tpye == 'ising':
            self.spin[index] *= -1
        elif self.tpye == 'heisenberg' or self.tpye == 'XY':
            self.spin[index] = 2 * np.random.rand(self.dimension) - 1
        elif self.tpye == 'potts':
            self.spin[index] = np.random.choice(range(self.p))
        else:
            raise ValueError('Invalid type of spin')

    def _change_delta_energy(self, index: Tuple[int, ...]):
        '''Get the delta energy of the site'''
        old_site_energy = self._get_site_energy(index)
        self._change_site_spin(index)
        new_site_energy = self._get_site_energy(index)
        detle_energy = new_site_energy - old_site_energy
        return detle_energy


class Heisenberg(Ising):

    def __init__(self, L, Jij=1, H=0, *args, **kwargs):
        super().__init__(L, Jij, H, dimension=3, *args, **kwargs)
        self._init_spin(type='heisenberg')


class XY(Ising):

    def __init__(self, L, Jij=1, H=0, *args, **kwargs):
        super().__init__(L, Jij, H, dimension=2, *args, **kwargs)
        self._init_spin(type='XY')


class Potts(Ising):

    def __init__(self, L, Jij=1, H=0, dimension=2, p=3, *args, **kwargs):
        # p is the number of states of the system
        self.p = p
        super().__init__(L, Jij, H, dimension, *args, **kwargs)
        self._init_spin(type='potts', p=p)
