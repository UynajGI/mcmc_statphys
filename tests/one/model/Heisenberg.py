#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@文件    :Heisenberg.py
@时间    :2023/07/06 22:11:00
@作者    :結凪
"""

from typing import Tuple
import numpy as np
import copy
from .Ising import Ising

__all__ = ["Heisenberg"]


class Heisenberg(Ising):
    """
    Classical Heisenberg model
    ==========================

    Example
    -------
    >>> import mcmc_statphys as mcsp
    >>> m = mcsp.model.Heisenberg(L=10, J=1)

    Description
    -----------

    The Classical Heisenberg model, developed by Werner Heisenberg, is the :math:`n = 3` case of the n-vector model, one of the models used in statistical physics to model ferromagnetism, and other phenomena.

    Definition
    ----------

    It can be formulated as follows: take a d-dimensional lattice, and a set of spins of the unit length

    .. math:: \vec{s}_i \in \mathbb{R}^3, |\vec{s}_i|=1

    each one placed on a lattice node.

    The model is defined through the following Hamiltonian:

    .. math:: H = -J \sum_{\langle i,j \rangle} \vec{s_i}\cdot\vec{s_j}

    References
    ----------

    -  [1] `Classical Heisenberg model -Wikipedia <https://en.wikipedia.org/wiki/Classical_Heisenberg_model>`__
    """

    def __init__(self, L: int, J: float = 1):
        """
        init the Heisenberg model

        Parameters
        ----------
        L : int
            The length of the lattice.
        J : float, optional
            The interaction strength, by default 1
        """
        super().__init__(L=L, J=J, H=0, dim=3)
        self._init_spin(type="heisenberg")

    def _init_spin(self, type="heisenberg"):
        """
        init the spin of the lattice

        Parameters
        ----------
        type : str, optional
            The type of the spin, by default "heisenberg"
        """
        self.spin = 2 * np.random.rand(self.L, self.L, self.L, self.dim) - 1
        self.spin = self.spin.astype(np.float32)
        self.type = type

    def _change_site_spin(self, index: Tuple[int, ...]):
        """
        change the spin of the site

        Parameters
        ----------
        index : Tuple[int, ...]
            The index of the site
        """
        self.spin[index] = 2 * np.random.rand(self.dim) - 1

    def _get_site_energy(self, index: Tuple[int, ...]) -> float:
        """
        get the energy of the site

        Parameters
        ----------
        index : Tuple[int, ...]
            The index of the site

        Returns
        -------
        float
            The energy of the site
        """
        neighbors_spin = super()._get_neighbor_spin(index)
        energy = 0
        for neighbor_spin in neighbors_spin:
            energy -= self.J * np.dot(self.spin[index], neighbor_spin)
        return energy
