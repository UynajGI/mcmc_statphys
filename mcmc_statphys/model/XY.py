#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@文件    :XY.py
@时间    :2023/07/11 23:51:11
@作者    :結凪
"""

from typing import Tuple
import numpy as np
from .Ising import Ising

__all__ = ["XY"]


class XY(Ising):
    """
    XY model
    ========

    The classical XY model (sometimes also called classical rotor
    (rotator) model or :math:`O(2)` model) is a lattice model of
    statistical mechanics. In general, the XY model can be seen as a
    specialization of Stanley’s n-vector model for n = 2.

    Definition
    ----------

    Given a :math:`D`-dimensional lattice :math:`\Lambda`, per each lattice
    site :math:`j\in \Lambda` there is a two-dimensional, unit-length vector
    :math:`\mathbf{s}_j=(\cos{\theta_j},\sin{\theta_j})`

    The spin configuration, :math:`\mathbf{s}=(\mathbf{s}_j)_{j\in\Lambda}`
    is an assignment of the angle :math:`-\pi < \theta_j \le \pi` for each
    :math:`j\in \Lambda`.

    Given a translation-invariant interaction :math:`J_{ij} = J(i − j)` and
    a point dependent external
    field\ :math:`\mathbf{h}_j=(h_j,0)`,
    the configuration energy is

    .. math::

    H = -\sum_{i\not ={j}}{J_{ij}\mathbf{s}_i\cdot\mathbf{s}_j} - \sum_{j}{\mathbf{h}_j\cdot\mathbf{s}_j}

    References
    ----------

    -  [1] `Classical XY model -
    Wikipedia <https://en.wikipedia.org/wiki/Classical_XY_model>`__
    """

    def __init__(self, L, Jij=1, H=0):
        super().__init__(L, Jij, H, dim=2)
        self._init_spin(type="XY")

    def _init_spin(self, type="XY"):
        """Initialize the spin of the system

        Args:
            type (str, optional): The type of the spin
        """
        self.spin = 2 * np.random.rand(self.L, self.L, self.dim) - 1
        self.spin = self.spin.astype(np.float32)
        self.type = type

    def _change_site_spin(self, index: Tuple[int, ...]):
        """Change the spin of the site

        Args:
            index (Tuple[int, ...]): The index of the site

        Raises:
            ValueError: Invalid type of spin
        """
        self.spin[index] = 2 * np.random.rand(self.dim) - 1

    def _get_site_energy(self, index: Tuple[int, ...]) -> float:
        """Get the energy of the site

        Args:
            index (Tuple[int, ...]): The index of the site

        Raises:
            ValueError: Invalid type of spin

        Returns:
            float: The energy of the site
        """
        neighbors_spin = super()._get_neighbor_spin(index)
        energy = 0
        for neighbor_spin in neighbors_spin:
            energy -= self.Jij * np.dot(self.spin[index], neighbor_spin)
            energy -= self.H * np.dot(self.spin[index], self.spin[index])
        return energy
