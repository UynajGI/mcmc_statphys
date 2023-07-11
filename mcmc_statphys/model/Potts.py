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
from .Ising import Ising

__all__ = ["Potts"]


class Potts(Ising):
    """
    Potts
    =====

    In statistical mechanics, the Potts model, a generalization of the Ising model, is a model of interacting spins on a crystalline lattice.[1] By studying the Potts model, one may gain insight into the behaviour of ferromagnets and certain other phenomena of solid-state physics. The strength of the Potts model is not so much that it models these physical systems well; it is rather that the one-dimensional case is exactly solvable, and that it has a rich mathematical formulation that has been studied extensively.

    Definition of Ising Model
    -------------------------

    The Potts model consists of spins that are placed on a lattice; the lattice is usually taken to be a two-dimensional rectangular Euclidean lattice, but is often generalized to other dimensions and lattice structures. Each spin can take on one of q different states, and the spins interact with their nearest neighbors. The Hamiltonian of the Potts model is given by

    .. math::H = -J_p \sum_{\langle i,j \rangle} \delta (s_i, s_j)

    where :math:`\delta (s_{i},s_{j})` is the Kronecker delta, which equals one whenever :math:`s_{i}=s_{j}` and zero otherwise.

    Ref
    ---

    -  [1] `Potts model -
    Wikipedia <https://en.wikipedia.org/wiki/Potts_model>`__
    """

    def __init__(self, L: int, Jij: float = 1, H: float = 0, dim: int = 2, p: int = 3):
        # p is the number of states of the system
        self.p = p
        super().__init__(L, Jij, H, dim)
        self._init_spin(type="potts", p=p)

    def _init_spin(self, type="potts"):
        """Initialize the spin of the system

        Args:
            type (str, optional): The type of the spin. Defaults to "potts".
        """
        self.spin = np.random.choice(range(self.p), size=(self.L,) * self.dim)
        self.type = type

    def _change_site_spin(self, index: Tuple[int, ...]):
        """Change the spin of the site

        Args:
            index (Tuple[int, ...]): The index of the site.
        """
        self.spin[index] = np.random.choice(range(self.p))

    def _get_site_energy(self, index: Tuple[int, ...]) -> float:
        """Get the energy of the site

        Args:
            index (Tuple[int, ...]): The index of the site.

        Returns:
            float: The energy of the site.
        """
        neighbors_spin = super()._get_neighbor_spin(index)
        energy = 0
        for neighbor_spin in neighbors_spin:
            if self.spin[index] == neighbor_spin:
                energy -= self.Jij
        return energy
