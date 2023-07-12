#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@文件    :Ising.py
@时间    :2023/07/06 21:25:05
@作者    :結凪
"""

from typing import Any, Tuple, Union
import numpy as np
import copy
import pandas as pd

__all__ = ["Ising"]


class Ising(object):
    """
    Ising
    =====

    Example
    -------
    >>> import mcmc_statphys as mcsp
    >>> m = mcsp.model.Ising(L=10, J=1, H=0, dim=2)

    Description
    -----------

    The Ising model is a mathematical model for describing ferromagnetism, named by physicists Ernst Ising and Wilhelm Lenz.
    The model consists of a set of discrete variables representing the magnetic dipole moments of the atomic "spins", which can be in two states (+1 or -1).
    The spins are arranged in a diagram (usually a lattice) so that each spin can interact with its neighbors.
    The same neighboring spin has a lower energy than a different neighboring spin;
    the system tends to have the lowest energy, but heat perturbs this tendency, resulting in a different structural phase. The model can be used as a realistic simplified model to identify phase transitions.

    Definition of Ising Model
    -------------------------

    Consider a set of lattice points, each of which has a set of neighboring lattice points (e.g. a graph) forming a lattice of one dimension.
    For each lattice point, there is a discrete variable, satisfying, that represents the spin of the point.
    A spin configuration is one that assigns a spin value to each lattice point.
    For any two neighboring lattice points, there is an interaction. In addition, a lattice point has an external magnetic field with which it interacts.
    The energy of a configuration is given by the Hamiltonian function

    .. math::
        H = -J \sum_{\langle i,j \rangle} s_i s_j - h \sum_i s_i

    The first summation is performed over adjacent spin pairs (each pair is counted only once), and the second summation is performed over all spins.

    Analytical and numerical methods for Ising models
    -------------------------------------------------

    The one-dimensional Ising model can be solved by Ising himself in his 1924 paper, and it has no phase transition.
    The two-dimensional square lattice Ising model is much more difficult and was not described analytically until 1944 by Lars Onsager.
    It is usually solved by transfer matrix methods, although some methods related to quantum field theory also exist.
    In the case of greater than four dimensions, the phase transition of the Ising model can be described by mean-field theory.
    In addition to analytical methods, the Ising model can also be solved numerically, for example by Monte Carlo simulations.
    This method can be used to generate spin configurations at different temperatures and to calculate relevant
    physical quantities such as magnetization strength, specific heat, magnetization rate, etc.

    References
    ----------

    -  [1] `Ising model -
    Wikipedia <https://en.wikipedia.org/wiki/Ising_model>`__
    -  [2] `Shekaari, A., & Jafari, M. (2021). Theory and Simulation of the
    Ising Model. <http://arxiv.org/abs/2105.00841>`__
    -  [3] `Ising Model -
    Scholarpedia <http://www.scholarpedia.org/article/Ising_model>`__
    """

    def __init__(self, L: int, J: float = 1, H: float = 0, dim: int = 2):
        """
        initialize the Ising model

        Parameters
        ----------
        L : int
            The length of the lattice
        J : float, optional
            The interaction between the neighbor, by default 1
        H : float, optional
            The external magnetic field, by default 0
        dim : int, optional
            The dimension of the lattice, by default 2
        """
        L = int(L)
        self.L: int = L  # The length of the lattice
        self.dim: int = dim  # The dimension of the lattice
        self.N: int = L**dim  # The number of the lattice
        self.J: float = J  # The interaction between the neighbor
        self.H: float = H  # The external magnetic field
        self.energy: float = 0  # The total energy of the system
        self.magnetization: float = 0  # The total magnetization of the system

        self._init_spin(type="ising")
        self._get_total_energy()
        self._get_total_magnetization()

    def _init_spin(self, type="ising"):
        """
        Initialize the spin of the system

        Parameters
        ----------
        type : str, optional
            The type of the spin, by default "ising"
        """
        self.spin = np.random.choice([-1, 1], size=(self.L,) * self.dim)
        self.type = type

    def _get_neighbor(self, index: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Get the neighbor of the site

        Parameters
        ----------
        index : Tuple[int, ...]
            The index of the site

        Returns
        -------
        Tuple[int, ...]
            The neighbor of the site
        """
        neighbors = []
        for i in range(self.dim):
            for j in [-1, 1]:
                neighbors.append(index[:i] + ((index[i] + j) % self.L,) + index[i + 1 :])
        neighbors = list(set(neighbors))  # remove the same neighbor
        return neighbors

    def _get_neighbor_spin(self, index: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Get the spin of the neighbor of the site

        Returns
        -------
        Tuple[int, ...]
            The spin of the neighbor of the site
        """
        neighbors = self._get_neighbor(index)
        neighbors_spin = []
        for neighbor in neighbors:
            neighbors_spin.append(self.spin[neighbor])
        return neighbors_spin

    def _get_site_energy(self, index: Tuple[int, ...]) -> float:
        """
        Get the energy of the site

        Parameters
        ----------
        index : Tuple[int, ...]
            The index of the site

        Returns
        -------
        float
            The energy of the site
        """
        neighbors_spin = self._get_neighbor_spin(index)
        energy = 0
        for neighbor_spin in neighbors_spin:
            energy -= self.J * np.dot(self.spin[index], neighbor_spin)
        energy -= self.H * self.spin[index]
        return energy

    def _get_total_energy(self) -> float:
        """
        Get the total energy of the system

        Returns
        -------
        float
            The total energy of the system
        """
        energy = 0
        for index in np.ndindex(self.spin.shape):
            energy += self._get_site_energy(index)
        self.energy = energy / 2
        return self.energy

    def _get_per_energy(self) -> float:
        """
        Get the per energy of the system

        Returns
        -------
        float
            The per energy of the system
        """
        self.energy = self._get_total_energy()
        return self.energy / self.N

    def _get_total_magnetization(self) -> float:
        """
        Get the magnetization of the system

        Returns
        -------
        float
            The magnetization of the system
        """
        self.magnetization = np.sum(self.spin, axis=tuple(range(self.dim)))
        return self.magnetization

    def _get_per_magnetization(self) -> float:
        """
        Get the per magnetization of the system

        Returns
        -------
        float
            The per magnetization of the system
        """
        return self._get_total_magnetization() / self.N

    def _change_site_spin(self, index: Tuple[int, ...]) -> None:
        """
        Change the spin of the site.

        Parameters
        ----------
        index : Tuple[int, ...]
            The index of the site
        """
        self.spin[index] *= -1

    def _change_delta_energy(self, index: Tuple[int, ...]) -> float:
        """
        Change the spin of the site and get the delta energy

        Parameters
        ----------
        index : Tuple[int, ...]
            The index of the site

        Returns
        -------
        float
            The delta energy of the site
        """
        old_site = self.spin[index]
        old_site_energy = self._get_site_energy(index)
        self._change_site_spin(index)
        new_site = self.spin[index]
        new_site_energy = self._get_site_energy(index)
        detle_energy = new_site_energy - old_site_energy
        self.energy += detle_energy
        self.magnetization += new_site - old_site
        return detle_energy

    def _random_walk(self) -> float:
        """
        Random walk of the system

        Returns
        -------
        float
            The delta energy of the system
        """
        site = tuple(np.random.randint(0, self.L, size=self.dim))
        detle_energy = self._change_delta_energy(site)
        return detle_energy

    def set_spin(self, spin: np.ndarray):
        """
        Set the spin of the system

        Parameters
        ----------
        spin : np.ndarray
            The spin of the system
        """
        self.spin = spin
        self._get_total_energy()
        self._get_total_magnetization()

    def _init_data(self) -> pd.DataFrame:
        """
        Initialize the data

        Returns
        -------
        pd.DataFrame
            The data
        """
        data: pd.DataFrame = pd.DataFrame(
            columns=[
                "uid",
                "iter",
                "T",
                "H",
                "energy",
                "magnetization",
                "spin",
            ]
        )
        data.set_index(["uid", "iter"], inplace=True)
        return data

    def _save_date(self, T: float, uid: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Save the data

        Parameters
        ----------
        T : float
            The temperature
        uid : str
            The uid of the data
        data : pd.DataFrame
            The data

        Returns
        -------
        pd.DataFrame
            The data
        """
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
