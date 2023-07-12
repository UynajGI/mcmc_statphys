#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@文件    :SKmodel.py
@时间    :2023/07/12 15:35:48
@作者    :結凪
"""

from typing import Tuple
import numpy as np
import copy
from .Ising import Ising

__all__ = ["SKmodel"]


class SKmodel(Ising):
    """
    Spin glass
    ==========

    Example
    -------
    >>> import mcmc_statphys as mcsp
    >>> m = mcsp.model.SKmodel(N=10, Jmean=0, Jsigma=1,Jform='norm')

    Description
    -----------

    In condensed matter physics, a spin glass is a magnetic state characterized by randomness, besides cooperative behavior in freezing of spins at a temperature called 'freezing temperature' Tf. In ferromagnetic solids, component atoms' magnetic spins all align in the same direction. Spin glass when contrasted with a ferromagnet is defined as "disordered" magnetic state in which spins are aligned randomly or without a regular pattern and the couplings too are random.

    The term "glass" comes from an analogy between the magnetic disorder in a spin glass and the positional disorder of a conventional, chemical glass, e.g., a window glass. In window glass or any amorphous solid the atomic bond structure is highly irregular; in contrast, a crystal has a uniform pattern of atomic bonds. In ferromagnetic solids, magnetic spins all align in the same direction; this is analogous to a crystal's lattice-based structure.

    The individual atomic bonds in a spin glass are a mixture of roughly equal numbers of ferromagnetic bonds (where neighbors have the same orientation) and antiferromagnetic bonds (where neighbors have exactly the opposite orientation: north and south poles are flipped 180 degrees). These patterns of aligned and misaligned atomic magnets create what are known as frustrated interactions – distortions in the geometry of atomic bonds compared to what would be seen in a regular, fully aligned solid. They may also create situations where more than one geometric arrangement of atoms is stable.

    Spin glasses and the complex internal structures that arise within them are termed "metastable" because they are "stuck" in stable configurations other than the lowest-energy configuration (which would be aligned and ferromagnetic). The mathematical complexity of these structures is difficult but fruitful to study experimentally or in simulations; with applications to physics, chemistry, materials science and artificial neural networks in computer science.

    Definition
    ----------

    See Wikipedia.

    References
    ----------

    -  [1] `Spin glass -Wikipedia <https://en.wikipedia.org/wiki/Spin_glass>`__

    """

    def __init__(self, N: int, Jmean: float = 0, Jsigma: float = 1, Jform: str = "norm") -> None:
        """
        init the SK model

        Parameters
        ----------
        N : int
            The length of the lattice.
        Jmean : float, optional
            The mean of the interaction strength, by default 0
        Jsigma : float, optional
            The sigma of the interaction strength, by default 1
        Jform : str, optional
            The form of the interaction strength, by default "norm"
        H : float, optional
            The external field strength, by default 0
        """
        super().__init__(L=N, dim=1, H=0)
        self.Jmean: float = Jmean
        self.Jsigma: float = Jsigma

        self._init_spin(type="SK")
        self._init_Jij(Jform=Jform)
        self._get_total_energy()
        self._get_total_magnetization()

    def _init_Jij(self, Jform: str) -> None:
        """
        init the interaction strength

        Parameters
        ----------
        Jform : str
            The form of the interaction strength.
        """
        if Jform == "norm":
            self.Jij = np.random.normal(self.Jmean / self.N, self.Jsigma / np.sqrt(self.N), (self.N, self.N))
            self.Jij = np.tril(self.Jij)
            self.Jij = self.Jij + self.Jij.T
            np.fill_diagonal(self.Jij, 0)
            self.Jij = self.Jij.astype(np.float32)
        elif Jform == "uniform":
            self.Jij = np.random.choice([-self.Jsigma, self.Jsigma], size=(self.N, self.N))
            self.Jij = np.tril(self.Jij)
            self.Jij = self.Jij + self.Jij.T
            np.fill_diagonal(self.Jij, 0)
            self.Jij = self.Jij.astype(np.float32)

    def _get_total_energy(self) -> float:
        """
        get the total energy of the system

        Returns
        -------
        float
            The total energy of the system.
        """
        self.energy = -1 / 2 * np.dot(self.spin, np.dot(self.Jij, self.spin))
        self.energy -= np.sum(self.H * self.spin)
        return self.energy

    def _change_delta_energy(self, index: Tuple[int, ...]) -> float:
        """
        change the spin of the site

        Parameters
        ----------
        index : Tuple[int, ...]
            The index of the site.

        Returns
        -------
        float
            The delta energy of the system.
        """
        old_energy = copy.deepcopy(self.energy)
        self._change_site_spin(index)
        new_energy = self._get_total_energy()
        detle_energy = new_energy - old_energy
        self.magnetization += 2 * self.spin[index]
        return detle_energy

    def _random_walk(self) -> float:
        """
        random walk

        Returns
        -------
        float
            The delta energy of the system.
        """
        site = np.random.randint(self.N)
        return self._change_delta_energy(site)
