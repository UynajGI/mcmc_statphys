#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@文件    :RFIsing.py
@时间    :2023/07/12 11:54:59
@作者    :結凪
"""

from typing import Any, Tuple
import numpy as np
from .Ising import Ising
import pandas as pd
import copy

__all__ = ["RFIsing"]


class RFIsing(Ising):
    """
    Random field Ising model
    ========================

    # TODO: add the description of the model
    """

    def __init__(self, L: int, J: float = 1, Hmean: float = 0, Hsigma: float = 1, Hform: str = "norm", dim: int = 2):
        """
        init the RFIsing model

        Parameters
        ----------
        L : int
            The length of the lattice.
        J : float, optional
            The interaction strength, by default 1
        Hmean : float, optional
            The mean of the H, by default 0
        Hsigma : float, optional
            The sigma of the H, by default 1
        Hform : str, optional
            The form of the H, "norm" or "uniform", by default "norm"
        dim : int, optional
            The dimension of the lattice, by default 2
        """
        H = self._init_H(Hmean=Hmean, Hsigma=Hsigma, Hform=Hform)
        super().__init__(L=L, J=J, H=H, dim=dim)
        self._init_spin(type="rfising")
        self._get_total_energy()
        self._get_total_magnetization()

    def _init_H(self, Hmean: float, Hsigma: float, Hform: str) -> np.ndarray:
        """
        init the H of the lattice

        Parameters
        ----------
        Hmean : float
            The mean of the H.
        Hsigma : float
            The sigma of the H.
        Hform : str
            The form of the H, "norm" or "uniform".

        Returns
        -------
        np.ndarray
            The H of the lattice.

        Raises
        ------
        ValueError
            Invalid Hform.
        """
        if Hform == "norm":
            H = np.random.normal(Hmean, Hsigma, (self.L,) * self.dim)
        elif Hform == "uniform":
            H = np.random.choice([-Hsigma, Hsigma], size=(self.L,) * self.dim)
        else:
            raise ValueError("Invalid Hform")
        return H

    def _get_site_energy(self, index: Tuple[int, ...]) -> float:
        """
        get the energy of the site

        Parameters
        ----------
        index : Tuple[int, ...]
            The index of the site.

        Returns
        -------
        float
            The energy of the site.
        """
        neighbors_spin = self._get_neighbor_spin(index)
        energy = 0
        for neighbor_spin in neighbors_spin:
            energy -= self.J * np.dot(self.spin[index], neighbor_spin)
        energy -= self.H[index] * self.spin[index]
        return energy

    def _init_data(self) -> pd.DataFrame:
        """
        init the data

        Returns
        -------
        pd.DataFrame
            The data.
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
        save the data

        Parameters
        ----------
        T : float
            The temperature.
        uid : str
            The uid of the data.
        data : pd.DataFrame
            The data.

        Returns
        -------
        pd.DataFrame
            The data.
        """
        if uid not in data.index.get_level_values("uid").values:
            data.loc[(uid, 1), :] = [
                T,
                0,
                self.energy,
                self.magnetization,
                0,
            ]
            data.at[(uid, 1), "spin"] = copy.deepcopy(self.spin)
            data.at[(uid, 1), "H"] = self.H
        else:
            iterplus = data.loc[uid].index.max() + 1
            data.loc[(uid, iterplus), :] = [
                T,
                0,
                self.energy,
                self.magnetization,
                0,
            ]
            data.at[(uid, iterplus), "spin"] = copy.deepcopy(self.spin)
            data.at[(uid, iterplus), "H"] = self.H
        return data
