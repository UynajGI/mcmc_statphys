#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@文件    :Demon.py
@时间    :2023/07/12 20:10:35
@作者    :結凪
"""


import copy
import numpy as np
import pandas as pd
import uuid
from tqdm import tqdm

__all__ = ["Demon"]


class Demon(object):
    """
    Demon algorithm
    ===============

    Example
    -------
    >>> import mcmc_statphys as mcsp
    >>> m = mcsp.model.Ising(L=10, dim=2)
    >>> from mcsp.algorithm import Demon
    >>> model = Ising(10)

    """

    def __init__(self, model):
        self.model = model
        self.name = "Demon"
        self.Es = 0
        self.Ed = 0
        self._rowmodel = copy.deepcopy(self.model)
        self._init_data()

    def _setup_uid(self, uid):
        if uid is None:
            uid = (uuid.uuid1()).hex
        else:
            if not self.data.empty:
                if uid not in self.data.index.get_level_values("uid").values:
                    self._reset_model()
                else:
                    self.model.set_spin(self.data.loc[uid].loc[self.data.loc[uid].index.max()].spin)
        return uid

    def _init_data(self):
        self.data: pd.DataFrame = pd.DataFrame(columns=["uid", "iter", "H", "Es", "Ed", "spin"])
        self.data.set_index(["uid", "iter"], inplace=True)

    def _save_date(self, uid):
        if uid not in self.data.index.get_level_values("uid").values:
            self.data.loc[(uid, 1), :] = [self.model.H, self.Es, self.Ed, 0]
            self.data.at[(uid, 1), "spin"] = copy.deepcopy(self.model.spin)
        else:
            iterplus = self.data.loc[uid].index.max() + 1
            self.data.loc[(uid, iterplus), :] = [self.model.H, self.Es, self.Ed, 0]
            self.data.at[(uid, iterplus), "spin"] = copy.deepcopy(self.model.spin)

    def _reset_model(self):
        self.model = copy.deepcopy(self._rowmodel)

    def iter_sample(self, uid: str = None) -> str:
        """
        iter_sample

        Parameters
        ----------
        uid : str, optional
            uid, by default None

        Returns
        -------
        str
            uid
        """
        uid = self._setup_uid(uid)
        self.Es = self.model.energy
        temp_model = copy.deepcopy(self.model)
        delta_E = self.model._random_walk()
        if delta_E <= 0:
            self.Ed += abs(delta_E)
            self.Es -= abs(delta_E)
        else:
            if self.Ed >= abs(delta_E):
                self.Ed -= abs(delta_E)
                self.Es += abs(delta_E)
            else:
                self.model = temp_model
        self._save_date(uid)
        return uid

    def equil_sample(
        self,
        max_iter: int = 1000,
        uid: str = None,
    ) -> str:
        """
        equil_sample

        Parameters
        ----------
        max_iter : int, optional
            The maximum number of iterations, by default 1000
        uid : str, optional
            uid, by default None

        Returns
        -------
        str
            uid
        """
        uid = self._setup_uid(uid)
        for iter in tqdm(range(max_iter), leave=False):
            self.iter_sample(uid)
        return uid

    def get_temperature(self, uid: str, t0: int = 1) -> list:
        """
        get_temperature

        Parameters
        ----------
        uid :
            uid
        t0 : int, optional
            The number of iterations to be discarded, by default 1

        Returns
        -------
        list
            Temperature list
        """
        J = self.model.Jij
        lst = []
        Tst = []
        forlst = self.data.loc[uid]["Ed"].tolist()[t0:]
        for i in forlst:
            lst.append(i)
            T = (1 / (J * 4) * np.log(1 + 4 * J / np.mean(lst))) ** (-1)
            Tst.append(T)
        return Tst
