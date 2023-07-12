#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@文件    :Tempering.py
@时间    :2023/07/12 19:49:36
@作者    :結凪
"""

from typing import Dict
import copy
from tqdm import tqdm
import numpy as np
import uuid
import pandas as pd
from .Metropolis import Metropolis

__all__ = ["Tempering"]


class Tempering(Metropolis):
    """
    Parallel tempering
    ==================

    Examples
    --------

    >>> import mcmc_statphys as mcsp
    >>> m = mcsp.model.Ising(L=10, dim=2)
    >>> f = mcsp.algorithm.Tempering(m)
    >>> Tmax, Tmin, Tlen = 1, 3, 5
    >>> f.param_sample(T: (Tmin,Tmax,Tlen), H0 = 0.0, max_iter = 1000, eq_iter = 1000, ac_from = "class")
    >>> f.data

    Description
    -----------

    Parallel tempering in physics and statistics is a computer simulation method typically used to find the lowest energy state of a system of many interacting particles. It addresses the problem that at high temperature one may have a stable state different from low temperature, whereas simulations at low temperature may become "stuck" in a metastable state. It does this by using the fact that the high temperature simulation may visit states typical of both stable and metastable low temperature states.

    More specifically, parallel tempering (also known as replica exchange MCMC sampling), is a simulation method aimed at improving the dynamic properties of Monte Carlo method simulations of physical systems, and of Markov chain Monte Carlo (MCMC) sampling methods more generally. The replica exchange method was originally devised by Robert Swendsen and J. S. Wang, then extended by Charles J. Geyer, and later developed further by Giorgio Parisi, Koji Hukushima and Koji Nemoto, and others. Y. Sugita and Y. Okamoto also formulated a molecular dynamics version of parallel tempering; this is usually known as replica-exchange molecular dynamics or REMD.

    Essentially, one runs N copies of the system, randomly initialized, at different temperatures. Then, based on the Metropolis criterion one exchanges configurations at different temperatures. The idea of this method is to make configurations at high temperatures available to the simulations at low temperatures and vice versa. This results in a very robust ensemble which is able to sample both low and high energy configurations. In this way, thermodynamical properties such as the specific heat, which is in general not well computed in the canonical ensemble, can be computed with great precision.

    References
    ----------

    -  [1] `Parallel tempering -Wikipedia <https://en.wikipedia.org/wiki/Parallel_tempering>`__

    """

    def __init__(self, model: object):
        super().__init__(model)
        self.name = "Tempering"

    # def iter_sample(self, T: float, uid: str = None, ac_from="class") -> str:
    #     uid = self._setup_uid(uid)
    #     super().iter_sample(T, uid, ac_from=ac_from)
    #     return uid

    # def equil_sample(self, T: float, max_iter: int = 1000, uid: str = None, ac_from="class") -> str:
    #     uid = self._setup_uid(uid)
    #     for iter in tqdm(range(max_iter), leave=False):
    #         self.iter_sample(T, uid, ac_from=ac_from)
    #     return uid

    def param_sample(
        self, T: tuple, H0: float = 0.0, max_iter: int = 1000, eq_iter: int = 1000, ac_from: str = "class"
    ) -> Dict[str, float]:
        self.model.H = H0
        Tmin, Tmax, Tlen = T
        T_lst = np.linspace(Tmin, Tmax, Tlen)
        algo_lst = [Metropolis(copy.deepcopy(self.model)) for T in T_lst]
        uid_lst = [uuid.uuid1().hex for T in T_lst]
        for iter in tqdm(range(max_iter), leave=False):
            for i_algo in range(len(algo_lst)):
                algo_lst[i_algo].equil_sample(T=T_lst[i_algo], max_iter=eq_iter, uid=uid_lst[i_algo], ac_from=ac_from)
            for i_T in range(len(T_lst) - 1):
                Delta = (1 / T_lst[i_T + 1] - 1 / T_lst[i_T]) * (
                    algo_lst[i_T].model.energy - algo_lst[i_T + 1].model.energy
                )
                if np.exp(-Delta) > np.random.rand():
                    uid_lst[i_T], uid_lst[i_T + 1] = uid_lst[i_T + 1], uid_lst[i_T]
                    algo_lst[i_T], algo_lst[i_T + 1] = algo_lst[i_T + 1], algo_lst[i_T]
        self.data = pd.concat([algo.data for algo in algo_lst])
        uid_param_dict: Dict = {"uid": uid_lst, "T": T_lst}
        self.param_list.append(uid_param_dict)
        return uid_param_dict
