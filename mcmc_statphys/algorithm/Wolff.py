#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@文件    :Wolff.py
@时间    :2023/07/12 19:49:23
@作者    :結凪
"""

from collections import deque
from typing import Dict

# here put the import lib
import numpy as np
from tqdm import tqdm
from .Metropolis import Metropolis
from .Metropolis import _rename

__all__ = ["Wolff"]


class Wolff(Metropolis):
    """
    Wolff algorithm
    ===============

    Examples
    --------

    >>> import mcmc_statphys as mcsp
    >>> m = mcsp.model.Ising(L=10, dim=2)
    >>> f = mcsp.algorithm.Wolff(m)
    >>> f.equil_sample(T=1.0,max_step=1000,uid="test")
    >>> f.data

    Description
    -----------

    The Wolff algorithm, named after Ulli Wolff, is an algorithm for Monte Carlo simulation of the Ising model and Potts model in which the unit to be flipped is not a single spin (as in the heat bath or Metropolis algorithms) but a cluster of them. This cluster is defined as the set of connected spins sharing the same spin states, based on the Fortuin-Kasteleyn representation.

    References
    ----------

    -  [1] `Wolff algorithm -Wikipedia <https://en.wikipedia.org/wiki/Wolff_algorithm>`__

    """

    def __init__(self, model: object):
        # TODO: 增加对于其他模型的支持
        if model.type != "ising" and model.type != "rfising":
            raise ValueError("The model must be Ising")
        super().__init__(model)
        self.name = "Wolff"

    def iter_sample(self, T: float, uid: str = None, **kwargs) -> str:
        """
        Iterative sampling

        Parameters
        ----------
        T : float
            The temperature
        uid : str, optional
            The uid of the data, by default None

        Returns
        -------
        str
            The uid of the data

        """
        uid = self._setup_uid(uid)
        cluster = set()
        neighbors = deque()
        # 随机选取一个点
        site = tuple(np.random.randint(0, self.model.L, size=self.model.dim))
        neighbors.append(site)
        cluster.add(site)
        while len(neighbors) > 0:
            neighbor = neighbors.pop()
            total_neighbors = self.model._get_neighbor(neighbor)
            for same_neighbor in total_neighbors:
                b1 = self.model.spin[same_neighbor] == self.model.spin[site]
                b2 = np.random.rand() < (1 - np.exp(-2 * self.model.J / T))
                b3 = same_neighbor not in cluster
                if b1 and b2 and b3:
                    cluster.add(same_neighbor)
                    neighbors.append(same_neighbor)
        for clip in cluster:
            old_site = self.model.spin[clip]
            old_site_energy = self.model._get_site_energy(clip)

            self.model.spin[clip] *= -1

            new_site = self.model.spin[clip]
            new_site_energy = self.model._get_site_energy(clip)
            self.model.energy += new_site_energy - old_site_energy
            self.model.magnetization += new_site - old_site

        self.data = self.model._save_date(T=T, uid=uid, data=self.data)
        return uid

    # def equil_sample(self, T: float, max_iter: int = 1000, uid: str = None) -> str:
    #     """
    #     Equilibrium sampling

    #     Parameters
    #     ----------
    #     T : float
    #         The temperature
    #     max_iter : int, optional
    #         The maximum number of iterations, by default 1000
    #     uid : str, optional
    #         The uid of the data, by default None

    #     Returns
    #     -------
    #     str
    #         The uid of the data
    #     """
    #     uid = self._setup_uid(uid)
    #     for iter in tqdm(range(max_iter), leave=False):
    #         self.iter_sample(T, uid)
    #     return uid

    # def param_sample(self, param: tuple, param_name: str or int = "T", stable: float = 0.0, max_iter: int = 1000):
    #     """_summary_

    #     Args:
    #         max_iter (int, optional): _description_. Defaults to 1000.

    #     Returns:
    #         _type_: _description_
    #     """
    #     self.parameter = _rename(param_name)
    #     param_lst = super()._init_paramlst(param)
    #     uid_lst = []
    #     for param in tqdm(param_lst):
    #         uid = self._setup_uid(None)
    #         uid_lst.append(uid)
    #         if self.parameter == "T":
    #             if self.model.type == "ising" or self.model.type == "potts":
    #                 self.model.H = stable
    #             self.equil_sample(param, max_iter=max_iter, uid=uid)
    #         elif self.parameter == "H":
    #             self.model.H = param
    #             self.equil_sample(stable, max_iter=max_iter, uid=uid)
    #     uid_param_dict: Dict = {
    #         "uid": uid_lst,
    #         "{param}".format(param=self.parameter): param_lst,
    #     }
    #     self.param_list.append(uid_param_dict)
    #     return uid_param_dict
