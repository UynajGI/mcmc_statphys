#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@文件    :Anneal.py
@时间    :2023/07/12 20:14:35
@作者    :結凪
"""

from typing import Dict
import copy
from tqdm import tqdm
from .Metropolis import Metropolis
from .Metropolis import _rename

__all__ = ["Anneal"]


class Anneal(Metropolis):
    """
    Simulated annealing
    ===================

    Example
    -------
    >>> import mcmc_statphys as mcsp
    >>> m = mcsp.model.Ising(L=10, dim=2)
    >>> from mcsp.algorithm import Anneal
    >>> m = Ising(10)
    >>> f = Anneal(m)
    >>> f.equil_sample(2.0, 1000)
    >>> f.data

    Description
    -----------

    Simulated annealing (SA) is a probabilistic technique for approximating the global optimum of a given function. Specifically, it is a metaheuristic to approximate global optimization in a large search space for an optimization problem. For large numbers of local optima, SA can find the global optima. It is often used when the search space is discrete (for example the traveling salesman problem, the boolean satisfiability problem, protein structure prediction, and job-shop scheduling). For problems where finding an approximate global optimum is more important than finding a precise local optimum in a fixed amount of time, simulated annealing may be preferable to exact algorithms such as gradient descent or branch and bound.

    Reference
    ---------

    -  [1] `Simulated annealing algorithm -Wikipedia <https://en.wikipedia.org/wiki/Simulated_annealing>`__

    """

    def __init__(self, model: object):
        super().__init__(model)
        self.name = "Anneal"

    # def iter_sample(self, T: float, uid: str = None, ac_from="class") -> object:
    #     """_summary_

    #     Args:
    #         T (float): _description_
    #         uid (str, optional): _description_. Defaults to None.
    #     """
    #     uid = self._setup_uid(uid)
    #     super().iter_sample(T, uid, ac_from=ac_from)
    #     return uid

    def equil_sample(
        self, targetT: float, max_iter: int = 1000, highT=None, dencyT=0.9, uid: str = None, ac_from="class"
    ):
        """
        Equilibrium sampling

        Parameters
        ----------

        targetT : float
            Target temperature.
        max_iter : int, optional
            Maximum number of iterations. The default is 1000.
        highT : float, optional
            High temperature. The default is None.
        dencyT : float, optional
            Density of temperature. The default is 0.9.
        uid : str, optional
            uid. The default is None.
        ac_from : str, optional
            Acceptance criterion. The default is "class".

        Returns
        -------

        uid : str
            uid
        """

        uid = self._setup_uid(uid)
        if highT is None:
            highT = targetT / (0.9**10)
        tempT = copy.deepcopy(highT)
        while highT < targetT:
            highT *= 2
            if highT > targetT:
                print(
                    "Your highT {old} < targetT {target}, we change highT = {new} now, please check your input next time.".format(
                        old=tempT, target=targetT, new=highT
                    )
                )
        T = copy.deepcopy(highT)
        while T > targetT:
            super().equil_sample(T, max_iter=max_iter, uid=uid, ac_from=ac_from)
            T = max(T * dencyT, targetT)
        return uid

    def param_sample(
        self,
        param: tuple,
        param_name: str or int = "T",
        stable: float = 0.0,
        max_iter: int = 1000,
        ac_from: str = "class",
    ):
        """_summary_

        Args:
            max_iter (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        self.parameter = _rename(param_name)
        param_lst = super()._init_paramlst(param)
        uid_lst = []
        for param in tqdm(param_lst):
            uid = self._setup_uid(None)
            uid_lst.append(uid)
            if self.parameter == "T":
                if self.model.type == "ising" or self.model.type == "potts":
                    self.model.H = stable
                self.equil_sample(param, max_iter=max_iter, uid=uid, ac_from=ac_from)
            elif self.parameter == "H":
                self.model.H = param
                self.equil_sample(stable, max_iter=max_iter, uid=uid, ac_from=ac_from)
        uid_param_dict: Dict = {
            "uid": uid_lst,
            "{param}".format(param=self.parameter): param_lst,
        }
        self.param_list.append(uid_param_dict)
        return uid_param_dict
