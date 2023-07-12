#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@文件    :Metropolis.py
@时间    :2023/07/12 11:55:19
@作者    :結凪
"""

import copy
from typing import List, Tuple, Dict, Union
import os
import numpy as np
import uuid
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import HTMLWriter
import statsmodels.tsa.stattools as stattools

__all__ = ["Metropolis"]


def _is_Flat(sequence: np.ndarray, epsilon: float = 0.1) -> bool:
    """Determine whether the sequence is flat / cn: 判断序列是否平坦

    Args:
        sequence (np.ndarray): sampling sequence / cn: 采样序列
        epsilon (float, optional): fluctuation. / cn: 涨落 (Defaults 0.1)

    Returns:
        bool: is flat / cn: 是否平坦
    """
    return np.std(sequence) / np.mean(sequence) < epsilon


def _sample_acceptance(delta_E: float, sample_Temperture: float, form: str = "class") -> bool:
    """
    Determine whether to accept the sample

    Parameters
    ----------
    delta_E: float
        Energy difference between the current state and the next state
    sample_Temperture: float
        Sample temperature
    form: str
        Acceptance form, "class" or "bath"

    Returns
    -------
    bool
        Whether to accept the sample
    """
    if form == "class":
        return np.random.rand() < np.exp(-delta_E / sample_Temperture)
    elif form == "bath":
        return np.random.rand() < 1 / (1 + np.exp(delta_E / sample_Temperture))


def _rename(column: str) -> str:
    """
    Rename the column name

    Parameters
    ----------
    column : str
        Column name

    Returns
    -------
    str
        New column name
    """
    if column == "E" or column == "e" or column == "Energy" or column == "energy":
        column = "energy"
    elif (
        column == "M"
        or column == "m"
        or column == "Magnetization"
        or column == "magnetization"
        or column == "Mag"
        or column == "mag"
        or column == "Magnet"
        or column == "magnet"
    ):
        column = "magnetization"
    elif (
        column == "S"
        or column == "s"
        or column == "Spin"
        or column == "spin"
        or column == "SpinMatrix"
        or column == "spinmatrix"
        or column == "Spinmatrix"
        or column == "spinMatrix"
    ):
        column = "spin"
    else:
        if column == "t" or column == "T" or column == "temperature" or column == "Temperature":
            column = "T"
        elif column == "h" or column == "H" or column == "field" or column == "Field":
            column = "H"
    return column


class Metropolis:
    """
    Metropolis algorithm
    ====================

    Example
    -------

    >>> import mcmc_statphys as mcsp
    >>> m = mcsp.model.Ising(L=10, dim=2)
    >>> f = mcsp.algorithm.Metropolis(m)
    >>> f.equil_sample(T=1.0,max_step=1000,uid="test")
    >>> f.data


    Description
    -----------

    In statistics and statistical physics, the Metropolis–Hastings algorithm is a Markov chain Monte Carlo (MCMC) method for obtaining a sequence of random samples from a probability distribution from which direct sampling is difficult. This sequence can be used to approximate the distribution (e.g. to generate a histogram) or to compute an integral (e.g. an expected value). Metropolis–Hastings and other MCMC algorithms are generally used for sampling from multi-dimensional distributions, especially when the number of dimensions is high. For single-dimensional distributions, there are usually other methods (e.g. adaptive rejection sampling) that can directly return independent samples from the distribution, and these are free from the problem of autocorrelated samples that is inherent in MCMC methods.

    References
    ----------

    -  [1] `Metropolis–Hastings algorithm -Wikipedia <https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm>`__


    """

    def __init__(self, model: object):
        self.model = model
        self._rowmodel = copy.deepcopy(model)  # row model
        self.name = "Metroplis"
        self.data = self.model._init_data()
        self.param_list = []

    def _reset_model(self):
        self.model = copy.deepcopy(self._rowmodel)

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

    def _init_paramlst(self, param: Tuple[float, float, int]) -> np.array:
        """
        init param list

        Parameters
        ----------
        param : Tuple[float, float, int]
            param_max, param_min, param_num

        Returns
        -------
        np.array
            param list
        """
        param_max, param_min, param_num = param
        return np.linspace(param_max, param_min, param_num)

    def iter_sample(self, T: float, uid: str = None, ac_from="class") -> str:
        """
        Iterative sampling

        Parameters
        ----------
        T : float
            Sample temperature
        uid : str, optional
            uid, by default None
        ac_from : str, optional
            Acceptance form, "class" or "bath", by default "class"

        Returns
        -------
        str
            uid
        """
        uid = self._setup_uid(uid)
        temp_model = copy.deepcopy(self.model)
        delta_E = self.model._random_walk()
        if not _sample_acceptance(delta_E, T, form=ac_from):
            self.model = temp_model
        self.data = self.model._save_date(T=T, uid=uid, data=self.data)
        return uid

    def equil_sample(self, T: float, max_iter: int = 1000, uid: str = None, ac_from="class") -> str:
        """
        Equilibrium sampling

        Parameters
        ----------
        T : float
            Sample temperature
        max_iter : int, optional
            Max iteration, by default 1000
        uid : str, optional
            uid, by default None
        ac_from : str, optional
            Acceptance form, "class" or "bath", by default "class"

        Returns
        -------
        str
            uid
        """
        uid = self._setup_uid(uid)
        for iter in tqdm(range(max_iter), leave=False):
            self.iter_sample(T, uid, ac_from=ac_from)
        return uid

    def param_sample(
        self,
        param: tuple,
        param_name: str or int = "T",
        stable: float = 0.0,
        max_iter: int = 1000,
        ac_from: str = "class",
    ) -> Dict:
        """
        Parameter sampling

        Parameters
        ----------
        param : tuple
            param_max, param_min, param_num
        param_name : str or int, optional
            param name, by default "T"
        stable : float, optional
            stable parameter, by default 0.0
        max_iter : int, optional
            Max iteration, by default 1000
        ac_from : str, optional
            Acceptance form, "class" or "bath", by default "class"

        Returns
        -------
        Dict
            uid_param_dict
        """
        self.parameter = _rename(param_name)
        param_lst = self._init_paramlst(param)
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

    def svd(self, uid: Union[str, dict, List[str]], norm: bool = True, t0: int = 0) -> np.array:
        """
        SVD

        Parameters
        ----------
        uid : Union[str, dict, List[str]]
            uid or uid list
        norm : bool, optional
            norm, by default True
        t0 : int, optional
            start time, by default 0

        Returns
        -------
        np.array
            svd

        Raises
        ------
        ValueError
            The key of the dict is not 'uid'.
        """
        # TODO: SVD 优化
        if isinstance(uid, str):
            data = self.data
            column: str = "spin"
            spin_lst = data.loc[uid][column][t0:]
            spin_matrix = []
            for spin in spin_lst:
                spin = spin.reshape(-1)
                spin_matrix.append(spin)
            spin_matrix = np.matrix(spin_matrix)
            _, s, _ = np.linalg.svd(spin_matrix, full_matrices=True)
            if norm:
                return s / np.linalg.norm(s)
            else:
                return s
        elif isinstance(uid, dict):
            if "uid" in uid.keys():
                uid_lst = uid["uid"]
            else:
                raise ValueError("The key of the dict is not 'uid'.")
            svd_lst = []
            for uid_item in uid_lst:
                svd_lst.append(self.svd(uid=uid_item, norm=norm))
            return svd_lst
        elif isinstance(uid, list):
            svd_lst = []
            for uid_item in uid:
                svd_lst.append(self.svd(uid=uid_item, norm=norm))
            return svd_lst

    def mean(self, uid: str, column: str, t0: int = 0, n: int = 1) -> float:
        """
        Mean

        Parameters
        ----------
        uid : str
            uid
        column : str
            column
        t0 : int, optional
            start time, by default 0
        n : int, optional
            power, by default 1

        Returns
        -------
        float
            mean
        """
        column = _rename(column)
        return np.mean(self.data.loc[uid][column][t0:] ** n)

    def std(self, uid: str, column: str, t0: int = 0) -> float:
        """
        Standard deviation

        Parameters
        ----------
        uid : str
            uid
        column : str
            column
        t0 : int, optional
            start time, by default 0

        Returns
        -------
        float
            standard deviation
        """
        column = _rename(column)
        return np.std(self.data.loc[uid][column][t0:])

    def var(self, uid: str, column: str, t0: int = 0) -> float:
        """
        Variance

        Parameters
        ----------
        uid : str
            uid
        column : str
            column
        t0 : int, optional
            start time, by default 0

        Returns
        -------
        float
            variance
        """
        column = _rename(column)
        return np.var(self.data.loc[uid][column][t0:])

    def norm(self, uid: str, column: str, t0: int = 0, ord: int = 2) -> float:
        """
        Norm

        Parameters
        ----------
        uid : str
            uid
        column : str
            column
        t0 : int, optional
            start time, by default 0
        ord : int, optional
            order, by default 2

        Returns
        -------
        float
            norm
        """
        column = _rename(column)
        return np.linalg.norm(self.data.loc[uid][column][t0:], ord=ord)

    def diff(self, uid: str, column: str, t0: int = 0, n: int = 1) -> np.array:
        """
        Difference

        Parameters
        ----------
        uid : str
            uid
        column : str
            column
        t0 : int, optional
            start time, by default 0
        n : int, optional
            power, by default 1

        Returns
        -------
        np.array
            difference
        """
        column = _rename(column)
        return np.diff(self.data.loc[uid][column][t0:], n)

    def cv(self, uid: str, column: str, t0: int = 0) -> float:
        """
        Coefficient of variation

        Parameters
        ----------
        uid : str
            uid
        column : str
            column
        t0 : int, optional
            start time, by default 0

        Returns
        -------
        float
            coefficient of variation
        """
        column = _rename(column)
        return self.std(uid, column, t0) / self.mean(uid, column, t0)

    def u4(self, uid: str, t0: int = 0) -> float:
        """
        U4

        Parameters
        ----------
        uid : str
            uid
        t0 : int, optional
            start time, by default 0

        Returns
        -------
        float
            U4
        """
        return 1 - self.mean(uid, "magnetization", t0, n=4) / (3 * self.mean(uid, "magnetization", t0, n=2) ** 2)

    def getcolumn(self, uid: str, column: str, t0: int = 0) -> np.array:
        """
        Get column

        Parameters
        ----------
        uid : str
            uid
        column : str
            column
        t0 : int, optional
            start time, by default 0

        Returns
        -------
        np.array
            column
        """
        column = _rename(column)
        return self.data.loc[uid][column][t0:]

    def autocorrelation(self, uid: str, column: str) -> Tuple[float, np.array]:
        """
        Autocorrelation

        Parameters
        ----------
        uid : str
            uid
        column : str
            column

        Returns
        -------
        Tuple[float, np.array]
            tau, autocorrelation
        """
        column = _rename(column)
        autocorrelation_list = stattools.acf(self.getcolumn(uid, column), nlags=len((self.getcolumn(uid, column))))
        tau = np.argmin(np.abs(autocorrelation_list - np.exp(-1)))
        return (tau, autocorrelation_list)

    def curve(self, uid: str, column: str, t0: int = 0):
        """
        Curve

        Parameters
        ----------
        uid : str
            uid
        column : str
            column
        t0 : int, optional
            start time, by default 0

        Attributes
        ----------
        matplotlib.pyplot.plot
        """
        data = self.data
        column = _rename(column)
        array = data.loc[uid][column][t0:]
        index = data.loc[uid].index
        plt.plot(index, array)

    def scatter(self, uid, column, t0: int = 0):
        """
        Scatter

        Parameters
        ----------
        uid : str
            uid
        column : str
            column
        t0 : int, optional
            start time, by default 0

        Attributes
        ----------
        matplotlib.pyplot.scatter
        """
        data = self.data
        column = _rename(column)
        array = data.loc[uid][column][t0:]
        index = data.loc[uid].index
        plt.scatter(index, array)

    # def param_plot(self, uid_dict: Dict[str, np.array], column: str, per: bool = True):
    #     """
    #     Parameter plot

    #     Parameters
    #     ----------
    #     uid_dict : Dict[str, np.array]
    #         uid_dict
    #     column : str
    #         column
    #     per : bool, optional
    #         per, by default True

    #     Raises
    #     ------
    #     ValueError
    #         _description_
    #     ValueError
    #         _description_
    #     """
    #     column = _rename(column)
    #     x = []
    #     y = []
    #     if isinstance(uid_dict, dict):
    #         param_name = list(uid_dict.keys())[1]
    #         for i in range(len(list(uid_dict.values())[0])):
    #             uid = list(uid_dict.values())[0][i]
    #             param = list(uid_dict.values())[1][i]
    #             if uid not in self.data.index.get_level_values("uid").values:
    #                 raise ValueError("Invalid uid.")
    #             x.append(param)
    #             y.append(self.mean(uid, column))
    #     else:
    #         raise ValueError("Invalid uid_dict.")
    #     if per:
    #         plt.plot(x, y / self.model.N, label=param_name)
    #     else:
    #         plt.plot(x, y, label=param_name)

    # def param_scatter(self, uid_dict: Dict[str, np.array], column: str, per: bool = True) -> None:
    #     """
    #     Draw a parametric scatter.
    #     """
    #     column = _rename(column)
    #     x = []
    #     y = []
    #     if isinstance(uid_dict, dict):
    #         param_name = list(uid_dict.keys())[1]
    #         for i in range(len(list(uid_dict.values())[0])):
    #             uid = list(uid_dict.values())[0][i]
    #             param = list(uid_dict.values())[1][i]
    #             if uid not in self.data.index.get_level_values("uid").values:
    #                 raise ValueError("Invalid uid.")
    #             x.append(param)
    #             y.append(self.mean(uid, column))
    #     else:
    #         raise ValueError("Invalid uid_dict.")
    #     if per:
    #         plt.scatter(x, y / self.model.N, label=param_name)
    #     else:
    #         plt.scatter(x, y, label=param_name)

    def imshow(self, uid: str, iter: int, cmap: str = "gray") -> None:
        """
        Show the spin.

        Parameters
        ----------
        uid : str
            uid
        iter : int
            iter
        cmap : str, optional
            cmap, by default "gray"
        Attributes
        ----------
        matplotlib.pyplot.imshow
        """
        spin = self.data.loc[(uid, iter), "spin"]
        plt.imshow(spin, cmap=cmap)
        plt.axis("off")
        plt.axis("equal")

    def animate(self, uid: str, save: bool = False, savePath: str = None) -> None:
        """
        Animate the spin.

        Parameters
        ----------
        uid : str
            uid
        save : bool, optional
            save, by default False
        savePath : str, optional
            savePath, by default None

        Attributes
        ----------
        matplotlib.animation.FuncAnimation
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        spin_lst = self.data.loc[uid, "spin"].tolist()

        def init():
            ax.imshow(spin_lst[0], cmap="gray")
            ax.axis("off")
            return ax

        def update(iter):
            ax.clear()
            ax.imshow(spin_lst[iter], cmap="gray")
            ax.set_title("iter: {}".format(iter))
            ax.axis("off")
            return ax

        ani = animation.FuncAnimation(fig=fig, func=update, init_func=init, frames=range(len(spin_lst)))
        if save:
            mywriter = HTMLWriter(fps=60)
            if savePath is None:
                if not os.path.exists(uid):
                    os.mkdir(uid)
                os.chdir(uid)
            else:
                if not os.path.exists(savePath):
                    os.mkdir(savePath)
                os.chdir(savePath)
            ani.save("myAnimation.html", writer=mywriter)
            plt.close()
