import copy
from typing import Dict, List
import os

# here put the import lib
import numpy as np
import pandas as pd
import uuid
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import HTMLWriter
import statsmodels.tsa.stattools as stattools

__all__ = ['Metropolis']


def _is_Flat(sequence: np.ndarray, epsilon: float = 0.1) -> bool:
    """Determine whether the sequence is flat / cn: 判断序列是否平坦

    Args:
        sequence (np.ndarray): sampling sequence / cn: 采样序列
        epsilon (float, optional): fluctuation. / cn: 涨落 (Defaults 0.1)

    Returns:
        bool: is flat / cn: 是否平坦
    """
    return np.std(sequence) / np.mean(sequence) < epsilon


def _sample_acceptance(delta_E: float,
                       sample_Temperture: float,
                       form: str = 'class') -> bool:
    """Determine whether to accept a new state / cn: 判断是否接受新的状态

    Args:
        delta_E (float): Energy change / cn: 能量变化
        sample_Temperture (float): Sample temperature / cn: 采样温度

    Returns:
        bool: accept or reject / cn: 接受或拒绝
    """
    if form == 'class':
        return np.random.rand() < np.exp(-delta_E / sample_Temperture)
    elif form == 'bath':
        return np.random.rand() < 1 / (1 + np.exp(delta_E / sample_Temperture))


def _rasie_parameter(minparam: float, maxparam: float, num: int):
    """Raise parameter / cn: 抛出参数

    Args:
        minparam (float): minparam / cn: 最小参数
        maxparam (float): maxparam / cn: 最大参数
        num (int): num / cn: 采样数量

    Raises:
        ValueError: min should be less than max
        ValueError: num should be greater than 0
        ValueError: min should be greater than 0
    """
    if minparam > maxparam:
        raise ValueError("min should be less than max")
    if num < 1:
        raise ValueError("num should be greater than 0")
    if minparam < 0:
        raise ValueError("min should be greater than 0")


def _rename(column):
    if column == "E" or column == "e" or column == "Energy" or column == "energy":
        column = "energy"
    elif column == "M" or column == "m" or column == "Magnetization" or column == "magnetization" or column == "Mag" or column == "mag" or column == "Magnet" or column == "magnet":
        column = "magnetization"
    elif column == "S" or column == "s" or column == "Spin" or column == "spin" or column == "SpinMatrix" or column == "spinmatrix" or column == "Spinmatrix" or column == "spinMatrix":
        column = "spin"
    else:
        if column == 't' or column == 'T' or column == 'temperature' or column == 'Temperature':
            column = 'T'
        elif column == 'h' or column == 'H' or column == 'field' or column == 'Field':
            column = 'H'
    return column


class Metropolis:
    """
    \tIn statistics and statistical physics, the Metropolis–Hastings algorithm\n
    is a Markov chain Monte Carlo (MCMC) method for obtaining a sequence of\n
    random samples from a probability distribution from which direct sampling is difficult.\n
    Details please see: https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm \n
    \tcn: 在统计学和统计物理学中，Metropolis-Hastings 算法是一种马尔可夫链蒙特卡洛 (MCMC) 方法，\n
    用于从难以直接采样的概率分布中获得一系列随机样本。\n
    """

    def __init__(self, model: object):
        self.model = model
        self._rowmodel = copy.deepcopy(model)
        self.name = "Metroplis"
        self._init_data()
        self.param_list = []

    def __str__(self):
        return self.name

    def _reset_model(self):
        self.model = copy.deepcopy(self._rowmodel)

    def _setup_uid(self, uid):
        if uid is None:
            uid = (uuid.uuid1()).hex
        else:
            if uid not in self.data.index.get_level_values("uid").values:
                self._reset_model()
            else:
                self.model.set_spin(self.data.loc[uid].loc[
                    self.data.loc[uid].index.max()].spin)
        return uid

    def _init_data(self):
        if self.model.type == "nvt":
            self.data: pd.DataFrame = pd.DataFrame(columns=[
                "uid",
                "iter",
                "T",
                "energy",
                "spin",
            ])
            self.data.set_index(["uid", "iter"], inplace=True)
        else:
            self.data: pd.DataFrame = pd.DataFrame(columns=[
                "uid",
                "iter",
                "T",
                "H",
                "energy",
                "magnetization",
                "spin",
            ])
            self.data.set_index(["uid", "iter"], inplace=True)

    def _save_date(self, T, uid):
        if self.model.type == 'nvt':
            if uid not in self.data.index.get_level_values("uid").values:
                self.data.loc[(uid, 1), :] = [
                    T,
                    self.model.energy,
                    0,
                ]
                self.data.at[(uid, 1), "spin"] = copy.deepcopy(self.model.spin)
            else:
                iterplus = self.data.loc[uid].index.max() + 1
                self.data.loc[(uid, iterplus), :] = [
                    T,
                    self.model.energy,
                    0,
                ]
                self.data.at[(uid, iterplus),
                             "spin"] = copy.deepcopy(self.model.spin)
        else:
            if uid not in self.data.index.get_level_values("uid").values:
                self.data.loc[(uid, 1), :] = [
                    T,
                    self.model.H,
                    self.model.energy,
                    self.model.magnetization,
                    0,
                ]
                self.data.at[(uid, 1), "spin"] = copy.deepcopy(self.model.spin)
            else:
                iterplus = self.data.loc[uid].index.max() + 1
                self.data.loc[(uid, iterplus), :] = [
                    T,
                    self.model.H,
                    self.model.energy,
                    self.model.magnetization,
                    0,
                ]
                self.data.at[(uid, iterplus),
                             "spin"] = copy.deepcopy(self.model.spin)

    def _init_paramlst(self, param):
        """Initialize parameter list / cn: 初始化参数列表"""
        param_max, param_min, param_num = param
        return np.linspace(param_max, param_min, param_num)

    def iter_sample(self, T: float, uid: str = None, ac_from='class') -> str:
        """Single sample / cn: 单次采样

        Args:
            T (float): Sample temperature / cn: 采样温度
            uid (str, optional): uid / cn: 唯一标识符 (Defaults None)

        Returns:
            object: model / cn: 模型
        """
        uid = self._setup_uid(uid)
        site = tuple(np.random.randint(0, self.model.L, size=self.model.dim))
        temp_model = copy.deepcopy(self.model)
        delta_E = self.model._change_delta_energy(site)
        if not _sample_acceptance(delta_E, T, form=ac_from):
            self.model = temp_model
        self._save_date(T, uid)
        return uid

    def equil_sample(self,
                     T: float,
                     max_iter: int = 1000,
                     uid: str = None,
                     ac_from='class') -> str:
        """Equilibrium sampling / cn: 平衡采样

        Args:
            T (float): Sample temperature / cn: 采样温度
            max_iter (int): Maximum iteration / cn: 最大迭代次数
            uid (str, optional): uid / cn: 唯一标识符 (Defaults None)
        """

        uid = self._setup_uid(uid)
        for iter in tqdm(range(max_iter), leave=False):
            self.iter_sample(T, uid, ac_from=ac_from)
        return uid

    def param_sample(self,
                     param: tuple,
                     param_name: str or int = 'T',
                     stable: float = 0.0,
                     max_iter: int = 1000,
                     ac_from: str = 'class') -> Dict:
        """_summary_

        Args:
            max_iter (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
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
                self.equil_sample(param,
                                  max_iter=max_iter,
                                  uid=uid,
                                  ac_from=ac_from)
            elif self.parameter == "H":
                self.model.H = param
                self.equil_sample(stable,
                                  max_iter=max_iter,
                                  uid=uid,
                                  ac_from=ac_from)
        uid_param_dict: Dict = {
            "uid": uid_lst,
            "{param}".format(param=self.parameter): param_lst,
        }
        self.param_list.append(uid_param_dict)
        return uid_param_dict

    def svd(self,
            uid: str or Dict or List[str],
            norm: bool = True,
            t0: int = 0) -> np.array:

        if isinstance(uid, str):
            data = self.data
            column: str = 'spin'
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
            if 'uid' in uid.keys():
                uid_lst = uid['uid']
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
        column = _rename(column)
        return np.mean(self.data.loc[uid][column][t0:]**n)

    def std(self, uid: str, column: str, t0: int = 0) -> float:
        column = _rename(column)
        return np.std(self.data.loc[uid][column][t0:])

    def var(self, uid: str, column: str, t0: int = 0) -> float:
        column = _rename(column)
        return np.var(self.data.loc[uid][column][t0:])

    def norm(self, uid: str, column: str, t0: int = 0, ord: int = 2) -> float:
        column = _rename(column)
        return np.linalg.norm(self.data.loc[uid][column][t0:], ord=ord)

    def diff(self, uid: str, column: str, t0: int = 0, n: int = 1) -> np.array:
        column = _rename(column)
        return np.diff(self.data.loc[uid][column][t0:], n)

    def cv(self, uid: str, column: str, t0: int = 0) -> float:
        column = _rename(column)
        return self.std(uid, column, t0) / self.mean(uid, column, t0)

    def u4(self, uid: str, t0: int = 0) -> float:
        return 1 - self.mean(uid, "magnetization", t0, n=4) / (
            3 * self.mean(uid, "magnetization", t0, n=2)**2)

    def getcolumn(self, uid: str, column: str, t0: int = 0) -> np.array:
        column = _rename(column)
        return self.data.loc[uid][column][t0:]

    def autocorrelation(self, uid: str, column: str):
        column = _rename(column)
        result = stattools.acf(self.getcolumn(uid, column),
                               nlags=len((self.getcolumn(uid, column))))
        return result

    def curve(self, uid, column, t0: int = 0) -> None:
        """
        Draw a curve.
        """
        data = self.data
        column = _rename(column)
        array = data.loc[uid][column][t0:]
        index = data.loc[uid].index
        plt.plot(index, array)

    def scatter(self,
                uid,
                column,
                t0: int = 0,
                s=None,
                c=None,
                marker=None,
                cmap=None,
                norm=None,
                vmin=None,
                vmax=None,
                alpha=None,
                linewidths=None,
                *,
                edgecolors=None,
                plotnonfinite=False,
                data=None,
                **kwargs) -> None:

        data = self.data
        column = _rename(column)
        array = data.loc[uid][column][t0:]
        index = data.loc[uid].index
        plt.scatter(index,
                    array,
                    s=s,
                    c=c,
                    marker=marker,
                    cmap=cmap,
                    norm=norm,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=alpha,
                    linewidths=linewidths,
                    edgecolors=edgecolors,
                    plotnonfinite=plotnonfinite,
                    data=data,
                    **kwargs)

    def param_plot(self,
                   uid_dict: Dict[str, np.array],
                   column: str,
                   per: bool = True) -> None:
        """
        Draw a parametric plot.
        """
        column = _rename(column)
        x = []
        y = []
        if isinstance(uid_dict, dict):
            param_name = list(uid_dict.keys())[1]
            for i in range(len(list(uid_dict.values())[0])):
                uid = list(uid_dict.values())[0][i]
                param = list(uid_dict.values())[1][i]
                if uid not in self.data.index.get_level_values("uid").values:
                    raise ValueError("Invalid uid.")
                x.append(param)
                y.append(self.mean(uid, column))
        else:
            raise ValueError("Invalid uid_dict.")
        if per:
            plt.plot(x, y / self.model.N, label=param_name)
        else:
            plt.plot(x, y, label=param_name)

    def param_scatter(self,
                      uid_dict: Dict[str, np.array],
                      column: str,
                      per: bool = True) -> None:
        """
        Draw a parametric scatter.
        """
        column = _rename(column)
        x = []
        y = []
        if isinstance(uid_dict, dict):
            param_name = list(uid_dict.keys())[1]
            for i in range(len(list(uid_dict.values())[0])):
                uid = list(uid_dict.values())[0][i]
                param = list(uid_dict.values())[1][i]
                if uid not in self.data.index.get_level_values("uid").values:
                    raise ValueError("Invalid uid.")
                x.append(param)
                y.append(self.mean(uid, column))
        else:
            raise ValueError("Invalid uid_dict.")
        if per:
            plt.scatter(x, y / self.model.N, label=param_name)
        else:
            plt.scatter(x, y, label=param_name)

    def imshow(self,
               uid: str,
               iter: int,
               cmap: str = 'gray',
               norm=None,
               aspect=None,
               interpolation=None,
               alpha=None,
               vmin=None,
               vmax=None,
               origin=None,
               extent=None,
               interpolation_stage=None,
               filternorm=True,
               filterrad=4.0,
               resample=None,
               url=None,
               data=None,
               **kwargs) -> None:
        """
        Draw a inshow.
        """
        spin = self.data.loc[(uid, iter), "spin"]
        plt.imshow(spin,
                   cmap=cmap,
                   norm=norm,
                   aspect=aspect,
                   interpolation=interpolation,
                   alpha=alpha,
                   vmin=vmin,
                   vmax=vmax,
                   origin=origin,
                   extent=extent,
                   interpolation_stage=interpolation_stage,
                   filternorm=filternorm,
                   filterrad=filterrad,
                   resample=resample,
                   url=url,
                   data=data,
                   **kwargs)
        plt.axis('off')
        plt.axis('equal')

    def animate(self,
                uid: str,
                save: bool = False,
                savePath: str = None) -> None:
        """
        Animate the spin.
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        spin_lst = self.data.loc[uid, 'spin'].tolist()

        def init():
            ax.imshow(spin_lst[0], cmap='gray')
            ax.axis('off')
            return ax

        def update(iter):
            ax.clear()
            ax.imshow(spin_lst[iter], cmap='gray')
            ax.set_title('iter: {}'.format(iter))
            ax.axis('off')
            return ax

        ani = animation.FuncAnimation(fig=fig,
                                      func=update,
                                      init_func=init,
                                      frames=range(len(spin_lst)))
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
            ani.save('myAnimation.html', writer=mywriter)
            plt.close()
