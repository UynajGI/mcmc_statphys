"""Main module."""
import copy
from collections import deque
from typing import Dict

# here put the import lib
import numpy as np
import pandas as pd
import uuid
from tqdm import tqdm


def is_Flat(sequence: np.ndarray, epsilon: float = 0.1) -> bool:
    """Determine whether the sequence is flat / cn: 判断序列是否平坦

    Args:
        sequence (np.ndarray): sampling sequence / cn: 采样序列
        epsilon (float, optional): fluctuation. / cn: 涨落 (Defaults 0.1)

    Returns:
        bool: is flat / cn: 是否平坦
    """
    return np.std(sequence) / np.mean(sequence) < epsilon


def sample_acceptance(delta_E: float, sample_Temperture: float) -> bool:
    """Determine whether to accept a new state / cn: 判断是否接受新的状态

    Args:
        delta_E (float): Energy change / cn: 能量变化
        sample_Temperture (float): Sample temperature / cn: 采样温度

    Returns:
        bool: accept or reject / cn: 接受或拒绝
    """
    if delta_E <= 0:
        return True
    else:
        return np.random.rand() < np.exp(-delta_E / sample_Temperture)


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

    def __str__(self):
        return self.name

    def _getnum(self, string: str, type: str):
        num = input("Input {s}{t}: ".format(s=string, t=type))
        while num == "":
            num = input("Input h_min('q' exit): ")
            num = eval(num)
            if num == "q":
                exit()
        return num

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
            self.data.at[(uid, iterplus), "spin"] = copy.deepcopy(self.model.spin)

    def iter_sample(self, T: float, uid: str = None) -> str:
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
        if not sample_acceptance(delta_E, T):
            self.model = temp_model
        self._save_date(T, uid)
        return uid

    def equil_sample(self,
                     T: float,
                     max_iter: int = 1000,
                     uid: str = None) -> str:
        """Equilibrium sampling / cn: 平衡采样

        Args:
            T (float): Sample temperature / cn: 采样温度
            max_iter (int): Maximum iteration / cn: 最大迭代次数
            uid (str, optional): uid / cn: 唯一标识符 (Defaults None)
        """

        uid = self._setup_uid(uid)
        for iter in tqdm(range(max_iter)):
            self.iter_sample(T, uid)
        return uid

    def param_sample(self, max_iter: int = 1000) -> Dict:
        """_summary_

        Args:
            max_iter (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        self._init_parameter()
        param_lst = self._init_paramlst()
        uid_lst = []
        for param in tqdm(param_lst):
            uid = self._setup_uid(None)
            uid_lst.append(uid)
            if self.parameter == "T":
                if self.model.type == "ising" or self.model.type == "potts":
                    self.model.H = self.H0
                self.equil_sample(param, max_iter=max_iter, uid=uid)
            elif self.parameter == "H":
                self.model.H = param
                self.equil_sample(self.T0, max_iter=max_iter, uid=uid)
            param_lst.set_description("{parameter} = {param}".format(
                parameter=self.parameter, param=param))
        uid_param_dict: Dict = {
            "uid": uid_lst,
            "{param}".format(param=self.parameter): param_lst,
        }
        return uid_param_dict

    def _init_parameter(self):
        """Initialize parameter / cn: 初始化参数

        Raises:
            ValueError: Invalid parameter / cn: 无效的参数
        """
        self.parameter = input("Input parameter T/H: (default: T)") or "T"
        if self.parameter != "T" and self.parameter != "H":
            raise ValueError("Invalid parameter")

    def _init_paramlst(self):
        """Initialize parameter list / cn: 初始化参数列表"""
        if self.parameter == "T":
            Tmin = float(self._getnum(string="T", type="_min"))
            Tmax = float(self._getnum(string="T", type="_max"))
            num = int(self._getnum(string="sample", type="_num"))
            if self.model.type == "ising" or self.model.type == "potts":
                self.H0 = float(self._getnum(string="H", type="_0"))
            _rasie_parameter(Tmin, Tmax, num)
            return np.linspace(Tmin, Tmax, num=num)
        elif self.parameter == "H":
            if self.model.type != "ising" or self.model.type != "potts":
                raise ValueError(
                    "The model {type} without outfield effect, can't change field, please change model."
                    .format(type=self.model.type))
            else:
                hmin = float(self._getnum(string="H", type="_min"))
                hmax = float(self._getnum(string="H", type="_max"))
                num = int(self._getnum(string="sample", type="_num"))
                self.T0 = float(self._getnum(string="T", type="_0"))
                _rasie_parameter(hmin, hmax, num)
                return np.linspace(hmin, hmax, num=num)


class Wolff(Metropolis):
    """The Wolff algorithm, named after Ulli Wolff, is an algorithm for Monte Carlo simulation\n
    of the Ising model and Potts model.\n
    Details please see: https://en.wikipedia.org/wiki/Wolff_algorithm \n
    cn: Wolff 算法，以 Ulli Wolff 命名，是一种蒙特卡洛模拟算法，用于 Ising 模型和 Potts 模型。\n
    详情请见：https://en.wikipedia.org/wiki/Wolff_algorithm
    """

    def __init__(self, model: object):
        if model.type != "ising":
            raise ValueError("The model must be Ising")
        super().__init__(model)
        self.name = "Wolff"

    def iter_sample(self, T: float, uid: str = None) -> object:
        """_summary_

        Args:
            T (float): _description_
            uid (str, optional): _description_. Defaults to None.

        Returns:
            object: _description_
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
                b2 = np.random.rand() < (1 - np.exp(-2 * self.model.Jij / T))
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
            self.model.energy += (new_site_energy - old_site_energy)
            self.model.magnetization += (new_site - old_site)

        self._save_date(T, uid)
        return uid

    def equil_sample(self, T: float, max_iter: int = 1000, uid: str = None):
        uid = self._setup_uid(uid)
        for iter in tqdm(range(max_iter)):
            self.iter_sample(T, uid)
        return uid

    def param_sample(self, max_iter: int = 1000):
        """_summary_

        Args:
            max_iter (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        super()._init_parameter()
        param_lst = super()._init_paramlst()
        uid_lst = []
        for param in tqdm(param_lst):
            uid = self._setup_uid(None)
            uid_lst.append(uid)
            if self.parameter == "T":
                if self.model.type == "ising" or self.model.type == "potts":
                    self.model.H = self.H0
                self.equil_sample(param, max_iter=max_iter, uid=uid)
            elif self.parameter == "H":
                self.model.H = param
                self.equil_sample(self.T0, max_iter=max_iter, uid=uid)
        uid_param_dict: Dict = {
            "uid": uid_lst,
            "{param}".format(param=self.parameter): param_lst,
        }
        return uid_param_dict


class Anneal(Metropolis):

    def __init__(self, model: object):
        super().__init__(model)
        self.name = "Anneal"

    def iter_sample(self, T: float, uid: str = None) -> object:
        """_summary_

        Args:
            T (float): _description_
            uid (str, optional): _description_. Defaults to None.
        """
        uid = self._setup_uid(uid)
        super().iter_sample(T, uid)
        return uid

    def equil_sample(
        self,
        targetT: float,
        max_iter: int = 1000,
        highT=None,
        dencyT=0.9,
        uid: str = None,
    ):
        """_summary_

        Args:
            targetT (float): _description_
            max_iter (int, optional): _description_. Defaults to 1000.
            highT (int, optional): _description_. Defaults to 10.
            dencyT (float, optional): _description_. Defaults to 0.9.
            uid (str, optional): _description_. Defaults to None.
        """
        uid = self._setup_uid(uid)
        if highT is None:
            highT = targetT / (0.9**10)
        tempT = copy.deepcopy(highT)
        while highT < targetT:
            highT *= 2
            if highT > targetT:
                print(
                    "Your highT {old} < targetT {target}, we change highT = {new} now, please check your input next time."
                    .format(old=tempT, target=targetT, new=highT))
        T = copy.deepcopy(highT)
        while T > targetT:
            super().equil_sample(T, max_iter=max_iter, uid=uid)
            T = max(T * dencyT, targetT)
        return uid

    def param_sample(self, max_iter: int = 1000):
        """_summary_

        Args:
            max_iter (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        super()._init_parameter()
        param_lst = super()._init_paramlst()
        uid_lst = []
        for param in tqdm(param_lst):
            uid = self._setup_uid(None)
            uid_lst.append(uid)
            if self.parameter == "T":
                if self.model.type == "ising" or self.model.type == "potts":
                    self.model.H = self.H0
                self.equil_sample(param, max_iter=max_iter, uid=uid)
            elif self.parameter == "H":
                self.model.H = param
                self.equil_sample(self.T0, max_iter=max_iter, uid=uid)
        uid_param_dict: Dict = {
            "uid": uid_lst,
            "{param}".format(param=self.parameter): param_lst,
        }
        return uid_param_dict
