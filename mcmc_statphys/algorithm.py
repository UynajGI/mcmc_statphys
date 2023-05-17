"""Main module."""
import copy
from collections import deque
from typing import List, Tuple, Dict

# here put the import lib
import numpy as np
import pandas as pd
import uuid
from tqdm import tqdm

# TODO[0.2.1](Changed): 重构模型类
# TODO[0.2.1](Added): 添加进度条功能 tqmd


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
            if uid not in self.iter_data.index.get_level_values("uid").values:
                self._reset_model()
            else:
                self.model.set_spin(
                    self.iter_data.loc[uid]
                    .loc[self.iter_data.loc[uid].index.max()]
                    .spin
                )
        return uid

    def _init_data(self):
        self.iter_data: pd.DataFrame = pd.DataFrame(
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
        self.iter_data.set_index(["uid", "iter"], inplace=True)

    def _save_date(self, T, uid):
        if uid not in self.iter_data.index.get_level_values("uid").values:
            self.iter_data.loc[(uid, 1), :] = [
                T,
                self.model.H,
                self.model._get_total_energy(),
                self.model._get_total_magnetization(),
                0,
            ]
            self.iter_data.at[(uid, 1), "spin"] = self.model.spin
        else:
            iterplus = self.iter_data.loc[uid].index.max() + 1
            self.iter_data.loc[(uid, iterplus), :] = [
                T,
                self.model.H,
                self.model._get_total_energy(),
                self.model._get_total_magnetization(),
                0,
            ]
            self.iter_data.at[(uid, iterplus), "spin"] = self.model.spin

    def iter_sample(self, T: float, uid: str = None) -> object:
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

    def equil_sample(
        self, T: float, max_iter: int = 1000, uid: str = None
    ) -> Tuple[List[float], List[float], List[np.ndarray]]:
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

    def param_sample(self, max_iter: int = 1000):
        """_summary_

        Args:
            max_iter (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        self._init_parameter()
        param_lst = tqdm(self._init_paramlst())
        uid_lst = []
        for param in param_lst:
            uid = self._setup_uid(None)
            uid_lst.append(uid)
            if self.parameter == "T":
                if self.model.tpye == "ising" or self.model.tpye == "potts":
                    self.model.H = self.H0
                self.equil_sample(param, max_iter=max_iter, uid=uid)
            elif self.parameter == "H":
                self.model.H = param
                self.equil_sample(self.T0, max_iter=max_iter, uid=uid)
            param_lst.set_description(
                "{parameter} = {param}".format(parameter=self.parameter, param=param)
            )
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
            if self.model.tpye == "ising" or self.model.tpye == "potts":
                self.H0 = float(self._getnum(string="H", type="_0"))
            _rasie_parameter(Tmin, Tmax, num)
            return np.linspace(Tmin, Tmax, num=num)
        elif self.parameter == "H":
            if self.model.tpye != "ising" or self.model.tpye != "potts":
                raise ValueError(
                    "The model {tpye} without outfield effect, can't change field, please change model.".format(
                        tpye=self.model.tpye
                    )
                )
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
        if model.tpye != "ising":
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
            self.model.spin[clip] *= -1
        self._save_date(T, uid)

    def equil_sample(self, T: float, max_iter: int = 1000, uid: str = None):
        uid = self._setup_uid(uid)
        for iter in tqdm(range(max_iter)):
            self.iter_sample(T, uid)

    def param_sample(self, max_iter: int = 1000):
        """_summary_

        Args:
            max_iter (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        super()._init_parameter()
        param_lst = tqdm(super()._init_paramlst())
        uid_lst = []
        for param in param_lst:
            uid = self._setup_uid(None)
            uid_lst.append(uid)
            if self.parameter == "T":
                if self.model.tpye == "ising" or self.model.tpye == "potts":
                    self.model.H = self.H0
                self.equil_sample(param, max_iter=max_iter, uid=uid)
            elif self.parameter == "H":
                self.model.H = param
                self.equil_sample(self.T0, max_iter=max_iter, uid=uid)
            param_lst.set_description(
                "{parameter} = {param}".format(parameter=self.parameter, param=param)
            )
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
        super().iter_sample(T, uid)

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
                    "Your highT {old} < targetT {target}, we change highT = {new} now, please check your input next time.".format(
                        old=tempT, target=targetT, new=highT
                    )
                )
        T = copy.deepcopy(highT)
        while T > targetT:
            super().equil_sample(T, max_iter=max_iter, uid=uid)
            T = max(T * dencyT, targetT)

    def param_sample(self, max_iter: int = 1000):
        """_summary_

        Args:
            max_iter (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        super()._init_parameter()
        param_lst = tqdm(super()._init_paramlst())
        uid_lst = []
        for param in param_lst:
            uid = self._setup_uid(None)
            uid_lst.append(uid)
            if self.parameter == "T":
                if self.model.tpye == "ising" or self.model.tpye == "potts":
                    self.model.H = self.H0
                self.equil_sample(param, max_iter=max_iter, uid=uid)
            elif self.parameter == "H":
                self.model.H = param
                self.equil_sample(self.T0, max_iter=max_iter, uid=uid)
            param_lst.set_description(
                "{parameter} = {param}".format(parameter=self.parameter, param=param)
            )
        uid_param_dict: Dict = {
            "uid": uid_lst,
            "{param}".format(param=self.parameter): param_lst,
        }
        return uid_param_dict


# TODO[0.2.1](Added): 添加本征态采样功能

# class Simulation:

#     def __init__(self, model: object):
#         self.model = model

#     def sample_acceptance(self, delta_E: float,
#                           sample_Temperture: float) -> bool:
#         """Determine whether to accept a new state / cn: 判断是否接受新的状态

#         Args:
#             delta_E (float): Energy change / cn: 能量变化
#             sample_Temperture (float): Sample temperature / cn: 采样温度

#         Returns:
#             bool: accept or reject / cn: 接受或拒绝
#         """
#         if delta_E <= 0:
#             return True
#         else:
#             return np.random.rand() < np.exp(-delta_E / sample_Temperture)

#     def is_Flat(self, sequence: np.ndarray, epsilon: float = 0.1) -> bool:
#         """Determine whether the sequence is flat / cn: 判断序列是否平坦

#         Args:
#             sequence (np.ndarray): sampling sequence / cn: 采样序列
#             epsilon (float, optional): fluctuation. / cn: 涨落 (Defaults 0.1)

#         Returns:
#             bool: is flat / cn: 是否平坦
#         """
#         return np.std(sequence) / np.mean(sequence) < epsilon

#     def iter_sample(self, sample_Temperture: float) -> object:
#         """single sample / cn: 单次采样

#         Args:
#             sample_Temperture (float): Sample temperature / cn: 采样温度

#         Returns:
#             object: model / cn: 模型
#         """
#         site = tuple(
#             np.random.randint(0, self.model.L, size=self.model.dim))
#         temp_model = copy.deepcopy(self.model)
#         delta_E = self.model._change_delta_energy(site)
#         if not self.sample_acceptance(delta_E, sample_Temperture):
#             self.model = temp_model
#         return self.model

#     def iter_wolff_sample(self, T: float) -> object:
#         """single wolff sample / cn: 单次 wolff 采样 \n
#         The Wolff algorithm, named after Ulli Wolff, is an algorithm for Monte Carlo simulation\n
#         of the Ising model and Potts model.\n
#         Details please see: https://en.wikipedia.org/wiki/Wolff_algorithm \n
#         cn: Wolff 算法，以 Ulli Wolff 命名，是一种蒙特卡洛模拟算法，用于 Ising 模型和 Potts 模型。\n
#         详情请见：https://en.wikipedia.org/wiki/Wolff_algorithm

#         Args:
#             T (float): Sample temperature / cn: 采样温度

#         Raises:
#             ValueError: Invalid type of spin / cn: 无效的自旋类型

#         Returns:
#             object: model / cn: 模型
#         """
#         if self.model.tpye != "ising":
#             # wolff 算法只适用于 ising 模型
#             raise ValueError("Invalid type of spin")
#         else:
#             cluster = set()
#             neighbors = deque()
#             # 随机选取一个点
#             site = tuple(
#                 np.random.randint(0, self.model.L, size=self.model.dim))
#             neighbors.append(site)
#             cluster.add(site)
#             while len(neighbors) > 0:
#                 neighbor = neighbors.pop()
#                 total_neighbors = self.model._get_neighbor(neighbor)
#                 for same_neighbor in total_neighbors:
#                     b1 = self.model.spin[same_neighbor] == self.model.spin[
#                         site]
#                     b2 = np.random.rand() < (1 -
#                                              np.exp(-2 * self.model.Jij / T))
#                     b3 = same_neighbor not in cluster
#                     if b1 and b2 and b3:
#                         cluster.add(same_neighbor)
#                         neighbors.append(same_neighbor)
#             for clip in cluster:
#                 self.model.spin[clip] *= -1
#         return self.model

#     def metropolis_sample(
#             self, T: float,
#             max_iter: int) -> Tuple[List[int], List[float], object]:
#         """Metropolis sampling / cn: Metropolis 采样
#             In statistics and statistical physics, the Metropolis-Hastings algorithm\n
#             is a Markov chain Monte Carlo (MCMC) method for obtaining a sequence of\n
#             random samples from a probability distribution from which direct sampling is difficult.\n
#             Details please see: https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm \n
#             cn: 在统计学和统计物理学中，Metropolis-Hastings 算法是一种马尔可夫链蒙特卡洛 (MCMC) 方法，\n
#             用于从难以直接采样的概率分布中获得一系列随机样本。\n

#         Args:
#             T (float): sample temperature / cn: 采样温度
#             max_iter (int): max iteration / cn: 最大迭代次数

#         Returns:
#             Tuple[List[int], List[float], object]: return energys, magnetizations, model / cn: 返回能量，磁化，模型
#         """
#         energys: List[float] = []
#         magnetizations: List[float] = []
#         with alive_bar(max_iter,
#                        manual=True,
#                        title="Metropolis Sample",
#                        force_tty=True) as bar:
#             for iter in range(max_iter):
#                 self.iter_sample(T)
#                 _, energy, magnetization = self.model._get_per_info()
#                 energys.append(energy)
#                 magnetizations.append(magnetization)
#                 bar(iter / max_iter)
#                 if iter % 1000 == 0:
#                     pass  # TODO[0.2.1](Added): 添加日志功能
#                 if self.is_Flat(energys) and iter > 5000:
#                     break
#             bar(1.0)
#         return (energys, magnetizations, self.model)

#     def simulate_anneal_sample(
#             self,
#             T_min: float,
#             Tdceny: float = 0.9,
#             T: float = 10,
#             max_iter: int = 5000) -> Tuple[List[int], List[float], object]:
#         """Simulated annealing sampling / cn: 模拟退火采样
#         Simulated annealing (SA) is a probabilistic technique for approximating\n
#         the global optimum of a given function. Specifically, it is a metaheuristic\n
#         to approximate global optimization in a large search space for an optimization problem.
#         Details please see: https://en.wikipedia.org/wiki/Simulated_annealing \n
#         cn: 模拟退火 (SA) 是一种概率技术，用于近似给定函数的全局最优解。\n
#         具体来说，它是一种元启发式算法，用于近似优化问题的大搜索空间中的全局优化。\n

#         Args:
#             T_min (float): Minimum temperature / cn: 最小温度
#             Tdceny (float, optional): Temperature decay / cn: 温度衰减 (Defaults 0.9)
#             T (float, optional): Initial temperature / cn: 初始温度 (Defaults 10)
#             max_iter (int, optional):  Maximum iteration / cn: 最大迭代次数 (Defaults 5000)

#         Returns:
#             Tuple[List[int], List[float], object]: return energys, magnetizations, model / cn: 返回能量，磁化，模型
#         """
#         energys_anneal: List[float] = []
#         magnetizations_anneal: List[float] = []
#         while T < T_min:
#             T *= 2
#         with alive_bar(
#                 title="Simulate Anneal",
#                 unknown="stars",
#                 spinner="message_scrolling",
#                 force_tty=True,
#         ) as bar:
#             while T > T_min:
#                 energys, magnetizations, _ = self.metropolis_sample(
#                     T, max_iter)
#                 energys_anneal.extend(energys)
#                 magnetizations_anneal.extend(magnetizations)
#                 T = max(Tdceny * T, T_min)
#                 bar()
#         return (energys_anneal, magnetizations_anneal, self.model)

#     def wolff_sample(self, T: float,
#                      max_iter: int) -> Tuple[List[float], List[float], object]:
#         """Wolff sampling / cn: Wolff 采样

#         Args:
#             T (float): sample temperature / cn: 采样温度
#             max_iter (int): max iteration / cn: 最大迭代次数

#         Returns:
#             Tuple[List[float], List[float], object]: return energys, magnetizations, model / cn: 返回能量，磁化，模型
#         """
#         energys_wolff: List[float] = []
#         magnetizations_wolff: List[float] = []
#         with alive_bar(max_iter, force_tty=True) as bar:
#             for iter in range(max_iter):
#                 self.iter_wolff_sample(T)
#                 if iter % 1000 == 0:
#                     pass  # TODO[0.2.1](Added): 添加日志功能
#                 _, energy, magnetization = self.model._get_per_info()
#                 energys_wolff.append(energy)
#                 magnetizations_wolff.append(magnetization)
#                 bar()
#         return (energys_wolff, magnetizations_wolff, self.model)

# class ParameterSample(Simulation):

#     def __init__(self, model: object, *args, **kwargs):
#         super().__init__(model)
#         self._rowmodel = copy.deepcopy(model)
#         self._init_parameter()
#         self._init_paramlst()

#     def _init_parameter(self):
#         """Initialize parameter / cn: 初始化参数

#         Raises:
#             ValueError: Invalid parameter / cn: 无效的参数
#             ValueError: Invalid algorithm / cn: 无效的算法
#         """
#         if not hasattr(self, "parameter"):
#             self.parameter = input("Input parameter T/H: (default: T)") or "T"
#             if self.parameter != "T" and self.parameter != "H":
#                 raise ValueError("Invalid parameter")
#         if not hasattr(self, "algorithm"):
#             algorithm = (input(
#                 "Input algorithm:\n\t[1] metropolis\n\t[2] wolff\n\t[3] anneal: (default: [1] metropolis)"
#             ) or "metropolis")
#             if algorithm == "1" or algorithm == "metropolis":
#                 self.algorithm = "metropolis"
#             elif algorithm == "2" or algorithm == "wolff":
#                 self.algorithm = "wolff"
#             elif algorithm == "3" or algorithm == "anneal":
#                 self.algorithm = "anneal"
#             else:
#                 raise ValueError("Invalid algorithm")

#     def _init_paramlst(self):
#         """Initialize parameter list / cn: 初始化参数列表"""
#         if self.parameter == "T":
#             # T_lst 不存在，input 初始化
#             if not hasattr(self, "Tlst"):
#                 Tmin = input("Input T_min: ")
#                 while Tmin == "":
#                     Tmin = input("Input T_min('q' exit): ")
#                     # 输入 q 退出程序
#                     if Tmin == "q":
#                         exit()
#                 Tmax = input("Input T_max('q' exit): ")
#                 while Tmax == "":
#                     Tmax = input("Input T_max: ")
#                     if Tmax == "q":
#                         exit()
#                 num = input("Input sample num: ")
#                 while num == "":
#                     num = input("Input sample num('q' exit): ")
#                     if num == "q":
#                         exit()
#                 Tmin = float(Tmin)
#                 Tmax = float(Tmax)
#                 num = int(num)
#                 # 判断输入是否合法
#                 if Tmin > Tmax:
#                     raise ValueError("T_min should be less than T_max")
#                 if num < 1:
#                     raise ValueError("num should be greater than 0")
#                 # 判断 Tmin 是否合法
#                 if Tmin < 0:
#                     raise ValueError("T_min should be greater than 0")
#                 self._init_Tlst(Tmin, Tmax, num)
#         elif self.parameter == "H":
#             if self.model.tpye != "ising" or self.model.tpye != "potts":
#                 raise ValueError(
#                     "The model {tpye} without outfield effect, can't change field, please change model."
#                     .format(tpye=self.model.tpye))
#             else:
#                 if not hasattr(self, "hlst"):
#                     hmin = input("Input h_min: ")
#                     hmax = input("Input h_max: ")
#                     num = input("Input sample num: ")
#                     self._init_hlst(hmin, hmax, num)
#                     if hmin > hmax:
#                         raise ValueError("h_min should be less than h_max")
#                     if num < 1:
#                         raise ValueError("num should be greater than 0")
#                     if hmin < 0:
#                         raise ValueError("h_min should be greater than 0")
#                     if num != int(num):
#                         num = int(num)
#                         print("num should be a positive integer, now num = ",
#                               num)
#                     self._init_hlst(hmin, hmax, num)
#         else:
#             raise ValueError("Invalid parameter")

#     def _init_Tlst(self, T_min: float, T_max: float, num: int):
#         """Initialize temperature list / cn: 初始化温度列表

#         Args:
#             T_min (float): T_min / cn: 最小温度
#             T_max (float): T_max / cn: 最大温度
#             num (int): sample num / cn: 采样数量
#         """
#         self.Tlst = np.linspace(T_min, T_max, num=num)

#     def _init_hlst(self, h_min: float, h_max: float, num: int):
#         """Initialize field list / cn: 初始化场强列表

#         Args:
#             h_min (float): h_min / cn: 最小场强
#             h_max (float): h_max / cn: 最大场强
#             num (int): sample num / cn: 采样数量
#         """
#         self.hlst = np.linspace(h_min, h_max, num=num)

#     def _init_model(self):
#         """Initialize model / cn: 初始化模型"""
#         self.model = copy.deepcopy(self._rowmodel)

#     def sample(
#             self,
#             max_iter: int = 10000
#     ) -> Tuple[List[float], List[float], List[float]]:
#         """Sample / cn: 采样

#         Args:
#             max_iter (int, optional): Maximum iteration / cn: 最大迭代次数 (Defaults 10000)

#         Returns:
#             Tuple[List[float], List[float], List[float]]: return parameter list, energy list, magnetization list / cn: 返回参数列表，能量列表，磁化列表
#         """
#         if self.parameter == "T":
#             H0 = input("Input H0: (default: 0)") or 0
#             self.model.H = H0
#             p_lst = self.Tlst
#         elif self.parameter == "H":
#             T0 = input("Input T0: (default: 1)") or 1
#             p_lst = self.hlst
#         else:
#             raise ValueError("Invalid parameter")

#         energy_lst = []
#         magnetization_lst = []

#         if self.algorithm == "metropolis":
#             if self.parameter == "T":
#                 for T in self.Tlst:
#                     self._init_model()
#                     energy, magnetization, _ = self.metropolis_sample(
#                         T=T, max_iter=max_iter)
#                     energy_lst.append(energy)
#                     magnetization_lst.append(magnetization)
#             elif self.parameter == "H":
#                 for H in self.hlst:
#                     self._init_model()
#                     self.model.H = H
#                     energy, magnetization, _ = self.metropolis_sample(
#                         T=T0, max_iter=max_iter)
#                     energy_lst.append(energy)
#                     magnetization_lst.append(magnetization)
#         elif self.algorithm == "wolff":
#             if self.parameter == "T":
#                 for T in self.Tlst:
#                     self._init_model()
#                     energy, magnetization, _ = self.wolff_sample(
#                         T=T, max_iter=max_iter)
#                     energy_lst.append(energy)
#                     magnetization_lst.append(magnetization)
#             elif self.parameter == "H":
#                 for H in self.hlst:
#                     self.model.H = H
#                     self._init_model()
#                     energy, magnetization, _ = self.wolff_sample(
#                         T=T0, max_iter=max_iter)
#                     energy_lst.append(energy)
#                     magnetization_lst.append(magnetization)
#         elif self.algorithm == "anneal":
#             if self.parameter == "T":
#                 for T in self.Tlst:
#                     self._init_model()
#                     energy, magnetization, _ = self.simulate_anneal_sample(
#                         T=T, max_iter=max_iter)
#                     energy_lst.append(energy)
#                     magnetization_lst.append(magnetization)
#             elif self.parameter == "H":
#                 for H in self.hlst:
#                     self._init_model()
#                     self.model.H = H
#                     energy, magnetization, _ = self.simulate_anneal_sample(
#                         T=T0, max_iter=max_iter)
#                     energy_lst.append(energy)
#                     magnetization_lst.append(magnetization)
#         else:
#             raise ValueError("Invalid algorithm")
#         return (p_lst, energy_lst, magnetization_lst)
