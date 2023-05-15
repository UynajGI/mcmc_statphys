"""Main module."""
import copy
from collections import deque
from typing import List, Tuple
from alive_progress import alive_bar

# here put the import lib
import numpy as np


# TODO[1.0.0](Changed): 重构模型类
# TODO[0.2.1](Added): 添加进度条功能 alive_progress
class Simulation():

    def __init__(self, model: object):
        self.model = model

    def sample_acceptance(self, delta_E: float,
                          sample_Temperture: float) -> bool:
        """ Determine whether to accept a new state / cn: 判断是否接受新的状态

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

    def iter_sample(self, sample_Temperture: float) -> object:
        """single sample / cn: 单次采样

        Args:
            sample_Temperture (float): Sample temperature / cn: 采样温度

        Returns:
            object: model / cn: 模型
        """
        site = tuple(
            np.random.randint(0, self.model.L, size=self.model.dimension))
        temp_model = copy.deepcopy(self.model)
        delta_E = self.model._change_delta_energy(site)
        if not self.sample_acceptance(delta_E, sample_Temperture):
            self.model = temp_model
        return self.model

    def iter_wolff_sample(self, T: float) -> object:
        """single wolff sample / cn: 单次 wolff 采样 \n
        The Wolff algorithm, named after Ulli Wolff, is an algorithm for Monte Carlo simulation\n
        of the Ising model and Potts model.\n
        Details please see: https://en.wikipedia.org/wiki/Wolff_algorithm \n
        cn: Wolff 算法，以 Ulli Wolff 命名，是一种蒙特卡洛模拟算法，用于 Ising 模型和 Potts 模型。\n
        详情请见：https://en.wikipedia.org/wiki/Wolff_algorithm

        Args:
            T (float): Sample temperature / cn: 采样温度

        Raises:
            ValueError: Invalid type of spin / cn: 无效的自旋类型

        Returns:
            object: model / cn: 模型
        """
        if self.model.tpye != 'ising':
            # wolff 算法只适用于 ising 模型
            raise ValueError('Invalid type of spin')
        else:
            cluster = set()
            neighbors = deque()
            # 随机选取一个点
            site = tuple(
                np.random.randint(0, self.model.L, size=self.model.dimension))
            neighbors.append(site)
            cluster.add(site)
            while len(neighbors) > 0:
                neighbor = neighbors.pop()
                total_neighbors = self.model._get_neighbor(neighbor)
                for same_neighbor in total_neighbors:
                    b1 = self.model.spin[same_neighbor] == self.model.spin[
                        site]
                    b2 = np.random.rand() < (1 -
                                             np.exp(-2 * self.model.Jij / T))
                    b3 = same_neighbor not in cluster
                    if b1 and b2 and b3:
                        cluster.add(same_neighbor)
                        neighbors.append(same_neighbor)
            for clip in cluster:
                self.model.spin[clip] *= -1
        return self.model

    def is_Flat(self, sequence: np.ndarray, epsilon: float = 0.1) -> bool:
        """ Determine whether the sequence is flat / cn: 判断序列是否平坦

        Args:
            sequence (np.ndarray): sampling sequence / cn: 采样序列
            epsilon (float, optional): fluctuation. / cn: 涨落 (Defaults 0.1)

        Returns:
            bool: is flat / cn: 是否平坦
        """
        return np.std(sequence) / np.mean(sequence) < epsilon

    def metropolis_sample(
            self, T: float,
            max_iter: int) -> Tuple[List[int], List[float], object]:
        """ Metropolis sampling / cn: Metropolis 采样
            In statistics and statistical physics, the Metropolis–Hastings algorithm\n
            is a Markov chain Monte Carlo (MCMC) method for obtaining a sequence of\n
            random samples from a probability distribution from which direct sampling is difficult.\n
            Details please see: https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm \n
            cn: 在统计学和统计物理学中，Metropolis-Hastings 算法是一种马尔可夫链蒙特卡洛 (MCMC) 方法，\n
            用于从难以直接采样的概率分布中获得一系列随机样本。\n

        Args:
            T (float): sample temperature / cn: 采样温度
            max_iter (int): max iteration / cn: 最大迭代次数

        Returns:
            Tuple[List[int], List[float], object]: return energys, magnetizations, model / cn: 返回能量，磁化，模型
        """
        energys: List[float] = []
        magnetizations: List[float] = []
        with alive_bar(max_iter,
                       manual=True,
                       title="Metropolis Sample",
                       force_tty=True) as bar:
            for iter in range(max_iter):
                self.iter_sample(T)
                _, energy, magnetization = self.model._get_per_info()
                energys.append(energy)
                magnetizations.append(magnetization)
                bar(iter / max_iter)
                if iter % 1000 == 0:
                    pass  # TODO[0.2.1](Added): 添加日志功能
                if self.is_Flat(energys) and iter > 5000:
                    break
            bar(1.0)
        return (energys, magnetizations, self.model)

    def simulate_anneal_sample(
            self,
            T_min: float,
            Tdceny: float = 0.9,
            T: float = 10,
            max_iter: int = 5000) -> Tuple[List[int], List[float], object]:
        """ Simulated annealing sampling / cn: 模拟退火采样
        Simulated annealing (SA) is a probabilistic technique for approximating\n
        the global optimum of a given function. Specifically, it is a metaheuristic\n
        to approximate global optimization in a large search space for an optimization problem.
        Details please see: https://en.wikipedia.org/wiki/Simulated_annealing \n
        cn: 模拟退火 (SA) 是一种概率技术，用于近似给定函数的全局最优解。\n
        具体来说，它是一种元启发式算法，用于近似优化问题的大搜索空间中的全局优化。\n

        Args:
            T_min (float): Minimum temperature / cn: 最小温度
            Tdceny (float, optional): Temperature decay / cn: 温度衰减 (Defaults 0.9)
            T (float, optional): Initial temperature / cn: 初始温度 (Defaults 10)
            max_iter (int, optional):  Maximum iteration / cn: 最大迭代次数 (Defaults 5000)

        Returns:
            Tuple[List[int], List[float], object]: return energys, magnetizations, model / cn: 返回能量，磁化，模型
        """
        energys_anneal: List[float] = []
        magnetizations_anneal: List[float] = []
        while T < T_min:
            T *= 2
        with alive_bar(title="Simulate Anneal",
                       unknown="stars",
                       spinner="message_scrolling",
                       force_tty=True) as bar:
            while T > T_min:
                energys, magnetizations, _ = self.metropolis_sample(
                    T, max_iter)
                energys_anneal.extend(energys)
                magnetizations_anneal.extend(magnetizations)
                T = max(Tdceny * T, T_min)
                bar()
        return (energys_anneal, magnetizations_anneal, self.model)

    def wolff_sample(self, T: float,
                     max_iter: int) -> Tuple[List[float], List[float], object]:
        """ Wolff sampling / cn: Wolff 采样

        Args:
            T (float): sample temperature / cn: 采样温度
            max_iter (int): max iteration / cn: 最大迭代次数

        Returns:
            Tuple[List[float], List[float], object]: return energys, magnetizations, model / cn: 返回能量，磁化，模型
        """
        energys_wolff: List[float] = []
        magnetizations_wolff: List[float] = []
        with alive_bar(max_iter, force_tty=True) as bar:
            for iter in range(max_iter):
                self.iter_wolff_sample(T)
                if iter % 1000 == 0:
                    pass  # TODO[0.2.1](Added): 添加日志功能
                _, energy, magnetization = self.model._get_per_info()
                energys_wolff.append(energy)
                magnetizations_wolff.append(magnetization)
                bar()
        return (energys_wolff, magnetizations_wolff, self.model)


class ParameterSample(Simulation):

    def __init__(self, model: object, *args, **kwargs):
        super().__init__(model)
        self.rowmodel = copy.deepcopy(model)
        self._init_parameter()
        self._init_paramlst()

    def _init_parameter(self):
        """ Initialize parameter / cn: 初始化参数

        Raises:
            ValueError: Invalid parameter / cn: 无效的参数
            ValueError: Invalid algorithm / cn: 无效的算法
        """
        if not hasattr(self, 'parameter'):
            self.parameter = input('Input parameter T/h: (default: T)') or 'T'
            if self.parameter != 'T' and self.parameter != 'h':
                raise ValueError('Invalid parameter')
        if not hasattr(self, 'algorithm'):
            algorithm = input(
                'Input algorithm:\n\t[1] metropolis\n\t[2] wolff\n\t[3] anneal: (default: [1] metropolis)'
            ) or 'metropolis'
            if algorithm == '1' or algorithm == 'metropolis':
                self.algorithm = 'metropolis'
            elif algorithm == '2' or algorithm == 'wolff':
                self.algorithm = 'wolff'
            elif algorithm == '3' or algorithm == 'anneal':
                self.algorithm = 'anneal'
            else:
                raise ValueError('Invalid algorithm')

    def _init_paramlst(self):
        """ Initialize parameter list / cn: 初始化参数列表
        """
        if self.parameter == 'T':
            # T_lst 不存在，input 初始化
            if not hasattr(self, 'Tlst'):
                Tmin = input('Input T_min: ')
                while Tmin == '':
                    Tmin = input('Input T_min(\'q\' exit): ')
                    # 输入 q 退出程序
                    if Tmin == 'q':
                        exit()
                Tmax = input('Input T_max(\'q\' exit): ')
                while Tmax == '':
                    Tmax = input('Input T_max: ')
                    if Tmax == 'q':
                        exit()
                num = input('Input sample num: ')
                while num == '':
                    num = input('Input sample num(\'q\' exit): ')
                    if num == 'q':
                        exit()
                Tmin = float(Tmin)
                Tmax = float(Tmax)
                num = int(num)
                # 判断输入是否合法
                if Tmin > Tmax:
                    raise ValueError('T_min should be less than T_max')
                if num < 1:
                    raise ValueError('num should be greater than 0')
                # 判断 Tmin 是否合法
                if Tmin < 0:
                    raise ValueError('T_min should be greater than 0')
                self._init_Tlst(Tmin, Tmax, num)
        elif self.parameter == 'h':
            if self.model.tpye != 'ising' or self.model.tpye != 'potts':
                raise ValueError(
                    'The model {tpye} without outfield effect, can\'t change field, please change model.'
                    .format(tpye=self.model.tpye))
            else:
                if not hasattr(self, 'hlst'):
                    hmin = input('Input h_min: ')
                    hmax = input('Input h_max: ')
                    num = input('Input sample num: ')
                    self._init_hlst(hmin, hmax, num)
                    if hmin > hmax:
                        raise ValueError('h_min should be less than h_max')
                    if num < 1:
                        raise ValueError('num should be greater than 0')
                    if hmin < 0:
                        raise ValueError('h_min should be greater than 0')
                    if num != int(num):
                        num = int(num)
                        print('num should be a positive integer, now num = ',
                              num)
                    self._init_hlst(hmin, hmax, num)
        else:
            raise ValueError('Invalid parameter')

    def _init_Tlst(self, T_min: float, T_max: float, num: int):
        """ Initialize temperature list / cn: 初始化温度列表

        Args:
            T_min (float): T_min / cn: 最小温度
            T_max (float): T_max / cn: 最大温度
            num (int): sample num / cn: 采样数量
        """
        self.Tlst = np.linspace(T_min, T_max, num=num)

    def _init_hlst(self, h_min: float, h_max: float, num: int):
        """ Initialize field list / cn: 初始化场强列表

        Args:
            h_min (float): h_min / cn: 最小场强
            h_max (float): h_max / cn: 最大场强
            num (int): sample num / cn: 采样数量
        """
        self.hlst = np.linspace(h_min, h_max, num=num)

    def _init_model(self):
        """ Initialize model / cn: 初始化模型
        """
        self.model = copy.deepcopy(self.rowmodel)

    def sample(
            self,
            max_iter: int = 10000
    ) -> Tuple[List[float], List[float], List[float]]:
        """ Sample / cn: 采样

        Args:
            max_iter (int, optional): Maximum iteration / cn: 最大迭代次数 (Defaults 10000)

        Returns:
            Tuple[List[float], List[float], List[float]]: return parameter list, energy list, magnetization list / cn: 返回参数列表，能量列表，磁化列表
        """
        if self.parameter == 'T':
            h0 = input('Input h0: (default: 0)') or 0
            self.model.H = h0
            p_lst = self.Tlst
        elif self.parameter == 'h':
            T0 = input('Input T0: (default: 1)') or 1
            p_lst = self.hlst
        else:
            raise ValueError('Invalid parameter')

        energy_lst = []
        magnetization_lst = []

        if self.algorithm == 'metropolis':
            if self.parameter == 'T':
                for T in self.Tlst:
                    self._init_model()
                    energy, magnetization, _ = self.metropolis_sample(
                        T=T, max_iter=max_iter)
                    energy_lst.append(energy)
                    magnetization_lst.append(magnetization)
            elif self.parameter == 'h':
                for h in self.hlst:
                    self._init_model()
                    self.model.H = h
                    energy, magnetization, _ = self.metropolis_sample(
                        T=T0, max_iter=max_iter)
                    energy_lst.append(energy)
                    magnetization_lst.append(magnetization)
        elif self.algorithm == 'wolff':
            if self.parameter == 'T':
                for T in self.Tlst:
                    self._init_model()
                    energy, magnetization, _ = self.wolff_sample(
                        T=T, max_iter=max_iter)
                    energy_lst.append(energy)
                    magnetization_lst.append(magnetization)
            elif self.parameter == 'h':
                for h in self.hlst:
                    self.model.H = h
                    self._init_model()
                    energy, magnetization, _ = self.wolff_sample(
                        T=T0, max_iter=max_iter)
                    energy_lst.append(energy)
                    magnetization_lst.append(magnetization)
        elif self.algorithm == 'anneal':
            if self.parameter == 'T':
                for T in self.Tlst:
                    self._init_model()
                    energy, magnetization, _ = self.simulate_anneal_sample(
                        T=T, max_iter=max_iter)
                    energy_lst.append(energy)
                    magnetization_lst.append(magnetization)
            elif self.parameter == 'h':
                for h in self.hlst:
                    self._init_model()
                    self.model.H = h
                    energy, magnetization, _ = self.simulate_anneal_sample(
                        T=T0, max_iter=max_iter)
                    energy_lst.append(energy)
                    magnetization_lst.append(magnetization)
        else:
            raise ValueError('Invalid algorithm')
        return (p_lst, energy_lst, magnetization_lst)

    # TODO[0.2.1](Added): 添加本征态采样功能
