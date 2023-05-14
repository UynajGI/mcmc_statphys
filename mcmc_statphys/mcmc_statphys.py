# -*- encoding: utf-8 -*-
'''
@File    :   MCMC.py
@Time    :   2023/05/13 21:07:49
@Author  :   UynajGI
@Version :   beta0.0.1
@Contact :   betterWL@hotmail.com
@License :   (CC-BY-4.0)Copyright 2023
'''

import copy
from collections import deque
from typing import List, Tuple

# here put the import lib
import numpy as np


class Simulation():

    def __init__(self, model: object):
        self.model = model

    def sample_acceptance(self, delta_E: float,
                          sample_Temperture: float) -> bool:
        if delta_E <= 0:
            return True
        else:
            return np.random.rand() < np.exp(-delta_E / sample_Temperture)

    def iter_sample(self, sample_Temperture: float) -> object:
        site = tuple(
            np.random.randint(0, self.model.L, size=self.model.dimension))
        temp_model = copy.deepcopy(self.model)
        delta_E = self.model._change_delta_energy(site)
        if not self.sample_acceptance(delta_E, sample_Temperture):
            self.model = temp_model
        return self.model

    def iter_wolff_sample(self, T):
        # wolff 算法
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

    def is_Flat(self, arrry: np.ndarray, epsilon: float = 0.1) -> bool:
        return np.std(arrry) / np.mean(arrry) < epsilon

    def metropolis_sample(
            self, T: float,
            max_iter: int) -> Tuple[List[int], List[float], List[float]]:
        energys: List[float] = []
        magnetizations: List[float] = []
        for iter in range(max_iter):
            self.iter_sample(T)
            _, energy, magnetization = self.model._get_per_info()
            energys.append(energy)
            magnetizations.append(magnetization)
            if iter % 1000 == 0:
                pass  # TODO(log): add log
            # 平衡判据，用变异系数
            if self.is_Flat(energys) and iter > 5000:
                break
        return (energys, magnetizations, self.model)

    def simulate_anneal_sample(
            self,
            T_min: float,
            Tdceny: float = 0.9,
            T: float = 10,
            max_iter: int = 5000
    ) -> Tuple[List[int], List[float], List[float]]:
        energys_anneal: List[float] = []
        magnetizations_anneal: List[float] = []
        # 温度不够高，升温
        while T < T_min:
            T *= 2
        # 开始模拟退火
        while T > T_min:
            energys, magnetizations, _ = self.metropolis_sample(T, max_iter)
            energys_anneal.extend(energys)
            magnetizations_anneal.extend(magnetizations)
            T = max(Tdceny * T, T_min)
            print(T)
        return (energys_anneal, magnetizations_anneal, self.model)

    def wolff_sample(self, T, max_iter):
        energys_wolff: List[float] = []
        magnetizations_wolff: List[float] = []
        for iter in range(max_iter):
            self.iter_wolff_sample(T)
            if iter % 1000 == 0:
                pass  # TODO(log): add log
            _, energy, magnetization = self.model._get_per_info()
            energys_wolff.append(energy)
            magnetizations_wolff.append(magnetization)
        return (energys_wolff, magnetizations_wolff, self.model)


class ParameterSample(Simulation):

    def __init__(self, model: object, *args, **kwargs):
        super().__init__(model)
        self.rowmodel = copy.deepcopy(model)
        self._init_parameter()
        self._init_paramlst()

    def _init_parameter(self):
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
        if self.parameter == 'T':
            # T_lst 不存在，input 初始化
            if not hasattr(self, 'Tlst'):
                Tmin = input('Input T_min: ')
                while Tmin == '':
                    Tmin = input('Input T_min: ')
                Tmax = input('Input T_max: ')
                while Tmax == '':
                    Tmax = input('Input T_max: ')
                num = input('Input sample num: ')
                while num == '':
                    num = input('Input sample num: ')
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
        self.Tlst = np.linspace(T_min, T_max, num=num)

    def _init_hlst(self, h_min: float, h_max: float, num: int):
        self.hlst = np.linspace(h_min, h_max, num=num)

    def _init_model(self):
        self.model = copy.deepcopy(self.rowmodel)

    def sample(self, max_iter: int = 10000):
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
