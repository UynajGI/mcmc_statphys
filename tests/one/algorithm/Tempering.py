from typing import Dict
import copy

# here put the import lib
from tqdm import tqdm
import numpy as np
import uuid
import pandas as pd
from .Metropolis import Metropolis

__all__ = ["Tempering"]


class Tempering(Metropolis):
    def __init__(self, model: object):
        super().__init__(model)
        self.name = "Tempering"

    def iter_sample(self, T: float, uid: str = None, ac_from="class") -> str:
        uid = self._setup_uid(uid)
        super().iter_sample(T, uid, ac_from=ac_from)
        return uid

    def equil_sample(self, T: float, max_iter: int = 1000, uid: str = None, ac_from="class") -> str:
        uid = self._setup_uid(uid)
        for iter in tqdm(range(max_iter), leave=False):
            self.iter_sample(T, uid, ac_from=ac_from)
        return uid

    def param_sample(
        self, T: tuple, H0: float = 0.0, max_iter: int = 1000, eq_iter: int = 1000, ac_from: str = "class"
    ) -> Dict:
        self.model.H = H0
        Tmax, Tmin, Tlen = T
        T_lst = np.linspace(Tmax, Tmin, Tlen)
        algo_lst = [Metropolis(copy.deepcopy(self.model)) for T in T_lst]
        uid_lst = [uuid.uuid1().hex for T in T_lst]
        iter = 0
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
