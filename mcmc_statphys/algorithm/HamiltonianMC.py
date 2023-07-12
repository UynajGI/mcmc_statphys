import copy
import uuid
import pandas as pd
import numpy as np
from tqdm import tqdm
from .Metropolis import _sample_acceptance

__all__ = ["HamiltonianMC"]


class HamiltonianMC:
    def __init__(self, model, positive_C=1.05, learning_rate=0.01):
        if model.type != "SK":
            raise ValueError("The model must be SKmodel")
        self.model = model
        self.learning_rate = learning_rate
        self.positive_C = positive_C
        self.name = "HamiltonianMC"
        self._init_positive()
        self._init_p()
        self._init_q()
        self._init_data()

    def _init_q(self):
        self.q = np.random.randn(self.model.N)

    def _init_p(self):
        self.p = np.random.randn(self.model.N)

    def _setup_uid(self, uid):
        if uid is None:
            uid = (uuid.uuid1()).hex
        else:
            if uid not in self.data.index.get_level_values("uid").values:
                self._init_q()
                self._init_p()
            else:
                self.q = copy.deepcopy(self.data.loc[uid].loc[self.data.loc[uid].index.max()]["q"])
                self.p = copy.deepcopy(self.data.loc[uid].loc[self.data.loc[uid].index.max()]["p"])
        return uid

    def _init_positive(self):
        if self.model.type == "SK":
            eigval, _ = np.linalg.eig(self.model.Jij)
            self.Jij_positive = self.model.Jij + (np.abs(np.min(eigval)) + self.positive_C) * np.eye(
                self.model.Jij.shape[0]
            )

    def _init_data(self):
        self.data: pd.DataFrame = pd.DataFrame(columns=["uid", "iter", "T", "H", "q", "p"])
        self.data.set_index(["uid", "iter"], inplace=True)

    def _save_date(self, T, uid):
        if uid not in self.data.index.get_level_values("uid").values:
            self.data.loc[(uid, 1), :] = [T, self.model.H, 0, 0]
            self.data.at[(uid, 1), "q"] = copy.deepcopy(self.q)
            self.data.at[(uid, 1), "p"] = copy.deepcopy(self.p)
        else:
            iterplus = self.data.loc[uid].index.max() + 1
            self.data.loc[(uid, iterplus), :] = [T, self.model.H, 0, 0]
            self.data.at[(uid, iterplus), "q"] = copy.deepcopy(self.q)
            self.data.at[(uid, iterplus), "p"] = copy.deepcopy(self.p)

    def _hamiltonian(self, T, J=1):
        ham = (
            1 / 2 * np.sum(self.p**2)
            + 1 / 2 * np.dot(self.q, np.dot(self.Jij_positive, self.q))
            - 1 / np.sqrt(J / T) * self.model.H * np.sum(self.q)
            - np.sum(np.log(np.cosh(np.sqrt(J / T) * np.dot(self.Jij_positive, self.q))))
        )
        return ham

    def _grid_p(self, T, J=1):
        return (
            -np.dot(self.Jij_positive, self.q)
            + 1 / np.sqrt(J / T) * self.model.H / T * np.ones(self.q.shape[0])
            + np.sqrt(J / T) * np.tanh(np.sqrt(J / T) * np.dot(self.Jij_positive, self.q))
        )

    def _leapfrog(self, T):
        self.p -= self.learning_rate / 2 * self._grid_p(T)
        self.q += self.learning_rate * self.p
        self.p -= self.learning_rate / 2 * self._grid_p(T)

    def iter_sample(self, T: float, uid: str = None, ac_from="class") -> str:
        uid = self._setup_uid(uid)
        # 保存上一次的状态
        q_old = copy.deepcopy(self.q)
        p_old = copy.deepcopy(self.p)
        hamiltonian_old = self._hamiltonian(T)
        # 采样
        self._leapfrog(T)
        # 计算能量差
        delta_E = self._hamiltonian(T) - hamiltonian_old
        # 判断是否接受
        if not _sample_acceptance(delta_E, T, form=ac_from):
            self.q = q_old
            self.p = p_old
        self._save_date(T, uid)
        return uid

    def equil_sample(self, T: float, max_iter: int = 1000, uid: str = None, ac_from="class") -> str:
        uid = self._setup_uid(uid)
        for iter in tqdm(range(max_iter), leave=False):
            self.iter_sample(T, uid, ac_from=ac_from)
        return uid

    def get_energy(self, uid: str, t0: int = 0) -> float:
        pass
