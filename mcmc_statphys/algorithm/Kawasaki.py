# here put the import lib
import numpy as np
from .Metropolis import Metropolis
from .Metropolis import _sample_acceptance
import copy

__all__ = ['Kawasaki']


class Kawasaki(Metropolis):

    def __init__(self, model: object, M: int):
        if abs(M) > model.N:
            raise ValueError("M must be less than N")
        self.M = M
        model = self._init_model(model)
        super().__init__(model)
        self.name = "Kawasaki"

    def _init_model(self, model):
        plus = int((model.N + self.M) / 2)
        # spin 变为一维
        shape = model.spin.shape
        model.spin = model.spin.reshape(model.N)
        model.spin[:plus] = 1
        model.spin[plus:] = -1
        np.random.shuffle(model.spin)
        # spin 变回原来的形状
        model.spin = model.spin.reshape(shape)
        model._get_total_energy()
        model._get_total_magnetization()
        if model.magnetization != self.M:
            print("we change the magnetization to {}".format(
                model.magnetization))
        return model

    def iter_sample(self, T: float, uid: str = None, ac_from='class') -> str:
        uid = self._setup_uid(uid)
        site = tuple(np.random.randint(0, self.model.L, size=self.model.dim))
        site2 = tuple(np.random.randint(0, self.model.L, size=self.model.dim))
        # 若两个 site 的 spin 相同，则重新选取
        while self.model.spin[site] == self.model.spin[site2]:
            site2 = tuple(
                np.random.randint(0, self.model.L, size=self.model.dim))
        temp_model = copy.deepcopy(self.model)
        delta_E = self.model._change_delta_energy(site)
        delta_E += self.model._change_delta_energy(site2)
        if not _sample_acceptance(delta_E, T, form=ac_from):
            self.model = temp_model
        self._save_date(T, uid)
        return uid
