# here put the import lib
import numpy as np
from .Metropolis import Metropolis
from .Metropolis import _sample_acceptance
import copy

__all__ = ["Kawasaki"]


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
            print("we change the magnetization to {}".format(model.magnetization))
        return model

    def iter_sample(self, T: float, uid: str = None, ac_from="class") -> str:
        uid = self._setup_uid(uid)
        _plus = np.argwhere(self.model.spin == 1)
        _minus = np.argwhere(self.model.spin == -1)
        # 从 plus 中随机选取一个
        _site = _plus[np.random.choice(np.arange(len(_plus)))]
        _site2 = _minus[np.random.choice(np.arange(len(_minus)))]
        _temp_model = copy.deepcopy(self.model)
        _delta_E = self.model._change_delta_energy(_site)
        _delta_E += self.model._change_delta_energy(_site2)
        if not _sample_acceptance(_delta_E, T, form=ac_from):
            self.model = _temp_model
        self._save_date(T, uid)
        return uid
