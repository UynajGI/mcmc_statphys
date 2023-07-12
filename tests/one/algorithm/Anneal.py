from typing import Dict
import copy

# here put the import lib
from tqdm import tqdm
from .Metropolis import Metropolis
from .Metropolis import _rename

__all__ = ["Anneal"]


class Anneal(Metropolis):
    def __init__(self, model: object):
        super().__init__(model)
        self.name = "Anneal"

    def iter_sample(self, T: float, uid: str = None, ac_from="class") -> object:
        """_summary_

        Args:
            T (float): _description_
            uid (str, optional): _description_. Defaults to None.
        """
        uid = self._setup_uid(uid)
        super().iter_sample(T, uid, ac_from=ac_from)
        return uid

    def equil_sample(
        self,
        targetT: float,
        max_iter: int = 1000,
        highT=None,
        dencyT=0.9,
        uid: str = None,
        ac_from="class",
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
            super().equil_sample(T, max_iter=max_iter, uid=uid, ac_from=ac_from)
            T = max(T * dencyT, targetT)
        return uid

    def param_sample(
        self,
        param: tuple,
        param_name: str or int = "T",
        stable: float = 0.0,
        max_iter: int = 1000,
        ac_from: str = "class",
    ):
        """_summary_

        Args:
            max_iter (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        self.parameter = _rename(param_name)
        param_lst = super()._init_paramlst(param)
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
