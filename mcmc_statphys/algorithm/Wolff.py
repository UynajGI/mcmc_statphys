from collections import deque
from typing import Dict

# here put the import lib
import numpy as np
from tqdm import tqdm
from .Metropolis import Metropolis
from .Metropolis import _rename

__all__ = ['Wolff']


class Wolff(Metropolis):
    """The Wolff algorithm, named after Ulli Wolff, is an algorithm for Monte Carlo simulation\n
    of the Ising model and Potts model.\n
    Details please see: https://en.wikipedia.org/wiki/Wolff_algorithm \n
    cn: Wolff 算法，以 Ulli Wolff 命名，是一种蒙特卡洛模拟算法，用于 Ising 模型和 Potts 模型。\n
    详情请见：https://en.wikipedia.org/wiki/Wolff_algorithm
    """

    def __init__(self, model: object):
        if model.type != "ising" and model.type != "rfising":
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
        for iter in tqdm(range(max_iter), leave=False):
            self.iter_sample(T, uid)
        return uid

    def param_sample(self,
                     param: tuple,
                     param_name: str or int = 'T',
                     stable: float = 0.0,
                     max_iter: int = 1000):
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
                self.equil_sample(param, max_iter=max_iter, uid=uid)
            elif self.parameter == "H":
                self.model.H = param
                self.equil_sample(stable, max_iter=max_iter, uid=uid)
        uid_param_dict: Dict = {
            "uid": uid_lst,
            "{param}".format(param=self.parameter): param_lst,
        }
        self.param_list.append(uid_param_dict)
        return uid_param_dict
