import numpy as np
import copy
from tqdm import tqdm

__all__ = ["WangLandau"]


class WangLandau:
    """
    WangLandau
    ==========
    WangLandau is the Wang-Landau algorithm
    """

    def __init__(self, model, overlap: float = 0.06):
        self.model = model
        self.name = "WangLandau"
        self.elst = []
        self.logG = []
        self.hist = []
        self.logF = 1
        self.overlap = overlap * self.model.N

    def _flat(self, array, epsilon=0.8):
        nparray = np.array(array)
        return min(nparray[nparray > 0]) > epsilon * np.mean(nparray[nparray > 0])

    def sample(self, epsilon=1e-8):
        count = 0
        total = int(np.log(epsilon) / np.log(0.5)) + 1
        with tqdm(total=total) as pbar:
            while self.logF > epsilon:
                if not self.elst:
                    self.elst.append(self.model.energy)
                    self.logG.append(1)
                    self.hist.append(1)
                else:
                    # site = tuple(np.random.randint(0, self.model.L, size=self.model.dim))
                    temp_model = copy.deepcopy(self.model)
                    self.model._random_walk()
                    index_old = np.argmin(np.abs(self.elst - temp_model.energy))
                    index_new = np.argmin(np.abs(self.elst - self.model.energy))
                    # self.elst[index_new] 是离当前能量太远
                    if not np.log(np.random.rand()) < self.logG[index_old] - self.logG[index_new]:
                        self.model = temp_model
                        index = index_old
                    else:
                        index = index_new

                    if abs(self.elst[index] - self.model.energy) > self.overlap:
                        if self.elst[index] - self.model.energy > 0:
                            self.elst = self.elst[:index] + [self.model.energy] + self.elst[index:]
                            self.logG = self.logG[:index] + [self.logF] + self.logG[index:]
                            self.hist = self.hist[:index] + [1] + self.hist[index:]
                        else:
                            self.elst = self.elst[: index + 1] + [self.model.energy] + self.elst[index + 1 :]
                            self.logG = self.logG[: index + 1] + [self.logF] + self.logG[index + 1 :]
                            self.hist = self.hist[: index + 1] + [1] + self.hist[index + 1 :]
                    else:
                        self.hist[index] += 1
                        self.logG[index] += self.logF
                count += 1
                if count % 1000 == 0:
                    narray = np.array(self.hist)
                    pbar.set_description(
                        "Now count: {c}e3; The logF is {f}; Min_h is {m}; 80% mean is {e}".format(
                            c=count // 1000,
                            f=self.logF,
                            m=min(narray[narray > 0]),
                            e=np.round(0.8 * np.mean(narray[narray > 0]), 3),
                        )
                    )
                if self._flat(self.hist) and count > 1e4:
                    self.hist = np.zeros(len(self.hist)).tolist()
                    self.logF /= 2
                    pbar.update(1)
                    count = 0
                    self.logG = np.array(self.logG)
                    self.logG -= min(self.logG)
                    self.logG = self.logG.tolist()
        return self.logG

    def logZ(self, T):
        elst = np.array(self.elst)
        logG = np.array(self.logG)
        logz = 0
        for i in range(len(elst)):
            logz += logz + np.log1p(np.exp(logG[i] - 1 / T * elst[i] - logz))

    def entropy_diff(self, T, epsilon=1e-8):
        s = self.energy_diff(T, epsilon=epsilon) / T + self.logZ(T)
        return s

    def energy_diff(self, T, epsilon=1e-8):
        beta = 1 / T

        e = -(self.logZ(1 / (beta + epsilon)) - self.logZ(1 / (beta - epsilon))) / (2 * epsilon)
        return e

    def heat_diff(self, T, epsilon=1e-8):
        beta = 1 / T
        e = self.energy_diff
        c = -(beta**2) * (e(1 / (beta + epsilon)) - e(1 / (beta - epsilon))) / (2 * epsilon)
        return c

    def energy(self, T):
        elst = np.array(self.elst)
        logG = np.array(self.logG)
        return np.dot(elst, np.exp(logG - 1 / T * elst)) / np.sum(np.exp(logG - 1 / T * elst))

    def heat(self, T):
        elst = np.array(self.elst)
        logG = np.array(self.logG)
        return (
            (np.dot(elst**2, np.exp(logG - 1 / T * elst))) / np.sum(np.exp(logG - 1 / T * elst)) - self.energy(T) ** 2
        ) / T**2
