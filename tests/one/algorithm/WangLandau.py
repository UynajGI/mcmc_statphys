import numpy as np
import copy
from tqdm import tqdm


class WangLandau():

    def __init__(self, model, glen=10):
        self.model = model
        self.name = "WangLandau"
        self.logG = np.log(np.ones(glen))
        self.hist = np.zeros(glen)
        self.elst = np.linspace(0, self.model.maxenergy, glen)
        self.logF = 1

    def _flat(self, array, epsilon=0.8):
        return min(array[array > 0]) > epsilon * np.mean(array[array > 0])

    def sample(self, epsilon=1e-8):
        count = 0
        total = int(np.log(epsilon) / np.log(0.5))
        with tqdm(total=total) as pbar:
            while self.logF > epsilon:
                site = tuple(
                    np.random.randint(0, self.model.L, size=self.model.dim))
                temp_model = copy.deepcopy(self.model)
                self.model._change_delta_energy(site)
                index_old = np.argmin(np.abs(self.elst - temp_model.energy))
                index_new = np.argmin(np.abs(self.elst - self.model.energy))
                if not np.log(np.random.rand()
                              ) < self.logG[index_old] - self.logG[index_new]:
                    self.model = temp_model
                    index = index_old
                else:
                    index = index_new
                self.hist[index] += 1
                self.logG[index] += self.logF
                count += 1
                if count % 1000 == 0:
                    pbar.set_description(
                        "Now count: {c}e3; The logF is {f}; Min_h is {m}; 80% mean is {e}"
                        .format(c=count // 1000,
                                f=self.logF,
                                m=min(self.hist[self.hist > 0]),
                                e=np.round(
                                    0.8 * np.mean(self.hist[self.hist > 0]),
                                    3)))
                if self._flat(self.hist) and count > 1e4:
                    self.hist = np.zeros(len(self.hist))
                    self.logF /= 2
                    pbar.update(1)
                    count = 0
                    self.logG -= min(self.logG)
        return self.logG

    def _log_partition(self, T):
        logZ = 0
        beta_lst = self.elst * (1 / T)
        z = self.logG - beta_lst
        logZ = np.log(np.sum(np.exp(z)))
        return logZ

    def energy(self, T, epsilon=1e-8):
        U = -T**2 * (self._log_partition(T + epsilon) -
                     self._log_partition(T - epsilon)) / (2 * epsilon)
        return U

    def heatcap(self, T, epsilon=1e-8):
        C = (self.energy(T + epsilon, epsilon=epsilon) -
             self.energy(T - epsilon, epsilon=epsilon)) / (2 * epsilon)
        return C
