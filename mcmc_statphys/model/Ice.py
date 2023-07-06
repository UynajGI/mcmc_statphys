import numpy as np

import copy
from .Ising import Ising
from typing import Tuple


class Ice(Ising):
    def __init__(self, L):
        super().__init__(L=L, dim=2)
        self.name = "Ice"
        self._init_spin()

    def _init_spin(self, type="ice"):
        # [up, down, left, right]
        # 1 mean to , -1 mean from
        # two 1 two -1
        L = self.L
        self.spin = np.zeros((L, L, 4))
        self.spin[:, :, 0] = -1
        self.spin[:, :, 1] = 1
        self.spin[:, :, 2] = -1
        self.spin[:, :, 3] = 1
        i, j = np.random.randint(0, L, size=2)
        direction = np.random.randint(0, 4)
        for iter in range(self.L**2):
            self.spin[i, j, direction] *= -1
            if direction == 0:
                i = (i - 1) % self.L
                val = copy.deepcopy(self.spin[i, j, 1])
                direction = np.random.choice(np.argwhere(self.spin[i, j] != val).reshape(-1))
                self.spin[i, j, 1] *= -1
            elif direction == 1:
                i = (i + 1) % self.L
                val = copy.deepcopy(self.spin[i, j, 0])
                direction = np.random.choice(np.argwhere(self.spin[i, j] != val).reshape(-1))
                self.spin[i, j, 0] *= -1
            elif direction == 2:
                j = (j - 1) % self.L
                val = copy.deepcopy(self.spin[i, j, 3])
                direction = np.random.choice(np.argwhere(self.spin[i, j] != val).reshape(-1))
                self.spin[i, j, 3] *= -1
            elif direction == 3:
                j = (j + 1) % self.L
                val = copy.deepcopy(self.spin[i, j, 2])
                direction = np.random.choice(np.argwhere(self.spin[i, j] != val).reshape(-1))
                self.spin[i, j, 2] *= -1

        self.type = type

    def _change_delta_energy(self, index: Tuple[int, ...]):
        path = []
        i, j = index
        direction = np.random.randint(0, 4)
        while (i, j) not in path:
            path.append((i, j))
            self.spin[i, j, direction] *= -1
            if direction == 0:
                i = (i - 1) % self.L
                val = copy.deepcopy(self.spin[i, j, 1])
                direction = np.random.choice(np.argwhere(self.spin[i, j] != val).reshape(-1))
                self.spin[i, j, 1] *= -1
            elif direction == 1:
                i = (i + 1) % self.L
                val = copy.deepcopy(self.spin[i, j, 0])
                direction = np.random.choice(np.argwhere(self.spin[i, j] != val).reshape(-1))
                self.spin[i, j, 0] *= -1
            elif direction == 2:
                j = (j - 1) % self.L
                val = copy.deepcopy(self.spin[i, j, 3])
                direction = np.random.choice(np.argwhere(self.spin[i, j] != val).reshape(-1))
                self.spin[i, j, 3] *= -1
            elif direction == 3:
                j = (j + 1) % self.L
                val = copy.deepcopy(self.spin[i, j, 2])
                direction = np.random.choice(np.argwhere(self.spin[i, j] != val).reshape(-1))
                self.spin[i, j, 2] *= -1
        indlen = path.index((i, j))
        for k in range(indlen):
            ind = indlen - k
            iup, jup = path[ind]
            idown, jdown = path[ind - 1]
            if idown == iup:
                if (jdown - jup) % self.L == (self.L - 1):
                    self.spin[iup, jup, 2] *= -1
                    self.spin[idown, jdown, 3] *= -1
                else:
                    self.spin[iup, jup, 3] *= -1
                    self.spin[idown, jdown, 2] *= -1
            else:
                if (idown - iup) % self.L == (self.L - 1):
                    self.spin[iup, jup, 0] *= -1
                    self.spin[idown, jdown, 1] *= -1
                else:
                    self.spin[iup, jup, 1] *= -1
                    self.spin[idown, jdown, 0] *= -1
        detle = len(path)
        # path = path[indlen:]
        return detle
