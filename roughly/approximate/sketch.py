# Embedding
# SRFT
# 

import numpy as np
from abc import ABCMeta, abstractmethod
from roughly.core.rng import gaussian, rademacher, spherical

class RangeSketch(metaclass=ABCMeta):
    def __init__(self, rng="gaussian"):
        if isinstance(rng, str):
            self.rng = eval(rng)
        elif isinstance(rng, function):
            self.rng = rng
        else:
            raise ValueError("'{}' is not a valid random number generation.".format(rng))

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def refine(self):
        pass

class StandardSketch(RangeSketch):
    def __init__(self, rng="gaussian"):
        super().__init__(rng)

    def compute(self, A, m=100):
        self.A = A
        self.m = m
        S = self.rng(A.shape[-1], m)
        self.Q, _ = np.linalg.qr(A @ S)
        return self.Q

    def refine(self, m=10):
        S = self.rng(self.A.shape[0], m)
        self.Q, _ = np.linalg.qr(np.hstack((self.Q, self.A @ S)))
        self.m += m
        return self.Q
