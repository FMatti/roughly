# Embedding
# SRFT
# 

import numpy as np
from abc import ABCMeta, abstractmethod
from roughly.core.rng import gaussian, rademacher, spherical

class RangeSketch(metaclass=ABCMeta):
    def __init__(self, rng="gaussian", orthogonal=True):
        self.orthogonal = orthogonal
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
    def __init__(self, rng="gaussian", orthogonal=True):
        super().__init__(rng, orthogonal)

    def compute(self, A, m=100, return_embedding=False):
        self.A = A
        self.m = m
        S = self.rng(A.shape[-1], m)
        self.Q = A @ S
        if self.orthogonal:
            self.Q, _ = np.linalg.qr(self.Q)
        if return_embedding:
            return self.Q, S
        return self.Q

    def refine(self, m=10, return_embedding=False):
        S = self.rng(self.A.shape[0], m)
        self.Q = np.hstack((self.Q, self.A @ S))
        if self.orthogonal:
            self.Q, _ = np.linalg.qr(self.Q)
        self.m += m
        if return_embedding:
            raise NotImplementedError
        return self.Q


