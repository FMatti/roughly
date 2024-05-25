# RSVD
# Nyström
# fun Nyström

import numpy as np
from abc import ABCMeta, abstractmethod
from roughly.core.rng import gaussian, rademacher, spherical
from .sketch import StandardSketch

class LowRankApproximator(metaclass=ABCMeta):
    def __init__(self, sketch=StandardSketch):
        self.sketch = sketch()

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def refine(self):
        pass

class RandomizedSVD(LowRankApproximator):
    def __init__(self, sketch=StandardSketch):
        super().__init__(sketch)

    def compute(self, A, m=100):
        self.A = A
        self.m = m
        self.Q = self.sketch.compute(A, m=m)
        U, self.S, self.V_H = np.linalg.svd(self.Q.T.conj() @ A, full_matrices=False)
        self.U = self.Q @ U
        return self.U, self.S, self.V_H

    def refine(self, m=10):
        self.Q = self.sketch.refine(m=m)
        U, self.S, self.V_H = np.linalg.svd(self.Q.T.conj() @ self.A, full_matrices=False)
        self.U = self.Q @ U
        self.m += m
        return self.U, self.S, self.V_H
