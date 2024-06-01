# quadratic
# Trace++ (Hutch++, Nyström++)
# SLQ
# Krylov-Aware

import numpy as np
from abc import ABCMeta, abstractmethod
from roughly.core.rng import gaussian, rademacher, spherical
from .sketch import StandardSketch


class TraceEstimator(metaclass=ABCMeta):
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

class HutchinsonTraceEstimator(TraceEstimator):
    def __init__(self, rng="gaussian"):
        super().__init__(rng)

    def compute(self, A, m=100):
        self.A = A
        self.m = m
        V = self.rng(self.A.shape[0], m)
        self.est = np.sum(self.A*(V @ V.T)) / m
        return self.est

    def refine(self, m=10):
        V = self.rng(self.A.shape[0], m)
        self.est = (self.est * self.m + np.sum(self.A*(V @ V.T))) / (m + self.m)
        self.m += m
        return self.est

class SubspaceProjectionEstimator(TraceEstimator):
    def __init__(self, rng="gaussian"):
        super().__init__(rng)
        self.sketch = StandardSketch(rng)

    def compute(self, A, m=100):
        self.A = A
        self.m = m
        self.Q = self.sketch.compute(A, m=m // 3)
        self.est = np.trace(self.Q.T @ A @ self.Q)
        return self.est

    def refine(self, m=10):
        self.Q = self.sketch.refine(m=m)
        self.est = np.trace(self.Q.T @ self.A @ self.Q)
        self.m += m
        return self.est

# Should be Low-Rank + Hutchinson correction, right?
class DeflatedTraceEstimator(TraceEstimator):
    def __init__(self, rng="gaussian"):
        super().__init__(rng)
        self.sketch = StandardSketch(rng)

    def compute(self, A, m=100):
        self.A = A
        self.m = m
        self.Q = self.sketch.compute(A, m=m // 3)
        G = self.rng(A.shape[-1], m // 3)
        G = G - self.Q @ (self.Q.T @ G)
        self.est = np.trace(self.Q.T @ A @ self.Q) + 1/G.shape[-1] * np.trace(G.T @ A @ G)
        return self.est

    def refine(self, m=10):
        self.Q = self.sketch.refine(m=m)
        G = self.rng(self.A.shape[-1], m)
        G = G - self.Q @ (self.Q.T @ G)
        self.est = np.trace(self.Q.T @ self.A @ self.Q) + 1/(self.m // 3 + G.shape[-1]) * np.trace(G.T @ self.A @ G)
        self.m += m
        return self.est


#def Input function handles not matrices (Xtrace, Quadratic, Low-rank, low-rank++)
