"""
trace.py
--------

Stochastic trace estimators.

Warning: Due to the non-linearity (with respect to the linear operator) of most
of these estimators, there is little hope for unification.
"""

import numpy as np
from abc import ABCMeta, abstractmethod

from roughly.core.random import gaussian, rademacher, spherical
from roughly.approximate.sketch import StandardSketch

class TraceEstimator(metaclass=ABCMeta):
    def __init__(self, rng="gaussian"):
        if isinstance(rng, str):
            self.rng = eval(rng)
        elif isinstance(rng, function):
            self.rng = rng
        else:
            raise ValueError("'{}' is not a valid random number generation.".format(rng))

    def preprocess(self, A, k, n=None, dtype=None):
        self.k = k
        self.n = A.shape[0] if n is None else n

        self.matvec = lambda x: A @ x if isinstance(A, np.ndarray) else A
        self.dtype = A.dtype if dtype is None else dtype

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def refine(self):
        pass

class HutchinsonTraceEstimator(TraceEstimator):
    def __init__(self, rng="gaussian"):
        super().__init__(rng)

    def compute(self, A, k=10, n=None, dtype=None):
        self.preprocess(A, k, n, dtype=dtype)
        V = self.rng(self.n, self.k)
        self.est = np.trace(V.T @ self.matvec(V)) / self.k
        return self.est

    def refine(self, k=1):
        V = self.rng(self.n, k)
        self.est = (self.est * self.k + np.trace(V.T @ self.matvec(V))) / (k + self.k)
        self.k += k
        return self.est

class SubspaceProjectionEstimator(TraceEstimator):
    def __init__(self, rng="gaussian"):
        self.sketch = StandardSketch(rng)
        super().__init__(rng)

    def compute(self, A, k=10, n=None, dtype=None):
        self.preprocess(A, k, n, dtype=dtype)
        self.Q = self.sketch.compute(A, k=k, n=self.n, dtype=self.dtype)
        self.est = np.trace(self.Q.T @ self.matvec(self.Q))
        return self.est

    def refine(self, k=1):
        self.Q = self.sketch.refine(k=k)
        self.est = np.trace(self.Q.T @ self.matvec(self.Q))
        self.k += k
        return self.est

class DeflatedTraceEstimator(TraceEstimator):
    def __init__(self, rng="gaussian"):
        self.sketch = StandardSketch(rng)
        super().__init__(rng)

    def compute(self, A, k=10, n=None, dtype=None):
        self.preprocess(A, k, n, dtype=dtype)
        self.Q = self.sketch.compute(A, k=k // 3, n=self.n, dtype=self.dtype)
        G = self.rng(self.n, k // 3)
        G = G - self.Q @ (self.Q.T @ G)
        self.est = np.trace(self.Q.T @ self.matvec(self.Q)) + 1 / G.shape[-1] * np.trace(G.T @ self.matvec(G))
        return self.est

    def refine(self, k=1):
        self.Q = self.sketch.refine(k=k)
        G = self.rng(self.n, k)
        G = G - self.Q @ (self.Q.T @ G)
        self.est = np.trace(self.Q.T @ self.matvec(self.Q)) + 1/(self.k // 3 + G.shape[-1]) * np.trace(G.T @ self.matvec(G))
        self.k += k
        return self.est

# TODO: class XTrace(TraceEstimator)

# TODO: class KrylovAwareTraceEstimator(TraceEstimator)
