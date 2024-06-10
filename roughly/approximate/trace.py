"""
trace.py
--------

Stochastic trace estimators.

Warning: Due to the non-linearity (with respect to the linear operator) of most
of these estimators, there is little hope for unification.
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Union

from roughly.core.random import gaussian, rademacher, spherical
from roughly.approximate.sketch import StandardSketch

class TraceEstimator(metaclass=ABCMeta):
    def __init__(self, rng : Union[str, function] = "gaussian"):
        if isinstance(rng, str):
            self.rng = eval(rng)
        elif isinstance(rng, function):
            self.rng = rng
        else:
            raise ValueError("'{}' is not a valid random number generation.".format(rng))

    def _preprocess(self, A, k, n=None, dtype=None):
        """
        Preprocess the input matrix/operator and determine problem dimensions.
        """
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
    """
    Implements the Girard-Hutchinson trace estimator for matrix or linear
    operator A [1].

    Parameters
    ----------
    rng : function or str
        Distribution to generator the randomized embedding with. Either a
        function whose arguments are the sizes of the axes of the randomized
        embedding or one of the following strings:
            - "gaussian": Standard Gaussian random numbers
            - "rademacher": Uniformly sample from {-1, +1}
            - "spherical": Spherically normalized standard Gaussians

    Attributes
    ----------
    .compute(A, X, k, ...)
        Compute the Girard-Hutchinson trace estimate.
    .refine(k, ...)
        Refine the Girard-Hutchinson trace estimate.

    Example
    -------
    >>> import numpy as np
    >>> from roughly.approximate.trace import HutchinsonTraceEstimator
    >>> estimator = HutchinsonTraceEstimator()
    >>> A = np.diag(np.ones(100))
    >>> t_A = np.trace(A)
    >>> t = estimator.compute(A, 100)
    >>> assert(abs(t_A - t) / abs(t_A) < 1)
    >>> t = estimator.refine(100)
    >>> assert(abs(t_A - t) / abs(t_A) < 1e-1)

    [1] Girard, A. (1989). "A fast ‘Monte-Carlo cross-validation’ procedure for
    large least squares problems with noisy data". Numerische Mathematik.
    56 (1): 1-23. doi:10.1007/BF01395775.
    """
    def __init__(self, rng : Union[str, function] = "gaussian"):
        super().__init__(rng)

    def compute(self, A : Union[np.ndarray, function], k : int = 10, n : Union[int, None] = None, dtype : Union[type, None] = None):
        """
        Compute the trace estimate.

        Parameters
        ----------
        A : np.ndarray of shape (n, n) or function
            The matrix or linear operator (given as function handle) for which a
            basis of the trace is computed.
        k : int >= 1
            The number of iterations in the Arnoldi method
        n : int >= 1
            The length of each component in the sketch (only needed if A is
            given as a function handle).
        dtype : [np.float, np.complex, ...]
            The expected data type of the Krylov basis. If None, dtype is
            inferred from A.

        Returns
        -------
        self.est : float
            Trace estimate.
        """
        self._preprocess(A, k, n, dtype=dtype)
        V = self.rng(self.n, self.k)
        self.est = np.trace(V.T @ self.matvec(V)) / self.k
        return self.est

    def refine(self, k : int = 1):
        """
        Refine the trace estimate.

        Parameters
        ----------
        k : int >= 1
            The number of iterations in the Arnoldi method

        Returns
        -------
        self.est : float
            Trace estimate.
        """
        V = self.rng(self.n, k)
        self.est = (self.est * self.k + np.trace(V.T @ self.matvec(V))) / (k + self.k)
        self.k += k
        return self.est

class SubspaceProjectionEstimator(TraceEstimator):
    """
    Implements the subspace projection for trace estimation [2].

    Parameters
    ----------
    rng : function or str
        Distribution to generator the randomized embedding with. Either a
        function whose arguments are the sizes of the axes of the randomized
        embedding or one of the following strings:
            - "gaussian": Standard Gaussian random numbers
            - "rademacher": Uniformly sample from {-1, +1}
            - "spherical": Spherically normalized standard Gaussians

    Attributes
    ----------
    .compute(A, k, ...)
        Compute the trace estimate with subspace target rank k.
    .refine(k, ...)
        Refine the trace estimate by increasing the subspace target rank by k.

    Example
    -------
    >>> import numpy as np
    >>> from roughly.approximate.trace import SubspaceProjectionEstimator
    >>> estimator = SubspaceProjectionEstimator()
    >>> A = np.diag(np.ones(100))
    >>> t_A = np.trace(A)
    >>> t = estimator.compute(A, 50)
    >>> assert(abs(t_A - t) / abs(t_A) < 1)
    >>> t = estimator.refine(50)
    >>> assert(abs(t_A - t) / abs(t_A) < 1e-1)

    [2] Meyer, R. A., Musco, C., Musco, C., and Woodruff, D. P. (2021). 
        "Hutch++: Optimal Stochastic Trace Estimation". Symposium on Simplicity
        in Algorithms (SOSA). doi:10.1137/1.9781611976496.16.
    """
    def __init__(self, rng : Union[str, function] = "gaussian"):
        self.sketch = StandardSketch(rng)
        super().__init__(rng)

    def compute(self, A : Union[np.ndarray, function], k : int = 10, n : Union[int, None] = None, dtype : Union[type, None] = None):
        """
        Compute the trace estimate.

        Parameters
        ----------
        A : np.ndarray of shape (n, n) or function
            The matrix or linear operator (given as function handle) for which a
            basis of the trace is computed.
        k : int >= 1
            The number of iterations in the Arnoldi method
        n : int >= 1
            The length of each component in the sketch (only needed if A is
            given as a function handle).
        dtype : [np.float, np.complex, ...]
            The expected data type of the Krylov basis. If None, dtype is
            inferred from A.

        Returns
        -------
        self.est : float
            Trace estimate.
        """
        self._preprocess(A, k, n, dtype=dtype)
        self.Q = self.sketch.compute(A, k=k, n=self.n, dtype=self.dtype)
        self.est = np.trace(self.Q.T @ self.matvec(self.Q))
        return self.est

    def refine(self, k : int = 1):
        """
        Refine the trace estimate.

        Parameters
        ----------
        k : int >= 1
            The number of iterations in the Arnoldi method

        Returns
        -------
        self.est : float
            Trace estimate.
        """
        self.Q = self.sketch.refine(k=k)
        self.est = np.trace(self.Q.T @ self.matvec(self.Q))
        self.k += k
        return self.est

class DeflatedTraceEstimator(TraceEstimator):
    """
    Implements the Hutch++ algorithm for trace estimation [2].

    Parameters
    ----------
    rng : function or str
        Distribution to generator the randomized embedding with. Either a
        function whose arguments are the sizes of the axes of the randomized
        embedding or one of the following strings:
            - "gaussian": Standard Gaussian random numbers
            - "rademacher": Uniformly sample from {-1, +1}
            - "spherical": Spherically normalized standard Gaussians

    Attributes
    ----------
    .compute(A, k, ...)
        Compute the Hutch++ trace estimate with k matrix-vector products.
    .refine(k, ...)
        Refine the Hutch++ trace estimate by adding k matrix-vector products.

    Example
    -------
    >>> import numpy as np
    >>> from roughly.approximate.trace import DeflatedTraceEstimator
    >>> estimator = DeflatedTraceEstimator()
    >>> A = np.diag(np.ones(100))
    >>> t_A = np.trace(A)
    >>> t = estimator.compute(A, 50)
    >>> assert(abs(t_A - t) / abs(t_A) < 1)
    >>> t = estimator.refine(50)
    >>> assert(abs(t_A - t) / abs(t_A) < 1e-1)

    [2] Meyer, R. A., Musco, C., Musco, C., and Woodruff, D. P. (2021). 
        "Hutch++: Optimal Stochastic Trace Estimation". Symposium on Simplicity
        in Algorithms (SOSA). doi:10.1137/1.9781611976496.16.
    """
    def __init__(self, rng : Union[str, function] = "gaussian"):
        self.sketch = StandardSketch(rng)
        super().__init__(rng)

    def compute(self, A : Union[np.ndarray, function], k : int = 10, n : Union[int, None] = None, dtype : Union[type, None] = None):
        """
        Compute the trace estimate.

        Parameters
        ----------
        A : np.ndarray of shape (n, n) or function
            The matrix or linear operator (given as function handle) for which a
            basis of the trace is computed.
        k : int >= 1
            The number of iterations in the Arnoldi method
        n : int >= 1
            The length of each component in the sketch (only needed if A is
            given as a function handle).
        dtype : [np.float, np.complex, ...]
            The expected data type of the Krylov basis. If None, dtype is
            inferred from A.

        Returns
        -------
        self.est : float
            Trace estimate.
        """
        self._preprocess(A, k, n, dtype=dtype)
        self.Q = self.sketch.compute(A, k=k // 3, n=self.n, dtype=self.dtype)
        G = self.rng(self.n, k // 3)
        G = G - self.Q @ (self.Q.T @ G)
        self.est = np.trace(self.Q.T @ self.matvec(self.Q)) + 1 / G.shape[-1] * np.trace(G.T @ self.matvec(G))
        return self.est

    def refine(self, k : int = 1):
        """
        Refine the trace estimate.

        Parameters
        ----------
        k : int >= 1
            The number of iterations in the Arnoldi method

        Returns
        -------
        self.est : float
            Trace estimate.
        """
        self.Q = self.sketch.refine(k=k)
        G = self.rng(self.n, k)
        G = G - self.Q @ (self.Q.T @ G)
        self.est = np.trace(self.Q.T @ self.matvec(self.Q)) + 1/(self.k // 3 + G.shape[-1]) * np.trace(G.T @ self.matvec(G))
        self.k += k
        return self.est

# TODO: class XTrace(TraceEstimator)

# TODO: class KrylovAwareTraceEstimator(TraceEstimator)
