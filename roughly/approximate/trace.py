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
    def __init__(self, rng : Union[str, callable] = "gaussian"):
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

        self.matvec = (lambda x: A @ x) if isinstance(A, np.ndarray) else A
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
    rng : callable or str
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
    def __init__(self, rng : Union[str, callable] = "gaussian"):
        super().__init__(rng)

    def compute(self, A : Union[np.ndarray, callable], k : int = 10, n : Union[int, None] = None, dtype : Union[type, None] = None):
        """
        Compute the trace estimate.

        Parameters
        ----------
        A : np.ndarray of shape (n, n) or callable
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
        self.trace : float
            Trace estimate.
        """
        self._preprocess(A, k, n, dtype=dtype)
        V = self.rng(self.n, self.k)
        self.trace = np.sum(V.conj() * self.matvec(V)) / self.k
        return self.trace

    def refine(self, k : int = 1):
        """
        Refine the trace estimate.

        Parameters
        ----------
        k : int >= 1
            The number of iterations in the Arnoldi method

        Returns
        -------
        self.trace : float
            Trace estimate.
        """
        V = self.rng(self.n, k)
        self.trace = (self.trace * self.k + np.sum(V.conj() * self.matvec(V))) / (k + self.k)
        self.k += k
        return self.trace

class SubspaceProjectionEstimator(TraceEstimator):
    """
    Implements the subspace projection for trace estimation [2].

    Parameters
    ----------
    rng : callable or str
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
    def __init__(self, rng : Union[str, callable] = "gaussian"):
        self.sketch = StandardSketch(rng)
        super().__init__(rng)

    def compute(self, A : Union[np.ndarray, callable], k : int = 10, n : Union[int, None] = None, dtype : Union[type, None] = None):
        """
        Compute the trace estimate.

        Parameters
        ----------
        A : np.ndarray of shape (n, n) or callable
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
        self.trace : float
            Trace estimate.
        """
        self._preprocess(A, k, n, dtype=dtype)
        self.Q = self.sketch.compute(self.matvec, k=k, n=self.n, dtype=self.dtype)
        self.trace = np.sum(self.Q.conj() * self.matvec(self.Q))
        return self.trace

    def refine(self, k : int = 1):
        """
        Refine the trace estimate.

        Parameters
        ----------
        k : int >= 1
            The number of iterations in the Arnoldi method

        Returns
        -------
        self.trace : float
            Trace estimate.
        """
        self.Q = self.sketch.refine(k=k)
        self.trace = np.sum(self.Q.conj() * self.matvec(self.Q))
        self.k += k
        return self.trace

class DeflatedTraceEstimator(TraceEstimator):
    """
    Implements the Hutch++ algorithm for trace estimation [2].

    Parameters
    ----------
    rng : callable or str
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
    def __init__(self, rng : Union[str, callable] = "gaussian"):
        self.sketch = StandardSketch(rng)
        super().__init__(rng)

    def compute(self, A : Union[np.ndarray, callable], k : int = 10, sketch_ratio : float = 2/3, n : Union[int, None] = None, dtype : Union[type, None] = None):
        """
        Compute the trace estimate.

        Parameters
        ----------
        A : np.ndarray of shape (n, n) or callable
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
        self.trace : float
            Trace estimate.
        """
        self._preprocess(A, k, n, dtype=dtype)
        k_sketch = round(k * sketch_ratio / 2)
        k_correction = k - k_sketch

        # Subspace projection
        self.Q = self.sketch.compute(self.matvec, k=k_sketch, n=self.n, dtype=self.dtype)
        self.S = self.rng(self.n, k_correction)

        # Trace correction
        G = self.S - self.Q @ (self.Q.T @ self.S)
        self.trace = np.sum(self.Q.conj() * self.matvec(self.Q)) + np.sum(G.conj() * self.matvec(G)) / k_correction
        return self.trace

    def refine(self, k : int = 1, sketch_ratio : float = 2/3):
        """
        Refine the trace estimate.

        Parameters
        ----------
        k : int >= 1
            The number of iterations in the Arnoldi method

        Returns
        -------
        self.trace : float
            Trace estimate.
        """
        k_sketch = round(k * sketch_ratio / 2)
        k_correction = k - k_sketch

        # Subspace projection
        self.Q = self.sketch.refine(k=k_sketch)
        self.S = np.hstack((self.S, self.rng(self.n, k_correction)))

        # Trace correction
        G = self.S - self.Q @ (self.Q.T @ self.S)
        self.trace = np.sum(self.Q.conj() * self.matvec(self.Q)) + np.sum(G.conj() * self.matvec(G)) / G.shape[-1]
        self.k += k
        return self.trace

# TODO: class XTrace(TraceEstimator)

# TODO: class KrylovAwareTraceEstimator(TraceEstimator)
