"""
lowrank.py
----------

Methods for randomized range sketching of a linear operator.
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from roughly.approximate.sketch import RangeSketch, StandardSketch

class LowRankApproximator(metaclass=ABCMeta):
    def __init__(self, sketch : RangeSketch = StandardSketch()):
        self.sketch = sketch

    def _preprocess(self, A, k):
        """
        Preprocess the input matrix/operator and determine problem dimensions.
        """
        self.k = k
        self.A = A

    @abstractmethod
    def _factorize(self):
        pass

    def compute(self, A : np.ndarray, k : int = 10):
        """
        Compute the low-rank factorization.

        Parameters
        ----------
        A : np.ndarray of shape (n, m) or function
            The matrix for which a low-rank factorization is computed.
        k : int >= 1
            The target rank.

        Returns
        -------
        U : np.ndarray of shape (n, k)
            The left factor of the decomposition.
        S : np.ndarray of shape (k, k)
            The core factor of the decomposition.
        Vh : np.ndarray of shape (k, m)
            The right factor of the decomposition.
        """
        self._preprocess(A, k)
        self.Q, self.G = self.sketch.compute(self.A, k=self.k, return_embedding=True)
        U, S, Vh = self._factorize()
        return U, S, Vh

    def refine(self, k : int = 1):
        """
        Refine the low-rank factorization to a rank higher by k.

        Parameters
        ----------
        k : int >= 1
            The amount by which the target rank is increased.

        Returns
        -------
        U : np.ndarray of shape (n, ... + k)
            The left factor of the decomposition.
        S : np.ndarray of shape (... + k, ... + k)
            The core factor of the decomposition.
        Vh : np.ndarray of shape (... + k, m)
            The right factor of the decomposition.
        """
        self.Q, self.G = self.sketch.refine(k=k, return_embedding=True)
        U, S, Vh = self._factorize()
        self.k += k
        return U, S, Vh

class RandomizedSVD(LowRankApproximator):
    """
    Randomized singular value decomposition (SVD) of a matrix [1]. Computes a
    factorization of the form

        A ~ U S Vh

    where U and Vh are orthogonal matrices and S is diagonal.

    Parameters
    ----------
    sketch : RangeSketch
        The sketching method used to approximate the range of the matrix.
    return_matrix : bool
        Whether to return the singular values as a diagonal matrix or a vector.

    Attributes
    ----------
    .compute(A, k, ...)
        Computes the rank-k randomized SVD of the matrix A.
    .refine(k, ...)
        Refines the randomized SVD to a rank increased by k.

    Example
    -------
    >>> import numpy as np
    >>> from roughly.approximate.lowrank import RandomizedSVD
    >>> svd = RandomizedSVD()
    >>> L = np.arange(10)
    >>> X = np.random.randn(100, 10)
    >>> A = X @ np.diag(L) @ np.linalg.pinv(X)
    >>> U, S, Vh = svd.compute(A, 10)
    >>> U, S, Vh = svd.refine(1)
    >>> assert(np.linalg.norm(A - U @ np.diag(S) @ Vh) < 1e-10)

    [1] Halko, N., Martinsson, P. G., and Tropp J. A. (2011). "Finding
    Structure with Randomness: Probabilistic Algorithms for Constructing Appro-
    ximate Matrix Decompositions". SIAM Review. 53 (2) doi:10.1137/090771806.
    """
    def __init__(self, sketch=StandardSketch(), return_matrix=True):
        self.return_matrix = return_matrix
        super().__init__(sketch)

    def _factorize(self):
        U, S, Vh = np.linalg.svd(self.Q.T.conj() @ self.A, full_matrices=False)
        U = self.Q @ U
        if self.return_matrix:
            S = np.diag(S)
        return U, S, Vh

class NystromApproximation(LowRankApproximator):
    """
    Nyström approximation of a Hermitian matrix [2], which is a factorization
    of the form

        A ~ U @ S @ Vh

    where U = A S, S = pinv(G.T @ A @ G) for a sketch embedding G, and Vh = U*.

    Parameters
    ----------
    sketch : RangeSketch
        The sketching method used to approximate the range of the matrix.

    Attributes
    ----------
    .compute(A, k, ...)
        Computes the rank-k randomized SVD of the matrix A.
    .refine(k, ...)
        Refines the randomized SVD to a rank increased by k.

    Example
    -------
    >>> import numpy as np
    >>> from roughly.approximate.lowrank import NystromApproximation
    >>> svd = NystromApproximation()
    >>> L = np.arange(10)
    >>> X = np.random.randn(100, 10)
    >>> A = X @ np.diag(L) @ np.linalg.pinv(X)
    >>> A = A + A.T
    >>> Q, Y = svd.compute(A, 9)
    >>> Q, Y = svd.refine(1)
    >>> assert(np.linalg.norm(A - Q @ Y @ Q.T) < 1e-10)

    [2] Gittens A. and Mahoney M. W. (2016). "Revisiting the Nyström Method for
        Improved Large-scale Machine Learning". roceedings of the 30th
        International Conference on Machine Learning, PMLR 28 (3): 567-575.
    """
    def __init__(self, sketch=StandardSketch(orthogonal=False)):
        super().__init__(sketch)

    def _factorize(self):
        S = np.linalg.pinv(self.G.T @ self.Q)
        return self.Q, S, self.Q.conj().T
