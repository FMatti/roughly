"""
sketch.py
---------

Methods for randomized range sketching of a linear operator.
"""

import numpy as np
from abc import ABCMeta, abstractmethod

from roughly.core.random import gaussian, rademacher, spherical

class RangeSketch(metaclass=ABCMeta):
    def __init__(self, rng="gaussian", orthogonal=True):
        self.orthogonal = orthogonal
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

class StandardSketch(RangeSketch):
    """
    Randomized range sketch.

    Parameters
    ----------
    rng : function or str
        Distribution to generator the randomized embedding with. Either a
        function whose arguments are the sizes of the axes of the randomized
        embedding or one of the following strings:
            - "gaussian": Standard Gaussian random numbers
            - "rademacher": Uniformly sample from {-1, +1}
            - "spherical": Spherically normalized standard Gaussians
    orthogonal : bool
        Whether to orthogonalize the range sketch or not.
    """
    def __init__(self, rng="gaussian", orthogonal=True):
        super().__init__(rng, orthogonal)

    def compute(self, A, k=10, n=None, dtype=None, return_embedding=False):
        """
        Compute a randomized range sketch for the linear operator A.

        Parameters
        ----------
        A : np.ndarray of shape (n, n) or function
            The matrix or linear operator (given as function handle) for which a
            basis of the Krylov subspace is computed.
        X : np.ndarray of shape (n) or (n, m)
            The starting vector(s) used as input to the Arnoldi method.
        k : int >= 1
            The size of the sketch.
        n : int >= 1
            The length of each component in the sketch (only needed if A is
            given as a function handle).
        dtype : [np.float, np.complex, ...]
            The expected data type of the Krylov basis. If None, dtype is
            inferred from A.
        return_embedding : bool
            Whether to return the embedding matrix or not.
            
        Returns
        -------
        self.Q : np.ndarray of shape (n, k)
            Approximated range of the linear operator A.
        """
        self.preprocess(A, k, n, dtype)

        # Generate embedding and sketch linear operator
        self.S = self.rng(self.n, k)
        self.Q = self.matvec(self.S)
        
        # Orthogonalize and return
        if self.orthogonal:
            self.Q, _ = np.linalg.qr(self.Q)
        if return_embedding:
            return self.Q, self.S
        return self.Q

    def refine(self, k=10, return_embedding=False):
        """
        Increase the size of a randomized range sketch for a linear operator.

        Parameters
        ----------
        k : int >= 1
            The size of the sketch.
            given as a function handle).
        return_embedding : bool
            Whether to return the embedding matrix or not.

        Returns
        -------
        self.Q : np.ndarray of shape (n, ... + k)
            Approximated range of a linear operator.
        """
        # Generate embedding and sketch linear operator
        S = self.rng(self.n, k)
        self.Q = np.hstack((self.Q, self.matvec(S)))

        # Orthogonalize and return
        if self.orthogonal:
            # TODO: Only orthogonalize last k columns
            self.Q, _ = np.linalg.qr(self.Q)
        self.k += k
        self.S = np.hstack((self.S, S))
        if return_embedding:
            return self.Q, self.S
        return self.Q

# TODO: class SubsampledRandomizedHadamardTransform(RangeSketch)
