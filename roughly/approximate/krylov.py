"""
krylov.py
---------

Collection of methods for computing Krylov-space matrix decompositions.
"""

import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod
from typing import Union

class KrylovDecomposition(metaclass=ABCMeta):
    def __init__(self):
        self.k = 0

    def _preprocess(self, A, X, k, dtype=None):
        """
        Preprocess the input matrix/operator and determine problem dimensions.
        """
        if X.ndim < 2:
            X = X[:, np.newaxis]
        self.n, self.m = X.shape
        self.k = k

        self.matvec = (lambda x: A @ x) if isinstance(A, np.ndarray) else A
        self.dtype = A.dtype if dtype is None else dtype
        return X

    @staticmethod
    def _zeropad(A, axes, sizes):
        """
        Pad np.ndarray with user specified number of zeros along selected axes.
        """
        pad = np.zeros((A.ndim, 2), dtype=int)
        pad[list(axes), 1] = sizes
        return np.pad(A, pad, mode="constant")

    @abstractmethod
    def _initialize(self):
        """
        Initialize the matrices which define the decomposition.
        """
        pass

    @abstractmethod
    def _iterate(self, j):
        """
        Perform the j-th iteration of the Krylov method.
        """
        pass

    @abstractmethod
    def _result(self):
        """
        Return the matrices which define the decomposition.
        """
        pass

    @abstractmethod
    def _extend(self, k):
        """
        Extend the decomposition matrices for k more iterations.
        """
        pass

    def compute(self, A : Union[np.ndarray, callable], X : np.ndarray, k : int = 10, dtype : bool = None):
        """
        Compute Krylov decomposition of a linear operator

            A @ U[:, :k] = U[:, :k+1] @ H

        for a single input vector, a set of input vectors, or an input matrix.

        Parameters
        ----------
        A : np.ndarray of shape (n, n) or callable
            The matrix or linear operator (given as function handle) for which a
            basis of the Krylov subspace is computed.
        X : np.ndarray of shape (n) or (n, m)
            The starting vector(s) used as input to the Arnoldi method.
        k : int >= 1
            The number of iterations in the Arnoldi method
        dtype : [np.float, np.complex, ...]
            The expected data type of the Krylov basis. If None, dtype is
            inferred from A.

        Returns
        -------
        self.U : np.ndarray of shape (n, k + 1) or (m, n, k + 1)
            Orthogonal basis of the Krylov subspace of A for X.
        self.H : np.ndarray of shape (k + 1, k) or (m, k + 1, k)
            Hessenberg matrix.
        """

        X = self._preprocess(A, X, k, dtype)
        self._initialize(X)

        for j in range(k):
            self._iterate(j)

        return self._result()

    def refine(self, k : int = 1):
        """
        Refine the existing Krylov decomposition of a linear operator,

        Parameters
        ----------
        k : int >= 1
            The number of additional iterations to perform with Krylov method.

        Returns
        -------
        U : np.ndarray of shape (n, ... + k + 1) or (m, n, ... + k + 1)
            Orthogonal basis of the Krylov subspace of A for X.
        H : np.ndarray of shape (... + k + 1, ... + k) or (m, ... + k + 1, ... + k)
            Hessenberg matrix.
        """

        assert self.k != 0, "Must first compute a Krylov decomposition."

        self._extend(k)

        for j in range(self.k, self.k + k):
            self._iterate(j)

        self.k += k

        return self._result()

class ArnoldiDecomposition(KrylovDecomposition):
    """
    Implements the Krylov-decomposition of a matrix or linear operator A with
    the Arnoldi method [1]. The decomposition consists of an orthogonal matrix U
    and a upper Hessenberg matrix H which satisfy

        A @ U[:, :k] = U[:, :k+1] @ H

    after k iterations of the Arnoldi method.

    Parameters
    ----------
    reorth_tol : float < 1
        The tolerance for reorthogonalizing the Krylov basis between iterations.

    Attributes
    ----------
    .compute(A, X, k, ...)
        Compute Krylov decomposition of a linear operator A for starting
        vector(s) X with k Arnoldi iterations.
    .refine(k, ...)
        Refine Krylov decomposition with k additional Arnoldi iterations.

    Example
    -------
    >>> import numpy as np
    >>> from roughly.approximate.krylov import ArnoldiDecomposition
    >>> decomposition = ArnoldiDecomposition()
    >>> A = np.random.randn(100, 100)
    >>> X = np.random.randn(100)
    >>> U, H = decomposition.compute(A, X, 10)
    >>> np.testing.assert_allclose(A @ U[:, :10] - U @ H, 0, atol=1e-10)
    >>> U, H = decomposition.refine(1)
    >>> np.testing.assert_allclose(A @ U[:, :11] - U @ H, 0, atol=1e-10)

    [1] Arnoldi, W. E. (1951). "The principle of minimized iterations in the
        solution of the matrix eigenvalue problem". Quarterly of Applied
        Mathematics. 9 (1): 17–29. doi:10.1090/qam/42792.
    """
    def __init__(self, reorth_tol : float = 0.7):
        self.reorth_tol = reorth_tol
        super().__init__()

    def _initialize(self, X):
        self.H = np.zeros((self.k + 1, self.k, self.m), dtype=self.dtype)
        self.U = np.empty((self.k + 1, self.n, self.m), dtype=self.dtype)

        self.U[0] = X / np.linalg.norm(X, axis=0)

    def _iterate(self, j):
        # Orthogonalize next iterate against previous iterates
        w = self.matvec(self.U[j])
        self.H[: j + 1, j] = np.einsum("ijk,jk->ik", self.U[: j + 1].conj(), w)
        u_tilde = w - np.einsum("ijk,ik->jk", self.U[: j + 1], self.H[: j + 1, j])

        # Reorthogonalize
        idx, = np.where(np.linalg.norm(u_tilde, axis=0) < self.reorth_tol * np.linalg.norm(w, axis=0))
        if len(idx) >= 1:
            h_hat = np.einsum("ijk,jk->ik", self.U[:j + 1, :, idx].conj(), u_tilde[:, idx])
            self.H[: j + 1, j, idx] += h_hat
            u_tilde[:, idx] -= np.einsum("ijk,ik->jk", self.U[:j + 1, :, idx].conj(), h_hat)

        # Form the outputs
        self.H[j + 1, j] = np.linalg.norm(u_tilde, axis=0)
        self.U[j + 1] = u_tilde / self.H[j + 1, j]

    def _result(self):
        U = np.einsum("ijk->kji", self.U)
        H = np.einsum("ijk->kij", self.H)

        if self.m == 1:
            return U[0], H[0]
        return U, H

    def _extend(self, k):
        self.H = self._zeropad(self.H, [0, 1], [k, k])
        self.U = self._zeropad(self.U, [0], [k])

class LanczosDecomposition(KrylovDecomposition):
    """
    Implements the Krylov-decomposition of a Hermitian matrix or linear operator
    A with the Lanczos method [2]. The decomposition consists of an orthogonal
    matrix U and a Hermitian tridiagonal matrix H which satisfy

        A @ U[:, :k] = U[:, :k+1] @ H

    after k iterations of the Lanczos method.

    Parameters
    ----------
    reorth_tol : float < 1
        The tolerance for reorthogonalizing the Krylov basis between iterations.
    return_matrix : bool
        Whether to return the (full) tridiagonal matrix H or arrays of its
        diagonal and off-diagonal elements.
    return_matrix : bool
        Whether to extend the orthogonal matrix U with one more column after
        the last iteration.

    Attributes
    ----------
    .compute(A, X, k, ...)
        Compute Krylov decomposition of a Hermitian linear operator A for
        starting vector(s) X with k Lanczos iterations.
    .refine(k, ...)
        Refine Krylov decomposition with k additional Lanczos iterations.

    Example
    -------
    >>> import numpy as np
    >>> from roughly.approximate.krylov import LanczosDecomposition
    >>> decomposition = LanczosDecomposition()
    >>> A = np.random.randn(100, 100)
    >>> A = A + A.T
    >>> X = np.random.randn(100)
    >>> U, H = decomposition.compute(A, X, 10)
    >>> np.testing.assert_allclose(A @ U[:, :10] - U @ H, 0, atol=1e-10)
    >>> U, H = decomposition.refine(1)
    >>> np.testing.assert_allclose(A @ U[:, :11] - U @ H, 0, atol=1e-10)

    [2] Lanczos, C. (1950). "An iteration method for the solution of the
        eigenvalue problem of linear differential and integral operators".
        Journal of Research of the National Bureau of Standards. 45 (4): 255–282.
        doi:10.6028/jres.045.026.
    """
    def __init__(self, reorth_tol : float = 0.7, return_matrix : bool = True, extend_matrix : bool = True):
        self.return_matrix = return_matrix
        self.extend_matrix = extend_matrix
        self.reorth_tol = reorth_tol
        super().__init__()

    def _initialize(self, X):

        # Initialize arrays for storing the block-tridiagonal elements
        self.a = np.empty((self.k, self.m), dtype=self.dtype)
        self.b = np.empty((self.k + 1, self.m), dtype=self.dtype)
        self.U = np.empty((self.k + 1, self.n, self.m), dtype=self.dtype)

        self.b[0] = np.linalg.norm(X, axis=0)
        self.U[0] = X / self.b[0]

    def _iterate(self, j):
        # New set of vectors
        w = self.matvec(self.U[j])
        self.a[j] = np.sum(self.U[j].conj() * w, axis=0)
        u_tilde = w - self.U[j] * self.a[j] - (self.U[j-1] * self.b[j] if j > 0 else 0)

        # reorthogonalization
        idx, = np.where(np.linalg.norm(u_tilde, axis=0) < self.reorth_tol * np.linalg.norm(w, axis=0))
        if len(idx) >= 1:
            h_hat = np.einsum("ijk,jk->ik", self.U[:j + 1, :, idx].conj(), u_tilde[:, idx])
            self.a[j, idx] += h_hat[-1]
            if j > 0:
                self.b[j - 1, idx] += h_hat[-2]
            u_tilde[:, idx] -= np.einsum("ijk,ik->jk", self.U[:j + 1, :, idx].conj(), h_hat)

        self.b[j + 1] = np.linalg.norm(u_tilde, axis=0)
        self.U[j + 1] = u_tilde / self.b[j + 1]

    def _result(self):
        U = np.einsum("ijk->kji", self.U)
        a = np.einsum("ij->ji", self.a)
        b = np.einsum("ij->ji", self.b)

        if self.return_matrix:
            T = np.zeros((self.m, self.k + 1, self.k), dtype=self.a.dtype)
            T[:, np.arange(self.k), np.arange(self.k)] = a
            T[:, np.arange(1, self.k + 1), np.arange(self.k)] = b[:, 1:]
            T[:, np.arange(self.k - 1), np.arange(1, self.k)] = b[:, 1:-1]
            if not self.extend_matrix:
                T = T[:, :self.k, :]
            if self.m == 1:
                return U[0], T[0]
            return U, T

        if self.m == 1:
            return U[0], a[0], b[0]
        return U, a, b

    def _extend(self, k):
        self.a = self._zeropad(self.a, [0], [k])
        self.b = self._zeropad(self.b, [0], [k])
        self.U = self._zeropad(self.U, [0], [k])

class BlockArnoldiDecomposition(ArnoldiDecomposition):
    """
    Implements the Krylov-decomposition of a matrix or linear operator A with
    the block Arnoldi method [3]. The decomposition consists of an orthogonal
    matrix U and a upper Hessenberg matrix H which satisfy

        A @ U[:, :k] = U[:, :k+1] @ H

    after k iterations of the Arnoldi method.

    Attributes
    ----------
    .compute(A, X, k, ...)
        Compute Krylov decomposition of a linear operator A for starting
        matrix X with k Arnoldi iterations.
    .refine(k, ...)
        Refine Krylov decomposition with k additional Arnoldi iterations.

    Example
    -------
    >>> import numpy as np
    >>> from roughly.approximate.krylov import BlockArnoldiDecomposition
    >>> decomposition = BlockArnoldiDecomposition()
    >>> A = np.random.randn(100, 100)
    >>> X = np.random.randn(100, 2)
    >>> U, H = decomposition.compute(A, X, 10)
    >>> np.testing.assert_allclose(A @ U[:, :-2] - U @ H, 0, atol=1e-10)
    >>> U, H = decomposition.refine(1)
    >>> np.testing.assert_allclose(A @ U[:, :-2] - U @ H, 0, atol=1e-10)

    [3] Jaimoukha, I. M. and Kasenally, E. M. (1994). "Krylov Subspace Methods
    for Solving Large Lyapunov Equations". SIAM Journal on Numerical Analysis.
    31 (1): 227-251. doi:10.1137/0731012.
    """
    def __init__(self):
        super().__init__()

    def _initialize(self, X):
        self.H = np.zeros((self.k + 1, self.k, self.m, self.m), dtype=self.dtype)
        self.U = np.empty((self.k + 1, self.n, self.m), dtype=self.dtype)

        self.U[0], _ = np.linalg.qr(X)

    def _iterate(self, j):

        # New set of vectors
        w = self.matvec(self.U[j])
        for i in range(j + 1):
            self.H[i, j] = self.U[i].conj().T @ w
            w = w - self.U[i] @ self.H[i, j]

        Q, R = np.linalg.qr(w)
        self.U[j + 1] = Q
        self.H[j + 1, j] = R

    def _result(self):
        U = np.einsum("ijk->jik", self.U).reshape(self.n, -1)
        H = np.einsum("ijkl->ikjl", self.H).reshape(((self.k + 1) * self.m, self.k * self.m))

        return U, H

class BlockLanczosDecomposition(LanczosDecomposition):
    """
    Implements the Krylov-decomposition of a Hermitian matrix or linear operator
    A with the block Lanczos method [4]. The decomposition consists of an
    orthogonal matrix U and a Hermitian tridiagonal matrix H which satisfy

        A @ U[:, :k] = U[:, :k+1] @ H

    after k iterations of the Lanczos method.

    Parameters
    ----------
    return_matrix : bool
        Whether to return the (full) tridiagonal matrix H or arrays of its
        diagonal and off-diagonal elements.
    return_matrix : bool
        Whether to extend the orthogonal matrix U with one more column after
        the last iteration.
    reorth_steps : int
        The number of iterations in which to reorthogonalize. To always
        reorthogonalize, use -1.

    Attributes
    ----------
    .compute(A, X, k, ...)
        Compute Krylov decomposition of a Hermitian linear operator A for
        starting vector(s) X with k Arnoldi iterations.
    .refine(k, ...)
        Refine Krylov decomposition with k additional Lanczos iterations.

    Example
    -------
    >>> import numpy as np
    >>> from roughly.approximate.krylov import BlockLanczosDecomposition
    >>> decomposition = BlockLanczosDecomposition()
    >>> A = np.random.randn(100, 100)
    >>> A = A + A.T
    >>> X = np.random.randn(100, 2)
    >>> U, H = decomposition.compute(A, X, 10)
    >>> np.testing.assert_allclose(A @ U[:, :-2] - U @ H, 0, atol=1e-10)
    >>> U, H = decomposition.refine(1)
    >>> np.testing.assert_allclose(A @ U[:, :-2] - U @ H, 0, atol=1e-10)

    [4] Montgomery, P. L. (1995). "A Block Lanczos Algorithm for Finding
        Dependencies over GF(2)". Lecture Notes in Computer Science. EUROCRYPT.
        Vol. 921. Springer-Verlag. pp. 106–120. doi:10.1007/3-540-49264-X_9.
    [5] Chen T. and Hallman, E. (2023). "Krylov-Aware Stochastic Trace
        Estimation". SIAM Journal on Matrix Analysis and ApplicationsVol.
        44 (3). doi:10.1137/22M1494257.
    """
    def __init__(self, return_matrix : bool = True, extend_matrix : bool = True, reorth_steps : int = -1):
        self.return_matrix = return_matrix
        self.extend_matrix = extend_matrix
        self.reorth_steps = reorth_steps
        super().__init__()

    def _initialize(self, X):

        # Initialize arrays for storing the block-tridiagonal elements
        self.a = np.empty((self.k, self.m, self.m), dtype=self.dtype)
        self.b = np.empty((self.k + 1, self.m, self.m), dtype=self.dtype)
        self.U = np.empty((self.k + 1, self.n, self.m), dtype=self.dtype)

        self.U[0], self.b[0] = np.linalg.qr(X)

    def _iterate(self, j):
        # New set of vectors
        w = self.matvec(self.U[j])
        self.a[j] = self.U[j].conj().T @ w
        u_tilde = w - self.U[j] @ self.a[j] - (self.U[j-1] @ self.b[j].conj().T if j > 0 else 0)

        # reorthogonalization
        if j > 0 and (self.reorth_steps > j or self.reorth_steps == -1):
            h_hat = np.swapaxes(self.U[:j], 0, 1).reshape(self.n, -1).conj().T @ u_tilde
            self.a[j] += h_hat[-1]
            u_tilde = u_tilde - np.swapaxes(self.U[:j], 0, 1).reshape(self.n, -1) @ h_hat

        # Pivoted QR
        z_tilde, R, p = sp.linalg.qr(u_tilde, pivoting=True, mode="economic")
        self.b[j + 1] = R[:, np.argsort(p)]

        # Orthogonalize again if R is rank deficient
        if self.reorth_steps > j or self.reorth_steps == -1:
            r = np.abs(np.diag(self.b[j + 1]))
            r_idx = np.nonzero(r < np.max(r) * 1e-10)[0]
            z_tilde[:, r_idx] = z_tilde[:, r_idx] - self.U[j] @ (self.U[j].conj().T @ z_tilde[:, r_idx])
            z_tilde[:, r_idx], R = np.linalg.qr(z_tilde[:, r_idx])
            z_tilde[:, r_idx] *= np.sign(np.diag(R))
            self.U[j + 1] = z_tilde

    def _result(self):

        U = np.einsum("ijk->jik", self.U).reshape(self.n, -1)

        if self.return_matrix:

            T = np.zeros(((self.k + 1)*self.m, self.k*self.m), dtype=self.dtype)

            x, y = np.meshgrid(np.arange(self.m), np.arange(self.m))
            idx = np.add.outer(self.m * np.arange(self.k), x).ravel()
            idy = np.add.outer(self.m * np.arange(self.k), y).ravel()
            T[idy, idx] = self.a.ravel()
            T[idy + self.m, idx] = self.b[1:self.k+1].ravel()
            T[idy[:-self.m ** 2], idx[:-self.m ** 2] + self.m] = np.einsum("ijk->ikj", self.b[1:self.k].conj()).ravel()

            if not self.extend_matrix:
                T = T[:-self.m, :]
            return U, T
        return U, self.a, self.b
