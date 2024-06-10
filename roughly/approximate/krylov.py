# Lanczos / Arnoldi

import numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod

class KrylovDecomposition(metaclass=ABCMeta):
    def __init__(self, reorth_tol=0.7):
        self.reorth_tol = reorth_tol

    def preprocess(self, A, X, k, dtype=None):
        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        self.n, self.m = X.shape
        self.k = k

        self.matvec = lambda x: A @ x if isinstance(A, np.ndarray) else A
        self.dtype = A.dtype if dtype is None else dtype

    @staticmethod
    def zeropad(A, axes, sizes):
        pad = [[0, 0] for _ in range(len(A.shape))]
        for axis, size in zip(axes, sizes):
            pad[axis][1] = size
        return np.pad(A, pad, mode="constant")

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def iterate(self):
        pass

    @abstractmethod
    def result(self):
        pass
   
    @abstractmethod
    def extend(self):
        pass

    def compute(self, A, X, k=100, dtype=None):
        """
        Lanczos algorithm for Hermitian matrices.

        Parameters
        ----------
        A : np.ndarray of shape (n, n) or function
            The matrix for which a basis of the Krylov subspace is computed or the
            function handle implementing the matrix vector products.
        X : np.ndarray of shape (n) or (n, m)
            The starting vector(s) used as input to the Arnoldi method.
        k : int > 1
            The number of iterations in the Arnoldi method

        Returns
        -------
        self.U : np.ndarray of shape (n, k + 1) or (m, n, k + 1)
            Orthogonal basis of the Krylov subspace of A and X.
        self.H : np.ndarray of shape (k + 1, k) or (m, k + 1, k)
            Hessenberg matrix.
        """

        self.preprocess(A, X, k, dtype)
        self.initialize(X)

        for j in range(k):
            self.iterate(j)

        return self.result()

    def refine(self, k=1):
        """
        Arnoldi method.

        Parameters
        ----------
        k : int >= 1
            The number of iterations in the Arnoldi method

        Returns
        -------
        U : np.ndarray of shape (n, k + 1) or (m, n, k + 1)
            Orthogonal basis of the Krylov subspace of A and X.
        H : np.ndarray of shape (k + 1, k) or (m, k + 1, k)
            Hessenberg matrix.
        """

        self.extend(k)

        for j in range(self.k, self.k + k):
            self.iterate(j)

        self.k += k

        return self.result()

class ArnoldiDecomposition(KrylovDecomposition):
    def __init__(self, reorth_tol=0.7):
        super().__init__(reorth_tol)

    def initialize(self, X):
        self.H = np.zeros((self.k + 1, self.k, self.m), dtype=self.dtype)
        self.U = np.empty((self.k + 1, self.n, self.m), dtype=self.dtype)

        self.U[0] = X / np.linalg.norm(X, axis=0)

    def iterate(self, j):
        # Orthogonalize next iterate against previous iterates
        w = self.matvec(self.U[j])
        self.H[: j + 1, j] = np.einsum("ijk,jk->ik", self.U[: j + 1].conj(), w)
        u_tilde = w - np.einsum("ijk,ik->jk", self.U[: j + 1], self.H[: j + 1, j])

        # Reorthogonalize
        idx = np.where(np.linalg.norm(u_tilde, axis=0) < self.reorth_tol * np.linalg.norm(w, axis=0))[0]
        if len(idx) >= 1:
            h_hat = np.einsum("ijk,jk->ik", self.U[:j + 1, :, idx].conj(), u_tilde[:, idx])
            self.H[: j + 1, j, idx] += h_hat
            u_tilde[:, idx] -= np.einsum("ijk,ik->jk", self.U[:j + 1, :, idx].conj(), h_hat)

        # Form the outputs
        self.H[j + 1, j] = np.linalg.norm(u_tilde, axis=0)
        self.U[j + 1] = u_tilde / self.H[j + 1, j]

    def result(self):
        U = np.einsum("ijk->kji", self.U)
        H = np.einsum("ijk->kij", self.H)

        return U, H

    def extend(self, k):
        self.H = self.zeropad(self.H, [0, 1], [k, k])
        self.U = self.zeropad(self.U, [0], [k])

class LanczosDecomposition(KrylovDecomposition):
    def __init__(self, reorth_tol=0.7, return_matrix=True, extend_matrix=True):
        self.return_matrix = return_matrix
        self.extend_matrix = extend_matrix
        super().__init__(reorth_tol)

    def initialize(self, X):

        # Initialize arrays for storing the block-tridiagonal elements
        self.a = np.empty((self.k, self.m), dtype=self.dtype)
        self.b = np.empty((self.k + 1, self.m), dtype=self.dtype)
        self.U = np.empty((self.k + 1, self.n, self.m), dtype=self.dtype)

        self.b[0] = np.linalg.norm(X, axis=0)
        self.U[0] = X / self.b[0]

    def iterate(self, j):
        # New set of vectors
        w = self.matvec(self.U[j])
        self.a[j] = np.sum(self.U[j].conj() * w, axis=0)
        u_tilde = w - self.U[j] * self.a[j] - (self.U[j-1] * self.b[j] if j > 0 else 0)

        # reorthogonalization
        idx = np.where(np.linalg.norm(u_tilde, axis=0) < self.reorth_tol * np.linalg.norm(w, axis=0))[0]
        if len(idx) >= 1:
            h_hat = np.einsum("ijk,jk->ik", self.U[:j + 1, :, idx].conj(), u_tilde[:, idx])
            self.a[j, idx] += h_hat[-1]
            if j > 0:
                self.b[j - 1, idx] += h_hat[-2]
            u_tilde[:, idx] -= np.einsum("ijk,ik->jk", self.U[:j + 1, :, idx].conj(), h_hat)

        self.b[j + 1] = np.linalg.norm(u_tilde, axis=0)
        self.U[j + 1] = u_tilde / self.b[j + 1]

    def result(self):
        U = np.einsum("ijk->kji", self.U)
        a = np.einsum("ij->ji", self.a)
        b = np.einsum("ij->ji", self.b)

        if self.return_matrix:
            T = np.zeros((self.m, self.k + 1, self.k), dtype=self.a.dtype)
            for i in range(self.m):
                T[i, :self.k, :self.k] = np.diag(b[i, 1:-1], 1) + np.diag(a[i]) + np.diag(b[i, 1:-1], -1)
                T[i, -1, -1] = b[i, -1]
            if not self.extend_matrix:
                T = T[:, :self.k, :]
            return U, T

        return U, a, b

    def extend(self, k):
        self.a = self.zeropad(self.a, [0], [k])
        self.b = self.zeropad(self.b, [0], [k])
        self.U = self.zeropad(self.U, [0], [k])

class BlockArnoldiDecomposition(ArnoldiDecomposition):
    def __init__(self, reorth_tol=0.7):
        super().__init__(reorth_tol)

    def initialize(self, X):
        self.H = np.zeros((self.k + 1, self.k, self.m, self.m), dtype=self.dtype)
        self.U = np.empty((self.k + 1, self.n, self.m), dtype=self.dtype)

        self.U[0], _ = np.linalg.qr(X)

    def iterate(self, j):

        # New set of vectors
        w = self.matvec(self.U[j])
        for i in range(j + 1):
            self.H[i, j] = self.U[i].conj().T @ w
            w = w - self.U[i] @ self.H[i, j]

        Q, R = np.linalg.qr(w)
        self.U[j + 1] = Q
        self.H[j + 1, j] = R

    def result(self):
        U = np.einsum("ijk->jik", self.U).reshape(self.n, -1)
        H = np.einsum("ijkl->ikjl", self.H).reshape(((self.k + 1) * self.m, self.k * self.m))

        return U, H

class BlockLanczosDecomposition(LanczosDecomposition):
    def __init__(self, reorth_tol=0.7, return_matrix=True, extend_matrix=True, reorth_steps=100):
        self.return_matrix = return_matrix
        self.extend_matrix = extend_matrix
        self.reorth_steps = 100
        if reorth_steps == -1:
            self.reorth_steps = 100
        super().__init__(reorth_tol)

    def initialize(self, X):

        # Initialize arrays for storing the block-tridiagonal elements
        self.a = np.empty((self.k, self.m, self.m), dtype=self.dtype)
        self.b = np.empty((self.k + 1, self.m, self.m), dtype=self.dtype)
        self.U = np.empty((self.k + 1, self.n, self.m), dtype=self.dtype)

        self.U[0], self.b[0] = np.linalg.qr(X)

    def iterate(self, j):
        # New set of vectors
        w = self.matvec(self.U[j])
        self.a[j] = self.U[j].conj().T @ w
        u_tilde = w - self.U[j] @ self.a[j] - (self.U[j-1] @ self.b[j].conj().T if j > 0 else 0)

        # reorthogonalization
        if j > 0 and self.reorth_steps > j:
            h_hat = np.swapaxes(self.U[:j], 0, 1).reshape(self.n, -1).conj().T @ u_tilde
            self.a[j] += h_hat[-1]
            u_tilde = u_tilde - np.swapaxes(self.U[:j], 0, 1).reshape(self.n, -1) @ h_hat

        # Pivoted QR
        z_tilde, R, p = sp.linalg.qr(u_tilde, pivoting=True, mode="economic")
        self.b[j + 1] = R[:, np.argsort(p)]

        # Orthogonalize again if R is rank deficient
        if self.reorth_steps > j:
            r = np.abs(np.diag(self.b[j + 1]))
            r_idx = np.nonzero(r < np.max(r) * 1e-10)[0]
            z_tilde[:, r_idx] = z_tilde[:, r_idx] - self.U[j] @ (self.U[j].conj().T @ z_tilde[:, r_idx])
            z_tilde[:, r_idx], R = np.linalg.qr(z_tilde[:, r_idx])
            z_tilde[:, r_idx] *= np.sign(np.diag(R))
            self.U[j + 1] = z_tilde

    def result(self):
 
        U = np.einsum("ijk->jik", self.U).reshape(self.n, -1)

        if self.return_matrix:
            T = np.zeros(((self.k + 1)*self.m, self.k*self.m), dtype=self.dtype)
            for i in range(self.k):
                T[i*self.m:(i+1)*self.m,i*self.m:(i+1)*self.m] = self.a[i]
                T[(i+1)*self.m:(i+2)*self.m, i*self.m:(i+1)*self.m] = self.b[i + 1]
                if i < self.k - 1:
                    T[i*self.m:(i+1)*self.m, (i+1)*self.m:(i+2)*self.m] = self.b[i + 1].conj().T
            if not self.extend_matrix:
                T = T[:-self.m, :]
            return U, T
        return U, self.a, self.b
