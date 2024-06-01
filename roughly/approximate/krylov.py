# Lanczos / Arnoldi

import numpy as np
from abc import ABCMeta, abstractmethod

class KrylovDecomposition(metaclass=ABCMeta):
    def __init__(self, reorth_tol=0.7):
        self.reorth_tol = reorth_tol

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def refine(self):
        pass

class ArnoldiDecomposition(KrylovDecomposition):
    def __init__(self, reorth_tol=0.7):
        super().__init__(reorth_tol)

    def compute(self, A, X, k=100, dtype=None):
        """
        Arnoldi method.

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

        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        n, m = X.shape

        self.matvec = lambda x: A @ x if isinstance(A, np.ndarray) else A
        self.dtype = A.dtype if dtype is None else dtype

        self.k = k

        self.H = np.zeros((k + 1, k, m), dtype=self.dtype)
        self.U = np.empty((k + 1, n, m), dtype=self.dtype)

        self.U[0] = X / np.linalg.norm(X, axis=0)

        # TODO: maybe replace with arnoldi_iterate(j)
        for j in range(k):

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

        self.U = np.swapaxes(self.U, 0, 2)
        self.H = np.einsum("ijk->kij", self.H)

        return self.U, self.H

    def refine(self, k=10):

        self.U = np.swapaxes(self.U, 0, 2)
        self.H = np.einsum("ijk->jki", self.H)

        self.H = np.pad(self.H, [(0, k), (0, k), (0, 0)], mode="constant")
        self.U = np.pad(self.U, [(0, k), (0, 0), (0, 0)], mode="constant")

        for j in range(self.k, self.k + k):

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

        self.U = np.swapaxes(self.U, 0, 2)
        self.H = np.einsum("ijk->kij", self.H)

        self.k += k

        return self.U, self.H

class LanczosDecomposition(KrylovDecomposition):
    def __init__(self, reorth_tol=0.7, return_matrix=True, extend_matrix=True):
        self.return_matrix = return_matrix
        self.extend_matrix = extend_matrix
        super().__init__(reorth_tol)

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

        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        n, m = X.shape

        self.matvec = lambda x: A @ x if isinstance(A, np.ndarray) else A
        self.dtype = A.dtype if dtype is None else dtype

        self.k = k
        self.m = m

        # Initialize arrays for storing the block-tridiagonal elements
        self.a = np.empty((k, m), dtype=self.dtype)
        self.b = np.empty((k + 1, m), dtype=self.dtype)
        self.U = np.empty((k + 1, n, m), dtype=self.dtype)

        self.b[0] = np.linalg.norm(X, axis=0)
        self.U[0] = X / self.b[0]

        for j in range(k):

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

        self.U = np.swapaxes(self.U, 0, 2)
        self.a = np.swapaxes(self.a, 0, 1)
        self.b = np.swapaxes(self.b, 0, 1)

        if self.return_matrix:
            T = np.zeros((m, k + 1, k), dtype=self.a.dtype)
            for i in range(m):
                T[i, :k, :k] = np.diag(self.b[i, 1:-1], 1) + np.diag(self.a[i]) + np.diag(self.b[i, 1:-1], -1)
                T[i, -1, -1] = self.b[i, -1]
            if not self.extend_matrix:
                T = T[:, :k, :]
            return self.U, T
        return self.U, self.a, self.b


    def refine(self, k=10):

        self.U = np.swapaxes(self.U, 0, 2)
        self.a = np.swapaxes(self.a, 0, 1)
        self.b = np.swapaxes(self.b, 0, 1)

        self.a = np.pad(self.a, [(0, k), (0, 0)], mode="constant")
        self.b = np.pad(self.b, [(0, k), (0, 0)], mode="constant")
        self.U = np.pad(self.U, [(0, k), (0, 0), (0, 0)], mode="constant")

        for j in range(self.k, self.k + k):

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

        self.U = np.swapaxes(self.U, 0, 2)
        self.a = np.swapaxes(self.a, 0, 1)
        self.b = np.swapaxes(self.b, 0, 1)

        self.k += k

        if self.return_matrix:
            T = np.zeros((self.m, self.k + 1, self.k), dtype=self.a.dtype)
            for i in range(self.m):
                T[i, :self.k, :self.k] = np.diag(self.b[i, 1:-1], 1) + np.diag(self.a[i]) + np.diag(self.b[i, 1:-1], -1)
                T[i, -1, -1] = self.b[i, -1]
            if not self.extend_matrix:
                T = T[:, :self.k, :]
            return self.U, T
        return self.U, self.a, self.b
