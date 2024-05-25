import numpy as np
import scipy as sp
import numba

# TODO: Add seeds
# TÒDO: Unify methods

def Arnoldi(A, X, k, reorth_tol=0.7, dtype=np.float64):
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
    U : np.ndarray of shape (n, k + 1) or (m, n, k + 1)
        Orthogonal basis of the Krylov subspace of A and X.
    H : np.ndarray of shape (k + 1, k) or (m, k + 1, k)
        Hessenberg matrix.
    """

    if len(X.shape) < 2:
        X = X.reshape(-1, 1)
    n, m = X.shape

    matvec = lambda x: A @ x if isinstance(A, np.ndarray) else A
    dtype = A.dtype if isinstance(A, np.ndarray) else dtype

    H = np.zeros((k + 1, k, m), dtype=dtype)
    U = np.empty((k + 1, n, m), dtype=dtype)

    U[0] = X / np.linalg.norm(X, axis=0)

    for j in range(k):

        # Orthogonalize next iterate against previous iterates
        w = matvec(U[j])
        H[: j + 1, j] = np.einsum("ijk,jk->ik", U[: j + 1].conj(), w)
        u_tilde = w - np.einsum("ijk,ik->jk", U[: j + 1], H[: j + 1, j])

        # Reorthogonalize
        idx = np.where(np.linalg.norm(u_tilde, axis=0) < reorth_tol * np.linalg.norm(w, axis=0))[0]
        if len(idx) >= 1:
            h_hat = np.einsum("ijk,jk->ik", U[:j + 1, :, idx].conj(), u_tilde[:, idx])
            H[: j + 1, j, idx] += h_hat
            u_tilde[:, idx] -= np.einsum("ijk,ik->jk", U[:j + 1, :, idx].conj(), h_hat)

        # Form the outputs
        H[j + 1, j] = np.linalg.norm(u_tilde, axis=0)
        U[j + 1] = u_tilde / H[j + 1, j]

    U = np.swapaxes(U, 0, 2)
    H = np.einsum("ijk->kij", H)

    return U, H


def BlockArnoldi(A, X, k, dtype=np.float64):

    if len(X.shape) < 2:
        X = X.reshape(-1, 1)
    n, m = X.shape

    matvec = lambda x: A @ x if isinstance(A, np.ndarray) else A
    dtype = A.dtype if isinstance(A, np.ndarray) else dtype

    H = np.zeros((k + 1, k, m, m), dtype=dtype)
    U = np.empty((k + 1, A.shape[0], m), dtype=dtype)

    U[0], _ = np.linalg.qr(X)

    for j in range(k):

        # New set of vectors
        w = matvec(U[j])
        for i in range(j + 1):
            H[i, j] = U[i].conj().T @ w
            w = w - U[i] @ H[i, j]

        Q, R = np.linalg.qr(w)
        U[j + 1] = Q
        H[j + 1, j] = R

    U = np.swapaxes(U, 0, 1).reshape(n, -1)
    H = np.swapaxes(H, 1, 2).reshape(((k + 1) * m, k * m))

    return U, H


def Lanczos(A, X, k, reorth_tol=0.7, return_matrix=False, extend_matrix=False, dtype=np.float64):

    if len(X.shape) < 2:
        X = X.reshape(-1, 1)
    n, m = X.shape

    matvec = lambda x: A @ x if isinstance(A, np.ndarray) else A
    dtype = A.dtype if isinstance(A, np.ndarray) else dtype

    # Initialize arrays for storing the block-tridiagonal elements
    a = np.empty((k, m), dtype=dtype)
    b = np.empty((k + 1, m), dtype=dtype)
    U = np.empty((k + 1, n, m), dtype=dtype)

    b[0] = np.linalg.norm(X, axis=0)
    U[0] = X / b[0]

    for j in range(k):

        # New set of vectors
        w = matvec(U[j])
        a[j] = np.sum(U[j].conj() * w, axis=0)
        u_tilde = w - U[j] * a[j] - (U[j-1] * b[j] if j > 0 else 0)

        # reorthogonalization
        idx = np.where(np.linalg.norm(u_tilde, axis=0) < reorth_tol * np.linalg.norm(w, axis=0))[0]
        if len(idx) >= 1:
            h_hat = np.einsum("ijk,jk->ik", U[:j + 1, :, idx].conj(), u_tilde[:, idx])
            a[j, idx] += h_hat[-1]
            if j > 0:
                b[j - 1, idx] += h_hat[-2]
            u_tilde[:, idx] -= np.einsum("ijk,ik->jk", U[:j + 1, :, idx].conj(), h_hat)

        b[j + 1] = np.linalg.norm(u_tilde, axis=0)
        U[j + 1] = u_tilde / b[j + 1]

    U = np.swapaxes(U, 0, 2)
    a = np.swapaxes(a, 0, 1)
    b = np.swapaxes(b, 0, 1)

    if return_matrix:
        T = np.zeros((m, k + 1, k), dtype=a.dtype)
        for i in range(m):
            T[i, :k, :k] = np.diag(b[i, 1:-1], 1) + np.diag(a[i]) + np.diag(b[i, 1:-1], -1)
            T[i, -1, -1] = b[i, -1]
        if not extend_matrix:
            T = T[:, :k, :]
        return U, T
    return U, a, b


def BlockLanczos(A, X, k, reorth_steps=-1, return_matrix=False, extend_matrix=False, dtype=np.float64):

    if len(X.shape) < 2:
        X = X.reshape(-1, 1)
    n, m = X.shape

    matvec = lambda x: A @ x if isinstance(A, np.ndarray) else A
    dtype = A.dtype if isinstance(A, np.ndarray) else dtype

    if reorth_steps == -1:
        reorth_steps = k

    # Initialize arrays for storing the block-tridiagonal elements
    a = np.empty((k, m, m), dtype=dtype)
    b = np.empty((k + 1, m, m), dtype=dtype)
    U = np.empty((k + 1, n, m), dtype=dtype)

    U[0], b[0] = np.linalg.qr(X)

    for j in range(k):

        # New set of vectors
        w = matvec(U[j])
        a[j] = U[j].conj().T @ w
        u_tilde = w - U[j] @ a[j] - (U[j-1] @ b[j].conj().T if j > 0 else 0)

        # reorthogonalization
        if j > 0 and reorth_steps > j:
            h_hat = np.swapaxes(U[:j], 0, 1).reshape(n, -1).conj().T @ u_tilde
            a[j] += h_hat[-1]
            u_tilde = u_tilde - np.swapaxes(U[:j], 0, 1).reshape(n, -1) @ h_hat

        # Pivoted QR
        z_tilde, R, p = sp.linalg.qr(u_tilde, pivoting=True, mode="economic")
        b[j + 1] = R[:, np.argsort(p)]

        # Orthogonalize again if R is rank deficient
        if reorth_steps > j:
            r = np.abs(np.diag(b[j + 1]))
            r_idx = np.nonzero(r < np.max(r) * 1e-10)[0]
            z_tilde[:, r_idx] = z_tilde[:, r_idx] - U[j] @ (U[j].conj().T @ z_tilde[:, r_idx])
            z_tilde[:, r_idx], R = np.linalg.qr(z_tilde[:, r_idx])
            z_tilde[:, r_idx] *= np.sign(np.diag(R))
            U[j + 1] = z_tilde

    U = np.swapaxes(U, 0, 1).reshape(n, -1)

    if return_matrix:
        T = np.zeros(((k + 1)*m, k*m), dtype=A.dtype)
        for i in range(k):
            T[i*m:(i+1)*m,i*m:(i+1)*m] = a[i]
            T[(i+1)*m:(i+2)*m, i*m:(i+1)*m] = b[i + 1]
            if i < k - 1:
                T[i*m:(i+1)*m, (i+1)*m:(i+2)*m] = b[i + 1].conj().T
        if not extend_matrix:
            T = T[:-m, :]
        return U, T
    return U, a, b
