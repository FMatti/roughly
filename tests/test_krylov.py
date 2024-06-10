import __context__

import numpy as np
from roughly.approximate.krylov import ArnoldiDecomposition, BlockArnoldiDecomposition, LanczosDecomposition, BlockLanczosDecomposition

def test_ArnoldiDecomposition():
    n = 100
    k = 10
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    for m in [1, 2, 3, 10, 20]:
        X = np.random.randn(n, m)

        decomposition = ArnoldiDecomposition()
        U, H = decomposition.compute(A, X, k=k)
        k_refined = k
        for k_add in [1, 2, 5]:
            k_refined += k_add
            U, H = decomposition.refine(k_add)
            if m == 1:
                np.testing.assert_allclose(A @ U[:, :k_refined] - U @ H, 0, atol=1e-10)
            else:
                for i in range(m):
                    np.testing.assert_allclose(A @ U[i, :, :k_refined] - U[i] @ H[i], 0, atol=1e-10)

def test_BlockArnoldiDecomposition():
    n = 100
    k = 10
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    for m in [1, 2, 3, 10, 20]:
        X = np.random.randn(n, m)

        decomposition = BlockArnoldiDecomposition()
        U, H = decomposition.compute(A, X, k=k)
        k_refined = k
        for k_add in [1, 2, 5]:
            k_refined += k_add
            U, H = decomposition.refine(k_add)
            np.testing.assert_allclose(A @ U[:, :-m] - U @ H, 0, atol=1e-10)

def test_LanczosDecomposition():
    n = 100
    k = 10
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    A = A + A.conj().T

    for m in [1, 2, 3, 10, 20]:
        X = np.random.randn(n, m)

        decomposition = LanczosDecomposition()
        U, H = decomposition.compute(A, X, k=k)
        k_refined = k
        for k_add in [1, 2, 5]:
            k_refined += k_add
            U, H = decomposition.refine(k_add)
            if m == 1:
                np.testing.assert_allclose(A @ U[:, :k_refined] - U @ H, 0, atol=1e-10)
            else:
                for i in range(m):
                    np.testing.assert_allclose(A @ U[i, :, :k_refined] - U[i] @ H[i], 0, atol=1e-10)

def test_BlockLanczosDecomposition():
    n = 100
    k = 10
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    A = A + A.conj().T

    for m in [1, 2, 3]:#, 10, 20]: <- basis becomes degenerate by design
        X = np.random.randn(n, m)

        decomposition = BlockLanczosDecomposition()
        U, H = decomposition.compute(A, X, k=k)
        k_refined = k
        for k_add in [1, 2, 5]:
            k_refined += k_add
            U, H = decomposition.refine(k_add)
            np.testing.assert_allclose(A @ U[:, :-m] - U @ H, 0, atol=1e-10)
