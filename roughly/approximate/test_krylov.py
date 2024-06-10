import numpy as np
from krylov import ArnoldiDecomposition, BlockArnoldiDecomposition, LanczosDecomposition, BlockLanczosDecomposition

def test_ArnoldiDecomposition():
    n = 100
    k = 10
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    for m in [1, 2, 3, 10, 20]:
        X = np.random.randn(n, m)

        ar = ArnoldiDecomposition()
        U, H = ar.compute(A, X, k=k)
        k_refined = k
        for k_add in [1, 2, 5]:
            k_refined += k_add
            U, H = ar.refine(k_add)
            for i in range(m):
                np.testing.assert_allclose(A @ U[i, :, :k_refined] - U[i] @ H[i], 0, atol=1e-10)

def test_BlockArnoldiDecomposition():
    n = 100
    k = 10
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    for m in [1, 2, 3, 10, 20]:
        X = np.random.randn(n, m)

        ar = BlockArnoldiDecomposition()
        U, H = ar.compute(A, X, k=k)
        k_refined = k
        for k_add in [1, 2, 5]:
            k_refined += k_add
            U, H = ar.refine(k_add)
            np.testing.assert_allclose(A @ U[:, :-m] - U @ H, 0, atol=1e-10)

def test_LanczosDecomposition():
    n = 100
    k = 10
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    A = A + A.conj().T

    for m in [1, 2, 3, 10, 20]:
        X = np.random.randn(n, m)

        ar = LanczosDecomposition()
        U, H = ar.compute(A, X, k=k)
        k_refined = k
        for k_add in [1, 2, 5]:
            k_refined += k_add
            U, H = ar.refine(k_add)
            for i in range(m):
                np.testing.assert_allclose(A @ U[i, :, :k_refined] - U[i] @ H[i], 0, atol=1e-10)

def test_BlockLanczosDecomposition():
    n = 100
    k = 10
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    A = A + A.conj().T

    for m in [1, 2, 3]:#, 10, 20]:
        X = np.random.randn(n, m)

        ar = BlockLanczosDecomposition()
        U, H = ar.compute(A, X, k=k)
        k_refined = k
        for k_add in [1, 2, 5]:
            k_refined += k_add
            U, H = ar.refine(k_add)
            np.testing.assert_allclose(A @ U[:, :-m] - U @ H, 0, atol=1e-10)
