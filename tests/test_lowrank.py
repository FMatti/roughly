import __context__

import numpy as np
from roughly.approximate.lowrank import RandomizedSVD, NystromApproximation
from roughly.approximate.sketch import StandardSketch

def approximation_quality(A, k, U, S, Vh):
    U_svd, S_svd, Vh_svd = np.linalg.svd(A)
    U_svd, S_svd, Vh_svd = U_svd[:, :k], S_svd[:k], Vh_svd[:k, :]
    error = np.linalg.norm(A - U @ np.diag(S) @ Vh, ord=2)
    error_svd = np.linalg.norm(A - U_svd @ np.diag(S_svd) @ Vh_svd, ord=2)
    return abs(error - error_svd) / error_svd

def test_RandomizedSVD():
    L = np.arange(10)
    X = np.random.randn(100, 10)
    A = X @ np.diag(L) @ np.linalg.pinv(X)

    for k in [10, 20, 30]:
        approximator = RandomizedSVD()
        U, S, Vh = approximator.compute(A, k)
        for k_add in [1, 2, 3]:
            U, S, Vh = approximator.refine(5)
            assert(np.linalg.norm(A - U @ S @ Vh) < 1e-10)

def test_NystromApproximation():
    L = np.arange(10)
    X = np.random.randn(100, 10)
    A = X @ np.diag(L) @ np.linalg.pinv(X)
    A = A + A.T

    for k in [10, 20, 30]:
        approximator = NystromApproximation()
        U, S, Vh = approximator.compute(A, k)
        for k_add in [1, 2, 3]:
            U, S, Vh = approximator.refine(5)
            assert(np.linalg.norm(A - U @ S @ Vh) < 1e-10)
