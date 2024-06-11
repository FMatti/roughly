import __context__

import numpy as np
from roughly.approximate.trace import HutchinsonTraceEstimator, SubspaceProjectionEstimator, DeflatedTraceEstimator

np.random.seed(42)

def test_HutchinsonTraceEstimator():
    n = 100
    k = 10
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    A = A + A.conj().T
    tr_A = np.trace(A)

    for k in [1, 2, 3, 10, 20]:
        estimator = HutchinsonTraceEstimator()
        estimator.compute(A, k)
        k_total = k
        for k_refine in range(10):
            k_total += k_refine
            t = estimator.refine(k=k_refine)
            print(abs(tr_A - t) / abs(tr_A) < 1 / np.sqrt(k_total))

def test_SubspaceProjectionEstimator():
    n = 100
    k = 10
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    A = A + A.conj().T
    tr_A = np.trace(A)

    for k in [1, 2, 3, 10, 20]:
        estimator = SubspaceProjectionEstimator()
        estimator.compute(A, k)
        k_total = k
        for k_refine in range(10):
            k_total += k_refine
            t = estimator.refine(k=k_refine)
            print(abs(tr_A - t) / abs(tr_A) < 1 / np.sqrt(k_total))

def test_DeflatedTraceEstimator():
    n = 100
    k = 10
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    A = A + A.conj().T
    tr_A = np.trace(A)

    for k in [1, 2, 3, 10, 20]:
        estimator = DeflatedTraceEstimator()
        estimator.compute(A, k)
        k_total = k
        for k_refine in range(10):
            k_total += k_refine
            t = estimator.refine(k=k_refine)
            if k_refine > 5:
                assert(abs(tr_A - t) / abs(tr_A) < 5 * n / k_total)
