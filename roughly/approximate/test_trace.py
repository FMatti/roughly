import numpy as np
from trace import HutchinsonTraceEstimator

def test_HutchinsonTraceEstimator():
    n = 100
    k = 10
    A = np.random.randn(n, n) # + 1j * np.random.randn(n, n)
    A = A + A.conj().T
    tr_A = np.trace(A)

    for k in [1, 2, 3, 10, 20]:
        estimator = HutchinsonTraceEstimator(rng="rademacher")
        estimator.compute(A, k)
        k_total = k
        for k_refine in range(10):
            k_total += k_refine
            t = estimator.refine(m=k_refine)
            print(tr_A - t < 10 / np.sqrt(k_total))

# TODO: Fix import issue and add other methods