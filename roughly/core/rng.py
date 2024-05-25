import numpy as np

def gaussian(*dim):
    return np.random.randn(*dim)

def rademacher(*dim):
    return 2 * np.random.randint(0, 2, dim) - 1

def spherical(*dim): 
    x = np.random.randn(*dim)
    x /= np.linalg.norm(x, axis=0)[:, np.newaxis] * np.sqrt(dim[0])
    return x
