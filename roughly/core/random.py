"""
random.py
---------

Random tensor generators.
"""
import numpy as np

def gaussian(*dim):
    """
    Standard Gaussian random tensor, where every entry follows the standard 
    normal distribution.

    Parameters
    ----------
    d0, d1, ..., dn : int
        The dimensions of the Gaussian random tensor.

    Returns 
    -------
    np.ndarray of shape (d0, d1, ..., dn)
        The Gaussian random tensor.
    """
    return np.random.randn(*dim)

def rademacher(*dim):
    """
    Rademacher random tensor, where every entry is sampled from {-1, 1} with
    equal probability.

    Parameters
    ----------
    d0, d1, ..., dn : int
        The dimensions of the Rademacher random tensor.

    Returns 
    -------
    np.ndarray of shape (d0, d1, ..., dn)
        The Rademacher random tensor.
    """
    return 2 * np.random.randint(0, 2, dim) - 1

def spherical(*dim): 
    """
    Spherically normalized Gaussian random tensor, where every entry is sampled
    from a standard normal distribution, and subsequently normalized with the
    norm along the last axis and its dimension.

    Parameters
    ----------
    d0, d1, ..., dn : int
        The dimensions of the spherical Gaussian random tensor.

    Returns 
    -------
    x : np.ndarray of shape (d0, d1, ..., dn)
        The spherical Gaussian random tensor.
    """
    x = np.random.randn(*dim)
    x /= np.linalg.norm(x, axis=0)[:, np.newaxis] * np.sqrt(dim[0])
    return x
