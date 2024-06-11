![roughly logo](graphics/icon.png)

# roughly

![](https://img.shields.io/badge/-Compatibility-gray?style=flat-square) &ensp;
![](https://img.shields.io/badge/Python_3.8+-white?style=flat-square&logo=python&color=white&logoColor=white&labelColor=gray)

![](https://img.shields.io/badge/-Dependencies-gray?style=flat-square)&ensp;
![](https://img.shields.io/badge/NumPy-white?style=flat-square&logo=numpy&color=white&logoColor=white&labelColor=gray)
![](https://img.shields.io/badge/SciPy-white?style=flat-square&logo=scipy&color=white&logoColor=white&labelColor=gray)

## About

A majority of algorithms in randomized numerical linear algebra are sequential and additive in nature. However, many implementations do not effectively exploit this structure. Often, once an insufficiently accurate result is observed, the computation is restarted from the beginning with a modified parameter set. This results in highly inefficient workflows.

The goal of _roughly_ is to collect the most widespread algorithms of randomized numerical linear algebra and wrap them into an easy to use package where previous computations are stored in memory and available for the user to be reused.

This project is based on the doctoral course [Advanced Scientific Programming in Python](https://github.com/JochenHinz/python_seminar) by Jochen Hinz.

## Example

Suppose you need to compute a basis of the Krylov subspace

$$
\mathcal{K}^{k}(\boldsymbol{A}, \boldsymbol{x}) = \operatorname{span}\left\{ \boldsymbol{x}, \boldsymbol{A}\boldsymbol{x}, \boldsymbol{A}^2\boldsymbol{x}, \dots, \boldsymbol{A}^{k-1}\boldsymbol{x} \right\}.
$$

We do this by running $k$ iterations of the Arnoldi method.

```python
import numpy as np
from roughly.approximate.krylov import ArnoldiDecomposition

arnoldi = ArnoldiDecomposition()

A = np.random.randn(100, 100)  # Example matrix
x = np.random.randn(100)  # Example starting vector
basis, _ = arnoldi.compute(A, x, k=10)
```

After $k$ iterations of the Arnoldi method you proceed with your computations, but realize your basis is not sufficient for these purposes. In these cases, _roughly_ makes it easy to "refine" the approximation with additional iterations.

```python
refined_basis, _ = arnoldi.refine(k=10)
```

The `refine()` attribute also makes convergence studies of the methods easier to compute.

## Usage

To install this package, simply use

```python
pip install https://github.com/FMatti/roughly.git
```

and then import it with

```python
import roughly as rly
```

## Features

Most implementations in roughly also work for linear operator only available as function handles instead of matrices. Currently, roughly implements the Arnoldi, Lanczos, and blocked versions of them; the randomized SVD and Nyström approximation; the randomized range sketch; and the Girard-Hutchinson, subspace projection, and Hutch++ algorithms.
