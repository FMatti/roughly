![roughly logo](graphics/icon.png)

# roughly

![](https://img.shields.io/badge/-Compatibility-gray?style=flat-square) &ensp;
![](https://img.shields.io/badge/Python_3.8+-white?style=flat-square&logo=python&color=white&logoColor=white&labelColor=gray)

![](https://img.shields.io/badge/-Dependencies-gray?style=flat-square)&ensp;
![](https://img.shields.io/badge/NumPy-white?style=flat-square&logo=numpy&color=white&logoColor=white&labelColor=gray)
![](https://img.shields.io/badge/SciPy-white?style=flat-square&logo=scipy&color=white&logoColor=white&labelColor=gray)

## About

A majority of algorithms in randomized numerical linear algebra are sequential and additive in nature. However, many implementations do not effectively exploit this structure. Often, once an insufficiently accurate result is observed, the computation is restarted from the beginning with a modified parameter set. This results in highly inefficient workflows.

The goal of roughly is to collect the most widespread algorithms of randomized numerical linear algebra and wrap them into an easy to use package where previous computations are stored in memory and available for the user to be reused.

This project was based on the course [Advanced Scientific Programming in Python](https://github.com/JochenHinz/python_seminar) by Jochen Hinz.

## Example

Computing a Krylov decomposition using the Arnoldi method:

```[python]
import numpy as np
from roughly.approximate.krylov import ArnoldiDecomposition

decomposition = ArnoldiDecomposition()

A = np.random.randn(100, 100)
X = np.random.randn(100)
U, H = decomposition.compute(A, X, k=10)

# ... do some calculations and realize the decomposition is not sufficient

U, H = decomposition.refine(k=10)
```

## Quick start

### Prerequisites

To install this package, simply use

```[python]
pip install https://github.com/FMatti/roughly.git
```

and then import using

```[python]
import roughly as rly
```

## Features

Most implementations in roughly also work for linear operator only available as function handles instead of matrices. Currently, roughly implements the Arnoldi, Lanczos, and blocked versions of them; the randomized SVD and Nyström approximation; the randomized range sketch; and the Girard-Hutchinson, subspace projection, and Hutch++ algorithms.
