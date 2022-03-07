"""
 Author:: Nemali Aditya <aditya.nemali@dzne.de>
==================
Kernel preprocessing
==================
This module contains function that perform a transformation (normalization and centering) on kernels
"""


import numpy as np
from utils.validation import check_square
from utils.exceptions import SquaredKernelError, LingError
def kernel_normalization(K):
    """normalize a squared kernel matrix
    Parameters
    ----------
    K : (n,n) ndarray,
        the squared kernel matrix.
    Returns
    -------
    Kn : ndarray,
         the normalized version of *K*.
    Notes
    -----
    Given a kernel K, the normalized version is defines as:

    .. math:: \hat{k}(x,z) = \frac{k(x,z)}{\sqrt{k(x,x)\cdot k(z,z)}}
    """
    if(check_square(K)):
        d = K.diagonal().copy().reshape(-1, 1)
        denom = np.dot(d, d.T)
        return K / denom ** 0.5
    else:
        raise SquaredKernelError(K.shape)

def kernel_centering(K):
    """move a squared kernel at the center of axis
    Parameters
    ----------
    K : (n,n) ndarray,
        the squared kernel matrix.

    Returns
    -------
    Kc : ndarray,
         the centered version of *K*.
    """
    if (check_square(K)):
        N = K.shape[0]
        I = np.ones(K.shape)
        C = np.diag(np.full(N, 1)) - (1.0 / N * I)
        return C.dot(K).dot(C)
    else:
        raise SquaredKernelError(K.shape)


def cholesky_factorise(y_cov):
    try:
        L = np.linalg.cholesky(y_cov)
    except:
        raise LingError()
    return L