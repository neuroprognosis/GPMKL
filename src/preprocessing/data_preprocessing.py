"""
 Author:: Nemali Aditya <aditya.nemali@dzne.de>
==================
data preprocessing

==================
This module contains function that perform a transformation (normalization and centering) over data and samples matrices
"""

import numpy as np
from scipy.stats import norm
'''
def normalization(X):
    """normalize a samples matrix (n,m) .. math:: \|X_i\|^2 = 1 \forall i \in [1..n]
    Parameters
    ----------
    X : (n,m) ndarray,
        where *n* is the number of samples and *m* is the number of features.
    Returns
    -------
    Xn : (n,m) ndarray,
         the normalized version of *X*.
    """
    return (X.T / np.linalg.norm(X, ord=2)).T


def centering(X):
    """move the data at the center of axis
    Parameters
    ----------
    X : (n,m) ndarray,
        where *n* is the number of samples and *m* is the number of features.
    Returns
    -------
    Xc : (n,m) ndarray,
         the centered version of *X*.
    """
    n = X.shape[0]
    uno = np.ones((n, 1))
    Xm = 1.0 / n * uno.T @ X
    return X - uno @ Xm
'''

def normalization(X):
    return  (X-X.mean())/(X.std())

def softmax(z, derivative=False):
    """
    Examples
    --------
    >>> z = np.log([1, 2, 5])
    >>> softmax(z)
    array([[ 0.125,  0.25 ,  0.625]])
    >>> z += 100.
    >>> softmax(z)
    array([[ 0.125,  0.25 ,  0.625]])
    >>> softmax(z, derivative=True)
    array([[ 0.109375,  0.1875  ,  0.234375]])
    """
    z = np.atleast_2d(z)
    # avoid numerical overflow by subtracting max
    e = np.exp(z - np.amax(z, axis=1, keepdims=True))
    y = e / np.sum(e, axis=1, keepdims=True)
    if derivative:
        return y * (1. - y) # element-wise
    return y

def one_hot_decision_function(y):
    """
    Examples
    --------
    >>> y = [[0.1, 0.4, 0.5],
    ...      [0.8, 0.1, 0.1],
    ...      [0.2, 0.2, 0.6],
    ...      [0.3, 0.4, 0.3]]
    >>> one_hot_decision_function(y)
    array([[ 0.,  0.,  1.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.]])
    """
    z = np.zeros_like(y)
    z[np.arange(len(z)), np.argmax(y, axis=1)] = 1
    return z

def one_hot(y):
    """Convert `y` to one-hot encoding.
    Examples
    --------
    >>> y = [2, 1, 0, 2, 0]
    >>> one_hot(y)
    array([[ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]])
    """
    n_classes = np.max(y) + 1
    return np.eye(n_classes)[y]

