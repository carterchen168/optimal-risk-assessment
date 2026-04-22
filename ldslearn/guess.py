"""
guess.py
========
Python equivalent of guess.m (ACCEPT idslearn subfolder).

PURPOSE
-------
Generates random initial-guess matrices for the EM LDS algorithm.
The diagonal form for A_g (instead of a dense randn matrix) was introduced
to help enforce stability of the initial model.
(Comment by Rodney A. Martin, 6/4/2010)

DEPENDENCIES
------------
    numpy
"""

import numpy as np


def guess(
    k: int,
    m: int,
    n: int,
    yt: np.ndarray,
):
    """
    Generate random initial-guess matrices for the EM LDS algorithm.

    Parameters
    ----------
    k  : int            — latent state dimension
    m  : int            — input dimension
    n  : int            — observation dimension
    yt : np.ndarray, shape (n, T)
         Observation matrix (columns = time steps).
         Used only to compute R_g via sample covariance.

    Returns
    -------
    A_g  : np.ndarray (k, k)  — diagonal, uniform random, stabilising guess
    B_g  : np.ndarray (k, m)  — random Gaussian
    C_g  : np.ndarray (n, k)  — random Gaussian
    D_g  : np.ndarray (n, m)  — random Gaussian
    Q_g  : np.ndarray (k, k)  — identity (process noise covariance)
    R_g  : np.ndarray (n, n)  — sample covariance of yt (obs noise covariance)
    x0_g : np.ndarray (k, 1)  — zero initial state
    P0_g : np.ndarray (k, k)  — identity initial state covariance
    """
    # Diagonal A with entries in [0, 1) keeps spectral radius < 1 (stability)
    A_g  = np.diag(np.random.rand(k))
    B_g  = np.random.randn(k, m)
    C_g  = np.random.randn(n, k)
    D_g  = np.random.randn(n, m)
    Q_g  = np.eye(k)
    # np.cov uses (T-1) denominator by default — matches MATLAB cov()
    R_g  = np.cov(yt)       # yt is (n, T); numpy cov operates over rows
    P0_g = np.eye(k)
    x0_g = np.zeros((k, 1))
    return A_g, B_g, C_g, D_g, Q_g, R_g, x0_g, P0_g