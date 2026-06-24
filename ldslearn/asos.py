"""
ldslearn/asos.py
------------------------------
Port of the ASOS Toolbox's own dependencies (James Martens, 2011,
Apache 2.0) — small utilities and doubling algorithms that
ApproxEStep/Step (ported separately) builds on.

Source: ASOS Toolbox, `logdet.m`, `pextract.m`, `eextract.m`,
`estruct.m`, `pstruct.m`, `KalmanDoubling.m`, `LyapDoubling.m`,
`SylvDoubling.m`.
"""

import numpy as np


class Struct:
    """Mirrors ldslearn.learn_kalman.Struct; duplicated here (rather than
    imported) to avoid a circular import once learn_kalman.py imports from
    this module for the ASOS EM branch."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def logdet(A):
    """Log-determinant of A, computed via LU/sign decomposition for
    numerical stability (mirrors ASOS Toolbox's logdet.m)."""
    sign, logabsdet = np.linalg.slogdet(A)
    if sign > 0:
        return logabsdet
    return logabsdet + np.log(complex(sign))


def pstruct(A, B, C, D, Q, R, pi_1, V_1):
    """Build a parameter Struct (ASOS Toolbox's pstruct.m)."""
    return Struct(A=A, B=B, C=C, D=D, Q=Q, R=R, pi_1=pi_1, V_1=V_1)


def pextract(params):
    """Unpack a parameter Struct (ASOS Toolbox's pextract.m)."""
    A, B, C, D, Q, R = params.A, params.B, params.C, params.D, params.Q, params.R
    insize = B.shape[1]
    outsize = C.shape[0]
    hidsize = A.shape[0]
    pi_1, V_1 = params.pi_1, params.V_1
    return A, B, C, D, Q, R, insize, outsize, hidsize, pi_1, V_1


def estruct(Ex_x_0, Ex_x_1, Ey_x_0, Eu_x_0, Eu_x_1, Ex_, Exx_, x1_):
    """Build a sufficient-statistics Struct (ASOS Toolbox's estruct.m)."""
    return Struct(Ex_x_0=Ex_x_0, Ex_x_1=Ex_x_1, Ey_x_0=Ey_x_0, Eu_x_0=Eu_x_0,
                  Eu_x_1=Eu_x_1, Ex_=Ex_, Exx_=Exx_, x1_=x1_)


def eextract(expt):
    """Unpack a sufficient-statistics Struct (ASOS Toolbox's eextract.m)."""
    return (expt.Ex_x_0, expt.Ex_x_1, expt.Ey_x_0, expt.Eu_x_0, expt.Eu_x_1,
            expt.Ex_, expt.Exx_, expt.x1_)


# ---------------------------------------------------------------------------
# Doubling algorithms
# ---------------------------------------------------------------------------

def LyapDoubling(E, M, T):
    """Solve the discrete Lyapunov fixed point Theta = E Theta E' + M via
    the doubling algorithm (ASOS Toolbox's LyapDoubling.m)."""
    Phi = E
    Theta = M
    for _ in range(T):
        incr = Phi @ Theta @ Phi.T
        Theta = Theta + incr
        if np.linalg.norm(incr.ravel()) < 1e-6 * np.linalg.norm(Theta.ravel()):
            break
        Phi = Phi @ Phi
    return Theta


def SylvDoubling(E, F, M, T):
    """Solve the Sylvester-type fixed point Theta = E Theta F + M via the
    doubling algorithm (ASOS Toolbox's SylvDoubling.m)."""
    Phi = E
    Upsilon = F
    Theta = M
    for _ in range(T):
        incr = Phi @ Theta @ Upsilon
        Theta = Theta + incr
        if np.linalg.norm(incr.ravel()) < 1e-6 * np.linalg.norm(Theta.ravel()):
            break
        Phi = Phi @ Phi
        Upsilon = Upsilon @ Upsilon
    return Theta


def KalmanDoubling(A, Q, C, R, T):
    """Steady-state Kalman filter/smoother gain quantities via the doubling
    algorithm (ASOS Toolbox's KalmanDoubling.m).

    A, Q : state dynamics matrix and dynamics-noise covariance.
    C, R : observation matrix and observation-noise covariance.
    """
    numhid = A.shape[0]

    # Forwards phase
    Phi = A.T
    Psi = C.T @ np.linalg.inv(R) @ C
    Theta = Q

    for _ in range(T):
        mat = np.linalg.inv(np.eye(numhid) + Psi @ Theta)
        incr = Phi.T @ Theta @ mat @ Phi
        Theta = Theta + incr
        if np.linalg.norm(incr.ravel()) < 1e-8 * np.linalg.norm(Theta.ravel()):
            break
        Psi = Psi + Phi @ mat @ Psi @ Phi.T
        Phi = Phi @ mat @ Phi

    # Asymptotic quantities from forwards phase
    V_0_1 = Theta
    invS = np.linalg.inv(C @ V_0_1 @ C.T + R)
    K = V_0_1 @ C.T @ invS
    V_0_0 = V_0_1 - K @ C @ V_0_1

    J = V_0_0 @ A.T @ np.linalg.inv(V_0_1)
    D = V_0_0 - J @ V_0_1 @ J.T

    # Backwards phase
    Phi = J.T
    Theta = D
    for _ in range(T):
        incr = Phi.T @ Theta @ Phi
        Theta = Theta + incr
        if np.linalg.norm(incr.ravel()) < 1e-8 * np.linalg.norm(Theta.ravel()):
            break
        Phi = Phi @ Phi
    V_0_T = Theta

    # Cross phase
    V_1_T = V_0_T @ J.T

    return V_0_0, V_0_1, V_0_T, V_1_T, K, J, invS
