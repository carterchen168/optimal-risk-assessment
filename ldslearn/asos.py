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


def vec(X):
    """Flatten to 1-D (mirrors ASOS Toolbox's vec.m). Only ever used inside
    norms and elementwise dot products, both order-invariant, so no
    Fortran-order matching with MATLAB's column-major vec is needed."""
    return np.asarray(X).ravel()


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


# ---------------------------------------------------------------------------
# ApproxEStep — precompute lag-indexed summary statistics (ASOS Toolbox's
# @ApproxEStep/ApproxEStep.m)
# ---------------------------------------------------------------------------


def ApproxEStep(y_, u_, klim, klag, edgesize, insize, outsize):
    """Precompute ASOS lag-indexed summary statistics (ASOS Toolbox's
    @ApproxEStep/ApproxEStep.m), once per sequence, ahead of the per-iteration
    `Step()` calls that consume the returned Struct.

    Ports the brute-force fallback formula that the MATLAB source leaves
    commented out as an FFT-equivalent but slower alternative, rather than
    the FFT path — same output, without the negative-lag index-reversal
    translation risk the FFT path introduces.
    """
    T = y_.shape[1]
    total = klim + 2
    y_y_ = np.zeros((outsize, outsize, total))
    u_u_ = np.zeros((insize, insize, total))
    u_y_ = np.zeros((insize, outsize, total))
    y_u_ = np.zeros((outsize, insize, total))
    for k in range(total):
        for t in range(edgesize, T - k - edgesize):
            y_y_[:, :, k] += y_[:, [t + k]] @ y_[:, [t]].T
            u_u_[:, :, k] += u_[:, [t + k]] @ u_[:, [t]].T
            u_y_[:, :, k] += u_[:, [t + k]] @ y_[:, [t]].T
            y_u_[:, :, k] += y_[:, [t + k]] @ u_[:, [t]].T

    return Struct(
        klim=klim, klag=klag, edgesize=edgesize, T=T,
        y_y_=y_y_, u_u_=u_u_, u_y_=u_y_, y_u_=y_u_,
        y_lead_=y_[:, 0:edgesize], u_lead_=u_[:, 0:edgesize],
        y_trail_=y_[:, T - edgesize:T], u_trail_=u_[:, T - edgesize:T],
        y_pre_=y_[:, edgesize:edgesize + klag],
        u_pre_=u_[:, edgesize:edgesize + klag],
        y_post_=y_[:, T - klag - edgesize:T - edgesize],
        u_post_=u_[:, T - klag - edgesize:T - edgesize],
    )


# ---------------------------------------------------------------------------
# Step — the core ASOS approximate E-step (ASOS Toolbox's
# @ApproxEStep/Step.m)
# ---------------------------------------------------------------------------
#
# Translation note: MATLAB's `Step.m` addresses its lag-indexed 3-D arrays
# (`x_x_`, `y_x_`, ...) via a `F+k` offset, where `F=1` is a pure 1-based
# storage offset and `k` is already a 0-based lag (0..klim+1). Below, `F` is
# dropped and `k` indexes those arrays directly. Position-indexed arrays
# (the klag-length `*_pre_`/`*_post_` slices, and within-edge `*_lead_`/
# `*_trail_` slices) are translated by shifting MATLAB's 1-based loop
# variable into Python's 0-based `range(N)` unchanged — the loop-variable
# shift and the storage-index shift cancel, so `(:, t)`, `(:, t+1)`,
# `(:, t-1)` carry over to `[:, t]`, `[:, t+1]`, `[:, t-1]` literally.
# MATLAB's `(:, end-c)` becomes Python's `[:, -1-c]` via negative indexing.


def Step(o, params):
    """Approximate-E-step update (ASOS Toolbox's @ApproxEStep/Step.m).

    Parameters
    ----------
    o : Struct — ApproxEStep-shaped summary-statistics object, with fields
        klim, klag, edgesize, T, y_y_, u_u_, u_y_, y_u_ (3-D, lag-indexed,
        shape (*, *, klim+2)), y_lead_/u_lead_/y_trail_/u_trail_ (shape
        (*, edgesize)), y_pre_/u_pre_/y_post_/u_post_ (shape (*, klag)).
    params : Struct — pstruct(A, B, C, D, Q, R, pi_1, V_1).

    Returns
    -------
    o : Struct — unchanged (mirrors MATLAB, which never mutates `o`).
    expt : Struct — sufficient statistics (estruct(...)).
    err : float — innovation-based residual.
    LL : float — approximate marginal log-likelihood contribution.
    """
    A, B, C, D, Q, R, insize, outsize, hidsize, pi_1, V_1 = pextract(params)
    pi_1 = np.asarray(pi_1).ravel()

    klim = o.klim
    klag = o.klag
    edgesize = o.edgesize

    y_y_ = o.y_y_
    u_u_ = o.u_u_
    u_y_ = o.u_y_
    y_u_ = o.y_u_

    y_lead_ = o.y_lead_
    u_lead_ = o.u_lead_
    y_trail_ = o.y_trail_
    u_trail_ = o.u_trail_

    y_pre_ = o.y_pre_
    u_pre_ = o.u_pre_
    y_post_ = o.y_post_
    u_post_ = o.u_post_

    T = o.T

    total = klim + 2

    x_x_ = np.zeros((hidsize, hidsize, total))
    x_y_ = np.zeros((hidsize, outsize, total))
    xT_xT_ = np.zeros((hidsize, hidsize, total))
    xT_x_ = np.zeros((hidsize, hidsize, total))
    xT_y_ = np.zeros((hidsize, outsize, total))
    x_u_ = np.zeros((hidsize, insize, total))
    xT_u_ = np.zeros((hidsize, insize, total))

    u_x_ = np.zeros((insize, hidsize, total))
    y_x_ = np.zeros((outsize, hidsize, total))
    x_xT_ = np.zeros((hidsize, hidsize, 2))
    u_xT_ = np.zeros((insize, hidsize, 2))

    V_0_0, V_0_1, V_0_T, V_1_T, K, J, invS = KalmanDoubling(A, Q, C, R, 50)

    CB = C @ B
    CA = C @ A
    H = A - K @ CA
    L = B - K @ CB
    P = np.eye(hidsize) - J @ A
    nKD = -K @ D
    nJB = -J @ B

    LL = 0.0
    err = 0.0

    # -----------------------------------------------------------------
    # Lead and pre first-order stats
    # -----------------------------------------------------------------
    x0_lead_ = np.zeros((hidsize, edgesize))
    V0_lead_ = np.zeros((hidsize, hidsize, edgesize))
    K_lead_ = np.zeros((hidsize, outsize, edgesize))
    J_lead_ = np.zeros((hidsize, hidsize, edgesize))
    x1_lead_ = np.zeros((hidsize, edgesize + 1))
    xT_lead_ = np.zeros((hidsize, edgesize + 1))
    V1_lead_ = np.zeros((hidsize, hidsize, edgesize + 1))
    VT_lead_ = np.zeros((hidsize, hidsize, edgesize + 1))
    Vc_lead_ = np.zeros((hidsize, hidsize, edgesize + 1))

    x1_pre_ = np.zeros((hidsize, klag))
    x0_pre_ = np.zeros((hidsize, klag))
    xT_pre_ = np.zeros((hidsize, klag))

    x1_lead_[:, 0] = pi_1
    V1_lead_[:, :, 0] = V_1
    for t in range(edgesize):
        invS_t = np.linalg.inv(C @ V1_lead_[:, :, t] @ C.T + R)
        K_lead_[:, :, t] = V1_lead_[:, :, t] @ C.T @ invS_t
        V0_lead_[:, :, t] = V1_lead_[:, :, t] - K_lead_[:, :, t] @ C @ V1_lead_[:, :, t]

        y_inov = y_lead_[:, [t]] - C @ x1_lead_[:, [t]] - D @ u_lead_[:, [t]]
        LL -= (y_inov.T @ invS_t @ y_inov).item() / 2
        LL += -outsize * np.log(2 * np.pi) / 2 + logdet(invS_t) / 2
        err += (y_inov.T @ y_inov).item()

        x0_lead_[:, [t]] = x1_lead_[:, [t]] + K_lead_[:, :, t] @ y_inov

        x1_lead_[:, [t + 1]] = A @ x0_lead_[:, [t]] + B @ u_lead_[:, [t]]
        V1_lead_[:, :, t + 1] = A @ V0_lead_[:, :, t] @ A.T + Q

    x1_pre_[:, [0]] = x1_lead_[:, [edgesize]]

    for t in range(klag):
        if t > 0:
            x1_pre_[:, [t]] = A @ x0_pre_[:, [t - 1]] + B @ u_pre_[:, [t - 1]]
        x0_pre_[:, [t]] = x1_pre_[:, [t]] + K @ (
            y_pre_[:, [t]] - C @ x1_pre_[:, [t]] - D @ u_pre_[:, [t]]
        )

    xT_pre_[:, [klag - 1]] = x0_pre_[:, [klag - 1]]
    for t in range(klag - 1, 0, -1):
        xT_pre_[:, [t - 1]] = x0_pre_[:, [t - 1]] + J @ (
            xT_pre_[:, [t]] - x1_pre_[:, [t]]
        )

    xT_lead_[:, [edgesize]] = xT_pre_[:, [0]]
    VT_lead_[:, :, edgesize] = V_0_T
    for t in range(edgesize, 0, -1):
        J_lead_[:, :, t - 1] = V0_lead_[:, :, t - 1] @ A.T @ np.linalg.inv(V1_lead_[:, :, t])
        xT_lead_[:, [t - 1]] = x0_lead_[:, [t - 1]] + J_lead_[:, :, t - 1] @ (
            xT_lead_[:, [t]] - x1_lead_[:, [t]]
        )
        VT_lead_[:, :, t - 1] = V0_lead_[:, :, t - 1] + J_lead_[:, :, t - 1] @ (
            VT_lead_[:, :, t] - V1_lead_[:, :, t]
        ) @ J_lead_[:, :, t - 1].T

    for t in range(1, edgesize + 1):
        Vc_lead_[:, :, t] = VT_lead_[:, :, t] @ J_lead_[:, :, t - 1].T

    # -----------------------------------------------------------------
    # Trail and post first-order stats
    # -----------------------------------------------------------------
    x0_trail_ = np.zeros((hidsize, edgesize))
    V0_trail_ = np.zeros((hidsize, hidsize, edgesize))
    K_trail_ = np.zeros((hidsize, outsize, edgesize))
    J_trail_ = np.zeros((hidsize, hidsize, edgesize))
    x1_trail_ = np.zeros((hidsize, edgesize))
    xT_trail_ = np.zeros((hidsize, edgesize))
    V1_trail_ = np.zeros((hidsize, hidsize, edgesize))
    VT_trail_ = np.zeros((hidsize, hidsize, edgesize))
    Vc_trail_ = np.zeros((hidsize, hidsize, edgesize))

    x1_post_ = np.zeros((hidsize, klag))
    x0_post_ = np.zeros((hidsize, klag))
    xT_post_ = np.zeros((hidsize, klag))

    for t in range(klag):
        if t > 0:
            x1_post_[:, [t]] = A @ x0_post_[:, [t - 1]] + B @ u_post_[:, [t - 1]]
        x0_post_[:, [t]] = x1_post_[:, [t]] + K @ (
            y_post_[:, [t]] - C @ x1_post_[:, [t]] - D @ u_post_[:, [t]]
        )

    if edgesize > 0:
        x1_trail_[:, [0]] = A @ x0_post_[:, [-1]] + B @ u_post_[:, [-1]]
        V1_trail_[:, :, 0] = V_0_1
        for t in range(edgesize):
            if t > 0:
                x1_trail_[:, [t]] = A @ x0_trail_[:, [t - 1]] + B @ u_trail_[:, [t - 1]]
                V1_trail_[:, :, t] = A @ V0_trail_[:, :, t - 1] @ A.T + Q
            invS_t = np.linalg.inv(C @ V1_trail_[:, :, t] @ C.T + R)
            K_trail_[:, :, t] = V1_trail_[:, :, t] @ C.T @ invS_t
            V0_trail_[:, :, t] = V1_trail_[:, :, t] - K_trail_[:, :, t] @ C @ V1_trail_[:, :, t]

            y_inov = y_trail_[:, [t]] - C @ x1_trail_[:, [t]] - D @ u_trail_[:, [t]]
            LL -= (y_inov.T @ invS_t @ y_inov).item() / 2
            LL += -outsize * np.log(2 * np.pi) / 2 + logdet(invS_t) / 2
            err += (y_inov.T @ y_inov).item()

            x0_trail_[:, [t]] = x1_trail_[:, [t]] + K_trail_[:, :, t] @ y_inov

        xT_trail_[:, [-1]] = x0_trail_[:, [-1]]
        VT_trail_[:, :, -1] = V0_trail_[:, :, -1]
        for t in range(edgesize - 1, 0, -1):
            J_trail_[:, :, t - 1] = V0_trail_[:, :, t - 1] @ A.T @ np.linalg.inv(V1_trail_[:, :, t])
            xT_trail_[:, [t - 1]] = x0_trail_[:, [t - 1]] + J_trail_[:, :, t - 1] @ (
                xT_trail_[:, [t]] - x1_trail_[:, [t]]
            )
            VT_trail_[:, :, t - 1] = V0_trail_[:, :, t - 1] + J_trail_[:, :, t - 1] @ (
                VT_trail_[:, :, t] - V1_trail_[:, :, t]
            ) @ J_trail_[:, :, t - 1].T

        xT_post_[:, [-1]] = x0_post_[:, [-1]] + (
            V_0_0 @ A.T @ np.linalg.inv(V1_trail_[:, :, 0])
        ) @ (xT_trail_[:, [0]] - x1_trail_[:, [0]])

        for t in range(1, edgesize):
            Vc_trail_[:, :, t] = VT_trail_[:, :, t] @ J_trail_[:, :, t - 1].T
        Vc_trail_[:, :, 0] = VT_trail_[:, :, 0] @ J.T
    else:
        xT_post_[:, [-1]] = x0_post_[:, [-1]]

    for t in range(klag - 1, 0, -1):
        xT_post_[:, [t - 1]] = x0_post_[:, [t - 1]] + J @ (
            xT_post_[:, [t]] - x1_post_[:, [t]]
        )

    # -----------------------------------------------------------------
    # Approximate u_x / x_u / y_x / x_y cross moments
    # -----------------------------------------------------------------
    u_x_[:, :, klim + 1] = (
        (u_y_[:, :, klim] - u_pre_[:, [klim]] @ y_pre_[:, [0]].T) @ K.T
        + u_u_[:, :, klim + 1] @ L.T
        + (u_u_[:, :, klim] - u_pre_[:, [klim]] @ u_pre_[:, [0]].T) @ nKD.T
        + u_pre_[:, [klim]] @ x0_pre_[:, [0]].T
    ) @ np.linalg.inv(np.eye(hidsize) - H.T)

    for k in range(klim, -1, -1):
        u_x_[:, :, k] = (
            u_x_[:, :, k + 1] @ H.T
            + (u_y_[:, :, k] - u_pre_[:, [k]] @ y_pre_[:, [0]].T) @ K.T
            + u_u_[:, :, k + 1] @ L.T
            + (u_u_[:, :, k] - u_pre_[:, [k]] @ u_pre_[:, [0]].T) @ nKD.T
            + u_pre_[:, [k]] @ x0_pre_[:, [0]].T
        )

    x_u_[:, :, 0] = u_x_[:, :, 0].T
    for k in range(1, klim + 2):
        x_u_[:, :, k] = (
            H @ (x_u_[:, :, k - 1] - x0_post_[:, [-1]] @ u_post_[:, [-k]].T)
            + K @ y_u_[:, :, k]
            + L @ (u_u_[:, :, k - 1] - u_post_[:, [-1]] @ u_post_[:, [-k]].T)
            + nKD @ u_u_[:, :, k]
        )

    y_x_[:, :, klim + 1] = (
        CA @ (-x0_post_[:, [-1]] @ x0_post_[:, [-1 - klim]].T)
        + CB @ (u_x_[:, :, klim] - u_post_[:, [-1]] @ x0_post_[:, [-1 - klim]].T)
        + D @ u_x_[:, :, klim + 1]
    )
    for k in range(klim, -1, -1):
        y_x_[:, :, k] = (
            y_x_[:, :, k + 1] @ H.T
            + (y_y_[:, :, k] - y_pre_[:, [k]] @ y_pre_[:, [0]].T) @ K.T
            + y_u_[:, :, k + 1] @ L.T
            + (y_u_[:, :, k] - y_pre_[:, [k]] @ u_pre_[:, [0]].T) @ nKD.T
            + y_pre_[:, [k]] @ x0_pre_[:, [0]].T
        )

    x_y_[:, :, 0] = y_x_[:, :, 0].T
    for k in range(1, klim + 1):
        x_y_[:, :, k] = (
            H @ (x_y_[:, :, k - 1] - x0_post_[:, [-1]] @ y_post_[:, [-k]].T)
            + K @ y_y_[:, :, k]
            + L @ (u_y_[:, :, k - 1] - u_post_[:, [-1]] @ y_post_[:, [-k]].T)
            + nKD @ u_y_[:, :, k]
        )

    # -----------------------------------------------------------------
    # Fixed-point solve for x_x_(:,:,klim) (Sylvester equation, via doubling)
    # -----------------------------------------------------------------
    const = (
        A @ (-x0_post_[:, [-1]] @ x0_post_[:, [-1 - klim]].T)
        + B @ (u_x_[:, :, klim] - u_post_[:, [-1]] @ x0_post_[:, [-1 - klim]].T)
    ) @ H.T + (x_y_[:, :, klim] - x0_pre_[:, [klim]] @ y_pre_[:, [0]].T) @ K.T \
        + x_u_[:, :, klim + 1] @ L.T \
        + (x_u_[:, :, klim] - x0_pre_[:, [klim]] @ u_pre_[:, [0]].T) @ nKD.T \
        + x0_pre_[:, [klim]] @ x0_pre_[:, [0]].T

    M = np.linalg.matrix_power(H, 2 * klim + 1)
    x_x_[:, :, klim] = np.zeros((hidsize, hidsize))
    tmp = const
    for _ in range(35):
        tmp = SylvDoubling(A, H.T, tmp, 50)
        x_x_[:, :, klim] = x_x_[:, :, klim] + tmp
        if np.linalg.norm(vec(tmp)) < 1e-12 * np.linalg.norm(vec(x_x_[:, :, klim])):
            break
        tmp = M @ tmp.T @ A.T @ C.T @ K.T

    # -----------------------------------------------------------------
    # Propagate the x_x_(:,:,klim) fixed point back into y_x_/x_y_
    # -----------------------------------------------------------------
    M = np.eye(hidsize)
    for k in range(klim, -1, -1):
        M = M @ H
        y_x_[:, :, k] = y_x_[:, :, k] + CA @ x_x_[:, :, klim] @ M.T

    for k in range(0, klim + 1):
        x_y_[:, :, k] = x_y_[:, :, k] + M @ x_x_[:, :, klim].T @ CA.T
        M = M @ H

    for k in range(klim - 1, -1, -1):
        x_x_[:, :, k] = (
            x_x_[:, :, k + 1] @ H.T
            + (x_y_[:, :, k] - x0_pre_[:, [k]] @ y_pre_[:, [0]].T) @ K.T
            + x_u_[:, :, k + 1] @ L.T
            + (x_u_[:, :, k] - x0_pre_[:, [k]] @ u_pre_[:, [0]].T) @ nKD.T
            + x0_pre_[:, [k]] @ x0_pre_[:, [0]].T
        )

    # -----------------------------------------------------------------
    # Smoothed (xT) cross moments
    # -----------------------------------------------------------------
    xT_x_[:, :, klim] = x_x_[:, :, klim]
    for k in range(klim - 1, -1, -1):
        xT_x_[:, :, k] = (
            J @ xT_x_[:, :, k + 1]
            + P @ (x_x_[:, :, k] - x0_post_[:, [-1]] @ x0_post_[:, [-1 - k]].T)
            + nJB @ (u_x_[:, :, k] - u_post_[:, [-1]] @ x0_post_[:, [-1 - k]].T)
            + xT_post_[:, [-1]] @ x0_post_[:, [-1 - k]].T
        )

    x_xT_[:, :, 0] = xT_x_[:, :, 0].T

    xT_u_[:, :, klim] = x_u_[:, :, klim]
    for k in range(klim - 1, -1, -1):
        xT_u_[:, :, k] = (
            J @ xT_u_[:, :, k + 1]
            + P @ (x_u_[:, :, k] - x0_post_[:, [-1]] @ u_post_[:, [-1 - k]].T)
            + nJB @ (u_u_[:, :, k] - u_post_[:, [-1]] @ u_post_[:, [-1 - k]].T)
            + xT_post_[:, [-1]] @ u_post_[:, [-1 - k]].T
        )

    u_xT_[:, :, 0] = xT_u_[:, :, 0].T

    const = J @ (
        (-xT_pre_[:, [0]] @ xT_pre_[:, [0]].T) @ J.T
        + xT_x_[:, :, 1] @ P.T
        + xT_u_[:, :, 1] @ nJB.T
    ) + P @ (x_xT_[:, :, 0] - x0_post_[:, [-1]] @ xT_post_[:, [-1]].T) \
        + nJB @ (u_xT_[:, :, 0] - u_post_[:, [-1]] @ xT_post_[:, [-1]].T) \
        + xT_post_[:, [-1]] @ xT_post_[:, [-1]].T

    xT_xT_[:, :, 0] = LyapDoubling(J, const, 50)
    xT_xT_[:, :, 1] = (
        xT_xT_[:, :, 0] - xT_pre_[:, [0]] @ xT_pre_[:, [0]].T
    ) @ J.T + xT_x_[:, :, 1] @ P.T + xT_u_[:, :, 1] @ nJB.T

    xT_y_[:, :, klim] = x_y_[:, :, klim]
    for k in range(klim - 1, -1, -1):
        xT_y_[:, :, k] = (
            J @ xT_y_[:, :, k + 1]
            + P @ (x_y_[:, :, k] - x0_post_[:, [-1]] @ y_post_[:, [-1 - k]].T)
            + nJB @ (u_y_[:, :, k] - u_post_[:, [-1]] @ y_post_[:, [-1 - k]].T)
            + xT_post_[:, [-1]] @ y_post_[:, [-1 - k]].T
        )

    # -----------------------------------------------------------------
    # Final log-likelihood / residual / sufficient-statistics outputs
    # -----------------------------------------------------------------
    predmat = (
        y_y_[:, :, 0] - y_pre_[:, [0]] @ y_pre_[:, [0]].T
        - 2 * y_x_[:, :, 1] @ CA.T
        + CA @ (x_x_[:, :, 0] - x0_post_[:, [-1]] @ x0_post_[:, [-1]].T) @ CA.T
        - 2 * y_u_[:, :, 1] @ CB.T
        + 2 * CB @ (u_x_[:, :, 0] - u_post_[:, [-1]] @ x0_post_[:, [-1]].T) @ CA.T
        + CB @ (u_u_[:, :, 0] - u_post_[:, [-1]] @ u_post_[:, [-1]].T) @ CB.T
        - 2 * (y_u_[:, :, 0] - y_pre_[:, [0]] @ u_pre_[:, [0]].T) @ D.T
        + 2 * D @ u_x_[:, :, 1] @ CA.T
        + 2 * D @ u_u_[:, :, 1] @ CB.T
        + D @ (u_u_[:, :, 0] - u_pre_[:, [0]] @ u_pre_[:, [0]].T) @ D.T
        + (y_pre_[:, [0]] - C @ x1_pre_[:, [0]] - D @ u_pre_[:, [0]])
        @ (y_pre_[:, [0]] - C @ x1_pre_[:, [0]] - D @ u_pre_[:, [0]]).T
    )

    LL += -0.5 * float(vec(invS) @ vec(predmat))
    LL += (T - 2 * edgesize) * (-outsize * np.log(2 * np.pi) + logdet(invS)) / 2
    err += float(np.trace(predmat))

    Ex_x_0 = (
        xT_xT_[:, :, 0]
        + xT_lead_[:, 0:edgesize] @ xT_lead_[:, 0:edgesize].T
        + xT_trail_ @ xT_trail_.T
        + V_0_T * (T - 2 * edgesize)
        + VT_lead_[:, :, 0:edgesize].sum(axis=2)
        + VT_trail_.sum(axis=2)
    )

    Ex_x_1 = (
        xT_xT_[:, :, 1]
        + xT_lead_[:, 1:edgesize + 1] @ xT_lead_[:, 0:edgesize].T
        + xT_trail_[:, 1:edgesize] @ xT_trail_[:, 0:edgesize - 1].T
        + V_1_T * (T - 2 * edgesize - 1)
        + Vc_lead_.sum(axis=2)
        + Vc_trail_.sum(axis=2)
    )
    if edgesize > 0:
        Ex_x_1 = Ex_x_1 + xT_trail_[:, [0]] @ xT_post_[:, [-1]].T

    Ey_x_0 = xT_y_[:, :, 0].T + y_lead_ @ xT_lead_[:, 0:edgesize].T + y_trail_ @ xT_trail_.T
    Eu_x_0 = u_xT_[:, :, 0] + u_lead_ @ xT_lead_[:, 0:edgesize].T + u_trail_ @ xT_trail_.T

    Ex_u_1 = (
        xT_u_[:, :, 1]
        + xT_lead_[:, 1:edgesize + 1] @ u_lead_.T
        + xT_trail_[:, 1:edgesize] @ u_trail_[:, 0:edgesize - 1].T
    )
    if edgesize > 0:
        Ex_u_1 = Ex_u_1 + xT_trail_[:, [0]] @ u_post_[:, [-1]].T

    Ex_start = xT_lead_[:, 0]
    if edgesize > 0:
        Ex_end = xT_trail_[:, -1]
    else:
        Ex_end = xT_post_[:, -1]

    Exx_start = Ex_start[:, None] @ Ex_start[None, :] + VT_lead_[:, :, 0]
    if edgesize > 0:
        Exx_end = Ex_end[:, None] @ Ex_end[None, :] + VT_trail_[:, :, -1]
    else:
        Exx_end = Ex_end[:, None] @ Ex_end[None, :] + V_0_T

    expt = estruct(
        Ex_x_0, Ex_x_1, Ey_x_0, Eu_x_0, Ex_u_1,
        Struct(start=Ex_start, end=Ex_end),
        Struct(start=Exx_start, end=Exx_end),
        None,
    )

    return o, expt, err, LL


def Step_out(o, in_struct):
    """ASOS iteration step without control inputs.

    Resolution (issue #014): the ASOS Toolbox has only one `Step.m`. Its
    algorithm is dimensionally generic in `insize` — every input-related
    term degrades to a zero-width array when there are no inputs. The
    Python stub split into `Step`/`Step_out` is therefore *not* a distinct
    algorithm: `Step_out` is `Step()` invoked with `insize=0` (`B`, `D`
    zero-width), matching the existing no-input call site
    (`learn_kalman.py`'s `in_struct` carries only `A, C, Q, R, initx, initV`
    — no `B`/`D`).
    """
    A, C, Q, R = in_struct['A'], in_struct['C'], in_struct['Q'], in_struct['R']
    initx, initV = in_struct['initx'], in_struct['initV']

    ss = A.shape[0]
    os = C.shape[0]
    B = np.zeros((ss, 0))
    D = np.zeros((os, 0))

    params = pstruct(A, B, C, D, Q, R, initx, initV)
    return Step(o, params)
