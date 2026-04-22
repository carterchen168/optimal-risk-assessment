import os
import time

import numpy as np
from scipy.linalg import (
    svd,
    inv,
    null_space,
    solve_discrete_lyapunov,    # dlyap equivalent
    solve_discrete_are,         # dare equivalent
)

from guess import guess
from ldsparamsidx import lds_params_idx
from learn_kalman import learn_kalman
from inverse_covariance_selection import inverse_covariance_selection


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _is_pos_def(M: np.ndarray) -> bool:
    """
    Return True if M is symmetric positive definite.
    Mirrors MATLAB isposdef() — uses a Cholesky factorisation attempt.
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


def _null(M: np.ndarray) -> np.ndarray:
    """
    Orthonormal basis for the null space of M.
    Returns shape (n, 0) when the null space is trivial.
    Mirrors MATLAB null().
    """
    return null_space(M)


# ---------------------------------------------------------------------------
# Remaining stubs (source files not yet provided)
# ---------------------------------------------------------------------------

def subid(y, u, i, n, *args):
    """
    STUB — N4SID subspace system identification.
    Original MATLAB signature:
      [A, B, C, D, K, Ro, AUX, ss, Q, S, R] = subid(y, u, i, n, [], [], 1)

    Parameters
    ----------
    y : np.ndarray (p, T)   — output sequence
    u : np.ndarray (m, T)   — input sequence
    i : int                 — block-row size (= 2*nmax)
    n : int                 — requested model order

    Returns
    -------
    A, B, C, D, K : np.ndarray or None
    Ro, AUX, ss   : auxiliary outputs
    Q, S, R       : noise covariance estimates (np.ndarray or None)
    """
    raise NotImplementedError(
        "subid() (N4SID) source was not provided. "
        "Implement or import it before calling lds_timeseries()."
    )


def learn_lds(y, n, i, algo):
    """
    STUB — learnLDS: SVD/subspace-based LDS initialiser.
    Original MATLAB signature:
      [A, C, Q, R, Xhat, Ymean] = learnLDS(y, n, i, algo)

    Returns
    -------
    A, C    : np.ndarray
    Q, R    : np.ndarray
    X_hat   : np.ndarray — state estimates
    Y_mean  : np.ndarray — output mean
    """
    raise NotImplementedError(
        "learnLDS() source was not provided. "
        "Implement or import it before calling lds_timeseries()."
    )


def learn_cg_model_em(beta, gamma1, A_prev, arg3, arg4):
    """
    STUB — learnCGModelEM: corrects an unstable A matrix.
    Original MATLAB signature:
      A_stable = learnCGModelEM(beta, gamma1, A_prev, 0, 1)

    Returns
    -------
    A_stable : np.ndarray — stable system matrix
    """
    raise NotImplementedError(
        "learnCGModelEM() source was not provided. "
        "Implement or import it before calling lds_timeseries()."
    )


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def lds_timeseries(params, nmax: int, y: list, u: list, learn_flag: bool):
    """
    Initialise and (optionally) learn an LDS model from multi-segment data.

    Parameters
    ----------
    params     : object
        Configuration namespace. Required fields:
          .initflag  : bool
          .inittype  : int   — 1 = linear regression, 2 = N4SID
          .distrib   : int   — 1 = local (skip directory change)
    nmax       : int
        Maximum / target model order.
    y          : list of np.ndarray, each (p, T_k)
        Output (observation) sequences, one per data segment.
    u          : list of np.ndarray, each (m, T_k), or [] / None
        Input (control) sequences aligned with y. Pass [] or None if absent.
    learn_flag : bool
        If True, run EM Kalman learning; otherwise use initialisation only.

    Returns
    -------
    params : object
        Updated with all learned/initialised LDS fields.
    """
    # ------------------------------------------------------------------
    # 1. Find the first usable data segment
    # ------------------------------------------------------------------
    k = 0
    params.initflag = True

    while params.initflag:
        p_k, ny = y[k].shape

        if u:
            m_k, nu = u[k].shape
        else:
            m_k = 0
            nu  = 0

        i = 2 * nmax

        all_zero_channel = np.any(np.all(y[k].T == 0, axis=0))
        too_short = (ny == 0) or ((ny - 2 * i + 1) < (2 * (m_k + p_k) * i))
        params.initflag = too_short or all_zero_channel

        if k >= len(y) - 1:
            break
        k += 1

    params.initIdx = k   # 0-indexed

    # ------------------------------------------------------------------
    # 2. Extract the chosen segment
    # ------------------------------------------------------------------
    y_seg = y[params.initIdx]
    u_seg = u[params.initIdx] if u else None

    p = y_seg.shape[0]
    m = u_seg.shape[0] if u_seg is not None else 0
    i = 2 * nmax

    # ------------------------------------------------------------------
    # 3. Initialise LDS parameters
    # ------------------------------------------------------------------
    if params.inittype == 1:
        # ---- Linear Regression initialisation ----
        beta, _, _, _ = np.linalg.lstsq(u_seg.T, y_seg.T, rcond=None)  # (m, p)

        U, s, Vt = svd(beta.T, full_matrices=False)  # U:(p,r), s:(r,), Vt:(r,m)
        S_mat = np.diag(s)
        V = Vt.T                                      # (m, r)
        n = nmax

        # C_d and B_d from truncated SVD — mirrors MATLAB if/elseif/else tree
        if n <= m:
            if n <= p:
                cd = U @ S_mat[:, :n]
            elif p <= m and n > p:
                cd = U @ np.hstack([S_mat[:, :p], 0.1 * np.random.randn(p, n - p)])
            else:   # p > m and n > p
                cd = U @ np.vstack([S_mat[:n, :n], 0.1 * np.random.randn(n - p, n)])
            bd = V[:n, :]
        else:       # n > m
            if p <= m:
                cd = U @ np.hstack([S_mat[:p, :p], 0.1 * np.random.randn(p, n - p)])
            else:   # p > m
                top    = np.hstack([S_mat[:m, :m], 0.1 * np.random.randn(m, n - m)])
                bottom = 0.1 * np.random.randn(p - m, n)
                cd = U @ np.vstack([top, bottom])
            bd = np.vstack([V, 0.1 * np.random.randn(n - m, m)])

        params.init.cd = cd
        params.init.bd = bd

        X_hat = u_seg.T @ bd.T                         # (T, n)
        Y_hat = X_hat @ cd.T                           # (T, p)
        Y_diff = y_seg.T - Y_hat                       # (T, p)
        R = (Y_diff.T @ Y_diff) / y_seg.shape[1]

        params.init.ad = 0.9 * np.eye(n)
        params.init.dd = np.zeros((p, m))

        t4 = np.cov(X_hat.T, bias=True)               # (n, n), MATLAB cov(...,1)
        Q  = t4 + np.eye(n) * np.max(np.abs(np.linalg.eigvals(t4))) * 0.01
        nsys = n

    else:
        # ---- N4SID initialisation ----
        nsys = nmax
        (params.init.ad, params.init.bd, params.init.cd, params.init.dd,
         params.init.K, Ro, AUX, ss, Q, S, R) = subid(y_seg, u_seg, i, nsys,
                                                        None, None, 1)

        if params.init.ad is None:
            # subid failed entirely — fall back to random guess
            (params.init.ad, params.init.bd, params.init.cd, params.init.dd,
             Q, R, _, _) = guess(nsys, m, p, y_seg)
            S = np.zeros((Q.shape[0], R.shape[0]))
        else:
            attempt = 0
            while (np.any(np.abs(np.linalg.eigvals(params.init.ad)) >= 1)
                   or Q is None or R is None):
                algo = min(attempt, 6) + 1
                if attempt > 4:
                    (params.init.ad, params.init.bd, params.init.cd,
                     params.init.dd, Q, R, _, _) = guess(nsys, m, p, y_seg)
                else:
                    (params.init.ad, params.init.cd, Q, R,
                     _, _) = learn_lds(y_seg, nsys, i, algo)
                S = np.zeros((Q.shape[0], R.shape[0]))
                attempt += 1

    # ------------------------------------------------------------------
    # 4. Symmetrise and store noise covariances
    # ------------------------------------------------------------------
    params.init.rvd = R
    Q = (Q + Q.T) / 2
    params.init.qwd = Q

    if params.inittype == 2:
        params.init.s = S
        params.init.N = np.block([[Q, S], [S.T, R]])

    # ------------------------------------------------------------------
    # 5. Steady-state covariance and initial state
    # ------------------------------------------------------------------
    params.init.xssd = solve_discrete_lyapunov(params.init.ad, params.init.qwd)

    A_minus_I = np.eye(params.init.ad.shape[0]) - params.init.ad
    if u_seg is not None:
        params.init.initx0 = inv(A_minus_I) @ params.init.bd @ u_seg[:, [0]]
    else:
        ns = _null(A_minus_I)
        if ns.shape[1] > 0:
            params.init.initx0 = ns
        else:
            params.init.initx0 = np.zeros((params.init.ad.shape[0], 1))

    # ------------------------------------------------------------------
    # 6. EM learning (optional)
    # ------------------------------------------------------------------
    if learn_flag:
        max_iter  = np.inf
        diag_q    = False
        diag_r    = False
        ar_mode   = False
        verbose   = False
        asos_flag = False

        t_start = time.process_time()

        # Mirror MATLAB: cd to ASOS dir when not distributed
        original_dir = os.getcwd()
        if params.distrib != 1:
            accept_dir = os.environ.get("ACCEPT_DIR", "")
            os.chdir(os.path.join(accept_dir, "ASOS"))

        params.learned, params.ll, llp = learn_kalman(
            y_seg, params, max_iter, diag_q, diag_r,
            ar_mode, verbose, u_seg, asos_flag
        )

        if params.distrib != 1:
            os.chdir(original_dir)

        elapsed = time.process_time() - t_start
        if verbose:
            print(f"It took {elapsed:.4f} sec to train the LDS")

        k_iters = params.learned.qwd.shape[2]

        # ---- Check stability of final A ----
        if np.max(np.abs(np.linalg.eigvals(params.learned.ad[:, :, -1]))) >= 1:
            if verbose:
                print("Unstable A, using Boots method to find stable matrix!")
            stable_A = learn_cg_model_em(
                params.learned.beta, params.learned.gamma1,
                params.learned.ad[:, :, -1], 0, 1
            )
            params.learned.ad = np.concatenate(
                [params.learned.ad, stable_A[:, :, np.newaxis]], axis=2
            )
            params.ka = k_iters + 1
        else:
            params.ka = k_iters

        # ---- Check positive-definiteness of Q ----
        if not _is_pos_def(params.learned.qwd[:, :, -1]):
            if verbose:
                print("Finding a positive definite Q")
            posdef_Q = inverse_covariance_selection(params.learned.qwd[:, :, -1], 0)
            params.learned.qwd = np.concatenate(
                [params.learned.qwd, posdef_Q[:, :, np.newaxis]], axis=2
            )
            params.kq = k_iters + 1
        else:
            params.kq = k_iters

        # ---- Check positive-definiteness of R ----
        if not _is_pos_def(params.learned.rvd[:, :, -1]):
            if verbose:
                print("Finding a positive definite R")
            posdef_R = inverse_covariance_selection(params.learned.rvd[:, :, -1], 0)
            params.learned.rvd = np.concatenate(
                [params.learned.rvd, posdef_R[:, :, np.newaxis]], axis=2
            )
            params.kr = k_iters + 1
        else:
            params.kr = k_iters

        # ---- Check positive-definiteness of P0 (xssd) ----
        if not _is_pos_def(params.learned.xssd[:, :, -1]):
            if verbose:
                print("Finding a positive definite P0")
            posdef_P = inverse_covariance_selection(params.learned.xssd[:, :, -1], 0)
            params.learned.xssd = np.concatenate(
                [params.learned.xssd, posdef_P[:, :, np.newaxis]], axis=2
            )
            params.kxinit = k_iters + 1
        else:
            params.kxinit = k_iters

        # ---- Extract final parameters via index bookmarks ----
        if u_seg is not None:
            a, q, c, r, params, b, d = lds_params_idx(params)
        else:
            a, q, c, r, params, _, _ = lds_params_idx(params)
            b, d = None, None

        params.initx0 = params.learned.initx0[:, -1]

    else:
        # ---- Learning disabled — use initialisation only ----
        params.learned = params.init
        q = params.learned.qwd
        r = params.learned.rvd
        a = params.learned.ad
        c = params.learned.cd
        if u_seg is not None:
            b = params.learned.bd
            d = params.learned.dd
        else:
            b, d = None, None
        params.initx0 = params.learned.initx0[:, -1]
        params.ll = []
        print("Using initial model only...learning disabled...")

    # ------------------------------------------------------------------
    # 7. Post-process: symmetrise, solve DARE, compute Kalman gain
    # ------------------------------------------------------------------
    q = (q + q.T) / 2
    r = (r + r.T) / 2

    params.xssdl = solve_discrete_lyapunov(a, q)
    params.adl   = a
    params.cdl   = c
    params.rvdl  = r
    params.qwdl  = q
    params.nd    = params.init.N

    if u_seg is not None:
        params.bdl = b
        params.ddl = d

    # Steady-state Kalman filter via DARE:
    #   solve  A'·P·A - P - A'·P·C'·(C·P·C' + R)^{-1}·C·P·A + Q = 0
    params.dare   = solve_discrete_are(a.T, c.T, q, r)
    params.kfgain = params.dare @ c.T @ inv(c @ params.dare @ c.T + r)
    params.pssd   = (np.eye(nsys) - params.kfgain @ params.cdl) @ params.dare

    return params