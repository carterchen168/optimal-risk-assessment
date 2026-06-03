import numpy as np
from scipy.linalg import solve_discrete_are

#    Copyright:
#
#            Peter Van Overschee, December 1995
#            peter.vanoverschee@esat.kuleuven.ac.be
#
#    N4SID subspace identification — SIPPY (sippy_unipi) backend.
#    The original manual MATLAB port has been replaced by a call to
#    sippy_unipi.system_identification() with id_method='N4SID'.
#    The 12-value return tuple and function signature are preserved unchanged
#    so that lds_timeseries.py continues to unpack the result without modification.


def _mydisp(sil: int, msg: str):
    if not sil:
        print(msg)


def subid(y, u, i, n=None, AUXin=None, W=None, sil=0):
    """
    General subspace identification via SIPPY (sippy_unipi).

    Parameters
    ----------
    y    : np.ndarray (l, N) — measured outputs
    u    : np.ndarray (m, N) or None/[] — measured inputs (None/[] → stochastic)
    i    : int — number of block rows (Hankel parameter); passed to SIPPY as SS_f
    n    : int or None — model order (None → SIPPY auto-selects)
    AUXin: ignored (legacy caching; SIPPY encapsulates this internally)
    W    : ignored (legacy weighting flag)
    sil  : int — 0 = verbose, 1 = silent

    Returns
    -------
    A, B, C, D  : np.ndarray — state-space matrices (B/D are None for stochastic)
    K           : np.ndarray or None — steady-state Kalman gain (innovations form)
    Ro          : np.ndarray or None — innovation covariance
    AUX         : None (stubbed; SIPPY encapsulates repeated-call optimisation)
    ss          : np.ndarray — zero vector of length n (stubbed singular values)
    Qs          : np.ndarray — process noise covariance from SIPPY
    Ss          : np.ndarray — cross-covariance from SIPPY
    Rs          : np.ndarray — measurement noise covariance from SIPPY
    cvx_flag    : bool — True if identified A has spectral radius >= 1
    """
    _mydisp(sil, ' ')
    _mydisp(sil, '   Subspace Identification (SIPPY/N4SID)')
    _mydisp(sil, '   ------------------------------------')

    # ------------------------------------------------------------------
    # Normalise input orientation to (l, N)
    # ------------------------------------------------------------------
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.shape[1] < y.shape[0]:
        y = y.T
    l, ny = y.shape

    # ------------------------------------------------------------------
    # Detect stochastic (output-only) vs deterministic/combined
    # ------------------------------------------------------------------
    if u is None or (hasattr(u, '__len__') and len(u) == 0):
        ds_flag = 2   # stochastic
    else:
        ds_flag = 1   # deterministic / combined
        if u.ndim == 1:
            u = u.reshape(1, -1)
        if u.shape[1] < u.shape[0]:
            u = u.T

    # ------------------------------------------------------------------
    # Call SIPPY
    # SIPPY uses time-first (N, channels) convention.
    # For the stochastic case a single-channel zero input is passed so that
    # SIPPY's N4SID routine runs; B and D are discarded afterward.
    # ------------------------------------------------------------------
    try:
        import sippy_unipi

        y_in = y.T                                           # (N, l)
        if ds_flag == 2:
            # SIPPY normalises inputs with std(u); passing zeros would divide by zero.
            # A linearly-spaced ramp has non-zero std and is uncorrelated with y,
            # so it does not bias A or C. B/D are discarded for the stochastic return.
            u_in = np.linspace(0.0, 1.0, ny).reshape(-1, 1)
        else:
            u_in = u.T                                       # (N, m)

        n_fixed = n if n is not None else np.nan

        sys_id = sippy_unipi.system_identification(
            y_in, u_in,
            id_method='N4SID',
            SS_fixed_order=n_fixed,
            SS_f=i,
            SS_p=i,
            SS_D_required=(ds_flag == 1),
        )

        A = np.array(sys_id.A)
        C = np.array(sys_id.C)
        n_id = A.shape[0]     # resolved order (may differ from requested n if auto)

        if ds_flag == 2:
            B = None
            D = None
        else:
            B = np.array(sys_id.B)
            D = np.array(sys_id.D)

        # SIPPY returns noise covariances and Kalman gain directly
        K_sippy = sys_id.K if hasattr(sys_id, 'K') else None
        Q_sippy = sys_id.Q if hasattr(sys_id, 'Q') else None
        R_sippy = sys_id.R if hasattr(sys_id, 'R') else None
        S_sippy = sys_id.S if hasattr(sys_id, 'S') else None

    except Exception as exc:
        _mydisp(sil, f'      SIPPY identification failed: {exc}')
        return (None,) * 11 + (False,)

    # ------------------------------------------------------------------
    # Validate / fall back for noise covariances
    # Q_est (n x n), R_est (l x l), Ss (n x l)
    # ------------------------------------------------------------------
    def _valid_matrix(M, shape):
        try:
            M = np.array(M, dtype=float)
            return M if M.shape == shape else None
        except Exception:
            return None

    Q_est = _valid_matrix(Q_sippy, (n_id, n_id))
    R_est = _valid_matrix(R_sippy, (l, l))
    Ss    = _valid_matrix(S_sippy, (n_id, l))

    # Fall back to data-driven estimates when SIPPY omits covariances
    if R_est is None:
        if ny > l:
            R_est = np.cov(y, bias=True) if l > 1 else np.array([[float(np.var(y))]])
        else:
            R_est = np.eye(l) * float(np.var(y.ravel()))
        R_est = (R_est + R_est.T) / 2 + np.eye(l) * 1e-8

    if Q_est is None:
        Q_est = np.eye(n_id) * float(np.mean(np.diag(R_est))) * 1e-2

    if Ss is None:
        Ss = np.zeros((n_id, l))

    # Symmetrise and regularise for positive-definiteness
    R_est = (R_est + R_est.T) / 2 + np.eye(l) * 1e-8
    Q_est = (Q_est + Q_est.T) / 2 + np.eye(n_id) * 1e-10

    # ------------------------------------------------------------------
    # Kalman gain (innovations/predictor form) and innovation covariance
    #
    # Use SIPPY's K when available; otherwise solve DARE:
    #   P = A P A' + Q - A P C' (C P C' + R)^{-1} C P A'
    #   K  = A P C' (C P C' + R)^{-1}
    #   Ro = C P C' + R
    # ------------------------------------------------------------------
    K = _valid_matrix(K_sippy, (n_id, l))

    try:
        P  = solve_discrete_are(A.T, C.T, Q_est, R_est)
        Ro = C @ P @ C.T + R_est
        if K is None:
            K = A @ P @ C.T @ np.linalg.inv(Ro)
    except Exception:
        Ro = None
        if K is None:
            K = None

    # ------------------------------------------------------------------
    # Stability flag — informational; caller (lds_timeseries.py) stabilises
    # via its learn_lds loop when spectral radius >= 1
    # ------------------------------------------------------------------
    cvx_flag_out = bool(np.max(np.abs(np.linalg.eigvals(A))) >= 1)

    # ------------------------------------------------------------------
    # Stub legacy caching / SVD variables
    # AUX_out: None (SIPPY encapsulates repeated-call caching internally)
    # ss_vec : zeros (singular values not exposed by SIPPY's N4SID)
    # ------------------------------------------------------------------
    AUX_out = None
    ss_vec  = np.zeros(n_id)

    if n is None:
        n = n_id   # update caller's order reference if it was auto-selected

    return A, B, C, D, K, Ro, AUX_out, ss_vec, Q_est, Ss, R_est, cvx_flag_out
