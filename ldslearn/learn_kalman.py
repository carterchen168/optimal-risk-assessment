import numpy as np
import math
from scipy.linalg import solve_triangular, cho_factor, cho_solve
from em_converged import em_converged


# ---------------------------------------------------------------------------
# Lightweight Struct (mirrors MATLAB dot-notation structs per AGENTS.md §3)
# ---------------------------------------------------------------------------

class Struct:
    """Lightweight MATLAB-style struct for parameter and result payloads."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Parameter struct builders
# ---------------------------------------------------------------------------

def pstruct(A, B, C, D, Q, R, initx, initV):
    """
    Build parameter struct for ExactEstep (with control inputs).

    Parameters
    ----------
    A, B, C, D : np.ndarray — system matrices
    Q, R       : np.ndarray — noise covariances
    initx      : np.ndarray (ss,) or (ss, 1) — initial state mean
    initV      : np.ndarray (ss, ss)          — initial state covariance
    """
    return Struct(A=A, B=B, C=C, D=D, Q=Q, R=R,
                  initx=np.atleast_1d(initx).ravel(),
                  initV=initV)


def pstruct_noinput(A, C, Q, R, initx, initV):
    """
    Build parameter struct for ExactEstep_noinput (no control inputs).
    """
    return Struct(A=A, C=C, Q=Q, R=R,
                  initx=np.atleast_1d(initx).ravel(),
                  initV=initV)


# ---------------------------------------------------------------------------
# Kalman smoother (Kalman forward filter + RTS backward smoother)
# ---------------------------------------------------------------------------

def kalman_smoother(y, A, C, Q, R, initx, initV, **kwargs):
    """
    Kalman forward filter followed by Rauch-Tung-Striebel (RTS) backward
    smoother for a linear dynamical system.

    State model  : x_t = A x_{t-1} + B u_{t-1} + w,  w ~ N(0, Q)
    Output model : y_t = C x_t     + D u_t     + v,  v ~ N(0, R)

    Parameters
    ----------
    y      : np.ndarray (os, T) — observations
    A      : np.ndarray (ss, ss)
    C      : np.ndarray (os, ss)
    Q      : np.ndarray (ss, ss) — process noise covariance
    R      : np.ndarray (os, os) — observation noise covariance
    initx  : np.ndarray (ss,) or (ss, 1) — prior mean of x_0
    initV  : np.ndarray (ss, ss)          — prior covariance of x_0
    **kwargs:
        B : np.ndarray (ss, is) — input matrix (state equation)
        D : np.ndarray (os, is) — feedthrough matrix (output equation)
        u : np.ndarray (is, T)  — input sequence

    Returns
    -------
    xsmooth       : np.ndarray (ss, T)
    Vsmooth       : np.ndarray (ss, ss, T)
    VVsmooth      : np.ndarray (ss, ss, T)   VVsmooth[:,:,t] = Cov(x_t, x_{t-1})
    loglik        : float
    perfect_loglik: float  — placeholder (0.0); TODO: ACCEPT AICS support
    aici_bias_cr  : float  — placeholder (0.0); TODO: ACCEPT AICS support
    """
    B = kwargs.get('B', None)
    D = kwargs.get('D', None)
    u = kwargs.get('u', None)
    has_input = (B is not None) and (u is not None)

    os, T = y.shape
    ss = A.shape[0]

    initx = np.atleast_1d(initx).ravel()     # (ss,)
    initV = np.atleast_2d(initV)             # (ss, ss)
    I_ss  = np.eye(ss)
    REG   = 1e-10 * np.eye(os)              # innovation cov regulariser

    # ------------------------------------------------------------------
    # Storage: forward filter
    # ------------------------------------------------------------------
    x_prior = np.zeros((ss, T))
    V_prior = np.zeros((ss, ss, T))
    x_filt  = np.zeros((ss, T))
    V_filt  = np.zeros((ss, ss, T))

    loglik = 0.0

    for t in range(T):
        # --- Prediction ---
        if t == 0:
            xp = initx.copy()
            Vp = initV.copy()
            if has_input:
                # No u_{-1}; initx already encodes the prior on x_0.
                pass
        else:
            xp = A @ x_filt[:, t - 1]
            if has_input:
                xp = xp + B @ u[:, t - 1]
            Vp = A @ V_filt[:, :, t - 1] @ A.T + Q

        x_prior[:, t]    = xp
        V_prior[:, :, t] = Vp

        # --- Innovation ---
        innov = y[:, t] - C @ xp
        if has_input and D is not None:
            innov = innov - D @ u[:, t]

        # --- Innovation covariance (regularised for stability) ---
        S = C @ Vp @ C.T + R + REG

        # Cholesky solve for numerical stability
        try:
            c_S, low = cho_factor(S, lower=True)
            K        = cho_solve((c_S, low), C @ Vp.T).T   # Vp C' S^{-1}
            innov_norm = float(
                innov @ cho_solve((c_S, low), innov)
            )
            log_det_S = 2.0 * np.sum(np.log(np.abs(np.diag(c_S))))
        except np.linalg.LinAlgError:
            # Fallback to lstsq if Cholesky fails
            K         = np.linalg.lstsq(S.T, (C @ Vp.T).T, rcond=None)[0].T
            innov_norm = float(innov @ np.linalg.lstsq(S, innov, rcond=None)[0])
            sign, log_det_S = np.linalg.slogdet(S)
            if sign <= 0:
                log_det_S = 0.0

        # Log-likelihood contribution
        loglik += -0.5 * (os * math.log(2.0 * math.pi) + log_det_S + innov_norm)

        # --- Update (Joseph stabilised form) ---
        ImKC = I_ss - K @ C
        x_filt[:, t]    = xp + K @ innov
        V_filt[:, :, t] = ImKC @ Vp @ ImKC.T + K @ R @ K.T

    # ------------------------------------------------------------------
    # Storage: backward smoother (RTS)
    # ------------------------------------------------------------------
    xsmooth  = np.zeros((ss, T))
    Vsmooth  = np.zeros((ss, ss, T))
    VVsmooth = np.zeros((ss, ss, T))   # VVsmooth[:,:,t] = Cov(x_t, x_{t-1})

    xsmooth[:, -1]     = x_filt[:, -1]
    Vsmooth[:, :, -1]  = V_filt[:, :, -1]

    for t in range(T - 2, -1, -1):
        Vp_next = V_prior[:, :, t + 1]
        # J = V_filt @ A' @ inv(V_prior_{t+1})
        J = np.linalg.lstsq(Vp_next.T, (A @ V_filt[:, :, t]).T, rcond=None)[0].T

        xsmooth[:, t]    = (x_filt[:, t]
                            + J @ (xsmooth[:, t + 1] - x_prior[:, t + 1]))
        Vsmooth[:, :, t] = (V_filt[:, :, t]
                            + J @ (Vsmooth[:, :, t + 1] - Vp_next) @ J.T)

        # Lagged cross-covariance: Cov(x_{t+1}, x_t)
        VVsmooth[:, :, t + 1] = J @ Vsmooth[:, :, t + 1]

    # TODO: implement perfect_loglik and aici_bias_cr for full ACCEPT AICS support
    perfect_loglik = 0.0
    aici_bias_cr   = 0.0

    return xsmooth, Vsmooth, VVsmooth, loglik, perfect_loglik, aici_bias_cr


# ---------------------------------------------------------------------------
# Exact E-step wrappers (delegates to _estep / _estep_input)
# ---------------------------------------------------------------------------

def ExactEstep_noinput(y, pstruct_obj):
    """
    Exact E-step without control inputs.

    Calls the Kalman smoother, accumulates sufficient statistics, and returns
    them packaged in a Struct matching the field names consumed by the EM loop
    in learn_kalman.

    Parameters
    ----------
    y           : np.ndarray (os, T)
    pstruct_obj : Struct built by pstruct_noinput(A, C, Q, R, initx, initV)

    Returns
    -------
    expt     : Struct with fields:
                 .Ex_x_1  (ss, ss)  — Σ x_t x_{t-1}'
                 .Ex_x_0  (ss, ss)  — Σ x_t x_t'
                 .Ey_x_0  (os, ss)  — Σ y_t x_t'
                 .Ex_     (ss, T)   — smoothed state sequence
                 .Exx_    Struct(.end, .start)  — endpoint outer products
    loglik_t : float
    err      : None
    """
    ps = pstruct_obj
    (beta, gamma, delta, gamma1, gamma2,
     x1, V1, xsmooth, loglik_t, perfect_loglik_t, aici_t) = _estep(
        y, ps.A, ps.C, ps.Q, ps.R,
        ps.initx[:, None], ps.initV, ar_mode=False
    )

    expt = Struct(
        Ex_x_1 = beta,                        # (ss, ss)
        Ex_x_0 = gamma,                       # (ss, ss)
        Ey_x_0 = delta,                       # (os, ss)
        Ex_    = xsmooth,                     # (ss, T)
        Exx_   = Struct(
            end   = gamma - gamma1,           # Σ x_T x_T' + V_T  (end outer product)
            start = gamma - gamma2            # Σ x_0 x_0' + V_0  (start outer product)
        )
    )
    return expt, loglik_t, None


def ExactEstep(y, u, pstruct_obj):
    """
    Exact E-step with control inputs.

    Parameters
    ----------
    y           : np.ndarray (os, T)
    u           : np.ndarray (is, T)
    pstruct_obj : Struct built by pstruct(A, B, C, D, Q, R, initx, initV)

    Returns
    -------
    expt     : Struct with fields as ExactEstep_noinput plus:
                 .Eu_x_0  (is, ss) — Σ_{t=0}^{T-2} u_t x_t'
                 .Ex_u_1  (ss, is) — Σ_{t=1}^{T-1} x_t u_{t-1}' (transposed psi)
    loglik_t : float
    err      : None
    """
    ps = pstruct_obj
    (beta, gamma, delta, gamma1, gamma2,
     xi, psi, x1, V1, xsmooth, loglik_t, perfect_loglik_t, aici_t) = _estep_input(
        y, u, ps.A, ps.B, ps.C, ps.D, ps.Q, ps.R,
        ps.initx[:, None], ps.initV, ar_mode=False
    )

    expt = Struct(
        Ex_x_1 = beta,
        Ex_x_0 = gamma,
        Ey_x_0 = delta,
        Ex_    = xsmooth,
        Exx_   = Struct(
            end   = gamma - gamma1,
            start = gamma - gamma2
        ),
        Eu_x_0 = xi,        # (is, ss)
        Ex_u_1 = psi.T      # (ss, is) — note the transpose matches MATLAB convention
    )
    return expt, loglik_t, None


# ---------------------------------------------------------------------------
# ASOS stubs (deferred — not used by the current ACCEPT pipeline)
# ---------------------------------------------------------------------------

def ApproxEStep(y, u, klim, window, n_particles, is_dim, os_dim):
    """STUB — Approximate E-step (ASOS mode). TODO: ASOS mode"""
    raise NotImplementedError("ApproxEStep() — ASOS mode not yet implemented.")


def Step(newparams_ex, pstruct_obj):
    """STUB — ASOS iteration step with inputs. TODO: ASOS mode"""
    raise NotImplementedError("Step() (ASOS) — not yet implemented.")


def Step_out(newparams_ex, in_struct):
    """STUB — ASOS iteration step without inputs. TODO: ASOS mode"""
    raise NotImplementedError("Step_out() (ASOS) — not yet implemented.")


# ---------------------------------------------------------------------------
# Private E-step helpers (inner functions of learn_kalman.m)
# ---------------------------------------------------------------------------

def _estep(y, A, C, Q, R, initx, initV, ar_mode):
    """
    Compute expected sufficient statistics for a single Kalman smoother pass
    (no control input case).

    Mirrors the inner `Estep` subfunction of learn_kalman.m.

    Parameters
    ----------
    y      : np.ndarray (os, T)
    A      : np.ndarray (ss, ss)
    C      : np.ndarray (os, ss)
    Q      : np.ndarray (ss, ss)
    R      : np.ndarray (os, os)
    initx  : np.ndarray (ss, 1)
    initV  : np.ndarray (ss, ss)
    ar_mode: bool — if True, treat observations as states (Gauss-Markov)

    Returns
    -------
    beta, gamma, delta, gamma1, gamma2 : np.ndarray
    x1     : np.ndarray (ss,)
    V1     : np.ndarray (ss, ss)
    loglik, perfect_loglik, aici_bias_cr : float
    """
    os, T = y.shape
    ss    = A.shape[0]

    if ar_mode:
        xsmooth    = y
        Vsmooth    = np.zeros((ss, ss, T))
        VVsmooth   = np.zeros((ss, ss, T))
        loglik     = 0.0
        perfect_loglik  = 0.0
        aici_bias_cr    = 0.0
    else:
        xsmooth, Vsmooth, VVsmooth, loglik, perfect_loglik, aici_bias_cr = \
            kalman_smoother(y, A, C, Q, R, initx, initV)

    delta  = np.zeros((os, ss))
    gamma  = np.zeros((ss, ss))
    beta   = np.zeros((ss, ss))

    for t in range(T):
        delta += y[:, [t]] @ xsmooth[:, [t]].T
        gamma += xsmooth[:, [t]] @ xsmooth[:, [t]].T + Vsmooth[:, :, t]
        if t > 0:
            beta += xsmooth[:, [t]] @ xsmooth[:, [t - 1]].T + VVsmooth[:, :, t]

    gamma1 = gamma - xsmooth[:, [-1]] @ xsmooth[:, [-1]].T - Vsmooth[:, :, -1]
    gamma2 = gamma - xsmooth[:, [0]]  @ xsmooth[:, [0]].T  - Vsmooth[:, :, 0]

    x1 = xsmooth[:, 0]
    V1 = Vsmooth[:, :, 0]

    return beta, gamma, delta, gamma1, gamma2, x1, V1, xsmooth, loglik, perfect_loglik, aici_bias_cr


def _estep_input(y, u, A, B, C, D, Q, R, initx, initV, ar_mode):
    """
    Compute expected sufficient statistics for a single Kalman smoother pass
    (with control input).

    Mirrors the inner `Estep_input` subfunction of learn_kalman.m.

    Parameters
    ----------
    y      : np.ndarray (os, T)
    u      : np.ndarray (is, T)
    A, B, C, D, Q, R, initx, initV : np.ndarray
    ar_mode: bool

    Returns
    -------
    beta, gamma, delta, gamma1, gamma2 : np.ndarray
    xi, psi : np.ndarray  — input cross-covariance sufficient statistics
    x1 : np.ndarray (ss,)
    V1 : np.ndarray (ss, ss)
    loglik, perfect_loglik, aici_bias_cr : float
    """
    os, T = y.shape
    ss    = A.shape[0]
    is_   = B.shape[1]

    if ar_mode:
        xsmooth  = y
        Vsmooth  = np.zeros((ss, ss, T))
        VVsmooth = np.zeros((ss, ss, T))
        loglik   = 0.0
        perfect_loglik = 0.0
        aici_bias_cr   = 0.0
    else:
        xsmooth, Vsmooth, VVsmooth, loglik, perfect_loglik, aici_bias_cr = \
            kalman_smoother(y, A, C, Q, R, initx, initV, B=B, D=D, u=u)

    delta = np.zeros((os, ss))
    gamma = np.zeros((ss, ss))
    beta  = np.zeros((ss, ss))
    xi    = np.zeros((is_, ss))
    psi   = np.zeros((is_, ss))

    for t in range(T):
        delta += y[:, [t]] @ xsmooth[:, [t]].T
        gamma += xsmooth[:, [t]] @ xsmooth[:, [t]].T + Vsmooth[:, :, t]
        if t > 0:
            beta += xsmooth[:, [t]] @ xsmooth[:, [t - 1]].T + VVsmooth[:, :, t]
            psi  += u[:, [t - 1]] @ xsmooth[:, [t]].T
        if t < T - 1:
            xi += u[:, [t]] @ xsmooth[:, [t]].T

    gamma1 = gamma - xsmooth[:, [-1]] @ xsmooth[:, [-1]].T - Vsmooth[:, :, -1]
    gamma2 = gamma - xsmooth[:, [0]]  @ xsmooth[:, [0]].T  - Vsmooth[:, :, 0]

    x1 = xsmooth[:, 0]
    V1 = Vsmooth[:, :, 0]

    return beta, gamma, delta, gamma1, gamma2, xi, psi, x1, V1, xsmooth, loglik, perfect_loglik, aici_bias_cr


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def learn_kalman(
    dataout,
    params,
    max_iter=10,
    diag_q=False,
    diag_r=False,
    ar_mode=False,
    verbose=True,
    datain=None,
    asos_flag=False,
    constr_fun=None,
):
    """
    Find ML parameters of a stochastic LDS via EM.

    Mirrors learn_kalman.m exactly, including the 3-D iteration-stacking
    convention for all parameter matrices (axis 2 = EM iteration index).

    Parameters
    ----------
    dataout   : list of np.ndarray, each (os, T_l)  — observation sequences
    params    : object
        Must have a `.init` sub-object with fields:
          .ad, .cd, .qwd, .rvd, .initx0, .xssd   — initial LDS matrices
          .bd, .dd                                  — (if datain is not None)
    max_iter  : int or float  — max EM iterations (default 10; pass np.inf for unlimited)
    diag_q    : bool — force Q diagonal (default False)
    diag_r    : bool — force R diagonal (default False)
    ar_mode   : bool — Gauss-Markov (C=I, R=0) mode (default False)
    verbose   : bool — print iteration info (default True)
    datain    : list of np.ndarray, each (is, T_l), or None — input sequences
    asos_flag : bool — use approximate ASOS E-step (default False)
    constr_fun: callable or None — optional constraint function called after M-step

    Returns
    -------
    params : object
        Updated with learned parameter arrays (3-D, axis 2 = iteration).
        Fields added: .ad, .cd, .qwd, .rvd, .initx0, .xssd,
                      .bd, .dd (if inputs given),
                      .delta, .gamma, .gamma1, .gamma2, .beta,
                      .alpha, .P1sum, .x1sum, .Tsum, .condQ
    LL     : list of float — log-likelihood at each EM iteration
    LLp    : list of float — "perfect" log-likelihood trajectory
    """
    thresh          = 1e-6
    check_increased = True

    # ------------------------------------------------------------------
    # Unpack initial parameters from params.init
    # ------------------------------------------------------------------
    # Parameters are stored as 3-D arrays (ss, ss, n_iters) so the full
    # learning history is preserved, matching the MATLAB convention.
    A     = params.init.ad[:, :, np.newaxis].copy()       # (ss, ss, 1)
    C     = params.init.cd[:, :, np.newaxis].copy()       # (os, ss, 1)
    Q     = params.init.qwd[:, :, np.newaxis].copy()      # (ss, ss, 1)
    R     = params.init.rvd[:, :, np.newaxis].copy()      # (os, os, 1)
    initx = np.atleast_1d(params.init.initx0).ravel()[:, np.newaxis].copy()  # (ss, 1)
    initV = params.init.xssd[:, :, np.newaxis].copy()     # (ss, ss, 1)

    params.condQ = np.array([np.linalg.cond(Q[:, :, 0])])

    has_input = datain is not None and len(datain) > 0
    if has_input:
        if not isinstance(datain, list):
            datain = [datain]
        B  = params.init.bd[:, :, np.newaxis].copy()      # (ss, is, 1)
        D  = params.init.dd[:, :, np.newaxis].copy()      # (os, is, 1)
        is_ = B.shape[1]
    else:
        is_ = 0

    if not isinstance(dataout, list):
        dataout = [dataout]

    N  = len(dataout)
    ss = A.shape[0]
    os = C.shape[0]

    # ------------------------------------------------------------------
    # Pre-compute deterministic statistics (observation cross-products)
    # ------------------------------------------------------------------
    if verbose:
        if asos_flag:
            print("Precomputing M-statistics for ASOS...")
        else:
            print("Precomputing deterministic statistics...")

    if has_input:
        zeta  = np.zeros((is_, is_))
        zeta1 = np.zeros((is_, is_))
        eta   = np.zeros((is_, os))
    else:
        ysum  = np.zeros((os, 1))

    alpha = np.zeros((os, os))
    Tsum_pre = 0

    new_params_asos = []   # for ASOS mode

    for ex in range(N):
        y = dataout[ex]
        if not has_input:
            u = None
            ysum += y.sum(axis=1, keepdims=True)
        else:
            u = datain[ex]
        T = y.shape[1]
        Tsum_pre += T

        if asos_flag:
            new_params_asos.append(
                ApproxEStep(y, u, params.klim, 2 * params.klim + 1, 25, is_, os)
            )
        else:
            for t in range(T):
                alpha += y[:, [t]] @ y[:, [t]].T
                if has_input:
                    eta   += u[:, [t]] @ y[:, [t]].T
                    zeta  += u[:, [t]] @ u[:, [t]].T
                    if t < T - 1:
                        zeta1 += u[:, [t]] @ u[:, [t]].T

    if asos_flag:
        alpha = sum(p.y_y_0 for p in new_params_asos)
        if has_input:
            zeta  = sum(p.u_u_0   for p in new_params_asos)
            zeta1 = sum(p.u_u_0_2 for p in new_params_asos)
            eta   = sum(p.u_y_0   for p in new_params_asos)

    # ------------------------------------------------------------------
    # EM loop
    # ------------------------------------------------------------------
    previous_loglik = -np.inf
    loglik          = 0.0
    converged       = False
    decrease        = False
    num_iter        = 1
    LL              = []
    LLp             = []
    aici            = []
    t_iter          = 0   # 0-indexed slice of current parameters (MATLAB t=1)

    while not converged and (num_iter <= max_iter):

        # --------------------------------------------------------------
        # E-step: accumulate sufficient statistics across all sequences
        # --------------------------------------------------------------
        delta  = np.zeros((os, ss))
        gamma  = np.zeros((ss, ss))
        gamma1 = np.zeros((ss, ss))
        gamma2 = np.zeros((ss, ss))
        beta   = np.zeros((ss, ss))
        P1sum  = np.zeros((ss, ss))
        x1sum  = np.zeros((ss, 1))
        loglik = 0.0
        perfect_loglik  = 0.0
        aici_bias_cr    = 0.0
        Tsum            = 0

        if has_input:
            xi  = np.zeros((is_, ss))
            xi1 = np.zeros((is_, ss))
            psi = np.zeros((is_, ss))

        # Slice current parameters (last axis = current iteration index)
        A_t     = A[:, :, t_iter]
        C_t     = C[:, :, t_iter]
        Q_t     = Q[:, :, t_iter]
        R_t     = R[:, :, t_iter]
        initx_t = initx[:, t_iter]
        initV_t = initV[:, :, t_iter]
        if has_input:
            B_t = B[:, :, t_iter]
            D_t = D[:, :, t_iter]

        for ex in range(N):
            y = dataout[ex]
            T = y.shape[1]
            Tsum += T

            if asos_flag:
                if has_input:
                    params_ex, out_ex, err, loglik_t = Step(
                        new_params_asos[ex],
                        pstruct(A_t, B_t, C_t, D_t, Q_t, R_t, initx_t, initV_t)
                    )
                else:
                    in_struct = dict(A=A_t, C=C_t, Q=Q_t, R=R_t,
                                     initx=initx_t, initV=initV_t)
                    params_ex, out_ex, err, loglik_t = Step_out(
                        new_params_asos[ex], in_struct
                    )
            else:
                if has_input:
                    u = datain[ex]
                    expt, loglik_t, err = ExactEstep(
                        y, u,
                        pstruct(A_t, B_t, C_t, D_t, Q_t, R_t, initx_t, initV_t)
                    )
                    beta_t   = expt.Ex_x_1
                    gamma_t  = expt.Ex_x_0
                    delta_t  = expt.Ey_x_0
                    gamma1_t = gamma_t - expt.Exx_.end
                    gamma2_t = gamma_t - expt.Exx_.start
                    xi_t     = expt.Eu_x_0
                    xi1_t    = xi_t - u[:, [-1]] @ expt.Ex_[:, [-1]].T
                    psi_t    = expt.Ex_u_1.T
                    x1       = expt.Ex_[:, 0]
                    V1       = expt.Exx_.start - x1[:, None] @ x1[None, :]
                    perfect_loglik_t = 0.0
                    aici_bias_cr_t   = 0.0

                else:
                    if y.size > 0:
                        expt, loglik_t, err = ExactEstep_noinput(
                            y,
                            pstruct_noinput(A_t, C_t, Q_t, R_t, initx_t, initV_t)
                        )
                        beta_t   = expt.Ex_x_1
                        gamma_t  = expt.Ex_x_0
                        delta_t  = expt.Ey_x_0
                        gamma1_t = gamma_t - expt.Exx_.end
                        gamma2_t = gamma_t - expt.Exx_.start
                        x1       = expt.Ex_[:, 0]
                        V1       = expt.Exx_.start - x1[:, None] @ x1[None, :]
                        perfect_loglik_t = 0.0
                        aici_bias_cr_t   = 0.0
                    else:
                        beta_t   = np.zeros((ss, ss))
                        gamma_t  = np.zeros((ss, ss))
                        delta_t  = np.zeros((os, ss))
                        gamma1_t = np.zeros((ss, ss))
                        gamma2_t = np.zeros((ss, ss))
                        x1       = np.zeros(ss)
                        V1       = np.zeros((ss, ss))
                        loglik_t = 0.0
                        perfect_loglik_t = 0.0
                        aici_bias_cr_t   = 0.0

                beta   += beta_t
                gamma  += gamma_t
                delta  += delta_t
                gamma1 += gamma1_t
                gamma2 += gamma2_t
                P1sum  += V1 + x1[:, None] @ x1[None, :]
                x1sum  += x1[:, None]
                perfect_loglik += perfect_loglik_t
                aici_bias_cr   += aici_bias_cr_t
                loglik         += loglik_t

                if has_input:
                    xi  += xi_t
                    xi1 += xi1_t
                    psi += psi_t

        # Aggregate ASOS outputs if needed
        if asos_flag:
            # (mirroring the MATLAB sum-of-struct-field logic)
            out_list = [out_ex]   # placeholder — real impl collects across ex loop
            gamma  = sum(o.gamma  for o in out_list)
            gamma1 = sum(o.gamma1 for o in out_list)
            gamma2 = sum(o.gamma2 for o in out_list)
            delta  = sum(o.delta  for o in out_list)
            beta   = sum(o.beta   for o in out_list)
            x1sum  = sum(o.x1sum  for o in out_list)
            P1sum  = sum(o.P1sum  for o in out_list)
            klim_vals = [o.klim for o in out_list]
            nonzero   = [k for k in klim_vals if k != 0]
            params.klim = float(np.mean(nonzero)) if nonzero else 0.0
            if has_input:
                xi  = sum(o.xi  for o in out_list)
                xi1 = sum(o.xi1 for o in out_list)
                psi = sum(o.psi for o in out_list)
        else:
            LLp.append(perfect_loglik)
            aici.append(aici_bias_cr)

        LL.append(loglik)
        if verbose:
            print(f"iteration {num_iter}, loglik = {loglik:.6f}")
        num_iter += 1

        # Bail out early if sufficient statistics contain NaN
        if np.any(np.isnan(gamma)) or np.any(np.isnan(delta)):
            converged = True
            params.ad    = A
            params.cd    = C
            params.rvd   = R
            params.qwd   = Q
            params.initx0 = initx
            params.xssd  = initV
            if has_input:
                params.bd = B
                params.dd = D
            return params, LL, LLp

        # --------------------------------------------------------------
        # M-step: update parameters analytically
        # --------------------------------------------------------------
        t_iter += 1            # new slice index
        Tsum1   = Tsum - N     # excludes last time step of each sequence

        if has_input:
            # Joint update of (A, B) and (C, D) via pseudo-inverse
            Lambda_xu  = np.block([[gamma,  xi.T ], [xi,  zeta ]])
            Lambda_xu1 = np.block([[gamma1, xi1.T], [xi1, zeta1]])
            Lambda_y   = np.vstack([delta.T, eta])
            Lambda_x   = np.vstack([beta.T,  psi])

            AB = Lambda_x.T @ np.linalg.pinv(Lambda_xu1)
            CD = Lambda_y.T @ np.linalg.pinv(Lambda_xu)

            A_new = AB[:, :ss]
            B_new = AB[:, ss:]
            C_new = CD[:, :ss]
            D_new = CD[:, ss:]

            Q_est = (gamma2 - AB @ Lambda_x) / Tsum1
            if not np.allclose(Q_est, Q_est.T):
                Q_est = (Q_est + Q_est.T) / 2
                if verbose:
                    print("Making Q a symmetric matrix")

            R_est = (alpha - CD @ Lambda_y) / Tsum
            if not np.allclose(R_est, R_est.T):
                R_est = (R_est + R_est.T) / 2
                if verbose:
                    print("Making R a symmetric matrix")

            A = np.concatenate([A, A_new[:, :, np.newaxis]], axis=2)
            B = np.concatenate([B, B_new[:, :, np.newaxis]], axis=2)
            C = np.concatenate([C, C_new[:, :, np.newaxis]], axis=2)
            D = np.concatenate([D, D_new[:, :, np.newaxis]], axis=2)
            Q = np.concatenate([Q, Q_est[:, :, np.newaxis]], axis=2)
            R = np.concatenate([R, R_est[:, :, np.newaxis]], axis=2)

        else:
            # No-input update
            # A = (gamma1 \ beta')' in MATLAB  →  solve gamma1 @ A.T = beta.T
            A_new = np.linalg.solve(gamma1, beta.T).T
            Q_est = (gamma2 - A_new @ beta.T) / Tsum1
            if not np.allclose(Q_est, Q_est.T):
                Q_est = (Q_est + Q_est.T) / 2
                if verbose:
                    print("Making Q a symmetric matrix")
            if diag_q:
                Q_est = np.diag(np.diag(Q_est))

            if not ar_mode:
                C_new = np.linalg.solve(gamma, delta.T).T
                R_est = (alpha - C_new @ delta.T) / Tsum
                if not np.allclose(R_est, R_est.T):
                    R_est = (R_est + R_est.T) / 2
                    if verbose:
                        print("Making R a symmetric matrix")
                if np.any(np.isnan(R_est)):
                    print("Debug: R_est contains NaN")
                if diag_r:
                    R_est = np.diag(np.diag(R_est))
                C = np.concatenate([C, C_new[:, :, np.newaxis]], axis=2)
                R = np.concatenate([R, R_est[:, :, np.newaxis]], axis=2)
            else:
                C = np.concatenate([C, C[:, :, [-1]]], axis=2)  # keep unchanged
                R = np.concatenate([R, R[:, :, [-1]]], axis=2)

            A = np.concatenate([A, A_new[:, :, np.newaxis]], axis=2)
            Q = np.concatenate([Q, Q_est[:, :, np.newaxis]], axis=2)

        # Update initial state and covariance
        initx_new = x1sum / N
        V_est     = P1sum / N - initx_new @ initx_new.T
        if not np.allclose(V_est, V_est.T):
            V_est = (V_est + V_est.T) / 2
            if verbose:
                print("Making P0 a symmetric matrix")
        initx = np.concatenate([initx, initx_new], axis=1)
        initV = np.concatenate([initV, V_est[:, :, np.newaxis]], axis=2)

        # Optional constraint function
        if constr_fun is not None:
            if has_input:
                (A[:, :, -1], C[:, :, -1], Q[:, :, -1], R[:, :, -1],
                 initx[:, -1], initV[:, :, -1]) = constr_fun(
                    A[:, :, -1], C[:, :, -1], Q[:, :, -1], R[:, :, -1],
                    initx[:, -1], initV[:, :, -1]
                )
            else:
                (A[:, :, -1], C[:, :, -1], Q[:, :, -1], R[:, :, -1],
                 initx[:, -1], initV[:, :, -1]) = constr_fun(
                    A[:, :, -1], C[:, :, -1], Q[:, :, -1], R[:, :, -1],
                    initx[:, -1], initV[:, :, -1]
                )

        params.condQ = np.append(params.condQ, np.linalg.cond(Q[:, :, -1]))

        # Store all current matrices on params (mirrors MATLAB end-of-loop update)
        params.ad    = A
        params.cd    = C
        params.rvd   = R
        params.qwd   = Q
        params.initx0 = initx
        params.xssd  = initV
        if has_input:
            params.bd    = B
            params.dd    = D
            params.xi    = xi
            params.xi1   = xi1
            params.psi   = psi
            params.zeta  = zeta
            params.zeta1 = zeta1
            params.eta   = eta
        params.delta  = delta
        params.gamma  = gamma
        params.gamma1 = gamma1
        params.gamma2 = gamma2
        params.beta   = beta
        params.alpha  = alpha
        params.P1sum  = P1sum
        params.x1sum  = x1sum
        params.Tsum   = Tsum

        # Convergence check
        converged, decrease = em_converged(
            loglik, previous_loglik, thresh, check_increased, verbose
        )
        previous_loglik = loglik
        converged = converged or decrease

        if decrease and verbose:
            print("Likelihood decreased!")
        if (converged or decrease or num_iter >= max_iter) and verbose:
            print("Done!")

    return params, LL, LLp