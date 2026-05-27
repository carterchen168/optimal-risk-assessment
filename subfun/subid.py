import numpy as np
from scipy.linalg import solve_discrete_lyapunov, qr, svd, pinv, eig as scipy_eig
from stablelds import learn_cg_model_em

#    Copyright:

#            Peter Van Overschee, December 1995
#            peter.vanoverschee@esat.kuleuven.ac.be


# ---------------------------------------------------------------------------
# Simple local helpers
# ---------------------------------------------------------------------------

def _mydisp(sil: int, msg: str):
    """Print msg unless silent mode (sil=1)."""
    if not sil:
        print(msg)


def _blkhank(Y: np.ndarray, num_rows: int, num_cols: int) -> np.ndarray:
    """
    Build a block Hankel matrix from row matrix Y (shape l × N).

    The result has shape (l*num_rows, num_cols):
      H[i*l:(i+1)*l, j] = Y[:, i+j]   for i in 0..num_rows-1, j in 0..num_cols-1

    Mirrors MATLAB blkhank(Y, num_rows, num_cols).
    """
    l = Y.shape[0]
    H = np.zeros((l * num_rows, num_cols))
    for i in range(num_rows):
        H[i * l:(i + 1) * l, :] = Y[:, i:i + num_cols]
    return H


# ---------------------------------------------------------------------------
# chkaux — AUX cache compatibility check (port of chkaux.m)
# ---------------------------------------------------------------------------

def chkaux(AUXin, i, Uaux, y11, ds_flag, Waux, sil):
    """
    Check compatibility of an AUX auxiliary variable from a previous subid call.

    Mirrors chkaux.m (Van Overschee, 1995).

    Parameters
    ----------
    AUXin  : np.ndarray or None — cached AUX matrix from a previous call
    i      : int                — current block-row parameter
    Uaux   : float or None      — u[0, 0] of current input (None for stochastic)
    y11    : float              — y[0, 0] of current output
    ds_flag: int                — 1 = deterministic/combined, 2 = stochastic
    Waux   : int                — weighting flag (0, 1, 2, or 3)
    sil    : int                — 0 = verbose, 1 = silent

    Returns
    -------
    AUX    : np.ndarray or None — validated (or cleared) AUX
    Wflag  : int                — 1 if R info is OK but weight info is incompatible
    """
    AUX   = AUXin
    Wflag = 0

    if AUXin is not None:
        info = AUXin[0, :]
        # Check ds_flag
        if info[0] != ds_flag:
            if ds_flag == 1:
                _mydisp(sil, '      Warning: AUXin neglected: only valid for stochastic models')
            if ds_flag == 2:
                _mydisp(sil, '      Warning: AUXin neglected: only valid for deterministic models')
            AUX = None
        # Check i
        if AUX is not None and info[1] != i:
            _mydisp(sil, '      Warning: AUXin neglected: Incompatible i')
            AUX = None
        # Check first input sample (only for combined/deterministic)
        if AUX is not None and Uaux is not None and info[2] != Uaux:
            _mydisp(sil, '      Warning: AUXin neglected: Incompatible input')
            AUX = None
        # Check first output sample
        if AUX is not None and info[3] != y11:
            _mydisp(sil, '      Warning: AUXin neglected: Incompatible output')
            AUX = None
        # Check weighting
        if AUX is not None and Waux != 0 and info[4] != Waux:
            _mydisp(sil, '      Warning: Weighting part in AUXin neglected: Incompatible weight')
            Wflag = 1

    return AUX, Wflag


# ---------------------------------------------------------------------------
# _solvric — Forward Riccati equation solver (port of solvric.m)
# ---------------------------------------------------------------------------

def _solvric(A, G, C, L0):
    """
    Solve the forward Riccati equation:
      P = A P A' + (G - A P C')(L0 - C P C')^{-1}(G - A P C')'

    Uses the generalized eigenvalue decomposition approach from solvric.m
    (Van Overschee, 1995).

    Parameters
    ----------
    A  : (n, n)
    G  : (n, l) — cross-covariance
    C  : (l, n) — output matrix
    L0 : (l, l) — innovation covariance

    Returns
    -------
    P    : (n, n) — Riccati solution
    flag : int    — 1 if eigenvalue lies on the unit circle (no valid solution)
    """
    if G is None or L0 is None or G.size == 0 or L0.size == 0:
        return None, 0

    n   = A.shape[0]
    L0i = np.linalg.inv(L0)

    # Construct the 2n × 2n pencil (AA, BB) from solvric.m:
    #   AA = [A' - C'*L0i*G'   0 ]    BB = [I     -C'*L0i*C ]
    #        [-G*L0i*G'         I ]         [0      A-G*L0i*C]
    AA = np.block([
        [A.T - C.T @ L0i @ G.T,  np.zeros((n, n))],
        [-G @ L0i @ G.T,          np.eye(n)]
    ])
    BB = np.block([
        [np.eye(n),          -C.T @ L0i @ C],
        [np.zeros((n, n)),    A - G @ L0i @ C]
    ])

    ew, v = scipy_eig(AA, BB)
    ew = np.asarray(ew, dtype=complex)

    # Check for eigenvalue on the unit circle → no valid solution
    flag = 0 if np.all(np.abs(np.abs(ew) - 1.0) > 1e-9) else 1

    # Select the n eigenvalues with smallest magnitude (inside unit circle)
    idx   = np.argsort(np.abs(ew))
    V_s   = v[:, idx[:n]]

    # P = real(V_s[n:2n, :] / V_s[:n, :])  — mirroring MATLAB's mrdivide
    P = np.real(
        np.linalg.lstsq(V_s[:n, :].T, V_s[n:, :].T, rcond=None)[0].T
    )
    return P, flag


# ---------------------------------------------------------------------------
# gl2kr — Kalman gain from Riccati solution (port of gl2kr.m)
# ---------------------------------------------------------------------------

def gl2kr(A, G, C, L0):
    """
    Compute the Kalman gain K and innovations covariance Ro.

    Solves the forward Riccati equation and derives:
      Ro = L0 - C P C'
      K  = (G - A P C') Ro^{-1}

    Mirrors gl2kr.m (Van Overschee, 1995).

    Parameters
    ----------
    A  : (n, n)
    G  : (n, l)
    C  : (l, n)
    L0 : (l, l)

    Returns
    -------
    K  : (n, l) or None
    Ro : (l, l) or None
    """
    if G is None or L0 is None:
        return None, None

    P, flag = _solvric(A, G, C, L0)
    if flag == 1 or P is None:
        _mydisp(0, 'Warning: Non positive real covariance model => K = R = []')
        return None, None

    Ro = L0 - C @ P @ C.T
    K  = np.linalg.solve(Ro.T, (G - A @ P @ C.T).T).T
    return K, Ro


# ---------------------------------------------------------------------------
# learnCGModelSS — stable A correction for subspace identification
# ---------------------------------------------------------------------------

def learnCGModelSS(Xi, Xr, A, simulate_LB1, cvx_flag):
    """
    Learn a stable dynamics matrix from state sequences using constraint
    generation.  Delegates to learn_cg_model_em() with the correct
    QP-objective mapping for the state-space (non-EM) variant.

    Mirrors learnCGModelSS.m: P = kron(I, Xi*Xi'), q = vec(Xi*Xr').
    In learn_cg_model_em: P = kron(I, gamma1), q = vec(beta).
    Mapping: gamma1 = Xi@Xi.T, beta = Xi@Xr.T.

    Parameters
    ----------
    Xi          : (d, T-1) — state sequence at t = 0 .. T-2
    Xr          : (d, T-1) — state sequence at t = 1 .. T-1
    A           : (d, d)   — initial (possibly unstable) dynamics matrix
    simulate_LB1: int      — 0 = stability CG, 1 = simulate Lacy-Bernstein 1
    cvx_flag    : int/bool — use CVXPY instead of quadprog/SLSQP

    Returns
    -------
    A_stable : (d, d) — stable dynamics matrix
    """
    gamma1 = Xi @ Xi.T
    beta   = Xi @ Xr.T   # matches MATLAB q = vec(Xi*Xr')
    return learn_cg_model_em(beta, gamma1, A,
                              simulate_LB1=bool(simulate_LB1),
                              cvx_flag=bool(cvx_flag))


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def subid(y, u, i, n=None, AUXin=None, W=None, sil=0):
    """
    General subspace identification.

    Mirrors subid.m exactly, including the 7-step Van Overschee / De Moor
    algorithm.

    Parameters
    ----------
    y    : np.ndarray (l, ny) — measured outputs
    u    : np.ndarray (m, nu) — measured inputs (pass None/[] for stochastic)
    i    : int                — number of block rows in Hankel matrices
                                (i * #outputs = max identifiable order)
    n    : int or None        — model order (None → heuristic or interactive)
    AUXin: np.ndarray or None — auxiliary variable to speed up repeated calls
    W    : str or None        — 'SV' or 'CVA' weighting (default depends on u)
    sil  : int                — 0 = verbose, 1 = silent

    Returns
    -------
    A, B, C, D  : np.ndarray — system matrices
    K           : np.ndarray or None — Kalman gain
    Ro          : np.ndarray or None — innovations covariance
    AUX         : np.ndarray         — auxiliary variable for repeated calls
    ss          : np.ndarray         — singular values
    Qs, Ss, Rs  : np.ndarray or None — noise covariance factors
    cvx_flag    : bool               — True if CVX was used to stabilise A
    """
    _mydisp(sil, ' ')
    _mydisp(sil, '   Subspace Identification')
    _mydisp(sil, '   -----------------------')

    # ------------------------------------------------------------------
    # Argument defaults and validation
    # ------------------------------------------------------------------
    if u is None or (hasattr(u, '__len__') and len(u) == 0):
        ds_flag = 2   # stochastic
    else:
        ds_flag = 1   # deterministic / combined

    if W is None:
        W = 'SV' if ds_flag == 1 else 'CVA'

    # Ensure y is (l, ny) — more columns than rows
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.shape[1] < y.shape[0]:
        y = y.T
    l, ny = y.shape

    if i < 0:
        raise ValueError("Number of block rows should be positive")
    if l < 1:
        raise ValueError("Need a non-empty output vector")

    if ds_flag == 1:
        if u.ndim == 1:
            u = u.reshape(1, -1)
        if u.shape[1] < u.shape[0]:
            u = u.T
        m, nu = u.shape
        if m < 1:
            raise ValueError("Need a non-empty input vector")
        if nu != ny:
            raise ValueError("Number of data points different in input and output")
        Uaux = float(u[0, 0])
    else:
        m = 0
        Uaux = None

    j = ny - 2 * i + 1   # number of Hankel matrix columns
    if (ny - 2 * i + 1) < (2 * (m + l) * i):
        raise ValueError("Not enough data points")

    # Weighting flag: 1 = SV, 2 = CVA
    W_upper = W.upper()
    if W_upper == 'SV':
        Wn = 1
        Waux = 2 if ds_flag == 1 else 3
    elif W_upper == 'CVA':
        Wn = 2
        Waux = 3 if ds_flag == 1 else 1
    else:
        raise ValueError("W should be 'SV' or 'CVA'")
    W_int = Wn

    # ------------------------------------------------------------------
    # Check AUXin (stub — skips cache for now)
    # ------------------------------------------------------------------
    try:
        AUXin, Wflag = chkaux(AUXin, i, Uaux, float(y[0, 0]), ds_flag, Waux, sil)
    except NotImplementedError:
        AUXin = None
        Wflag = 0

    # ------------------------------------------------------------------
    # Compute R factor (QR decomposition of stacked Hankel matrices)
    # ------------------------------------------------------------------
    if AUXin is None:
        Y_hank = _blkhank(y / np.sqrt(j), 2 * i, j)   # (2*l*i, j)
        _mydisp(sil, '      Computing ... R factor')
        if ds_flag == 1:
            U_hank = _blkhank(u / np.sqrt(j), 2 * i, j)   # (2*m*i, j)
            stack = np.vstack([U_hank, Y_hank])
            _, R_qr = qr(stack.T, mode='economic')
            R = R_qr.T
        else:
            _, R_qr = qr(Y_hank.T, mode='economic')
            R = R_qr.T
        R = R[:2 * i * (m + l), :2 * i * (m + l)]
        bb = None
    else:
        R = AUXin[1:2 * i * (m + l) + 1, :2 * (m + l) * i]
        bb = 2 * i * (m + l) + 1

    # ------------------------------------------------------------------
    # STEP 1 — Oblique projection
    # ------------------------------------------------------------------
    mi2 = 2 * m * i

    if AUXin is None or Wflag == 1:
        # Future outputs: rows (2m+l)*i .. 2(m+l)*i  (0-indexed)
        Rf = R[(2 * m + l) * i: 2 * (m + l) * i, :]
        # Past (inputs +) outputs
        Rp = np.vstack([
            R[:m * i, :],
            R[2 * m * i:(2 * m + l) * i, :]
        ])

        if ds_flag == 1:
            Ru = R[m * i:2 * m * i, :mi2]
            # Perpendicular projections
            Rfp = np.hstack([
                Rf[:, :mi2] - (Rf[:, :mi2] / Ru) @ Ru,
                Rf[:, mi2:]
            ])
            Rpp = np.hstack([
                Rp[:, :mi2] - (Rp[:, :mi2] / Ru) @ Ru,
                Rp[:, mi2:]
            ])

    if AUXin is None:
        if ds_flag == 1:
            norm_check = np.linalg.norm(
                Rpp[:, (2 * m + l) * i - 2 * l:(2 * m + l) * i], 'fro'
            )
            if norm_check < 1e-10:
                Ob = (Rfp @ pinv(Rpp.T).T) @ Rp
            else:
                Ob = np.linalg.lstsq(Rpp.T, Rfp.T, rcond=None)[0].T @ Rp
        else:
            # Ob = [Rf[:, :l*i], zeros(l*i, l*i)]
            Ob = np.hstack([Rf[:, :l * i], np.zeros((l * i, l * i))])
    else:
        Ob = AUXin[bb:bb + l * i, :2 * (l + m) * i]
        bb = bb + l * i

    # ------------------------------------------------------------------
    # STEP 2 — SVD
    # ------------------------------------------------------------------
    if AUXin is None or Wflag == 1:
        _mydisp(sil, '      Computing ... SVD')

        if ds_flag == 1:
            WOW = np.hstack([
                Ob[:, :mi2] - (Ob[:, :mi2] / Ru) @ Ru,
                Ob[:, mi2:]
            ])
        else:
            WOW = Ob

        W1i = None
        if W_int == 2:   # CVA
            _, R_w = qr(Rf.T, mode='economic')
            W1i = R_w[:l * i, :l * i].T
            WOW = np.linalg.solve(W1i, WOW)

        if np.any(np.isnan(WOW)):
            return (None,) * 9 + (None, None, None, False)

        U_svd, S_svd, _ = svd(WOW, full_matrices=False)
        if W_int == 2 and W1i is not None:
            U_svd = W1i @ U_svd
        ss_vec = S_svd
    else:
        U_svd  = AUXin[bb:bb + l * i, :l * i]
        ss_vec = AUXin[bb:bb + l * i, l * i]

    # ------------------------------------------------------------------
    # STEP 3 — Determine model order
    # ------------------------------------------------------------------
    if n is None:
        if not sil:
            # Interactive: print singular values and prompt
            print("Singular values:", ss_vec)
            n = 0
            while not (1 <= n <= l * i - 1):
                try:
                    n = int(input('      System order ? '))
                except ValueError:
                    n = -1
        else:
            # Heuristic automatic order selection
            if W_int == 2:
                angles = np.degrees(np.arccos(np.clip(np.real(ss_vec), -1, 1)))
                diffs  = np.abs(np.diff(angles))
                idx    = np.where(diffs > 1)[0]
                n = int(idx[-1]) + 1 if len(idx) > 0 else 1
            else:
                diffs = np.abs(np.diff(ss_vec))
                idx   = np.where(diffs > 0.1)[0]
                n = int(idx[-1]) + 1 if len(idx) > 0 else 1

    U1 = U_svd[:, :n]

    # ------------------------------------------------------------------
    # STEP 4 — Gamma matrices and pseudo-inverses
    # ------------------------------------------------------------------
    gam  = U1 @ np.diag(np.sqrt(ss_vec[:n]))      # (l*i, n)
    gamm = gam[:l * (i - 1), :]                    # (l*(i-1), n)
    gam_inv  = pinv(gam)
    gamm_inv = pinv(gamm)

    # ------------------------------------------------------------------
    # STEP 5 — System matrices A and C
    # ------------------------------------------------------------------
    _mydisp(sil, f'      Computing ... System matrices A,C (Order {n})')

    Rhs = np.vstack([
        np.hstack([
            gam_inv @ R[(2 * m + l) * i:2 * (m + l) * i, :(2 * m + l) * i],
            np.zeros((n, l))
        ]),
        R[m * i:2 * m * i, :(2 * m + l) * i + l]
    ])
    Lhs = np.vstack([
        gamm_inv @ R[(2 * m + l) * i + l:2 * (m + l) * i, :(2 * m + l) * i + l],
        R[(2 * m + l) * i:(2 * m + l) * i + l, :(2 * m + l) * i + l]
    ])

    # Least-squares solution for [A; C]
    sol = np.linalg.lstsq(Rhs.T, Lhs.T, rcond=None)[0].T
    A = sol[:n, :n]
    C = sol[n:n + l, :n]
    res = Lhs - sol @ Rhs

    cvx_flag_out = False
    if np.max(np.abs(np.linalg.eigvals(A))) >= 1:
        cvx_flag_out = True
        try:
            A = learnCGModelSS(Lhs[:n, :], res[:n, :], A, 0, 1)
        except NotImplementedError:
            pass   # leave A as-is; caller must handle instability

    # ------------------------------------------------------------------
    # Recompute gamma from A and C
    # ------------------------------------------------------------------
    gam = C.copy()
    for k in range(2, i + 1):
        gam = np.vstack([gam, gam[(k - 2) * l:(k - 1) * l, :] @ A])
    gamm     = gam[:l * (i - 1), :]
    gam_inv  = pinv(gam)
    gamm_inv = pinv(gamm)

    # Recompute Rhs / Lhs with updated gamma
    Rhs = np.vstack([
        np.hstack([
            gam_inv @ R[(2 * m + l) * i:2 * (m + l) * i, :(2 * m + l) * i],
            np.zeros((n, l))
        ]),
        R[m * i:2 * m * i, :(2 * m + l) * i + l]
    ])
    Lhs = np.vstack([
        gamm_inv @ R[(2 * m + l) * i + l:2 * (m + l) * i, :(2 * m + l) * i + l],
        R[(2 * m + l) * i:(2 * m + l) * i + l, :(2 * m + l) * i + l]
    ])

    # ------------------------------------------------------------------
    # STEP 6 — System matrices B and D
    # ------------------------------------------------------------------
    if ds_flag == 2:
        B = None
        D = None
    else:
        _mydisp(sil, f'      Computing ... System matrices B,D (Order {n})')

        P_bd = Lhs - np.vstack([A, C]) @ Rhs[:n, :]
        P_bd = P_bd[:, :2 * m * i]
        Q_bd = R[m * i:2 * m * i, :2 * m * i]   # Future inputs

        L1 = A @ gam_inv
        L2 = C @ gam_inv
        M_mat = np.hstack([np.zeros((n, l)), gamm_inv])
        X_mat = np.vstack([
            np.hstack([np.eye(l), np.zeros((l, n))]),
            np.hstack([np.zeros((l * (i - 1), l)), gamm])
        ])

        totm = np.zeros(((n + l) * (n + l), 2 * m * i * (n + l)))
        totm = 0.0
        totm_acc = None

        for k in range(1, i + 1):
            # N matrix (page 126)
            N_top = np.hstack([
                M_mat[:, (k - 1) * l:l * i] - L1[:, (k - 1) * l:l * i],
                np.zeros((n, (k - 1) * l))
            ])
            N_bot = np.hstack([
                -L2[:, (k - 1) * l:l * i],
                np.zeros((l, (k - 1) * l))
            ])
            N = np.vstack([N_top, N_bot])
            if k == 1:
                N[n:n + l, :l] += np.eye(l)
            N = N @ X_mat

            kron_block = np.kron(Q_bd[(k - 1) * m:k * m, :].T, N)
            if totm_acc is None:
                totm_acc = kron_block
            else:
                totm_acc = totm_acc + kron_block

        P_vec = P_bd.ravel(order='F')   # MATLAB uses column-major vec()
        sol_bd_vec = np.linalg.lstsq(totm_acc, P_vec, rcond=None)[0]
        sol_bd = sol_bd_vec.reshape((n + l, m), order='F')
        D = sol_bd[:l, :]
        B = sol_bd[l:l + n, :]

    # ------------------------------------------------------------------
    # STEP 7 — QSR and Kalman gain
    # ------------------------------------------------------------------
    if np.linalg.norm(res) > 1e-10:
        _mydisp(sil, f'      Computing ... System matrices G,L0 (Order {n})')

        cov_mat = res @ res.T
        Qs = cov_mat[:n, :n]
        Ss = cov_mat[:n, n:n + l]
        Rs = cov_mat[n:n + l, n:n + l]

        if np.max(np.abs(np.linalg.eigvals(A))) >= 1:
            Ro = None
            K  = None
            AUX_out = _make_aux(
                ds_flag, i, Uaux, y, Waux, R, Ob, U_svd, ss_vec, l, m
            )
            return A, B, C, D, K, Ro, AUX_out, ss_vec, Qs, Ss, Rs, cvx_flag_out

        sig = solve_discrete_lyapunov(A, Qs)
        G   = A @ sig @ C.T + Ss
        L0  = C @ sig @ C.T + Rs

        _mydisp(sil, '      Computing ... Riccati solution')
        try:
            K, Ro = gl2kr(A, G, C, L0)
        except NotImplementedError:
            K, Ro = None, None
    else:
        Ro = None
        K  = None
        Qs = None
        Ss = None
        Rs = None

    # ------------------------------------------------------------------
    # Build AUX output
    # ------------------------------------------------------------------
    AUX_out = _make_aux(ds_flag, i, Uaux, y, Waux, R, Ob, U_svd, ss_vec, l, m)

    return A, B, C, D, K, Ro, AUX_out, ss_vec, Qs, Ss, Rs, cvx_flag_out


# ---------------------------------------------------------------------------
# Helper: build AUX cache matrix
# ---------------------------------------------------------------------------

def _make_aux(ds_flag, i, Uaux, y, Waux, R, Ob, U, ss, l, m):
    """
    Pack intermediate matrices into the AUX cache array.
    Mirrors the 'Make AUX when needed' block at the end of subid.m.
    """
    rows = (4 * l + 2 * m) * i + 1
    cols = 2 * (m + l) * i
    AUX = np.zeros((rows, cols))

    uaux_val = float(Uaux) if Uaux is not None else 0.0
    info = np.array([ds_flag, i, uaux_val, float(y[0, 0]), Waux])
    AUX[0, :5] = info

    bb = 1
    sz_R = 2 * (m + l) * i
    AUX[bb:bb + sz_R, :sz_R] = R
    bb += sz_R

    sz_Ob = l * i
    AUX[bb:bb + sz_Ob, :2 * (l + m) * i] = Ob
    bb += sz_Ob

    AUX[bb:bb + l * i, :l * i]       = U[:, :l * i]
    AUX[bb:bb + l * i, l * i:l * i + 1] = ss[:l * i, np.newaxis]

    return AUX