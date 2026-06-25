import numpy as np
from .learnCGModelEM import learn_cg_model_em


# ---------------------------------------------------------------------------
# learnCGModel — constraint-generation stable dynamics (state-sequence form)
# ---------------------------------------------------------------------------

def learnCGModel(S1, S2, simulate_LB1):
    """
    Learn a stable LDS dynamics matrix from state sequences using constraint
    generation.

    Delegates to learn_cg_model_em() with the correct QP-objective mapping.

    From learnCGModel.m:
      P = kron(I, S1*S1'),  q = vec((S2*S1')') = vec(S1*S2')
      initial M = pinv(S1') * S2'

    Mapping to learn_cg_model_em:
      gamma1 = S1 @ S1.T   (matches P = kron(I, gamma1))
      beta   = S1 @ S2.T   (matches q = vec(beta) = vec(S1*S2'))
      A_init = pinv(S1.T) @ S2.T   (unconstrained LS solution)

    Parameters
    ----------
    S1          : np.ndarray (n, T-1) — states at t = 0..T-2
    S2          : np.ndarray (n, T-1) — states at t = 1..T-1
    simulate_LB1: int                 — 0 = stability CG, 1 = simulate LB-1

    Returns
    -------
    Ahat : np.ndarray (n, n) — stable dynamics matrix
    """
    gamma1 = S1 @ S1.T
    beta   = S1 @ S2.T                        # matches MATLAB q = vec(S1*S2')
    A_init = np.linalg.pinv(S1.T) @ S2.T      # unconstrained MLE (MATLAB pinv(S1')*S2')
    return learn_cg_model_em(beta, gamma1, A_init,
                              simulate_LB1=bool(simulate_LB1), cvx_flag=False)


# ---------------------------------------------------------------------------
# learnLB1Model — Lacy-Bernstein 1 (SDP via CVXPY)
# ---------------------------------------------------------------------------

def learnLB1Model(S1, S2):
    """
    Learn a stable LDS dynamics matrix using the Lacy-Bernstein 1 algorithm.

    Implements the SDP from:
      Lacy & Bernstein, "Subspace Identification with Guaranteed Stability
      using Constrained Optimization", ACC 2002.

    Requires CVXPY with an SDP-capable solver (e.g. SCS or MOSEK).

    Parameters
    ----------
    S1 : np.ndarray (n, T-1) — states at t = 0..T-2
    S2 : np.ndarray (n, T-1) — states at t = 1..T-1

    Returns
    -------
    Ahat : np.ndarray (n, n) — stable dynamics matrix
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "learnLB1Model() requires cvxpy (pip install cvxpy). "
            "TODO: implement without CVXPY."
        ) from exc

    n, L = S1.shape

    # --- helper: projection onto orthogonal complement of col(M) ---
    def piperp(M):
        # M is (2n, L); projects onto null space of M rows
        MM = M @ M.T
        return np.eye(M.shape[1]) - M.T @ np.linalg.pinv(MM) @ M

    # --- commutation (vec-permutation) matrix permtrans(m, n) ---
    def permtrans(m, n):
        P = np.zeros((m * n, m * n))
        inds = np.reshape(np.arange(m * n), (n, m)).T.ravel()
        for i in range(m * n):
            P[i, inds[i]] = 1
        return P

    U = np.zeros_like(S1)    # (n, L) zero input
    SU = np.vstack([S1, U])  # (2n, L)

    SU_R = SU.T @ np.linalg.pinv(SU @ SU.T)  # (L, 2n)

    # Build the block Ax matrix (see learnLB1Model.m for derivation)
    t1  = np.zeros((n * L, 1))
    t2  = np.kron(np.eye(n), piperp(SU)) @ permtrans(L, n)
    t3  = np.zeros((n * L, n * n))
    t4  = t3.copy()
    t5  = np.zeros((n * L, 4 * n * n))
    t6  = np.zeros((n * n, 1))
    t7  = permtrans(n, n) @ np.kron(np.eye(n), (SU_R @ np.vstack([np.eye(n), np.zeros((n, n))])).T) @ permtrans(L, n)
    t8  = np.zeros((n * n, n * n))
    t9  = t8.copy()
    t10 = np.hstack([np.zeros((n * n, n * n)), np.eye(n * n)]) @ np.kron(np.eye(2 * n), np.hstack([np.eye(n), np.zeros((n, n))]))
    t11 = t6.copy()
    t12 = t7.copy()
    t13 = t8.copy()
    t14 = t8.copy()
    t15 = permtrans(n, n) @ np.hstack([np.eye(n * n), np.zeros((n * n, n * n))]) @ np.kron(np.eye(2 * n), np.hstack([np.zeros((n, n)), np.eye(n)]))
    t16 = t6.copy()
    t17 = np.zeros((n * n, L * n))
    t18 = np.eye(n * n)
    t19 = t8.copy()
    t20 = -np.hstack([np.eye(n * n), np.zeros((n * n, n * n))]) @ np.kron(np.eye(2 * n), np.hstack([np.eye(n), np.zeros((n, n))]))
    t21 = t6.copy()
    t22 = t17.copy()
    t23 = t18.copy()
    t24 = t18.copy()
    t25 = np.zeros((n * n, 4 * n * n))
    t26 = t6.copy()
    t27 = t17.copy()
    t28 = t8.copy()
    t29 = t8.copy()
    t30 = np.hstack([np.zeros((n * n, n * n)), np.eye(n * n)]) @ np.kron(np.eye(2 * n), np.hstack([np.zeros((n, n)), np.eye(n)]))

    Ax = np.block([
        [t1,  t2,  t3,  t4,  t5 ],
        [t6,  t7,  t8,  t9,  t10],
        [t11, t12, t13, t14, t15],
        [t16, t17, t18, t19, t20],
        [t21, t22, t23, t24, t25],
        [t26, t27, t28, t29, t30],
    ])

    def vec(M):
        return M.ravel(order='F')

    tt1 = vec(piperp(SU) @ S2.T)
    tt2 = vec(S2 @ SU_R @ np.vstack([np.eye(n), np.zeros((n, n))]))
    tt3 = tt2.copy()
    tt4 = np.zeros(n * n)
    tt5 = vec(np.eye(n))
    tt6 = tt5.copy()

    bx = np.concatenate([tt1, tt2, tt3, tt4, tt5, tt6])

    N   = 6 * n * n + n * L + 1
    cx  = np.zeros(N)
    cx[0] = 1.0

    # Variable index ranges
    z1_inds = [0]
    z2_inds = list(range(1, 1 + n * L))
    z3_inds = list(range(z2_inds[-1] + 1, z2_inds[-1] + 1 + n * n))
    z4_inds = list(range(z3_inds[-1] + 1, z3_inds[-1] + 1 + n * n))
    z5_inds = list(range(z4_inds[-1] + 1, z4_inds[-1] + 1 + 4 * n * n))

    z  = cp.Variable(N)
    constraints = [
        Ax @ z == bx,
        z[z1_inds[0]] >= cp.norm(z[z2_inds]),
        cp.reshape(z[z3_inds], (n, n), order='F') >> 0,
        cp.reshape(z[z4_inds], (n, n), order='F') >> 0,
        cp.reshape(z[z5_inds], (2 * n, 2 * n), order='F') >> 0,
    ]
    prob = cp.Problem(cp.Minimize(cx @ z), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ('optimal', 'optimal_inaccurate') or z.value is None:
        # Fallback: least-squares (possibly unstable)
        return S2 @ np.linalg.pinv(S1)

    Z5  = z.value[z5_inds].reshape((2 * n, 2 * n), order='F')
    Ahat = Z5[:n, n:2 * n]
    return Ahat


# ---------------------------------------------------------------------------
# learnLB2Model — Lacy-Bernstein 2 (SDP via CVXPY)
# ---------------------------------------------------------------------------

def learnLB2Model(S1, S2):
    """
    Learn a stable LDS dynamics matrix using the Lacy-Bernstein 2 algorithm.

    Implements the SDP from:
      Lacy & Bernstein, "Subspace Identification with Guaranteed Stability
      using Constrained Optimization", IEEE TAC 2003.

    Ported directly from learnLB2Model.m (vectorised linear-equality + SOC +
    PSD form), mirroring the translation style used in learnLB1Model() above
    — not the (non-convex, A @ Z product-of-variables) formulation this
    function previously had.

    Parameters
    ----------
    S1 : np.ndarray (n, T-1) — states at t = 0..T-2
    S2 : np.ndarray (n, T-1) — states at t = 1..T-1

    Returns
    -------
    Ahat : np.ndarray (n, n) — stable dynamics matrix
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "learnLB2Model() requires cvxpy (pip install cvxpy). "
            "TODO: implement without CVXPY."
        ) from exc

    n = S1.shape[0]
    delta = 0.001   # from learnLB2Model.m

    U = np.zeros_like(S1)            # (n, L) zero input
    tmp1 = np.vstack([S1, U])        # (2n, L)
    term2 = tmp1.T @ np.linalg.pinv(tmp1 @ tmp1.T)   # (L, 2n)
    term3 = S2 @ term2                                # (n, 2n)
    X1 = term3[:, :n]                                 # (n, n)

    def vec(M):
        return M.ravel(order='F')

    # --- block Ax matrix (see learnLB2Model.m for derivation) ---
    t1 = np.zeros((n * n, 1))
    t2 = -np.eye(n * n)
    t3 = np.kron(np.hstack([np.zeros((n, n)), np.eye(n)]),
                 np.hstack([-np.eye(n), X1]))          # (n*n, 4*n*n)

    t4 = np.zeros((n * n, 1))
    t5 = np.zeros((n * n, n * n))
    t6 = (np.kron(np.hstack([np.zeros((n, n)), np.eye(n)]),
                   np.hstack([np.zeros((n, n)), np.eye(n)]))
          - np.kron(np.hstack([np.eye(n), np.zeros((n, n))]),
                     np.hstack([np.eye(n), np.zeros((n, n))])))

    Ax = np.block([
        [t1, t2, t3],
        [t4, t5, t6],
    ])

    bx = delta * np.concatenate([np.zeros(n * n), vec(np.eye(n))])

    N = 5 * n * n + 1   # 4*n*n (z3) + n*n (z2) + 1 (z1)

    z1_inds = [0]
    z2_inds = list(range(1, 1 + n * n))
    z3_inds = list(range(z2_inds[-1] + 1, z2_inds[-1] + 1 + 4 * n * n))

    cx = np.zeros(N)
    cx[0] = 1.0   # lambda = 0 → no regularisation term on P's diagonal

    z = cp.Variable(N)
    constraints = [
        Ax @ z == bx,
        z[z1_inds[0]] >= cp.norm(z[z2_inds]),
        cp.reshape(z[z3_inds], (2 * n, 2 * n), order='F') >> 0,
    ]
    prob = cp.Problem(cp.Minimize(cx @ z), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ('optimal', 'optimal_inaccurate') or z.value is None:
        # Fallback: least-squares (possibly unstable)
        return S2 @ np.linalg.pinv(S1)

    Z3 = z.value[z3_inds].reshape((2 * n, 2 * n), order='F')
    P = Z3[n:2 * n, n:2 * n]
    Q = Z3[:n, n:2 * n]
    Ahat = Q @ np.linalg.inv(P)
    return Ahat


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def learn_lds(Y, n, d, algo=1):
    """
    Learn an LDS from observations using Hankel-SVD subspace identification.

    Parameters
    ----------
    Y    : np.ndarray (m, t) — m-dimensional observations, t time steps
    n    : int               — latent state dimension
    d    : int               — number of observations per Hankel column
                               (controls temporal context window)
    algo : int               — dynamics matrix learning method:
             1 — Constraint Generation (stable, default)
             2 — Least Squares (fast, may be unstable)
             3 — Lacy-Bernstein 1 (stable)
             4 — Lacy-Bernstein 1 simulated via CG
             5 — Lacy-Bernstein 2 (stable)

    Returns
    -------
    Ahat  : np.ndarray (n, n) — dynamics matrix
    Chat  : np.ndarray (m, n) — observation matrix
    Qhat  : np.ndarray (n, n) — process noise covariance
    Rhat  : np.ndarray (m, m) or None
            Observation noise covariance.
            Not computed when m > 100 (matches original behaviour).
    Xhat  : np.ndarray (n, t-d+1) — latent state sequence estimate
    Ymean : np.ndarray (m, 1)     — observation mean (must be added back
                                     when simulating from the model)
    """
    m, t = Y.shape

    # ------------------------------------------------------------------
    # 1. Subtract mean
    # ------------------------------------------------------------------
    Ymean = Y.mean(axis=1, keepdims=True)   # (m, 1)
    Y = Y - Ymean                            # zero-mean observations

    # ------------------------------------------------------------------
    # 2. Build Hankel matrix  D  of shape (d*m, tau)
    #    D[(i-1)*m : i*m, j] = Y[:, j + i]   (0-indexed, i in 1..d, j in 0..tau-1)
    # ------------------------------------------------------------------
    tau = t - d + 1
    D = np.zeros((d * m, tau))
    for i in range(1, d + 1):
        for j in range(tau):
            D[(i - 1) * m: i * m, j] = Y[:, (j) + (i - 1)]
    D = D.T   # (tau, d*m)

    # ------------------------------------------------------------------
    # 3. SVD of Hankel matrix
    #    MATLAB branch: if #cols < #rows → use economy SVD on D directly,
    #                   else           → use economy SVD on D'
    #    In both cases: V (right singular vectors → state basis)
    #                   U (left  singular vectors → observation basis Chat)
    # ------------------------------------------------------------------
    if D.shape[1] < D.shape[0]:           # more rows than cols
        V_full, S_full, Ut_full = np.linalg.svd(D, full_matrices=False)
        # MATLAB: [V,S,U] = svd(D, 0)  → V is left, U is right
        V = V_full[:, :n]                 # (tau, n)
        S = np.diag(S_full[:n])           # (n, n)
        U = Ut_full[:n, :].T             # (d*m, n)
    else:                                 # more cols than rows (transposed path)
        U_full, S_full, Vt_full = np.linalg.svd(D.T, full_matrices=False)
        # MATLAB: [U,S,V] = svd(D', 0)
        V = Vt_full[:n, :].T             # (tau, n)
        S = np.diag(S_full[:n])           # (n, n)
        U = U_full[:, :n]                 # (d*m, n)

    Xhat = S @ V.T                        # (n, tau) — state sequence
    Chat = U[:m, :]                       # (m, n)   — observation matrix

    # ------------------------------------------------------------------
    # 4. Estimate dynamics matrix
    # ------------------------------------------------------------------
    S1 = Xhat[:, :-1]   # (n, tau-1) — states at t = 0..tau-2
    S2 = Xhat[:, 1:]    # (n, tau-1) — states at t = 1..tau-1

    if algo == 1:       # Constraint Generation (stable)
        Ahat = learnCGModel(S1, S2, 0)

    elif algo == 2:     # Least Squares (may be unstable)
        Ahat = S2 @ np.linalg.pinv(S1)

    elif algo == 3:     # Lacy-Bernstein 1
        Ahat = learnLB1Model(S1, S2)

    elif algo == 4:     # Lacy-Bernstein 1 simulated via CG
        Ahat = learnCGModel(S1, S2, 1)

    elif algo == 5:     # Lacy-Bernstein 2
        Ahat = learnLB2Model(S1, S2)

    else:
        raise ValueError(
            f"Invalid algo parameter {algo}. "
            "Valid choices: 1 (CG), 2 (LS), 3 (LB1), 4 (LB1-CG), 5 (LB2)."
        )

    # ------------------------------------------------------------------
    # 5. Estimate noise covariances
    # ------------------------------------------------------------------
    What = S2 - Ahat @ S1                 # process noise residuals (n, tau-1)
    Qhat = np.cov(What)                   # (n, n)

    if m <= 100:
        Vhat = Y[:, :tau] - Chat @ Xhat  # observation noise residuals (m, tau)
        Rhat = np.cov(Vhat)               # (m, m)
    else:
        Rhat = None   # too large to compute — matches MATLAB behaviour

    return Ahat, Chat, Qhat, Rhat, Xhat, Ymean