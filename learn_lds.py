import numpy as np


# ---------------------------------------------------------------------------
# Stubs for dynamics-matrix learning sub-methods
# ---------------------------------------------------------------------------

def learnCGModel(S1, S2, simulate_LB1):
    """
    STUB — Constraint-generation dynamics matrix (state-sequence variant).
    Original MATLAB signature:
      Ahat = learnCGModel(S1, S2, simulate_LB1)

    Parameters
    ----------
    S1          : np.ndarray (n, T-1) — state sequence at t = 0..T-2
    S2          : np.ndarray (n, T-1) — state sequence at t = 1..T-1
    simulate_LB1: int — 0 = standard CG, 1 = simulate Lacy-Bernstein 1

    Returns
    -------
    Ahat : np.ndarray (n, n) — stable dynamics matrix
    """
    raise NotImplementedError(
        "learnCGModel() source was not provided. "
        "Implement or import it before using algo=1 or algo=4 in learn_lds()."
    )


def learnLB1Model(S1, S2):
    """
    STUB — Lacy-Bernstein 1 dynamics matrix learning.
    Original MATLAB signature:
      Ahat = learnLB1Model(S1, S2)

    Returns
    -------
    Ahat : np.ndarray (n, n) — stable dynamics matrix
    """
    raise NotImplementedError(
        "learnLB1Model() source was not provided. "
        "Implement or import it before using algo=3 in learn_lds()."
    )


def learnLB2Model(S1, S2):
    """
    STUB — Lacy-Bernstein 2 dynamics matrix learning.
    Original MATLAB signature:
      Ahat = learnLB2Model(S1, S2)

    Returns
    -------
    Ahat : np.ndarray (n, n) — stable dynamics matrix
    """
    raise NotImplementedError(
        "learnLB2Model() source was not provided. "
        "Implement or import it before using algo=5 in learn_lds()."
    )


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