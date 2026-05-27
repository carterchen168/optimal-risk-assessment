import numpy as np
from scipy.sparse.linalg import svds


# ---------------------------------------------------------------------------
# Solver helpers
# ---------------------------------------------------------------------------

def _solve_qp_scipy(P, q, G, h):
    """
    Solve:  min  m' P m - 2 q' m
            s.t. G m <= h
    using scipy SLSQP (always available).

    P : (d², d²) PSD matrix
    q : (d²,)    linear term
    G : (k, d²)  inequality constraint matrix
    h : (k,)     RHS
    """
    from scipy.optimize import minimize

    d2 = P.shape[0]
    x0 = np.zeros(d2)

    def objective(m):
        return float(m @ P @ m - 2 * q @ m)

    def gradient(m):
        return 2 * P @ m - 2 * q

    constraints = []
    for row_g, val_h in zip(G, h):
        constraints.append({
            'type': 'ineq',
            'fun':  lambda m, g=row_g, b=val_h: float(b - g @ m),
            'jac':  lambda m, g=row_g: -g
        })

    result = minimize(
        objective, x0, jac=gradient,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False}
    )
    return result.x, result.fun


def _solve_qp_quadprog(P, q, G, h):
    """
    Solve via the `quadprog` package (pip install quadprog).
    quadprog minimises: 0.5 x' G x + a' x  s.t. C' x >= b
    We have: min m' P m - 2 q' m, G_ineq m <= h
    → H = 2P, a = -2q, C' = -G_ineq, b = -h
    """
    import quadprog
    H  = 2.0 * P + 1e-9 * np.eye(P.shape[0])   # ensure PD
    a  = -2.0 * q
    C  = -G.T    # (d², k)
    b  = -h
    sol = quadprog.solve_qp(H, a, C, b)
    return sol[0], sol[1]


def _solve_qp_cvxpy(P, q, G, h, d):
    """
    Solve via CVXPY (matches MATLAB CVX formulation directly).
    min  m' P m - 2 q' m   s.t.  -G m + h >= 0
    """
    import cvxpy as cp
    d2 = d * d
    m  = cp.Variable(d2)
    objective   = cp.Minimize(cp.quad_form(m, P) - 2 * q @ m)
    constraints = [-G @ m + h >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    return m.value, prob.value


def _solve_qp(P, q, G, h, d, cvx_flag):
    """Dispatch to the best available QP solver."""
    if cvx_flag:
        return _solve_qp_cvxpy(P, q, G, h, d)
    try:
        return _solve_qp_quadprog(P, q, G, h)
    except ImportError:
        return _solve_qp_scipy(P, q, G, h)


# ---------------------------------------------------------------------------
# Helper: eigenvalue magnitudes
# ---------------------------------------------------------------------------

def _get_eigenthings(M):
    """
    Return (eigenvalues, max_magnitude, min_magnitude).
    Mirrors the inner get_eigenthings() subfunction of learnCGModelEM.m.
    """
    evals = np.linalg.eigvals(M)
    mags  = np.abs(evals)
    return evals, float(np.max(mags)), float(np.min(mags))


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def learn_cg_model_em(beta, gamma1, A, simulate_LB1=False, cvx_flag=False):
    """
    Learn a stable LDS dynamics matrix using Constraint Generation (EM form).

    Parameters
    ----------
    beta        : np.ndarray (d, d) — EM sufficient statistic  Σ x_t x_{t-1}'
    gamma1      : np.ndarray (d, d) — EM sufficient statistic  Σ x_{t-1} x_{t-1}'
    A           : np.ndarray (d, d) — initial unconstrained MLE dynamics matrix
    simulate_LB1: bool — if True, simulate Lacy-Bernstein 1 by watching
                         for top singular value ≤ 1 instead of spectral radius < 1
    cvx_flag    : bool — if True, use CVXPY instead of quadprog/SLSQP

    Returns
    -------
    M : np.ndarray (d, d) — stable dynamics matrix (transposed to match
                             MATLAB's return convention: M = M')
    """
    sv_eps  = 0.0004999999999
    max_iter = 1000
    tol_bin  = 1e-5

    d = gamma1.shape[0]

    # ------------------------------------------------------------------
    # QP objective matrices (equations 27b-e from Boots' M.S. Thesis)
    # P = I_d ⊗ gamma1,  q = vec(beta)
    # We minimise  m'Pm - 2q'm  (ignoring the constant r=0)
    # ------------------------------------------------------------------
    P = np.kron(np.eye(d), gamma1)
    q = beta.ravel(order='F')     # MATLAB vec() is column-major

    # Constraints: initially empty
    G = np.zeros((0, d * d))
    h = np.zeros(0)

    # Initial solution is the unconstrained MLE
    M = A.copy()

    _, max_e, min_e = _get_eigenthings(M)
    Morig = M.copy()

    # Check whether we're already done before any QP iteration
    # scipy.sparse.linalg.svds returns (U, s, Vt) — Vt is already transposed.
    # MATLAB svds(M,1) returns [u, s, v] where u,v are column vectors.
    # Outer product u*v' (MATLAB) = u_vec @ v_vt  (Python, since v_vt = v.T)
    _, s_top, _ = svds(M, k=1)
    s_top = float(s_top)

    if simulate_LB1:
        if s_top <= 1 + sv_eps:
            return M.T
    else:
        if max_e < 1 and min_e > -1:
            return M.T

    # Add first constraint from the initial unstable solution
    u_vec, _, v_vt = svds(M, k=1)   # u_vec: (d,1), v_vt: (1,d)
    ebar = (u_vec @ v_vt).ravel(order='F')   # (d,1)@(1,d) = (d,d) → vec
    G = np.vstack([G, ebar[np.newaxis, :]])
    h = np.append(h, 1.0)

    # ------------------------------------------------------------------
    # Constraint generation loop
    # ------------------------------------------------------------------
    Mprev = M.copy()
    for iteration in range(max_iter):
        m_vec, _ = _solve_qp(P, q, G, h, d, cvx_flag)

        if m_vec is None:
            break

        Mprev = M.copy()
        M = m_vec.reshape((d, d), order='F')

        _, max_e, min_e = _get_eigenthings(M)
        u_vec, s_arr, v_vt = svds(M, k=1)
        s_top = float(s_arr)

        if simulate_LB1:
            if s_top <= 1 + sv_eps:
                break
        else:
            if max_e < 1 and min_e > -1:
                break

        # Add constraint from current largest singular vectors
        ebar = (u_vec @ v_vt).ravel(order='F')
        G = np.vstack([G, ebar[np.newaxis, :]])
        h = np.append(h, 1.0)

    # ------------------------------------------------------------------
    # Binary-search refinement (stability mode only)
    # ------------------------------------------------------------------
    if not simulate_LB1:
        Mbest  = M.copy()
        Morig  = Mprev.copy()
        lo, hi = 0.0, 1.0

        while hi - lo > tol_bin:
            alpha = lo + (hi - lo) / 2
            Mbest = (1 - alpha) * M + alpha * Morig
            maxeig = float(np.max(np.abs(np.linalg.eigvals(Mbest))))
            if maxeig > 1:
                hi = alpha
            elif maxeig < 1:
                lo = alpha
            else:
                break

        # Step slightly inside the stability boundary
        alpha_final = lo + tol_bin
        alpha_orig  = alpha - tol_bin
        M = (1 - alpha_final) * M + alpha_orig * Morig

    return M.T   # MATLAB convention: return M'