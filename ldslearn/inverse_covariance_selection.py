import numpy as np

try:
    import cvxpy as cp
except ImportError as exc:
    raise ImportError(
        "cvxpy is required for inverse_covariance_selection(). "
        "Install it with:  pip install cvxpy"
    ) from exc


def inverse_covariance_selection(S: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """
    Find the nearest positive-definite covariance matrix to S by solving
    the inverse covariance selection problem.

    Mirrors inverse_covariance_selection.m exactly, replacing CVX with CVXPY.

    Parameters
    ----------
    S       : np.ndarray (n, n)
        Input (possibly non-positive-definite) covariance matrix.
    epsilon : float
        Minimum eigenvalue floor for the solution (default 0).
        Corresponds to the `epsilon` constraint in the MATLAB CVX formulation.

    Returns
    -------
    Q : np.ndarray (n, n)
        Positive-definite covariance matrix Q = X⁻¹, where X is the
        optimal precision matrix found by the solver.

    Raises
    ------
    ValueError
        If the optimisation problem is infeasible or the solver fails.
    """
    n = S.shape[0]

    # Decision variable: symmetric positive-definite precision matrix X
    X = cp.Variable((n, n), symmetric=True)

    # Objective: trace(S @ X) - log_det(X)
    # (minimising negative log-likelihood of a Gaussian with precision X)
    objective = cp.Minimize(cp.trace(S @ X) - cp.log_det(X))

    # Constraint: X - epsilon*I is PSD  →  smallest eigenvalue of X >= epsilon
    constraints = [X - epsilon * np.eye(n) >> 0]

    prob = cp.Problem(objective, constraints)

    # Try SCS first (ships with cvxpy), fall back to CLARABEL if available
    solvers_to_try = [cp.SCS, cp.CLARABEL]
    solved = False
    for solver in solvers_to_try:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and X.value is not None:
                solved = True
                break
        except cp.SolverError:
            continue

    if not solved or X.value is None:
        raise ValueError(
            f"inverse_covariance_selection: solver failed (status={prob.status}). "
            "Try a different epsilon or check that S is well-conditioned."
        )

    Q = np.linalg.inv(X.value)
    return Q