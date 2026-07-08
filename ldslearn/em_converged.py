import math

import numpy as np


def em_converged(
    loglik: float,
    previous_loglik: float,
    threshold: float = 1e-4,
    check_increased: bool = True,
    verbose: bool = False,
) -> tuple[bool, bool]:
    """
    Check whether the EM algorithm has converged.

    Parameters
    ----------
    loglik          : float — log-likelihood at current iteration
    previous_loglik : float — log-likelihood at previous iteration
    threshold       : float — relative change threshold (default 1e-4)
    check_increased : bool  — if True, flag a likelihood decrease as
                              a convergence signal (default True)
    verbose         : bool  — if True, print a message on decrease

    Returns
    -------
    converged : bool — True if EM should stop
    decrease  : bool — True if log-likelihood decreased (possible with MAP)
    """
    converged = False
    decrease  = False

    if check_increased:
        if loglik - previous_loglik < -1e-3:   # allow a little numerical imprecision
            if verbose:
                print(
                    f"****** likelihood decreased from "
                    f"{previous_loglik:.4f} to {loglik:.4f}!"
                )
            decrease  = True
            converged = False
            return converged, decrease

    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik   = (abs(loglik) + abs(previous_loglik) + np.finfo(float).eps) / 2

    # inf/inf -> nan on the first iteration (previous_loglik = -inf) or a
    # non-finite loglik; nan < threshold is False, handled by the check below.
    with np.errstate(invalid="ignore"):
        if (delta_loglik / avg_loglik) < threshold:
            converged = True

    # Degenerate / non-finite / complex log-likelihood → also stop
    if not np.isreal(loglik) or math.isinf(loglik) or math.isnan(loglik):
        converged = True

    return converged, decrease