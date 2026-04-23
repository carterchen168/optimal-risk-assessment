import numpy as np
from scipy.stats import norm
from sklearn.metrics import auc as sk_auc
from detectopt.predopt.Lasearch import Lasearch


class Struct:
    """Lightweight class for dot-notation payloads."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# TODO: implement and wire these probability/covariance helpers translated from optalarm_osrelease.
try:
    from detectopt.opt.createVunceventmatrix import createVunceventmatrix  # type: ignore
except Exception:
    createVunceventmatrix = None

# TODO: implement and wire this helper translated from optalarm_osrelease.
try:
    from detectopt.opt.createValarmmatrix import createValarmmatrix  # type: ignore
except Exception:
    createValarmmatrix = None

# TODO: implement and wire this helper translated from optalarm_osrelease.
try:
    from detectopt.opt.unceventprob import unceventprob  # type: ignore
except Exception:
    unceventprob = None

def _to_1d_float(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _to_scalar_float(x, default: float = 0.0) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _safe_variance(v: float, fallback: float = 1.0) -> float:
    if not np.isfinite(v) or v <= 0.0:
        return float(fallback)
    return float(v)


def _estimate_stationary_var(lds_params) -> float:
    """Estimate stationary output variance from LDS covariance fields."""
    cdl = np.asarray(getattr(lds_params, "cdl", [[1.0]]), dtype=float)
    rvdl = np.asarray(getattr(lds_params, "rvdl", [[0.0]]), dtype=float)

    if hasattr(lds_params, "xssdl"):
        xssdl = np.asarray(lds_params.xssdl, dtype=float)
        y_cov = cdl @ xssdl @ cdl.T + rvdl
    elif hasattr(lds_params, "dare"):
        dare = np.asarray(lds_params.dare, dtype=float)
        y_cov = cdl @ dare @ cdl.T + rvdl
    else:
        y_cov = rvdl

    return _safe_variance(_to_scalar_float(y_cov), fallback=1.0)


def _estimate_preddist(lds_params, dstep: int, yss: float) -> float:
    """Extract predictive alarm variance preddist = Valarm[dstep-1, dstep-1].

    Reads from lds_params.Valarm if already set; otherwise falls back to yss.
    Mirrors MATLAB: lds_params.preddist = lds_params.Valarm(dstep, dstep).
    """
    if hasattr(lds_params, "Valarm"):
        try:
            valarm = np.asarray(lds_params.Valarm, dtype=float)
            if valarm.ndim == 2 and valarm.shape[0] >= dstep and valarm.shape[1] >= dstep:
                return _safe_variance(float(valarm[dstep - 1, dstep - 1]), fallback=yss)
        except Exception:
            pass
    return float(yss)


def _estimate_pc(lds_params, x: float, dstep: int, preddist: float) -> float:
    """Estimate probability of event for class-mixture ROC synthesis."""
    if callable(unceventprob):
        try:
            params_local = Struct(**vars(lds_params))
            params_local.fixed = x
            params_local.dstep = int(dstep)
            pc_val = unceventprob(params_local)
            if isinstance(pc_val, tuple):
                pc_val = pc_val[0]
            return float(np.clip(_to_scalar_float(pc_val, default=0.1), 1e-6, 1.0 - 1e-6))
        except Exception:
            pass

    sigma = np.sqrt(_safe_variance(preddist, fallback=1.0))
    # Fallback: exceedance probability under a Gaussian predictive residual model.
    pc_val = 2.0 * norm.sf(abs(float(x)) / sigma)
    return float(np.clip(pc_val, 1e-6, 1.0 - 1e-6))


def predtrainopt(params, lds_params, dstep, x):
    """
    Translation of ACCEPT/detectopt/predopt/predtrainopt.m.

    Predictive alarm training: computes ROC/AUC for the predictive (flag=2)
    alarm family using Kalman-filter-based predictive distributions.

    Returns:
        auc, fp, tp, pa, pca, laval
    """
    dstep = int(dstep)
    x = float(np.asarray(x, dtype=float).reshape(-1)[0])

    # Mirror MATLAB field assignments on the LDS struct.
    lds_params.fixed = x
    lds_params.dstep = dstep

    # Build uncertainty covariance matrices when helpers are available.
    if callable(createVunceventmatrix):
        try:
            lds_params.Vuncevent = createVunceventmatrix(lds_params)
        except Exception:
            pass

    if callable(createValarmmatrix):
        try:
            lds_params.Valarm = createValarmmatrix(
                lds_params, np.arange(1, dstep + 1), np.arange(1, dstep + 1)
            )
        except Exception:
            pass

    yss = _estimate_stationary_var(lds_params)
    preddist = _estimate_preddist(lds_params, dstep, yss)

    lds_params.yss = yss
    lds_params.preddist = preddist

    pc = _estimate_pc(lds_params, x=x, dstep=dstep, preddist=preddist)
    lds_params.pc = pc

    # flag=2 selects the predictive alarm path inside Lasearch.
    fp, tp, pa, pca, laval = Lasearch(lds_params, pc, params.tol, 2)
    fp = _to_1d_float(fp)
    tp = _to_1d_float(tp)
    pa = _to_1d_float(pa)
    pca = _to_1d_float(pca)
    laval = _to_1d_float(laval)
    # Replace manual MATLAB trapezoid AUC with sklearn per project rule.
    auc_val = float(sk_auc(fp, tp)) if fp.size > 1 and tp.size > 1 else 0.5
    return auc_val, fp, tp, pa, pca, laval
