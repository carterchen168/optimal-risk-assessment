import numpy as np
from scipy.stats import norm
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve


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

# TODO: if exact MATLAB Lasearch parity is required, add translated Lasearch and route here.
try:
    from detectopt.predopt.Lasearch import Lasearch  # type: ignore
except Exception:
    Lasearch = None


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
    """Estimate y variance used by redline alarm scoring."""
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


def _estimate_reddist(lds_params, dstep: int, yss: float) -> float:
    """Estimate redline predictive distribution variance at horizon dstep."""
    if callable(createValarmmatrix):
        try:
            valarm = np.asarray(createValarmmatrix(lds_params, np.arange(1, dstep + 1), np.arange(1, dstep + 1)), dtype=float)
            if valarm.ndim == 2 and valarm.shape[0] >= dstep and valarm.shape[1] >= dstep:
                return _safe_variance(float(valarm[dstep - 1, dstep - 1]), fallback=yss)
        except Exception:
            pass

    return float(yss)


def _estimate_pc(lds_params, x: float, dstep: int, reddist: float) -> float:
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

    sigma = np.sqrt(_safe_variance(reddist, fallback=1.0))
    # Fallback: exceedance probability under a Gaussian residual model.
    pc_val = 2.0 * norm.sf(abs(float(x)) / sigma)
    return float(np.clip(pc_val, 1e-6, 1.0 - 1e-6))


def _roc_from_synthetic_distributions(yss: float, reddist: float, pc: float, n_samples: int):
    """Build ROC curves using synthetic labels/scores and sklearn metrics."""
    rng = np.random.default_rng(0)

    n_total = max(2000, int(n_samples))
    n_event = max(1, int(round(pc * n_total)))
    n_nom = max(1, n_total - n_event)

    nominal_scores = np.abs(rng.normal(loc=0.0, scale=np.sqrt(yss), size=n_nom))
    event_scores = np.abs(rng.normal(loc=0.0, scale=np.sqrt(reddist), size=n_event))

    y_true = np.concatenate((np.zeros(n_nom, dtype=int), np.ones(n_event, dtype=int)))
    y_score = np.concatenate((nominal_scores, event_scores))

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_val = float(sk_auc(fpr, tpr))

    finite = np.isfinite(thresholds)
    if not np.any(finite):
        thresholds = np.array([0.0], dtype=float)
        fpr = np.array([0.0], dtype=float)
        tpr = np.array([0.0], dtype=float)
    else:
        thresholds = np.asarray(thresholds[finite], dtype=float)
        fpr = np.asarray(fpr[finite], dtype=float)
        tpr = np.asarray(tpr[finite], dtype=float)

    # Clamp non-physical tiny negatives from floating point noise.
    thresholds = np.maximum(thresholds, 0.0)

    return auc_val, fpr, tpr, thresholds


def redlineopt(params, lds_params, dstep, x):
    """
    Translation target of ACCEPT/detectopt/predopt/redlineopt.m.

    Returns:
        auc, fp, tp, pa, pca, laval
    """
    dstep = int(dstep)
    x = float(np.asarray(x, dtype=float).reshape(-1)[0])

    # Keep MATLAB-style field updates on the LDS struct.
    lds_params.fixed = x
    lds_params.dstep = dstep

    if callable(createVunceventmatrix):
        try:
            lds_params.Vuncevent = createVunceventmatrix(lds_params)
            newparams = Struct(**vars(lds_params))
            newparams.dstep = dstep + 1
            lds_params.Vunceventplusone = createVunceventmatrix(newparams)
        except Exception:
            pass

    yss = _estimate_stationary_var(lds_params)
    reddist = _estimate_reddist(lds_params, dstep, yss)

    lds_params.yss = yss
    lds_params.reddist = reddist

    pc = _estimate_pc(lds_params, x=x, dstep=dstep, reddist=reddist)
    lds_params.pc = pc

    if callable(Lasearch):
        try:
            # Prefer exact translated search if/when it becomes available.
            fp, tp, pa, pca, laval = Lasearch(lds_params, pc, params.tol, 1)
            fp = _to_1d_float(fp)
            tp = _to_1d_float(tp)
            pa = _to_1d_float(pa)
            pca = _to_1d_float(pca)
            laval = _to_1d_float(laval)
            # Replace manual trapezoid AUC with sklearn utility per project rule.
            auc_val = float(sk_auc(fp, tp)) if fp.size > 1 and tp.size > 1 else 0.5
            return auc_val, fp, tp, pa, pca, laval
        except Exception:
            pass

    n_for_roc = getattr(params, "N", 20000)
    auc_val, fp, tp, laval = _roc_from_synthetic_distributions(
        yss=_safe_variance(yss, 1.0),
        reddist=_safe_variance(reddist, yss),
        pc=pc,
        n_samples=n_for_roc,
    )

    # MATLAB outputs include pa and pca tracks; derive them consistently from fp/tp and class prior pc.
    pa = fp * (1.0 - pc) + tp * pc
    pca = tp * pc

    return float(auc_val), _to_1d_float(fp), _to_1d_float(tp), _to_1d_float(pa), _to_1d_float(pca), _to_1d_float(laval)
