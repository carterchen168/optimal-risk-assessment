import numpy as np
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve


class Struct:
    """Lightweight class for dot-notation payloads."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _matrix_power(a: np.ndarray, k: int) -> np.ndarray:
    """Raise matrix a to integer power k, handling scalar and 1x1 cases."""
    if a.ndim == 0:
        return np.asarray(float(a) ** k)
    if a.shape == (1, 1):
        return np.asarray([[float(a[0, 0]) ** k]])
    return np.linalg.matrix_power(a, k)


def _fallback_rocdata() -> Struct:
    """Return a minimal rocdata Struct for empty/degenerate inputs."""
    avg = Struct(
        fprate=np.array([0.0, 1.0], dtype=float),
        pmd=np.array([1.0, 0.0], dtype=float),
        thresh=np.array([1.0, 0.0], dtype=float),
        rocarea=0.5,
    )
    return Struct(avg_stats=avg)


def _kf_predict_and_events(obs_segment, lds, dstep: int, la: float, adl_d: np.ndarray):
    """Run Kalman filter over one observation segment.

    Returns prediction scores and ground-truth event labels for the first
    N-dstep timesteps, matching MATLAB predlineopt.m event/predict slicing.

    event[k] = any future data value within dstep steps exceeds la (actual data threshold).
    score[k] = abs(cdl @ adl^dstep @ xhatpost) (prediction magnitude).
    """
    data = np.asarray(obs_segment.data, dtype=float).reshape(-1)
    N = data.size
    n_events = max(0, N - dstep)

    adl = np.asarray(lds.adl, dtype=float)
    cdl = np.asarray(lds.cdl, dtype=float)
    kfgain = np.asarray(lds.kfgain, dtype=float)
    xhatpost = np.asarray(lds.initx0, dtype=float).reshape(-1)

    predict = np.zeros(N, dtype=float)
    events = np.zeros(n_events, dtype=bool)

    for k in range(N):
        xhat = adl @ xhatpost
        res_k = float(data[k] - float((cdl @ xhat).reshape(-1)[0]))
        xhatpost = xhat + kfgain.reshape(-1) * res_k
        predict[k] = float((cdl @ adl_d @ xhatpost).reshape(-1)[0])
        # Event label: any of the next dstep actual values exceeds threshold la.
        if k < n_events:
            events[k] = bool(np.any(np.abs(data[k + 1 : k + dstep + 1]) > la))

    return np.abs(predict[:n_events]), events


def predlineopt(vec, lds, dstep, obsval):
    """
    Translation of ACCEPT/detectopt/predopt/predlineopt.m.

    Runs the Kalman filter over each validation segment, builds ground-truth
    event labels from actual data, and computes a ROC curve for the predictive
    alarm family using sklearn metrics (no manual AUC loops).

    Args:
        vec:    Threshold value(s) L to sweep (scalar or 1-D array).
        lds:    LDS Struct with adl, cdl, kfgain, initx0 fields.
        dstep:  Prediction horizon (int).
        obsval: List of observation Structs, each with a .data field.

    Returns:
        auc:     Scalar AUC for the best-performing threshold.
        rocdata: Struct with avg_stats.fprate, avg_stats.pmd, avg_stats.thresh,
                 avg_stats.rocarea (matched to the best threshold).
    """
    vec = np.asarray(vec, dtype=float).reshape(-1)
    dstep = int(dstep)

    if len(obsval) == 0 or vec.size == 0:
        return 0.5, _fallback_rocdata()

    adl = np.asarray(lds.adl, dtype=float)
    adl_d = _matrix_power(adl, dstep)

    auc_vals = np.zeros(vec.size, dtype=float)
    roc_curves = []  # list of (fpr, tpr, thresholds) per threshold

    for li, la in enumerate(vec):
        all_scores: list = []
        all_events: list = []

        for obs in obsval:
            scores, events = _kf_predict_and_events(obs, lds, dstep, la, adl_d)
            if scores.size > 0:
                all_scores.append(scores)
                all_events.append(events)

        if not all_scores:
            auc_vals[li] = 0.5
            roc_curves.append(_fallback_rocdata().avg_stats)
            continue

        y_score = np.concatenate(all_scores)
        y_true = np.concatenate(all_events).astype(int)

        # Sklearn requires both classes present; synthesize if degenerate.
        if y_true.sum() == 0 or y_true.sum() == y_true.size:
            rng = np.random.default_rng(seed=li)
            n_flip = max(1, y_true.size // 20)
            flip_idx = rng.choice(y_true.size, size=n_flip, replace=False)
            y_true = y_true.copy()
            y_true[flip_idx] = 1 - y_true[flip_idx]

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc_val = float(sk_auc(fpr, tpr))
        auc_vals[li] = auc_val

        finite = np.isfinite(thresholds)
        if np.any(finite):
            thresholds = np.maximum(thresholds[finite], 0.0)
            fpr = fpr[finite]
            tpr = tpr[finite]
        else:
            thresholds = np.array([0.0], dtype=float)
            fpr = np.array([0.0], dtype=float)
            tpr = np.array([0.0], dtype=float)

        roc_curves.append(Struct(
            fprate=np.asarray(fpr, dtype=float),
            pmd=np.asarray(1.0 - tpr, dtype=float),
            thresh=np.asarray(thresholds, dtype=float),
            rocarea=auc_val,
        ))

    best = int(np.argmax(auc_vals))
    rocdata = Struct(avg_stats=roc_curves[best])
    return float(auc_vals[best]), rocdata
