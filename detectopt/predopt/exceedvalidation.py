import numpy as np
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve


class Struct:
    """Lightweight class for dot-notation payloads."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _to_1d_float(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _to_1d_int(x) -> np.ndarray:
    return np.asarray(x, dtype=int).reshape(-1)


def exceedvalidation(x, dstep, obsval):
    """
    Translation target of ACCEPT/detectopt/predopt/exceedvalidation.m (function name exceedopt).

    Signature required by dispatcher:
        auc, rocdata = exceedvalidation(x, dstep, obsval)
    """
    dstep = int(dstep)
    x = float(np.asarray(x, dtype=float).reshape(-1)[0])

    rocdata = Struct(exceedscore=[])

    all_events = []
    all_scores = []

    for obs in obsval:
        data = _to_1d_float(obs.data)
        n = data.size

        # MATLAB condition k <= length(data)-dstep maps to event length n-dstep in 0-based indexing.
        event_len = max(0, n - dstep)

        if event_len == 0:
            rocdata.exceedscore.append(np.array([], dtype=float))
            continue

        events = np.zeros(event_len, dtype=int)
        for k in range(event_len):
            # MATLAB checks future samples: data(k+1 : k+dstep) in 1-based indexing.
            future_window = np.abs(data[k + 1 : k + dstep + 1])
            events[k] = int(np.any(future_window > x))

        # Score is current residual magnitude aligned to event index.
        scores = np.abs(data[:event_len])

        all_events.append(events)
        all_scores.append(scores)
        rocdata.exceedscore.append(scores)

    if len(all_events) == 0:
        avg_stats = Struct(
            fprate=np.array([0.0], dtype=float),
            pmd=np.array([1.0], dtype=float),
            thresh=np.array([x], dtype=float),
            rocarea=0.0,
        )
        rocdata.avg_stats = avg_stats
        return 0.0, rocdata

    y_true = _to_1d_int(np.concatenate(all_events))
    y_score = _to_1d_float(np.concatenate(all_scores))

    if y_true.size == 0:
        avg_stats = Struct(
            fprate=np.array([0.0], dtype=float),
            pmd=np.array([1.0], dtype=float),
            thresh=np.array([x], dtype=float),
            rocarea=0.0,
        )
        rocdata.avg_stats = avg_stats
        return 0.0, rocdata

    if np.unique(y_true).size < 2:
        fprate = np.array([0.0, 1.0], dtype=float)
        tpr = np.array([0.0, 1.0], dtype=float)
        thresh = np.array([np.max(y_score) if y_score.size > 0 else x, 0.0], dtype=float)
        auc_val = 0.5
    else:
        fprate, tpr, thresh = roc_curve(y_true, y_score)
        auc_val = float(sk_auc(fprate, tpr))

    pmd = 1.0 - np.asarray(tpr, dtype=float)

    rocdata.avg_stats = Struct(
        fprate=np.asarray(fprate, dtype=float),
        pmd=np.asarray(pmd, dtype=float),
        thresh=np.asarray(thresh, dtype=float),
        rocarea=float(auc_val),
    )

    return float(auc_val), rocdata
