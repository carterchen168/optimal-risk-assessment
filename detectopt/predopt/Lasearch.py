import time

import numpy as np
from scipy.linalg import LinAlgError, solve
from scipy.stats import norm


class Struct:
    """Lightweight class for dot-notation payloads."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# TODO: implement and wire these helpers if exact MATLAB conditional alarm parity is required.
try:
    from detectopt.opt.unceventalarmprobred import unceventalarmprobred  # type: ignore
except Exception:
    unceventalarmprobred = None

try:
    from detectopt.opt.unceventalarmprobpred import unceventalarmprobpred  # type: ignore
except Exception:
    unceventalarmprobpred = None


def _to_1d_float(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _to_scalar_float(x, default: float = 0.0) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _safe_variance(value: float, fallback: float = 1.0) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return float(fallback)
    return float(value)


def _threshold_upper_bound(params, flag: int) -> float:
    dstep = max(1, int(_to_scalar_float(getattr(params, "dstep", 1), default=1)))

    if flag == 1:
        yss = _safe_variance(_to_scalar_float(getattr(params, "yss", 1.0), default=1.0), fallback=1.0)
        vuncevent = np.asarray(getattr(params, "Vuncevent", np.eye(dstep)), dtype=float)
        vunceventplusone = np.asarray(getattr(params, "Vunceventplusone", np.eye(dstep + 1)), dtype=float)

        if (
            vuncevent.ndim == 2
            and vuncevent.shape[0] >= dstep
            and vuncevent.shape[1] >= dstep
            and vunceventplusone.ndim == 2
            and vunceventplusone.shape[0] >= dstep + 1
            and vunceventplusone.shape[1] >= dstep + 1
        ):
            row = np.asarray(vunceventplusone[0, 1 : dstep + 1], dtype=float).reshape(1, -1)
            col = np.asarray(vunceventplusone[1 : dstep + 1, 0], dtype=float).reshape(-1, 1)
            try:
                correction = float(np.asarray(row @ solve(vuncevent, col), dtype=float).reshape(-1)[0])
                return _safe_variance(yss - correction, fallback=np.sqrt(yss))
            except (LinAlgError, ValueError):
                pass

        return float(np.sqrt(yss))

    preddist = _safe_variance(
        _to_scalar_float(getattr(params, "preddist", getattr(params, "yss", 1.0)), default=1.0),
        fallback=1.0,
    )
    valarm = np.asarray(getattr(params, "Valarm", np.eye(max(dstep, 1))), dtype=float)
    vuncevent = np.asarray(getattr(params, "Vuncevent", np.eye(dstep)), dtype=float)

    if (
        valarm.ndim == 2
        and valarm.shape[0] >= dstep
        and valarm.shape[1] >= dstep
        and vuncevent.ndim == 2
        and vuncevent.shape[0] >= dstep
        and vuncevent.shape[1] >= dstep
    ):
        row = np.asarray(valarm[dstep - 1, :dstep], dtype=float).reshape(1, -1)
        col = np.asarray(valarm[:dstep, dstep - 1], dtype=float).reshape(-1, 1)
        try:
            correction = float(np.asarray(row @ solve(vuncevent, col), dtype=float).reshape(-1)[0])
            return _safe_variance(preddist - correction, fallback=np.sqrt(preddist))
        except (LinAlgError, ValueError):
            pass

    return float(np.sqrt(preddist))


def _tail_probability(threshold: float, variance: float) -> float:
    sigma = np.sqrt(_safe_variance(variance, fallback=1.0))
    return float(2.0 * norm.cdf(-float(threshold) / sigma))


def _conditional_alarm_probability(params, pc: float, threshold: float, pa: float, flag: int, anum: float) -> float:
    if flag == 1 and callable(unceventalarmprobred):
        try:
            pca, _ = unceventalarmprobred(params, pc, threshold, pa)
            return float(np.clip(_to_scalar_float(pca, default=pa), 0.0, 1.0))
        except Exception:
            pass

    if flag == 2 and callable(unceventalarmprobpred):
        try:
            pca, _ = unceventalarmprobpred(params, pc, threshold, pa)
            return float(np.clip(_to_scalar_float(pca, default=pa), 0.0, 1.0))
        except Exception:
            pass

    return float(2.0 * norm.cdf(-float(threshold) / np.sqrt(_safe_variance(anum, fallback=1.0))))


def _derive_fp_tp(patrack: np.ndarray, pcatrack: np.ndarray, pc: float, nondegeneracyflag: bool):
    if nondegeneracyflag:
        fp = (patrack - pcatrack) / max(1.0 - pc, np.finfo(float).eps)
        tp = pcatrack / max(pc, np.finfo(float).eps)
    else:
        fp = pcatrack
        tp = patrack
    return np.clip(fp, 0.0, 1.0), np.clip(tp, 0.0, 1.0)


def _choose_thresholds(laval: np.ndarray, fp: np.ndarray, tp: np.ndarray, iteration: int, tol: int):
    if iteration < tol + 1:
        return 0.5 * (laval[:-1] + laval[1:])

    fp_change = np.where(np.abs(np.diff(fp)) > 0.01)[0]
    tp_change = np.where(np.abs(np.diff(tp)) > 0.01)[0]

    if fp_change.size == 0 and tp_change.size == 0:
        return np.array([], dtype=float)

    if fp_change.size > 0:
        laminfp = laval[int(fp_change[0])]
        lamaxfp = laval[int(fp_change[-1]) + 1]
    else:
        laminfp = lamaxfp = None

    if tp_change.size > 0:
        lamintp = laval[int(tp_change[0])]
        lamaxtp = laval[int(tp_change[-1]) + 1]
    else:
        lamintp = lamaxtp = None

    if laminfp is not None and lamintp is not None and lamaxfp is not None and lamaxtp is not None:
        return np.unique(np.concatenate((np.linspace(laminfp, lamaxfp, 100), np.linspace(lamintp, lamaxtp, 100))))
    if laminfp is not None and lamaxfp is not None:
        return np.linspace(laminfp, lamaxfp, 100)
    if lamintp is not None and lamaxtp is not None:
        return np.linspace(lamintp, lamaxtp, 100)
    return np.array([], dtype=float)


def _remove_invalid_points(laval: np.ndarray, patrack: np.ndarray, pcatrack: np.ndarray, pc: float):
    valid = np.ones(laval.size, dtype=bool)
    valid &= np.isfinite(laval)
    valid &= np.isfinite(patrack)
    valid &= np.isfinite(pcatrack)
    valid &= patrack >= -np.finfo(float).eps
    valid &= pcatrack >= -np.finfo(float).eps
    valid &= pcatrack <= pc + np.finfo(float).eps
    valid &= pcatrack <= patrack + np.finfo(float).eps
    return laval[valid], patrack[valid], pcatrack[valid]


def _run_search(params, pc: float, tol: int, flag: int):
    nondegeneracyflag = float(_to_scalar_float(getattr(params, "fixed", 0.0), default=0.0)) > 0.0
    lamax = float(_to_scalar_float(getattr(params, "fixed", 0.0), default=0.0))
    if not np.isfinite(lamax):
        lamax = 0.0

    variance_bound = _threshold_upper_bound(params, flag)
    if not np.isfinite(variance_bound) or variance_bound <= 0.0:
        variance_bound = 1.0

    laval = np.array([0.0, lamax], dtype=float)
    patrack = np.array([pc, 0.0], dtype=float)
    pcatrack = np.array([pc, 0.0], dtype=float)
    latime = np.array([0.0, 0.0], dtype=float)

    if laval[0] > laval[1]:
        laval = laval[::-1]
        patrack = patrack[::-1]
        pcatrack = pcatrack[::-1]

    iteration = 1
    max_points = int(_to_scalar_float(getattr(params, "N", 20000), default=20000))

    while True:
        fp, tp = _derive_fp_tp(patrack, pcatrack, pc, nondegeneracyflag)
        la = _choose_thresholds(laval, fp, tp, iteration, tol)
        if la.size == 0:
            break

        for threshold in la:
            pa = _tail_probability(threshold, variance_bound)
            pca = _conditional_alarm_probability(params, pc, float(threshold), pa, flag, variance_bound)
            laval = np.append(laval, float(threshold))
            patrack = np.append(patrack, pa)
            pcatrack = np.append(pcatrack, pca)
            latime = np.append(latime, 0.0)

        order = np.argsort(laval)
        laval = laval[order]
        patrack = patrack[order]
        pcatrack = pcatrack[order]

        fp, tp = _derive_fp_tp(patrack, pcatrack, pc, nondegeneracyflag)

        if iteration > tol:
            first_la = laval[laval > 0.0]
            last_la = laval[laval < lamax]
            if first_la.size > 1 and last_la.size > 1:
                fp_done = fp.size < 2 or np.max(np.abs(np.diff(fp))) < 0.01
                tp_done = tp.size < 2 or np.max(np.abs(np.diff(tp))) < 0.01
                if fp_done and tp_done:
                    break

        if laval.size > max_points:
            break

        iteration += 1
        if iteration > max(2 * max(tol, 1), 1) + 50:
            break

    laval, patrack, pcatrack = _remove_invalid_points(laval, patrack, pcatrack, pc)
    if laval.size == 0:
        laval = np.array([0.0], dtype=float)
        patrack = np.array([pc], dtype=float)
        pcatrack = np.array([pc], dtype=float)

    order = np.argsort(laval)
    laval = laval[order]
    patrack = patrack[order]
    pcatrack = pcatrack[order]

    fp, tp = _derive_fp_tp(patrack, pcatrack, pc, nondegeneracyflag)
    remfrac = 0.0 if pcatrack.size == 0 else float(np.sum(np.diff(pcatrack) > 0.01 * pc) / pcatrack.size)

    return fp, tp, patrack, pcatrack, laval, latime, remfrac


def Lasearch(params, pc, tol, flag, *, full_output: bool = False):
    """Port of ACCEPT/detectopt/predopt/Lasearch.m.

    The MATLAB helper returns seven outputs, but the current Python callers only
    consume the first five. By default this function returns the caller-facing
    five-tuple ``(fp, tp, patrack, pcatrack, laval)`` so existing imports keep
    working. Pass ``full_output=True`` to receive the full diagnostic payload.
    """

    tol = int(_to_scalar_float(tol, default=1))
    flag = int(_to_scalar_float(flag, default=1))
    pc = float(np.clip(_to_scalar_float(pc, default=0.0), 0.0, 1.0))

    fp, tp, patrack, pcatrack, laval, latime, remfrac = _run_search(params, pc, tol, flag)

    if full_output:
        return fp, tp, patrack, pcatrack, laval, latime, remfrac

    return fp, tp, patrack, pcatrack, laval


Lasearchtest = Lasearch
