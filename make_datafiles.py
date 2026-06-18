import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from typing import Any, Tuple


class Struct:
    """Dot-notation container — mirrors MATLAB struct behaviour."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _get(params, key, default=None):
    """Read a param from either a Struct (dot-notation) or a plain dict."""
    if isinstance(params, dict):
        return params.get(key, default)
    return getattr(params, key, default)


def _set(params, key, value):
    """Write a param to either a Struct or a plain dict."""
    if isinstance(params, dict):
        params[key] = value
    else:
        setattr(params, key, value)


def make_datafiles(params: Any, sev: int) -> Tuple:
    """
    Port of MATLAB make_datafiles.m.

    Returns
    -------
    tr, secondary, train_cell, rawdata_secondary, rawdata_tr,
    params, Statistics, feature_indices, fullfeatures

    When sev==2 and 'svr' in params.algo, also runs SVR preliminary
    hyperparameter optimisation (MATLAB make_datafiles.m lines 98-254):
      Step 1 — data-stats estimation of C, epsilon, sigma
      Step 2 — 1-D sigma grid search (100 log-spaced points)
      Step 3 — 3-D while-loop refinement until NMSE <= 0.5
    """

    # ── data loader ──────────────────────────────────────────────────────────
    def _load(path: str):
        if not path or not os.path.isdir(path):
            return np.array([])
        rows = []
        for f in glob.glob(os.path.join(path, '*.csv')):
            rows.append(pd.read_csv(f).values)
        for f in glob.glob(os.path.join(path, '*.mat')):
            try:
                md = sio.loadmat(f, struct_as_record=False, squeeze_me=True)
                for k, v in md.items():
                    if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim == 2:
                        rows.append(v)
            except Exception as e:
                print(f"Warning: could not load {f}: {e}")
        return np.vstack(rows) if rows else np.array([])

    # ── training data ─────────────────────────────────────────────────────────
    train_data = _load(_get(params, 'nompath', ''))
    rawdata_tr = train_data.copy() if train_data.size > 0 else np.array([])

    header = _get(params, 'header', [])
    target_name = _get(params, 'targetName', '')
    target_idx = 0
    if target_name and header:
        try:
            target_idx = list(header).index(target_name)
        except ValueError:
            target_idx = 0

    n_cols = train_data.shape[1] if train_data.size > 0 else len(header) if header else 0
    feature_indices = _get(params, 'channelContineous', list(range(n_cols)))
    fullfeatures = np.arange(train_data.shape[1]) if train_data.size > 0 else np.array([])

    if train_data.size > 0 and feature_indices:
        tr_y = train_data[:, target_idx] if target_idx < train_data.shape[1] else np.zeros(len(train_data))
        tr_x = train_data[:, feature_indices]
    else:
        tr_x = np.empty((0, len(feature_indices) if feature_indices else 0))
        tr_y = np.array([])

    train_cell = Struct(
        x=[tr_x[i:i+1, :] for i in range(len(tr_y))] if len(tr_y) > 0 else [],
        y=[np.array([tr_y[i]]) for i in range(len(tr_y))] if len(tr_y) > 0 else [],
    )
    tr = Struct(
        x=tr_x,
        y=tr_y,
        header=[header[i] for i in feature_indices] if header else [],
    )
    Statistics = Struct(
        mean=np.mean(tr_x, axis=0) if tr_x.size > 0 else np.array([]),
        std=np.std(tr_x, axis=0) if tr_x.size > 0 else np.array([]),
        min=np.min(tr_x, axis=0) if tr_x.size > 0 else np.array([]),
        max=np.max(tr_x, axis=0) if tr_x.size > 0 else np.array([]),
        samples=len(tr_y),
    )

    # ── SVR preliminary optimisation (sev==2 only) ───────────────────────────
    # MATLAB make_datafiles.m lines 98-254
    algo = _get(params, 'algo', [])
    algo_list = list(algo) if not isinstance(algo, list) else algo
    if sev == 2 and 'svr' in algo_list and tr_x.size > 0:
        from regressopt import modelopttest, optimsearch

        # Normalise for estimation — MATLAB zscoreStream normalises both X and y.
        _scaler = StandardScaler()
        _xs = _scaler.fit_transform(tr_x.astype(float))
        _y  = StandardScaler().fit_transform(tr_y.astype(float).reshape(-1, 1)).ravel()
        _N  = len(_y)

        # Step 1: data-stats estimation (Cherkassky & Ma 2004 heuristics)
        # MATLAB line 103 prepends prod(x') interaction term when multivariate.
        # Skipped — numerically unstable for high-dim inputs; washes out after grid search.
        _reg = _xs.copy()
        for _j in range(1, 21):
            _reg = np.hstack([_reg, _xs ** _j])
        _lp, _, _, _ = np.linalg.lstsq(_reg, _y, rcond=None)
        _resid  = _reg @ _lp - _y
        _varest = float((_resid @ _resid) / max(_N - 21, 1))
        _mean_y = float(np.mean(_y))

        _set(params, 'C',       float(max(abs(_mean_y + 3 * np.sqrt(_varest)),
                                           abs(_mean_y - 3 * np.sqrt(_varest)))))
        _set(params, 'epsilon', float(3 * np.sqrt(_varest * np.log(_N) / _N)))
        _ranges = _xs.max(axis=0) - _xs.min(axis=0)
        _set(params, 'sigma',   float(np.mean(_ranges * 0.3) ** (1.0 / _xs.shape[1])))

        _tr_s    = Struct(x=_xs, y=_y)
        _trtest  = Struct(x=[_xs], y=[_y])
        _svr_idx = algo_list.index('svr')

        # Step 2: 1-D sigma grid search
        _hp   = np.logspace(-10, 10, 100)
        _jmse = [modelopttest(float(s), params, _svr_idx, _tr_s, _trtest) for s in _hp]
        _set(params, 'sigma', float(_hp[int(np.argmin(_jmse))]))

        # Step 3: 3-D while-loop refinement (MATLAB parity + 10-attempt guard)
        _attempt = 0
        while float(min(_jmse)) > 0.5 and _attempt < 10:
            _x0 = np.array([_get(params, 'sigma'), _get(params, 'C'), _get(params, 'epsilon')])
            _x_opt, _, _, _ = optimsearch(_x0, params, _tr_s, _trtest, _svr_idx)
            _set(params, 'sigma',   float(_x_opt[0]))
            _set(params, 'C',       float(_x_opt[1]))
            _set(params, 'epsilon', float(_x_opt[2]))
            _jmse = [modelopttest(float(s), params, _svr_idx, _tr_s, _trtest) for s in _hp]
            _attempt += 1

    # ── secondary data (validation sev==2, test sev==3) ──────────────────────
    sec_path = _get(params, 'valpath' if sev == 2 else 'testpath', '')
    sec_data = _load(sec_path)
    rawdata_secondary = sec_data.copy() if sec_data.size > 0 else np.array([])

    if sec_data.size > 0 and feature_indices:
        sec_y = sec_data[:, target_idx] if target_idx < sec_data.shape[1] else np.zeros(len(sec_data))
        sec_x = sec_data[:, feature_indices]
    else:
        sec_x = np.empty((0, len(feature_indices) if feature_indices else 0))
        sec_y = np.array([])

    secondary = Struct(x=sec_x, y=sec_y, header=tr.header)

    return (tr, secondary, train_cell, rawdata_secondary, rawdata_tr,
            params, Statistics, feature_indices, fullfeatures)
