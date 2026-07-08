"""
tests/ldslearn/test_lds_timeseries.py
---------------------------------------
Pytest suite for ldslearn/lds_timeseries.py.
"""

import importlib.util
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('user_input_ressarch', MagicMock())

_ldslearn_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ldslearn")
# lds_timeseries.py imports its siblings (guess, ldsparamsidx, learn_kalman) as
# bare modules, so ldslearn/ must be importable on sys.path before exec_module runs.
sys.path.insert(0, _ldslearn_dir)

_mod_path = os.path.join(_ldslearn_dir, "lds_timeseries.py")
_spec = importlib.util.spec_from_file_location("ldslearn.lds_timeseries", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

lds_timeseries = _mod.lds_timeseries
learn_lds = _mod.learn_lds


def test_learn_kalman_call_site_signature_and_unpacking():
    # lds_timeseries.py's EM-learning branch (learn_flag=True) calls:
    #   params.learned, params.ll, llp, aici = learn_kalman(
    #       y_seg, params, max_iter, diag_q, diag_r, ar_mode, verbose, u_seg, asos_flag
    #   )
    # a positional call with y_seg as a raw (not list-wrapped) 2-D array and
    # u_seg=None (the only live calling convention today). Exercise that exact
    # call directly against the real learn_kalman() to catch any future
    # positional-signature drift or return-arity mismatch between the two
    # files, without running the full lds_timeseries() pipeline (stability
    # fixes, DARE, ASOS-dir handling) — that end-to-end path is #017's scope.
    ss, os_, T = 2, 3, 40
    rng = np.random.default_rng(70)
    A = np.diag(rng.uniform(0.3, 0.7, size=ss))
    C = rng.standard_normal((os_, ss))
    Q = np.eye(ss) * 0.05
    R = np.eye(os_) * 0.05

    x_prev = np.zeros(ss)
    y_seg = np.zeros((os_, T))
    for t in range(T):
        w = rng.multivariate_normal(np.zeros(ss), Q)
        x_prev = A @ x_prev + w
        v = rng.multivariate_normal(np.zeros(os_), R)
        y_seg[:, t] = C @ x_prev + v

    learn_kalman = _mod.learn_kalman
    params = SimpleNamespace(init=SimpleNamespace(
        ad=A, cd=C, qwd=Q, rvd=R, initx0=np.zeros(ss), xssd=np.eye(ss),
    ))

    # max_iter kept small (unlike lds_timeseries.py's hardcoded np.inf) so
    # this stays fast — only the call convention is under test here, not EM
    # convergence behavior (already covered by tests/ldslearn/test_learn_kalman.py).
    result = learn_kalman(y_seg, params, 3, False, False, False, False, None, False)

    assert len(result) == 4
    learned, ll, llp, aici = result
    assert len(aici) == len(ll)
    assert len(llp) == len(ll)
