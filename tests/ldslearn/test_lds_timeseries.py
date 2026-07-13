"""
tests/ldslearn/test_lds_timeseries.py
---------------------------------------
Pytest suite for ldslearn/lds_timeseries.py.
"""

import contextlib
import importlib.util
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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

# lds_timeseries.py's own `from learn_kalman import learn_kalman` (a bare
# import, resolved via the sys.path.insert above) populates sys.modules
# under the plain name "learn_kalman" as a side effect of exec_module above
# — reuse that exact module object (not a second independent load) so that
# patching em_converged on it actually affects the learn_kalman() call
# lds_timeseries() makes internally.
_lk_mod = sys.modules['learn_kalman']


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


def _simulate_lds_with_input(ss, os_, is_, T, seed):
    """Simulate a small stable input-driven LDS, returning (y, u) as (channels, T) arrays."""
    rng = np.random.default_rng(seed)
    A = np.diag(rng.uniform(0.3, 0.7, size=ss))
    B = rng.standard_normal((ss, is_)) * 0.1
    C = rng.standard_normal((os_, ss))
    D = rng.standard_normal((os_, is_)) * 0.1
    Q = np.eye(ss) * 0.05
    R = np.eye(os_) * 0.05

    u = rng.standard_normal((is_, T))
    x_prev = np.zeros(ss)
    y = np.zeros((os_, T))
    for t in range(T):
        w = rng.multivariate_normal(np.zeros(ss), Q)
        x_prev = A @ x_prev + B @ u[:, t] + w
        v = rng.multivariate_normal(np.zeros(os_), R)
        y[:, t] = C @ x_prev + D @ u[:, t] + v
    return y, u


@pytest.mark.parametrize("inittype,learn_flag,asosflag,seed", [
    (1, False, False, 100),
    (2, False, False, 101),
    (1, True, False, 102),
    (2, True, False, 103),
    (1, True, True, 104),
    (2, True, True, 105),
])
def test_lds_timeseries_full_matrix(inittype, learn_flag, asosflag, seed):
    # Full matrix of #017's acceptance criteria: both inittype branches
    # (1=linear regression, 2=N4SID), both learn_flag values, and
    # asosflag=True (only meaningful combined with learn_flag=True, since
    # asosflag only affects the internal EM loop). ss/os_/is_ mirror
    # tests/ldslearn/test_learn_kalman.py's asosflag fixtures (edgesize=25
    # hardcoded at the ASOS call site requires T > 2*edgesize+klim+1 = 53).
    ss, os_, is_, T = 2, 3, 2, 150
    y, u = _simulate_lds_with_input(ss, os_, is_, T, seed=seed)

    params = SimpleNamespace(inittype=inittype, distrib=1)
    if asosflag:
        params.klim = 2

    # lds_timeseries.py hardcodes max_iter=np.inf for the EM branch (MATLAB
    # parity); force convergence after the first M-step so the test stays
    # fast and deterministic regardless of asosflag's non-monotonic
    # likelihood, same pattern as
    # test_learn_kalman.py::test_learn_kalman_stops_early_when_converged.
    em_ctx = (
        patch.object(_lk_mod, "em_converged", return_value=(True, False))
        if learn_flag else contextlib.nullcontext()
    )
    with em_ctx:
        result = lds_timeseries(params, ss, [y], [u], learn_flag, asosflag)

    assert result.adl.shape == (ss, ss)
    assert result.cdl.shape == (os_, ss)
    assert result.qwdl.shape == (ss, ss)
    assert result.rvdl.shape == (os_, os_)
    assert result.bdl.shape == (ss, is_)
    assert result.ddl.shape == (os_, is_)

    assert np.max(np.abs(np.linalg.eigvals(result.adl))) < 1

    assert result.dare.shape == (ss, ss)
    assert result.kfgain.shape == (ss, os_)
    assert result.pssd.shape == (ss, ss)
    assert np.all(np.isfinite(result.dare))
    assert np.all(np.isfinite(result.kfgain))
    assert np.all(np.isfinite(result.pssd))


@pytest.mark.parametrize("set_params_asos,expect_asos", [
    (True, True),
    (False, False),
])
def test_lds_timeseries_asosflag_defaults_from_params_asos(set_params_asos, expect_asos):
    # aux_input.py's user-config prompt sets params.asos (and params.klim)
    # directly on the same params object testoptloop_ressarch.py's run()
    # passes into lds_timeseries() — the real pipeline never passes an
    # explicit asosflag argument, so lds_timeseries() must honor
    # params.asos when asosflag is left at its default (None).
    ss, os_, is_, T = 2, 3, 2, 150
    y, u = _simulate_lds_with_input(ss, os_, is_, T, seed=200)

    params = SimpleNamespace(inittype=1, distrib=1)
    if set_params_asos:
        params.asos = True
        params.klim = 2
    else:
        params.asos = False

    with patch.object(_lk_mod, "em_converged", return_value=(True, False)), \
         patch.object(_lk_mod, "ApproxEStep", wraps=_lk_mod.ApproxEStep) as spy_asos:
        lds_timeseries(params, ss, [y], [u], True)  # asosflag left at default (None)

    assert spy_asos.called is expect_asos


def test_lds_timeseries_asosflag_missing_klim_raises():
    # #019: params.klim is unconditionally read deep inside learn_kalman.py's
    # asos_flag branch — lds_timeseries() must raise a clear error at its own
    # boundary instead of letting the caller hit an AttributeError later.
    ss, os_, is_, T = 2, 3, 2, 150
    y, u = _simulate_lds_with_input(ss, os_, is_, T, seed=210)

    params = SimpleNamespace(inittype=1, distrib=1)  # no params.klim set

    with pytest.raises(ValueError, match="klim"):
        lds_timeseries(params, ss, [y], [u], True, True)


def test_lds_timeseries_asosflag_segment_too_short_raises():
    # #019: the segment-selection loop's own too_short check is tied to nmax
    # and has nothing to do with ASOS's T > 2*edgesize + klim + 1 requirement
    # (edgesize=25 hardcoded). T=50 with klim=2 (threshold 53) passes the
    # nmax-based selection loop (nmax=ss=2, os_=3, is_=2 -> that check only
    # rejects ny < 47) but must still be rejected by the new ASOS-specific
    # check before reaching ApproxEStep(), where it would otherwise silently
    # wrap via a negative-index slice instead of raising.
    ss, os_, is_, T = 2, 3, 2, 50
    y, u = _simulate_lds_with_input(ss, os_, is_, T, seed=220)

    params = SimpleNamespace(inittype=1, distrib=1)
    params.klim = 2

    with pytest.raises(ValueError, match="too short"):
        lds_timeseries(params, ss, [y], [u], True, True)


def test_lds_timeseries_q_fallback_survives_solver_failure():
    # #019: the Q positive-definiteness fallback previously had no try/except
    # around inverse_covariance_selection, unlike the structurally identical
    # P0 (xssd) fallback a few lines below it. Force the Q check to fail and
    # the solver to raise, and confirm lds_timeseries() still returns a
    # symmetric positive-definite qwdl via the eigenvalue-nudge fallback.
    ss, os_, is_, T = 2, 3, 2, 150
    y, u = _simulate_lds_with_input(ss, os_, is_, T, seed=230)

    params = SimpleNamespace(inittype=1, distrib=1)

    def raise_solver_error(*args, **kwargs):
        raise ValueError("forced solver failure")

    # _is_pos_def is called exactly 3 times in the learn_flag branch, in
    # order: Q, R, xssd (P0). Force only the Q check to fail.
    with patch.object(_lk_mod, "em_converged", return_value=(True, False)), \
         patch.object(_mod, "_is_pos_def", side_effect=[False, True, True]), \
         patch.object(_mod, "inverse_covariance_selection", side_effect=raise_solver_error):
        result = lds_timeseries(params, ss, [y], [u], True, False)

    # Must not raise: qwdl is symmetric positive definite via the fallback.
    np.linalg.cholesky(result.qwdl)
