"""
tests/test_modelsearch.py
--------------------------
Pytest suite for regressopt/modelsearch.py (optimsearch and _make_time_guard).

Covers:
  - _make_time_guard: passthrough when max_time is None, stop-signal timing,
    and base_cb argument forwarding
  - Bounds computation: x0=0 epsilon guard, negative x0, single-element list cast
  - Solver routing: opt_idx 1-5 + default (L-BFGS-B) + distrib=1 branching
  - History contracts: localhistory population via callback, globalhistory keys
  - Return contract: 4-tuple shape and types
"""

import sys
import importlib.util
import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module isolation
# ---------------------------------------------------------------------------

sys.modules.setdefault('user_input_ressarch', MagicMock())

_path = os.path.join(os.path.dirname(__file__), "..", "regressopt", "modelsearch.py")
_spec = importlib.util.spec_from_file_location("regressopt.modelsearch", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

optimsearch = _mod.optimsearch
_make_time_guard = _mod._make_time_guard


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class Struct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_params(opt_idx=6, max_time=None, distrib=2):
    regress = Struct(optIdx=opt_idx, maxtime=max_time)
    return Struct(regress=regress, distrib=distrib, algo=['gp'], filelength=[5])


def _make_tr_trtest(n=5, n_features=2):
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n, n_features))
    y = rng.standard_normal(n)
    tr = Struct(x=x, y=y)
    trtest = Struct(x=[x], y=[y])
    return tr, trtest


_MOCK_RESULT = MagicMock(x=np.array([1.0]), fun=0.5, success=True, message='converged')


# ---------------------------------------------------------------------------
# Module-level autouse fixture: isolate modelopttest from real ML execution
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_modelopttest():
    """Replace modelopttest with a cheap quadratic so solvers have a real objective."""
    _mod.modelopttest = lambda x, *a, **kw: float(np.sum(np.asarray(x) ** 2))
    yield


# ---------------------------------------------------------------------------
# 1. _make_time_guard
# ---------------------------------------------------------------------------

class TestTimeGuard:
    """
    Verifies _make_time_guard: passthrough when max_time is None, stop-signal
    logic based on monotonic elapsed time, and forwarding of args to base_cb.
    """

    def test_returns_base_cb_when_max_time_none(self):
        """_make_time_guard(None, cb) returns the same cb object unchanged."""
        cb = MagicMock()
        assert _make_time_guard(None, cb) is cb

    def test_returns_none_when_both_args_none(self):
        """_make_time_guard(None) returns None when no base_cb is supplied."""
        assert _make_time_guard(None) is None

    def test_returns_callable_when_max_time_set(self):
        """_make_time_guard returns a callable when max_time is a positive number."""
        guard = _make_time_guard(10.0)
        assert callable(guard)

    def test_guard_returns_false_before_timeout(self):
        """Guard callback returns False (keep running) when elapsed < max_time."""
        with patch.object(_mod, 'time') as mock_time:
            mock_time.monotonic.side_effect = [0.0, 5.0]  # start=0, now=5
            guard = _make_time_guard(10.0)
            result = guard(np.array([1.0]))
        assert result is False

    def test_guard_returns_true_after_timeout(self):
        """Guard callback returns True (stop solver) when elapsed > max_time."""
        with patch.object(_mod, 'time') as mock_time:
            mock_time.monotonic.side_effect = [0.0, 20.0]  # start=0, now=20
            guard = _make_time_guard(10.0)
            result = guard(np.array([1.0]))
        assert result is True

    def test_base_cb_receives_same_args(self):
        """Wrapped guard forwards the xk positional arg to base_cb unchanged."""
        received = []

        def base_cb(xk, *args, **kwargs):
            received.append(xk)

        with patch.object(_mod, 'time') as mock_time:
            mock_time.monotonic.side_effect = [0.0, 1.0]
            guard = _make_time_guard(10.0, base_cb)
            xk = np.array([3.0, 4.0])
            guard(xk)

        assert len(received) == 1
        assert np.array_equal(received[0], xk)


# ---------------------------------------------------------------------------
# 2. Bounds computation
# ---------------------------------------------------------------------------

class TestBoundsComputation:
    """
    Verifies the log-scale epsilon-guarded bounds formula. Patches the default
    L-BFGS-B solver to capture bounds without running real optimization.
    """

    def test_bounds_finite_when_x0_zero(self):
        """x0=[0]: eps guard prevents log10(0)=-inf, so all bounds stay finite."""
        params = _make_params(opt_idx=6)
        tr, trtest = _make_tr_trtest()
        with patch.object(_mod, 'minimize') as mock_min:
            mock_min.return_value = _MOCK_RESULT
            optimsearch([0.0], params, tr, trtest, 0)
            captured_bounds = mock_min.call_args.kwargs['bounds']
        lowers = [b[0] for b in captured_bounds]
        uppers = [b[1] for b in captured_bounds]
        assert np.all(np.isfinite(lowers))
        assert np.all(np.isfinite(uppers))

    def test_bounds_positive_when_x0_negative(self):
        """x0<0: abs() guard ensures log10 receives positive input; bounds stay positive."""
        params = _make_params(opt_idx=6)
        tr, trtest = _make_tr_trtest()
        with patch.object(_mod, 'minimize') as mock_min:
            mock_min.return_value = _MOCK_RESULT
            optimsearch([-5.0], params, tr, trtest, 0)
            captured_bounds = mock_min.call_args.kwargs['bounds']
        lowers = [b[0] for b in captured_bounds]
        uppers = [b[1] for b in captured_bounds]
        assert np.all(np.array(lowers) > 0)
        assert np.all(np.array(uppers) > 0)

    def test_x0_list_cast_to_float_ndarray(self):
        """x0 as a Python list is cast to a float ndarray before the solver call."""
        params = _make_params(opt_idx=6)
        tr, trtest = _make_tr_trtest()
        with patch.object(_mod, 'minimize') as mock_min:
            mock_min.return_value = _MOCK_RESULT
            optimsearch([1.0], params, tr, trtest, 0)
            # second positional arg to minimize is x0
            call_x0 = mock_min.call_args.args[1]
        assert isinstance(call_x0, np.ndarray)
        assert call_x0.dtype == float


# ---------------------------------------------------------------------------
# 3. Solver routing
# ---------------------------------------------------------------------------

class TestSolverRouting:
    """
    Verifies that each opt_idx dispatches to the correct scipy solver, and that
    the distrib=1 fork routes opt_idx=1 to differential_evolution instead of shgo.
    """

    def _call(self, opt_idx, distrib=2):
        params = _make_params(opt_idx=opt_idx, distrib=distrib)
        tr, trtest = _make_tr_trtest()
        return params, tr, trtest

    def test_optidx1_routes_to_shgo(self):
        """opt_idx=1, distrib=2 (default) dispatches to shgo (Global Search)."""
        params, tr, trtest = self._call(opt_idx=1)
        with patch.object(_mod, 'shgo') as mock_shgo:
            mock_shgo.return_value = _MOCK_RESULT
            optimsearch([1.0], params, tr, trtest, 0)
        mock_shgo.assert_called_once()

    def test_optidx2_routes_to_dual_annealing(self):
        """opt_idx=2 dispatches to dual_annealing (Simulated Annealing)."""
        params, tr, trtest = self._call(opt_idx=2)
        with patch.object(_mod, 'dual_annealing') as mock_da:
            mock_da.return_value = _MOCK_RESULT
            optimsearch([1.0], params, tr, trtest, 0)
        mock_da.assert_called_once()

    def test_optidx3_routes_to_differential_evolution(self):
        """opt_idx=3 dispatches to differential_evolution (Genetic Algorithm)."""
        params, tr, trtest = self._call(opt_idx=3)
        with patch.object(_mod, 'differential_evolution') as mock_de:
            mock_de.return_value = _MOCK_RESULT
            optimsearch([1.0], params, tr, trtest, 0)
        mock_de.assert_called_once()

    def test_optidx4_routes_to_nelder_mead(self):
        """opt_idx=4 dispatches to minimize(..., method='Nelder-Mead') (Pattern Search)."""
        params, tr, trtest = self._call(opt_idx=4)
        with patch.object(_mod, 'minimize') as mock_min:
            mock_min.return_value = _MOCK_RESULT
            optimsearch([1.0], params, tr, trtest, 0)
        mock_min.assert_called_once()
        assert mock_min.call_args.kwargs['method'] == 'Nelder-Mead'

    def test_optidx5_routes_to_basinhopping(self):
        """opt_idx=5 dispatches to basinhopping (Multistart)."""
        params, tr, trtest = self._call(opt_idx=5)
        with patch.object(_mod, 'basinhopping') as mock_bh:
            mock_bh.return_value = _MOCK_RESULT
            optimsearch([1.0], params, tr, trtest, 0)
        mock_bh.assert_called_once()

    def test_default_routes_to_lbfgsb(self):
        """opt_idx=6 (default) dispatches to minimize(..., method='L-BFGS-B') (local fmincon)."""
        params, tr, trtest = self._call(opt_idx=6)
        with patch.object(_mod, 'minimize') as mock_min:
            mock_min.return_value = _MOCK_RESULT
            optimsearch([1.0], params, tr, trtest, 0)
        mock_min.assert_called_once()
        assert mock_min.call_args.kwargs['method'] == 'L-BFGS-B'

    def test_distrib1_optidx1_routes_to_differential_evolution(self):
        """distrib=1, opt_idx=1 must use differential_evolution, not shgo."""
        params, tr, trtest = self._call(opt_idx=1, distrib=1)
        with patch.object(_mod, 'differential_evolution') as mock_de, \
             patch.object(_mod, 'shgo') as mock_shgo:
            mock_de.return_value = _MOCK_RESULT
            optimsearch([1.0], params, tr, trtest, 0)
        mock_de.assert_called_once()
        mock_shgo.assert_not_called()

    def test_distrib1_other_routes_to_shgo(self):
        """distrib=1, opt_idx!=1 falls back to shgo (not differential_evolution)."""
        params, tr, trtest = self._call(opt_idx=3, distrib=1)
        with patch.object(_mod, 'shgo') as mock_shgo, \
             patch.object(_mod, 'differential_evolution') as mock_de:
            mock_shgo.return_value = _MOCK_RESULT
            optimsearch([1.0], params, tr, trtest, 0)
        mock_shgo.assert_called_once()
        mock_de.assert_not_called()


# ---------------------------------------------------------------------------
# 4. History contracts
# ---------------------------------------------------------------------------

class TestHistoryContracts:
    """
    Verifies localhistory and globalhistory output contracts. localhistory is
    populated by callback_tracker when the solver fires the callback; globalhistory
    keys are unconditionally present.
    """

    def test_globalhistory_always_has_required_keys(self):
        """globalhistory must always carry 'exitflag' and 'output' regardless of solver."""
        params = _make_params(opt_idx=6)
        tr, trtest = _make_tr_trtest()
        with patch('regressopt.modelsearch.minimize') as mock_min:
            mock_min.return_value = _MOCK_RESULT
            _, _, _, ghist = optimsearch([1.0], params, tr, trtest, 0)
        assert 'exitflag' in ghist
        assert 'output' in ghist

    def test_localhistory_populated_when_callback_invoked(self):
        """When solver fires the callback, localhistory records x and fval entries."""
        params = _make_params(opt_idx=2, max_time=None)  # dual_annealing path
        tr, trtest = _make_tr_trtest()

        def fake_dual_annealing(objective, bounds, callback=None, x0=None):
            if callback is not None:
                callback(np.array([1.0]))
            return MagicMock(x=np.array([1.0]), fun=0.5, success=True, message='ok')

        with patch.object(_mod, 'dual_annealing', side_effect=fake_dual_annealing):
            _, _, lhist, _ = optimsearch([1.0], params, tr, trtest, 0)

        assert len(lhist['x']) == 1
        assert len(lhist['fval']) == 1
        assert np.array_equal(lhist['x'][0], np.array([1.0]))
        assert isinstance(lhist['fval'][0], float)

    def test_localhistory_empty_when_solver_has_no_callback(self):
        """shgo path passes no callback arg; localhistory stays empty."""
        params = _make_params(opt_idx=1, distrib=2)
        tr, trtest = _make_tr_trtest()
        with patch.object(_mod, 'shgo') as mock_shgo:
            mock_shgo.return_value = _MOCK_RESULT
            _, _, lhist, _ = optimsearch([1.0], params, tr, trtest, 0)
        assert lhist['x'] == []
        assert lhist['fval'] == []


# ---------------------------------------------------------------------------
# 5. Return contract
# ---------------------------------------------------------------------------

class TestReturnContract:
    """
    Verifies the 4-tuple (x, fval, localhistory, globalhistory) return shape
    and element types from optimsearch.
    """

    def test_return_is_four_tuple(self):
        """optimsearch always returns exactly 4 values."""
        params = _make_params(opt_idx=6)
        tr, trtest = _make_tr_trtest()
        with patch.object(_mod, 'minimize') as mock_min:
            mock_min.return_value = _MOCK_RESULT
            result = optimsearch([1.0], params, tr, trtest, 0)
        assert len(result) == 4

    def test_x_is_ndarray_fval_is_float(self):
        """x is a numpy ndarray and fval is a Python float."""
        params = _make_params(opt_idx=6)
        tr, trtest = _make_tr_trtest()
        with patch.object(_mod, 'minimize') as mock_min:
            mock_min.return_value = _MOCK_RESULT
            x, fval, _, _ = optimsearch([1.0], params, tr, trtest, 0)
        assert isinstance(x, np.ndarray)
        assert isinstance(fval, float)
