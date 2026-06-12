"""
tests/test_modelopttest.py
---------------------------
Pytest suite for regressopt/modelopttest.py (modelopttest function).

Covers:
  - Return contract: finite non-negative float on valid data
  - Single-fold fallback: no filelength → train_mask == test_mask (full set)
  - Multi-fold CV: fold call count and sizes match filelengths exactly
  - Training set excludes test fold (multi-fold)
  - DOF computation: total_samples − num_folds (multi), total_samples (single)
  - Variance: ddof=1 (MATLAB Bessel correction, differs from ddof=0 for small N)
  - Zero variance guard: constant y → inf
  - Zero DOF guard: filelength=[1]*N → inf
  - NaN predictions: prints warning, function does not raise
  - tst batch-wrapping contract: tstpart.x and tstpart.y are lists of length 1
  - algo_list selection: params.algo[algIdx] is passed as a single-element list
  - Integration smoke test: real Ridge ('lin') returns finite non-negative score
"""

import sys
import importlib.util
import os
from unittest.mock import MagicMock
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module isolation
# ---------------------------------------------------------------------------

sys.modules.setdefault('user_input_ressarch', MagicMock())

_path = os.path.join(os.path.dirname(__file__), "..", "regressopt", "modelopttest.py")
_spec = importlib.util.spec_from_file_location("regressopt.modelopttest", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

modelopttest = _mod.modelopttest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class Struct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


_ALL_ALGOS = ['gp', 'svr', 'libsvr', 'knn', 'btree', 'lin', 'quad', 'bnet', 'elm', 'ransac']
_LIN_IDX = 5  # params.algo[5] == 'lin' → Ridge (algo_idx 6 in mainREGcode)


def _make_params(filelength=None, algos=None):
    p = Struct()
    p.filelength = filelength if filelength is not None else [30, 20]
    p.algo = algos if algos is not None else _ALL_ALGOS
    return p


def _make_tr_tst(n=50, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = X @ [1.0, -0.5, 0.3] + 0.1 * rng.standard_normal(n)
    tr = Struct(x=X, y=y)
    tst = Struct(x=[X], y=[y])
    return tr, tst


def _mock_engine_zeros(x, trpart, tstpart, algo_list, params):
    """Predict zeros for every fold."""
    output = Struct(yhat=[np.zeros_like(tstpart.y[0])])
    return output, {}


def _mock_engine_perfect(x, trpart, tstpart, algo_list, params):
    """Predict the true y (zero residuals)."""
    output = Struct(yhat=[tstpart.y[0].copy()])
    return output, {}


# ---------------------------------------------------------------------------
# 1. Return contract
# ---------------------------------------------------------------------------

class TestReturnContract:
    def test_returns_float(self, monkeypatch):
        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', _mock_engine_zeros)
        params = _make_params()
        tr, tst = _make_tr_tst()
        result = modelopttest(1.0, params, _LIN_IDX, tr, tst)
        assert isinstance(result, float)

    def test_returns_finite_non_negative(self, monkeypatch):
        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', _mock_engine_zeros)
        params = _make_params()
        tr, tst = _make_tr_tst()
        result = modelopttest(1.0, params, _LIN_IDX, tr, tst)
        assert np.isfinite(result)
        assert result >= 0.0

    def test_perfect_predictions_return_zero(self, monkeypatch):
        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', _mock_engine_perfect)
        params = _make_params()
        tr, tst = _make_tr_tst()
        result = modelopttest(1.0, params, _LIN_IDX, tr, tst)
        assert result == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Single-fold fallback
# ---------------------------------------------------------------------------

class TestSingleFoldFallback:
    def test_no_filelength_attribute_calls_engine_once(self, monkeypatch):
        calls = []

        def engine(x, trpart, tstpart, algo_list, params):
            calls.append(len(tstpart.y[0]))
            return Struct(yhat=[np.zeros_like(tstpart.y[0])]), {}

        p = Struct()
        p.algo = ['lin']
        tr, tst = _make_tr_tst(n=20)
        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', engine)
        modelopttest(1.0, p, 0, tr, tst)

        assert len(calls) == 1

    def test_no_filelength_train_and_test_are_full_set(self, monkeypatch):
        """Single-fold: train_mask == test_mask, both arrays span all N samples."""
        sizes = []

        def engine(x, trpart, tstpart, algo_list, params):
            sizes.append({'tr': len(trpart.y), 'tst': len(tstpart.y[0])})
            return Struct(yhat=[np.zeros_like(tstpart.y[0])]), {}

        p = Struct()
        p.algo = ['lin']
        tr, tst = _make_tr_tst(n=20)
        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', engine)
        modelopttest(1.0, p, 0, tr, tst)

        assert sizes[0]['tr'] == 20
        assert sizes[0]['tst'] == 20

    def test_empty_filelength_triggers_fallback(self, monkeypatch):
        calls = []

        def engine(x, trpart, tstpart, algo_list, params):
            calls.append(len(tstpart.y[0]))
            return Struct(yhat=[np.zeros_like(tstpart.y[0])]), {}

        params = _make_params(filelength=[])
        tr, tst = _make_tr_tst(n=15)
        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', engine)
        modelopttest(1.0, params, _LIN_IDX, tr, tst)

        assert len(calls) == 1
        assert calls[0] == 15


# ---------------------------------------------------------------------------
# 3. Multi-fold boundaries
# ---------------------------------------------------------------------------

class TestMultiFoldBoundaries:
    def test_fold_count_and_sizes_match_filelength(self, monkeypatch):
        calls = []

        def engine(x, trpart, tstpart, algo_list, params):
            calls.append(len(tstpart.y[0]))
            return Struct(yhat=[np.zeros_like(tstpart.y[0])]), {}

        params = _make_params(filelength=[10, 20, 15])
        tr, tst = _make_tr_tst(n=45)
        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', engine)
        modelopttest(1.0, params, _LIN_IDX, tr, tst)

        assert calls == [10, 20, 15]

    def test_training_excludes_test_fold(self, monkeypatch):
        """Each fold's training set size = total − fold_size."""
        sizes = []

        def engine(x, trpart, tstpart, algo_list, params):
            sizes.append({'tr': len(trpart.y), 'tst': len(tstpart.y[0])})
            return Struct(yhat=[np.zeros_like(tstpart.y[0])]), {}

        params = _make_params(filelength=[20, 30])
        tr, tst = _make_tr_tst(n=50)
        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', engine)
        modelopttest(1.0, params, _LIN_IDX, tr, tst)

        assert sizes[0] == {'tr': 30, 'tst': 20}
        assert sizes[1] == {'tr': 20, 'tst': 30}

    def test_fold_test_sets_partition_full_dataset(self, monkeypatch):
        """Union of all fold test sets equals the full dataset with no overlap."""
        n = 45
        rng = np.random.default_rng(3)
        X = rng.standard_normal((n, 2))
        y_tagged = np.arange(n, dtype=float)  # unique value per sample

        tr = Struct(x=X, y=y_tagged)
        tst = Struct(x=[X], y=[y_tagged])

        seen = []

        def engine(x, trpart, tstpart, algo_list, params):
            seen.extend(tstpart.y[0].tolist())
            return Struct(yhat=[np.zeros_like(tstpart.y[0])]), {}

        params = _make_params(filelength=[10, 20, 15])
        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', engine)
        modelopttest(1.0, params, _LIN_IDX, tr, tst)

        assert sorted(seen) == list(range(n))


# ---------------------------------------------------------------------------
# 4 & 5. DOF computation
# ---------------------------------------------------------------------------

class TestDegreesOfFreedom:
    def _run_fixed_residual(self, monkeypatch, params, n, residual):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 2))
        y = rng.standard_normal(n)
        tr = Struct(x=X, y=y)
        tst = Struct(x=[X], y=[y])

        def engine(x, trpart, tstpart, algo_list, p):
            pred = tstpart.y[0] - residual
            return Struct(yhat=[pred]), {}

        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', engine)
        score = modelopttest(1.0, params, _LIN_IDX, tr, tst)
        return score, y

    def test_dof_multi_fold_is_total_minus_num_folds(self, monkeypatch):
        n, filelength, residual = 50, [30, 20], 1.0
        params = _make_params(filelength=filelength)
        score, y = self._run_fixed_residual(monkeypatch, params, n, residual)

        dof = n - len(filelength)
        expected = (residual ** 2 * n) / (dof * np.var(y, ddof=1))
        assert score == pytest.approx(expected, rel=1e-10)

    def test_dof_single_fold_is_total_samples(self, monkeypatch):
        n, residual = 20, 2.0
        p = Struct()
        p.algo = _ALL_ALGOS  # no filelength attr → single-fold fallback
        score, y = self._run_fixed_residual(monkeypatch, p, n, residual)

        expected = (residual ** 2 * n) / (n * np.var(y, ddof=1))
        assert score == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# 6. Variance uses ddof=1 (MATLAB Bessel correction)
# ---------------------------------------------------------------------------

class TestVarianceBesselCorrection:
    def test_score_matches_ddof1_not_ddof0(self, monkeypatch):
        """For small N, ddof=1 and ddof=0 produce measurably different scores."""
        n, residual = 10, 0.5
        filelength = [5, 5]
        rng = np.random.default_rng(7)
        X = rng.standard_normal((n, 2))
        y = rng.standard_normal(n)

        params = _make_params(filelength=filelength)
        tr = Struct(x=X, y=y)
        tst = Struct(x=[X], y=[y])

        def engine(x, trpart, tstpart, algo_list, p):
            return Struct(yhat=[tstpart.y[0] - residual]), {}

        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', engine)
        score = modelopttest(1.0, params, _LIN_IDX, tr, tst)

        dof = n - len(filelength)
        sse = residual ** 2 * n
        expected_ddof1 = sse / (dof * np.var(y, ddof=1))
        expected_ddof0 = sse / (dof * np.var(y, ddof=0))

        assert score == pytest.approx(expected_ddof1, rel=1e-10)
        assert abs(score - expected_ddof0) > 1e-10


# ---------------------------------------------------------------------------
# 7. Zero variance guard
# ---------------------------------------------------------------------------

class TestZeroVarianceGuard:
    def test_constant_y_returns_inf(self, monkeypatch):
        n = 20
        X = np.ones((n, 2))
        y = np.full(n, 3.14)

        params = _make_params(filelength=[10, 10])
        tr = Struct(x=X, y=y)
        tst = Struct(x=[X], y=[y])

        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', _mock_engine_zeros)
        assert modelopttest(1.0, params, _LIN_IDX, tr, tst) == float('inf')


# ---------------------------------------------------------------------------
# 8. Zero DOF guard
# ---------------------------------------------------------------------------

class TestZeroDOFGuard:
    def test_one_sample_per_fold_returns_inf(self, monkeypatch):
        """filelength=[1]*N → DOF = N − N = 0 → inf."""
        n = 5
        rng = np.random.default_rng(1)
        X = rng.standard_normal((n, 2))
        y = rng.standard_normal(n)

        params = _make_params(filelength=[1] * n)
        tr = Struct(x=X, y=y)
        tst = Struct(x=[X], y=[y])

        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', _mock_engine_zeros)
        assert modelopttest(1.0, params, _LIN_IDX, tr, tst) == float('inf')


# ---------------------------------------------------------------------------
# 9. NaN predictions warning
# ---------------------------------------------------------------------------

class TestNaNPredictions:
    def _engine_nan(self, x, trpart, tstpart, algo_list, params):
        return Struct(yhat=[np.full_like(tstpart.y[0], np.nan)]), {}

    def test_nan_prints_warning_to_stdout(self, monkeypatch, capsys):
        params = _make_params(filelength=[25, 25])
        tr, tst = _make_tr_tst(n=50)
        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', self._engine_nan)
        modelopttest(1.0, params, _LIN_IDX, tr, tst)

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "NaN" in captured.out

    def test_nan_does_not_raise(self, monkeypatch):
        params = _make_params(filelength=[25, 25])
        tr, tst = _make_tr_tst(n=50)
        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', self._engine_nan)
        modelopttest(1.0, params, _LIN_IDX, tr, tst)  # must not raise


# ---------------------------------------------------------------------------
# 10. tst batch-wrapping contract
# ---------------------------------------------------------------------------

class TestBatchWrappingContract:
    def test_tstpart_x_and_y_are_single_element_lists(self, monkeypatch):
        received = []

        def engine(x, trpart, tstpart, algo_list, params):
            received.append(tstpart)
            return Struct(yhat=[np.zeros_like(tstpart.y[0])]), {}

        params = _make_params(filelength=[20, 30])
        tr, tst = _make_tr_tst(n=50)
        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', engine)
        modelopttest(1.0, params, _LIN_IDX, tr, tst)

        for tstpart in received:
            assert isinstance(tstpart.x, list)
            assert isinstance(tstpart.y, list)
            assert len(tstpart.x) == 1
            assert len(tstpart.y) == 1


# ---------------------------------------------------------------------------
# 11. algo_list selection
# ---------------------------------------------------------------------------

class TestAlgoListSelection:
    def test_algidx_selects_correct_algo_string(self, monkeypatch):
        """params.algo[algIdx] is forwarded as a single-element list."""
        algo_names = ['lin', 'svr', 'knn']
        params = _make_params(filelength=[30], algos=algo_names)
        tr, tst = _make_tr_tst(n=30)

        for idx, expected_name in enumerate(algo_names):
            received = []

            def engine(x, trpart, tstpart, algo_list, p, _n=expected_name):
                received.append(algo_list)
                return Struct(yhat=[np.zeros_like(tstpart.y[0])]), {}

            monkeypatch.setattr(_mod, 'mainREGcode_ressarch', engine)
            modelopttest(1.0, params, idx, tr, tst)

            assert received[0] == [expected_name]

    def test_algo_list_is_always_length_one(self, monkeypatch):
        received = []

        def engine(x, trpart, tstpart, algo_list, params):
            received.append(algo_list)
            return Struct(yhat=[np.zeros_like(tstpart.y[0])]), {}

        params = _make_params(filelength=[30, 20])
        tr, tst = _make_tr_tst(n=50)
        monkeypatch.setattr(_mod, 'mainREGcode_ressarch', engine)
        modelopttest(1.0, params, _LIN_IDX, tr, tst)

        for algo_list in received:
            assert isinstance(algo_list, list)
            assert len(algo_list) == 1


# ---------------------------------------------------------------------------
# 12. Integration smoke test (real Ridge via 'lin')
# ---------------------------------------------------------------------------

class TestIntegrationRidge:
    def test_ridge_returns_finite_non_negative_score(self):
        """End-to-end: real Ridge ('lin') through modelopttest returns a valid score."""
        rng = np.random.default_rng(42)
        n = 60
        X = rng.standard_normal((n, 3))
        y = X @ [2.0, -1.0, 0.5] + 0.2 * rng.standard_normal(n)

        params = _make_params(filelength=[30, 30], algos=_ALL_ALGOS)
        tr = Struct(x=X, y=y)
        tst = Struct(x=[X], y=[y])

        score = modelopttest(1.0, params, _LIN_IDX, tr, tst)

        assert np.isfinite(score)
        assert score >= 0.0
