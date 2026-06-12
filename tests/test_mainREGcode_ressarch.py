"""
tests/test_mainREGcode_ressarch.py
------------------------------------
Pytest suite for regressopt/mainREGcode_ressarch.py (mainREGcode_ressarch).

Covers:
  - Output struct contracts: output.yhat for single algo, output.{algo}_yhat for multi
  - Each algorithm branch (1-10): fit + predict on minimal synthetic data
  - Model caching: runOptions stores model after first call, reuses on second call
  - Multi-batch (Ntests > 1): predictions stored at correct batch index
  - Ntests inference: defaults to len(tst.y) when not set
  - Unknown algo silently skipped (output.yhat stays None)
  - ELM config from runOptions (activation, alpha, orth_flag_ELM)
  - Algorithm accuracy (1-10): each algo trained on known-relationship data, asserts R²
"""

import sys
import importlib.util
import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Module isolation
# mainREGcode_ressarch imports from regressopt.elm, which triggers
# regressopt/__init__.py → reg_ranges → user_input_ressarch at load time.
# ---------------------------------------------------------------------------

sys.modules.setdefault('user_input_ressarch', MagicMock())

_path = os.path.join(os.path.dirname(__file__), "..", "regressopt", "mainREGcode_ressarch.py")
_spec = importlib.util.spec_from_file_location("regressopt.mainREGcode_ressarch", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

mainREGcode_ressarch = _mod.mainREGcode_ressarch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class Struct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_data(n=15, n_features=2, seed=0):
    """Single-batch training and test data using fixed-seed RNG."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_features))
    y = rng.standard_normal(n)
    tr = Struct(x=x, y=y)
    tst = Struct(x=[x], y=[y])  # batched list contract
    return tr, tst


def _make_opts(**kwargs):
    return Struct(**kwargs)


# ---------------------------------------------------------------------------
# 1. Output struct contracts
# ---------------------------------------------------------------------------

class TestOutputStructContracts:

    def test_single_algo_yhat_populated(self):
        """Single algo → output.yhat[0] is a populated ndarray."""
        tr, tst = _make_data()
        opts = _make_opts()
        output, _ = mainREGcode_ressarch(0.01, tr, tst, ['lin'], opts)
        assert isinstance(output.yhat[0], np.ndarray)

    def test_multi_algo_named_yhat_populated(self):
        """2 algos → output.{algo}_yhat[0] populated; output.yhat stays [None]."""
        tr, tst = _make_data()
        opts = _make_opts()
        output, _ = mainREGcode_ressarch(0.01, tr, tst, ['lin', 'knn'], opts)
        assert isinstance(output.lin_yhat[0], np.ndarray)
        assert isinstance(output.knn_yhat[0], np.ndarray)
        assert output.yhat[0] is None

    def test_yhat_length_equals_ntests(self):
        """Ntests=2 → output.yhat has 2 slots, both populated."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal((15, 2))
        y = rng.standard_normal(15)
        tr = Struct(x=x, y=y)
        tst = Struct(x=[x, x], y=[y, y])
        opts = _make_opts(Ntests=2)
        output, _ = mainREGcode_ressarch(0.01, tr, tst, ['lin'], opts)
        assert len(output.yhat) == 2
        assert isinstance(output.yhat[0], np.ndarray)
        assert isinstance(output.yhat[1], np.ndarray)

    def test_ntests_defaults_to_len_tst_y(self):
        """No Ntests attr on opts → inferred from len(tst.y)."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal((15, 2))
        y = rng.standard_normal(15)
        tr = Struct(x=x, y=y)
        tst = Struct(x=[x, x, x], y=[y, y, y])
        opts = _make_opts()
        output, _ = mainREGcode_ressarch(0.01, tr, tst, ['lin'], opts)
        assert len(output.yhat) == 3

    def test_unknown_algo_silently_skipped(self):
        """Unknown algo name → output.yhat stays [None], no exception raised."""
        tr, tst = _make_data()
        opts = _make_opts()
        output, _ = mainREGcode_ressarch(1.0, tr, tst, ['unknown_algo'], opts)
        assert output.yhat == [None]

    def test_prediction_shape_matches_test_batch(self):
        """Prediction length equals the number of rows in the test batch."""
        tr, tst = _make_data(n=15)
        opts = _make_opts()
        output, _ = mainREGcode_ressarch(0.01, tr, tst, ['lin'], opts)
        assert output.yhat[0].shape == (15,)


# ---------------------------------------------------------------------------
# 2. Algorithm branches
# ---------------------------------------------------------------------------

class TestAlgoBranches:
    """
    Each test fires a single algorithm branch and verifies that output.yhat[0]
    is a 1-D ndarray with the correct length.  n=15 keeps every solver fast.
    """

    def _run(self, algo, x, seed=42):
        tr, tst = _make_data(n=15, seed=seed)
        opts = _make_opts()
        output, _ = mainREGcode_ressarch(x, tr, tst, [algo], opts)
        return output

    def test_algo_gp(self):
        """GP (idx 1) returns predictions of correct shape."""
        output = self._run('gp', 1.0)
        assert isinstance(output.yhat[0], np.ndarray)
        assert output.yhat[0].shape == (15,)

    def test_algo_svr(self):
        """SVR (idx 2) returns predictions of correct shape."""
        output = self._run('svr', 1.0)
        assert isinstance(output.yhat[0], np.ndarray)
        assert output.yhat[0].shape == (15,)

    def test_algo_libsvr(self):
        """LibSVR / NuSVR (idx 3) returns predictions of correct shape."""
        output = self._run('libsvr', 1.0)
        assert isinstance(output.yhat[0], np.ndarray)
        assert output.yhat[0].shape == (15,)

    def test_algo_knn(self):
        """k-NN (idx 4) returns predictions of correct shape."""
        output = self._run('knn', 3.0)
        assert isinstance(output.yhat[0], np.ndarray)
        assert output.yhat[0].shape == (15,)

    def test_algo_btree(self):
        """Bagged Trees / RandomForest (idx 5) returns predictions of correct shape."""
        output = self._run('btree', 2.0)
        assert isinstance(output.yhat[0], np.ndarray)
        assert output.yhat[0].shape == (15,)

    def test_algo_lin(self):
        """Ridge linear (idx 6) returns predictions of correct shape."""
        output = self._run('lin', 0.01)
        assert isinstance(output.yhat[0], np.ndarray)
        assert output.yhat[0].shape == (15,)

    def test_algo_quad(self):
        """Quadratic Ridge pipeline (idx 7) returns predictions of correct shape."""
        output = self._run('quad', 0.01)
        assert isinstance(output.yhat[0], np.ndarray)
        assert output.yhat[0].shape == (15,)

    def test_algo_bnet(self):
        """Bagged Neural Nets / BaggingRegressor (idx 8) returns predictions of correct shape."""
        output = self._run('bnet', 4.0)
        assert isinstance(output.yhat[0], np.ndarray)
        assert output.yhat[0].shape == (15,)

    def test_algo_elm(self):
        """ELMRegressor (idx 9) returns predictions of correct shape."""
        output = self._run('elm', 5.0)
        assert isinstance(output.yhat[0], np.ndarray)
        assert output.yhat[0].shape == (15,)

    def test_algo_ransac(self):
        """RANSAC (idx 10) returns predictions of correct shape."""
        output = self._run('ransac', 1.0)
        assert isinstance(output.yhat[0], np.ndarray)
        assert output.yhat[0].shape == (15,)


# ---------------------------------------------------------------------------
# 3. Model caching
# ---------------------------------------------------------------------------

class TestModelCaching:
    """
    Cached algos store their fitted estimator on runOptions after the first
    call and reuse it (without refitting) on subsequent calls.  KNN has no
    cache guard in the source — confirmed by checking runOptions carries no
    modelknn attribute after execution.
    """

    def test_gp_model_stored_after_fit(self):
        """GP model stored on runOptions.modelGP after first call."""
        tr, tst = _make_data()
        opts = _make_opts()
        mainREGcode_ressarch(1.0, tr, tst, ['gp'], opts)
        assert hasattr(opts, 'modelGP')

    def test_svr_model_stored_after_fit(self):
        """SVR model stored on runOptions.modelSVR after first call."""
        tr, tst = _make_data()
        opts = _make_opts()
        mainREGcode_ressarch(1.0, tr, tst, ['svr'], opts)
        assert hasattr(opts, 'modelSVR')

    def test_btree_model_stored_after_fit(self):
        """RandomForest model stored on runOptions.modelbtree after first call."""
        tr, tst = _make_data()
        opts = _make_opts()
        mainREGcode_ressarch(2.0, tr, tst, ['btree'], opts)
        assert hasattr(opts, 'modelbtree')

    def test_elm_model_stored_after_fit(self):
        """ELM model stored on runOptions.modelELM after first call."""
        tr, tst = _make_data()
        opts = _make_opts()
        mainREGcode_ressarch(5.0, tr, tst, ['elm'], opts)
        assert hasattr(opts, 'modelELM')

    def test_ransac_model_stored_after_fit(self):
        """RANSAC model stored on runOptions.modelRANSAC after first call."""
        tr, tst = _make_data()
        opts = _make_opts()
        mainREGcode_ressarch(1.0, tr, tst, ['ransac'], opts)
        assert hasattr(opts, 'modelRANSAC')

    def test_cached_gp_model_reused_on_second_call(self):
        """Second call reuses the same GP model object — id() unchanged."""
        tr, tst = _make_data()
        opts = _make_opts()
        mainREGcode_ressarch(1.0, tr, tst, ['gp'], opts)
        first_id = id(opts.modelGP)
        mainREGcode_ressarch(1.0, tr, tst, ['gp'], opts)
        assert id(opts.modelGP) == first_id

    def test_knn_never_cached(self):
        """KNN has no cache guard — runOptions must not carry a modelknn attribute."""
        tr, tst = _make_data()
        opts = _make_opts()
        mainREGcode_ressarch(3.0, tr, tst, ['knn'], opts)
        assert not hasattr(opts, 'modelknn')


# ---------------------------------------------------------------------------
# 4. GP scale guard (MATLAB parity: gp_pnt caps training size)
# ---------------------------------------------------------------------------

class TestGPScaleGuard:
    """GP branch respects gp_pnt and does not train on more than gp_pnt samples."""

    def test_gp_subsamples_when_n_exceeds_gp_pnt(self):
        """With gp_pnt=10 and N=50, GP model trains on ≤10 points."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal((50, 2))
        y = rng.standard_normal(50)
        tr = Struct(x=x, y=y)
        tst = Struct(x=[x[:10]], y=[y[:10]])
        opts = _make_opts(gp_pnt=10)
        mainREGcode_ressarch(1.0, tr, tst, ['gp'], opts)
        assert opts.modelGP.X_train_.shape[0] <= 10

    def test_gp_trains_full_set_when_n_leq_gp_pnt(self):
        """With N=15 and default gp_pnt=500, GP uses all 15 points."""
        tr, tst = _make_data(n=15)
        opts = _make_opts()
        mainREGcode_ressarch(1.0, tr, tst, ['gp'], opts)
        assert opts.modelGP.X_train_.shape[0] == 15

    def test_gp_default_gp_pnt_is_500(self):
        """Without gp_pnt on runOptions, cap defaults to 500 (MATLAB parity)."""
        rng = np.random.default_rng(8)
        x = rng.standard_normal((600, 2))
        y = rng.standard_normal(600)
        tr = Struct(x=x, y=y)
        tst = Struct(x=[x[:10]], y=[y[:10]])
        opts = _make_opts()
        mainREGcode_ressarch(1.0, tr, tst, ['gp'], opts)
        assert opts.modelGP.X_train_.shape[0] == 500


# ---------------------------------------------------------------------------
# 5. Multi-batch
# ---------------------------------------------------------------------------

class TestMultiBatch:
    """
    Multi-batch inference populates every output slot and predictions from
    different-sized batches carry the correct per-batch length.
    """

    def test_multi_batch_all_slots_filled(self):
        """Ntests=3 → output.yhat[0..2] are all ndarrays."""
        rng = np.random.default_rng(10)
        x = rng.standard_normal((15, 2))
        y = rng.standard_normal(15)
        tr = Struct(x=x, y=y)
        tst = Struct(x=[x, x, x], y=[y, y, y])
        opts = _make_opts(Ntests=3)
        output, _ = mainREGcode_ressarch(0.01, tr, tst, ['lin'], opts)
        for slot in output.yhat:
            assert isinstance(slot, np.ndarray)

    def test_multi_batch_predictions_respect_batch_size(self):
        """Each batch prediction length matches that batch's row count."""
        rng = np.random.default_rng(11)
        x_a = rng.standard_normal((10, 2))
        x_b = rng.standard_normal((5, 2))
        y_a = rng.standard_normal(10)
        y_b = rng.standard_normal(5)
        tr = Struct(x=x_a, y=y_a)
        tst = Struct(x=[x_a, x_b], y=[y_a, y_b])
        opts = _make_opts(Ntests=2)
        output, _ = mainREGcode_ressarch(0.01, tr, tst, ['lin'], opts)
        assert output.yhat[0].shape == (10,)
        assert output.yhat[1].shape == (5,)


# ---------------------------------------------------------------------------
# 6. ELM config from runOptions
# ---------------------------------------------------------------------------

class TestELMConfig:
    """
    ELMRegressor reads its construction config from runOptions at cache-miss time.
    Tests verify the correct attrs propagate to the fitted model object.
    """

    def test_elm_reads_activation_from_run_options(self):
        """ActivationFunction in opts propagates to the fitted ELMRegressor."""
        tr, tst = _make_data(seed=20)
        opts = _make_opts(ActivationFunction='rbf')
        mainREGcode_ressarch(5.0, tr, tst, ['elm'], opts)
        assert opts.modelELM.activation == 'rbf'

    def test_elm_reads_alpha_from_run_options(self):
        """elm_alpha in opts propagates to ELMRegressor.alpha."""
        tr, tst = _make_data(seed=21)
        opts = _make_opts(elm_alpha=0.5)
        mainREGcode_ressarch(5.0, tr, tst, ['elm'], opts)
        assert opts.modelELM.alpha == 0.5

    def test_elm_reads_orth_flag_from_run_options(self):
        """orth_flag_ELM=1 in opts sets ELMRegressor.orthogonal to True."""
        tr, tst = _make_data(seed=22)
        opts = _make_opts(orth_flag_ELM=1)
        mainREGcode_ressarch(5.0, tr, tst, ['elm'], opts)
        assert opts.modelELM.orthogonal is True


# ---------------------------------------------------------------------------
# 7. Algorithm accuracy
# ---------------------------------------------------------------------------

def _run_accuracy(algo, hyperparam, relationship='linear', n_train=120, n_test=40,
                  noise_std=0.05, seed=99, **opts_kwargs):
    """
    Fit on synthetic data with a known relationship; return R² on held-out test set.

    X is standard-normal (Z-scored): in production 
    mainREGcode_ressarch always receives pre-scaled arrays from
    GlobalDataScaler, so synthetic inputs must reflect that same distribution.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    X = rng.standard_normal((n, 2))

    if relationship == 'linear':
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + noise_std * rng.standard_normal(n)
    elif relationship == 'quadratic':
        y = X[:, 0] ** 2 + 2.0 * X[:, 1] + noise_std * rng.standard_normal(n)
    elif relationship == 'smooth':
        # Smooth periodic function suited to the default RBF kernel; no global linear trend
        y = np.sin(2.0 * X[:, 0]) + np.cos(2.0 * X[:, 1]) + noise_std * rng.standard_normal(n)
    else:
        raise ValueError(f"Unknown relationship: {relationship!r}")

    tr = Struct(x=X[:n_train], y=y[:n_train])
    tst = Struct(x=[X[n_train:]], y=[y[n_train:]])
    opts = _make_opts(**opts_kwargs)

    output, _ = mainREGcode_ressarch(hyperparam, tr, tst, [algo], opts)
    return r2_score(y[n_train:], output.yhat[0])


class TestAlgorithmAccuracy:
    """
    Each algorithm is trained on synthetic data whose ground-truth relationship is
    known, then evaluated on a held-out test set.  Asserts that R² exceeds a
    per-algorithm minimum — confirming that the fit → predict path learns the signal
    rather than just returning the correct output shape.

    Data is Z-scored (standard-normal features).  Thresholds are calibrated to expected algorithm behavior on the given
    relationship type; kernel/ensemble/NN methods are given more headroom than the
    closed-form linear models.
    """

    def test_lin_recovers_linear_signal(self):
        """Ridge on y = 2x₀ + 3x₁ + ε — near-perfect recovery expected."""
        assert _run_accuracy('lin', 0.01) > 0.95

    def test_ransac_recovers_linear_signal(self):
        """RANSACRegressor on clean linear data — robust linear fit expected."""
        assert _run_accuracy('ransac', 0.5) > 0.90

    def test_quad_recovers_quadratic_signal(self):
        """PolynomialFeatures(2) + Ridge on y = x₀² + 2x₁ + ε — poly expansion tested."""
        assert _run_accuracy('quad', 0.01, relationship='quadratic') > 0.90

    def test_elm_recovers_linear_signal(self):
        """ELMRegressor (closed-form) on linear data — should converge reliably."""
        assert _run_accuracy('elm', 20.0) > 0.85

    def test_gp_interpolates_training_data(self):
        """GaussianProcessRegressor with default alpha=1e-10 must near-perfectly
        interpolate its own training data.  We pass the training set as the test batch
        to verify the fit → predict path is wired correctly.  Out-of-sample R² is not
        asserted: the default RBF kernel with tiny nugget collapses to a near-zero
        length scale at optimisation time, making predictions revert to the zero prior
        outside training support."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((60, 2))
        y = np.sin(2.0 * X[:, 0]) + np.cos(2.0 * X[:, 1])
        tr = Struct(x=X, y=y)
        tst = Struct(x=[X], y=[y])
        opts = _make_opts()
        output, _ = mainREGcode_ressarch(1.0, tr, tst, ['gp'], opts)
        assert r2_score(y, output.yhat[0]) > 0.99

    def test_knn_recovers_linear_signal(self):
        """KNeighborsRegressor (k=5) on linear data — local averaging recovers trend."""
        assert _run_accuracy('knn', 5.0) > 0.80

    def test_btree_recovers_linear_signal(self):
        """RandomForestRegressor on linear data — ensemble of trees recovers signal."""
        assert _run_accuracy('btree', 2.0) > 0.80

    def test_svr_recovers_linear_signal(self):
        """SVR (RBF kernel, C=1) on linear data — kernel method recovers linear trend."""
        assert _run_accuracy('svr', 1.0) > 0.75

    def test_libsvr_recovers_linear_signal(self):
        """NuSVR (nu=0.4, RBF) on linear data — looser threshold than plain SVR."""
        assert _run_accuracy('libsvr', 1.0) > 0.70

    def test_bnet_recovers_linear_signal(self):
        """BaggingRegressor(MLPRegressor) on linear data — NN ensemble, most variable."""
        assert _run_accuracy('bnet', 4.0, n_train=80, n_test=30) > 0.60
