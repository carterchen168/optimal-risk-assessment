"""
tests/test_preprocessing.py
----------------------------
Pytest suite for regressopt/preprocessing.py (GlobalDataScaler).

Covers:
  - Z-score normalization math and edge cases
  - Feature matrix / target vector dimensional contracts
  - Data-leakage prevention across train / validation / test splits
  - Scaler state persistence (save / load)
"""

import importlib.util
import os
import sys
import tempfile

import numpy as np
import pytest

# Import directly from the module file to avoid regressopt/__init__.py side
# effects (it opens params.txt at import time via user_input_ressarch).
_mod_path = os.path.join(os.path.dirname(__file__), "..", "..", "regressopt", "preprocessing.py")
_spec = importlib.util.spec_from_file_location("regressopt.preprocessing", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GlobalDataScaler = _mod.GlobalDataScaler


# ---------------------------------------------------------------------------
# 1. Z-score normalization
# ---------------------------------------------------------------------------

class TestZScoreNormalization:

    def test_fit_produces_zero_mean_unit_std(self):
        """fit_global_baselines shifts X and y to zero mean and unit std."""
        scaler = GlobalDataScaler()
        X = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
        y = np.array([1.0, 2.0, 3.0])

        X_scaled, y_scaled = scaler.fit_global_baselines(X, y)

        assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-7)
        assert np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-7)
        assert np.allclose(y_scaled.mean(), 0.0, atol=1e-7)
        assert np.allclose(y_scaled.std(), 1.0, atol=1e-7)

    def test_handles_zero_variance_feature_column(self):
        """Constant feature column produces zeros, not NaN (sklearn scale=1 fallback)."""
        scaler = GlobalDataScaler()
        # Column 0 is constant; column 1 varies normally.
        X = np.array([[5.0, 10.0], [5.0, 20.0], [5.0, 30.0]])
        y = np.array([1.0, 2.0, 3.0])

        X_scaled, _ = scaler.fit_global_baselines(X, y)

        assert not np.isnan(X_scaled).any()
        assert np.all(X_scaled[:, 0] == 0.0)

    def test_handles_zero_variance_target(self):
        """Constant target vector produces zeros, not NaN."""
        scaler = GlobalDataScaler()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([7.0, 7.0, 7.0])

        _, y_scaled = scaler.fit_global_baselines(X, y)

        assert not np.isnan(y_scaled).any()

    def test_2d_target_input_raveled_to_1d(self):
        """A column-vector target (n, 1) is accepted and output as 1-D."""
        scaler = GlobalDataScaler()
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 3))
        y_2d = rng.standard_normal((10, 1))

        _, y_scaled = scaler.fit_global_baselines(X, y_2d)

        assert y_scaled.ndim == 1
        assert y_scaled.shape == (10,)


# ---------------------------------------------------------------------------
# 2. Feature matrix / target vector dimensional contracts
# ---------------------------------------------------------------------------

class TestFeatureAndTargetSplitting:

    def test_fit_output_shapes(self):
        """fit_global_baselines preserves (n_samples, n_features) for X and (n_samples,) for y."""
        scaler = GlobalDataScaler()
        n_samples, n_features = 5, 4
        rng = np.random.default_rng(1)
        X = rng.standard_normal((n_samples, n_features))
        y = rng.standard_normal(n_samples)

        X_scaled, y_scaled = scaler.fit_global_baselines(X, y)

        assert X_scaled.shape == (n_samples, n_features)
        assert y_scaled.shape == (n_samples,)

    def test_transform_output_shapes(self):
        """transform_evaluation_data preserves shapes of an unseen split."""
        scaler = GlobalDataScaler()
        rng = np.random.default_rng(2)
        scaler.fit_global_baselines(rng.standard_normal((20, 3)), rng.standard_normal(20))

        X_val = rng.standard_normal((10, 3))
        y_val = rng.standard_normal(10)
        X_scaled, y_scaled = scaler.transform_evaluation_data(X_val, y_val)

        assert X_scaled.shape == (10, 3)
        assert y_scaled.shape == (10,)

    def test_transform_x_only_returns_none_y(self):
        """transform_evaluation_data with no raw_y returns (X_scaled, None)."""
        scaler = GlobalDataScaler()
        rng = np.random.default_rng(3)
        scaler.fit_global_baselines(rng.standard_normal((20, 3)), rng.standard_normal(20))

        X_val = rng.standard_normal((5, 3))
        X_scaled, y_out = scaler.transform_evaluation_data(X_val)

        assert X_scaled.shape == (5, 3)
        assert y_out is None


# ---------------------------------------------------------------------------
# 3. Data-leakage prevention
# ---------------------------------------------------------------------------

class TestDataPartitioningLeakage:

    def test_transform_before_fit_raises_valueerror(self):
        """Calling transform_evaluation_data on an unfitted scaler raises ValueError."""
        scaler = GlobalDataScaler()
        X = np.random.default_rng(4).standard_normal((5, 2))

        with pytest.raises(ValueError):
            scaler.transform_evaluation_data(X)

    def test_validation_uses_training_statistics(self):
        """Validation output is computed from training mean/std, never re-fitted."""
        scaler = GlobalDataScaler()
        # Training data: small values clustered near 1–5
        X_tr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_tr = np.array([10.0, 20.0, 30.0])
        scaler.fit_global_baselines(X_tr, y_tr)

        # Validation data has a very different distribution (large values)
        X_val = np.array([[100.0, 200.0], [300.0, 400.0]])
        X_val_scaled, _ = scaler.transform_evaluation_data(X_val)

        # Training mean/std applied → val output should NOT be zero-mean
        assert not np.allclose(X_val_scaled.mean(axis=0), 0.0, atol=1e-3)

        # Must agree exactly with sklearn's own transform using the training scaler
        expected = scaler.scaler_x.transform(X_val)
        assert np.allclose(X_val_scaled, expected)

    def test_repeated_transforms_are_deterministic(self):
        """Calling transform_evaluation_data twice on the same data gives identical results."""
        scaler = GlobalDataScaler()
        rng = np.random.default_rng(5)
        scaler.fit_global_baselines(rng.standard_normal((30, 4)), rng.standard_normal(30))

        X_val = rng.standard_normal((10, 4))
        X1, _ = scaler.transform_evaluation_data(X_val)
        X2, _ = scaler.transform_evaluation_data(X_val)

        assert np.array_equal(X1, X2)

    def test_is_fitted_flag_lifecycle(self):
        """is_fitted is False before fit and True immediately after."""
        scaler = GlobalDataScaler()
        assert scaler.is_fitted is False

        scaler.fit_global_baselines(
            np.random.default_rng(6).standard_normal((10, 2)),
            np.random.default_rng(6).standard_normal(10),
        )
        assert scaler.is_fitted is True


# ---------------------------------------------------------------------------
# 4. Scaler persistence
# ---------------------------------------------------------------------------

class TestScalerPersistence:

    def test_save_load_produces_identical_transforms(self):
        """A loaded scaler reproduces byte-identical outputs to the original."""
        scaler = GlobalDataScaler()
        rng = np.random.default_rng(7)
        scaler.fit_global_baselines(rng.standard_normal((20, 3)), rng.standard_normal(20))

        X_probe = rng.standard_normal((5, 3))
        y_probe = rng.standard_normal(5)
        X_before, y_before = scaler.transform_evaluation_data(X_probe, y_probe)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            scaler.save_scaler_state(tmp_path)

            scaler2 = GlobalDataScaler()
            scaler2.load_scaler_state(tmp_path)

            X_after, y_after = scaler2.transform_evaluation_data(X_probe, y_probe)

            assert np.allclose(X_before, X_after, atol=1e-12)
            assert np.allclose(y_before, y_after, atol=1e-12)
            assert scaler2.is_fitted is True
        finally:
            os.unlink(tmp_path)
