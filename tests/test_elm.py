"""
tests/test_elm.py
------------------
Pytest suite for regressopt/elm.py (ELMRegressor and internal helpers).

Covers:
  - Internal min-max normalization math and zero-range protection
  - Hidden-layer activation matrix computations for RBF and projection branches
  - Public ELMRegressor pipeline (fit and predict dimensional contracts)
  - Orthogonal initialization logic (QR vs. column normalization)
  - Primal vs. Dual regularized matrix solve transitions
"""

import importlib.util
import os
import numpy as np
import pytest

# Import directly from the module file to match preprocessing test isolation pattern
_mod_path = os.path.join(os.path.dirname(__file__), "..", "regressopt", "elm.py")
_spec = importlib.util.spec_from_file_location("regressopt.elm", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_to_2d_float = _mod._to_2d_float
_minmax_scale = _mod._minmax_scale
_minmax_reverse = _mod._minmax_reverse
_activation_matrix = _mod._activation_matrix
ELMRegressor = _mod.ELMRegressor


# ---------------------------------------------------------------------------
# 1. Internal min-max scaling mathematics
# ---------------------------------------------------------------------------

class TestMinMaxScalingMath:

    def test_minmax_scale_maps_correctly_to_bounds(self):
        """_minmax_scale accurately maps input matrices feature-wise to [-1, 1]."""
        x = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
        x_min = np.array([10.0, 100.0])
        x_max = np.array([30.0, 300.0])

        x_scaled = _minmax_scale(x, x_min, x_max)

        expected = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        assert np.allclose(x_scaled, expected, atol=1e-7)

    def test_minmax_scale_handles_zero_range_protection(self):
        """_minmax_scale clamps zero ranges to 1.0 to guarantee division safety without NaNs."""
        x = np.array([[5.0, 10.0], [5.0, 20.0], [5.0, 30.0]])
        x_min = np.array([5.0, 10.0])
        x_max = np.array([5.0, 30.0])

        x_scaled = _minmax_scale(x, x_min, x_max)

        assert not np.isnan(x_scaled).any()
        # For column 0: range is 0 -> clamped to 1. 2*(5-5)/1 - 1 = -1
        assert np.all(x_scaled[:, 0] == -1.0)

    def test_minmax_reverse_invertibility(self):
        """_minmax_reverse exactly reconstructs original unscaled space variations."""
        y_scaled = np.array([[-1.0], [0.0], [1.0]])
        y_min = np.array([10.0])
        y_max = np.array([30.0])

        y_original = _minmax_reverse(y_scaled, y_min, y_max)

        expected = np.array([[10.0], [20.0], [30.0]])
        assert np.allclose(y_original, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# 2. Hidden-layer activation matrices
# ---------------------------------------------------------------------------

class TestActivationMatrixGeneration:

    def test_rbf_branch_pairwise_squared_distances(self):
        """RBF activation handles squared Euclidean spatial distances and scales via biases."""
        # 2 samples, 2 features
        x = np.array([[0.0, 0.0], [1.0, 1.0]])
        # 1 hidden unit (center vector at origin)
        weights = np.array([[0.0], [0.0]])
        bias = np.array([0.5])  # Variance scaling factor

        H = _activation_matrix(x, weights, bias, activation='rbf')

        # Distance squared to origin: row 0 = 0, row 1 = 2
        # V = -dist_sq * bias -> row 0 = 0, row 1 = -2 * 0.5 = -1
        # H = exp(V) -> row 0 = 1, row 1 = exp(-1)
        expected = np.array([[1.0], [np.exp(-1.0)]])
        assert np.allclose(H, expected, atol=1e-7)

    def test_projection_branches_evaluate_correctly(self):
        """Projection paths map input matrices correctly through sig, sin, and tanh fields."""
        x = np.array([[1.0, -1.0]])
        weights = np.array([[2.0], [3.0]])
        bias = np.array([1.0])

        # hidden_input = x @ weights + bias = (1*2 + -1*3) + 1 = 0.0
        H_sin = _activation_matrix(x, weights, bias, activation='sin')
        H_tanh = _activation_matrix(x, weights, bias, activation='tanh')
        H_sig = _activation_matrix(x, weights, bias, activation='sig')

        assert np.allclose(H_sin, np.sin(0.0), atol=1e-7)
        assert np.allclose(H_tanh, np.tanh(0.0), atol=1e-7)
        assert np.allclose(H_sig, 0.5, atol=1e-7)

    def test_sigmoid_activation_handles_extreme_inputs_safely(self):
        """_activation_matrix clips extreme sigmoid inputs to prevent NumPy overflow warnings."""
        # 3 samples (extreme negative, extreme positive, infinite positive), 1 feature
        x = np.array([[-1000.0], [1000.0], [np.inf]])
        weights = np.array([[1.0]])
        bias = np.array([0.0])

        # Under raw NumPy math, exp(1000) throws a RuntimeWarning.
        # With the clip safetynet, it stays structurally quiet and mathematically sound.
        try:
            with np.errstate(all='raise'):
                H = _activation_matrix(x, weights, bias, activation='sig')
        except FloatingPointError as e:
            pytest.fail(f"Sigmoid activation raised a floating point exception: {e}")

        # Verify expected structural convergence:
        # -1000 clipped to -60 -> 1 / (1 + exp(60)) -> effectively 0.0
        # +1000 clipped to  60 -> 1 / (1 + exp(-60)) -> effectively 1.0
        # +inf  clipped to  60 -> 1 / (1 + exp(-60)) -> effectively 1.0
        assert np.allclose(H[0, 0], 0.0, atol=1e-7)
        assert np.allclose(H[1, 0], 1.0, atol=1e-7)
        assert np.allclose(H[2, 0], 1.0, atol=1e-7)


# ---------------------------------------------------------------------------
# 3. Model pipeline contracts & regularized solves
# ---------------------------------------------------------------------------

class TestELMRegressorPipeline:

    def test_fit_and_predict_dimensional_contracts(self):
        """fit and predict preserve target dimensional layout matrices for Multi-Output targets."""
        scaler = ELMRegressor(hidden_units=5, activation='sig', random_state=42)
        rng = np.random.default_rng(42)
        
        X_train = rng.uniform(-5, 5, size=(20, 3))
        y_train = rng.uniform(10, 20, size=(20, 2))  # 2 Target dimensions

        scaler.fit(X_train, y_train)
        
        X_test = rng.uniform(-5, 5, size=(10, 3))
        predictions = scaler.predict(X_test)

        assert predictions.shape == (10, 2)
        assert scaler.input_weights_.shape == (3, 5)
        assert scaler.bias_.shape == (5,)
        assert scaler.output_weights_.shape == (5, 2)

    def test_raveled_1d_output_for_single_target(self):
        """A single target vector shape tracking evaluates output down to a flat 1-D array."""
        scaler = ELMRegressor(hidden_units=4, random_state=0)
        rng = np.random.default_rng(0)
        
        X_train = rng.normal(size=(15, 2))
        y_train = rng.normal(size=(15, 1))

        scaler.fit(X_train, y_train)
        predictions = scaler.predict(rng.normal(size=(5, 2)))

        assert predictions.ndim == 1
        assert predictions.shape == (5,)

    def test_adaptive_primal_solve_transition(self):
        """System solves in Primal form over square dimensions when hidden_units <= samples."""
        # samples N=10, hidden_units=4 -> (hidden_units <= samples) -> Primal
        scaler = ELMRegressor(hidden_units=4, alpha=0.1, random_state=1)
        rng = np.random.default_rng(1)
        X = rng.normal(size=(10, 2))
        y = rng.normal(size=(10, 1))

        scaler.fit(X, y)
        # Verifies it executed normal path tracking successfully
        assert scaler.output_weights_.shape == (4, 1)

    def test_adaptive_dual_solve_transition(self):
        """System solves in Dual space over dimensions when hidden_units > samples."""
        # samples N=5, hidden_units=10 -> (hidden_units > samples) -> Dual
        scaler = ELMRegressor(hidden_units=10, alpha=0.1, random_state=2)
        rng = np.random.default_rng(2)
        X = rng.normal(size=(5, 2))
        y = rng.normal(size=(5, 1))

        scaler.fit(X, y)
        assert scaler.output_weights_.shape == (10, 1)


# ---------------------------------------------------------------------------
# 4. Orthogonal initialization constraints
# ---------------------------------------------------------------------------

class TestOrthogonalWeightInitialization:

    def test_orthogonal_qr_underdetermined_subspace(self):
        """QR path enforces rigid Column Orthonormal contracts when hidden_units <= features."""
        # features=4, hidden_units=3 -> QR path tracking executes
        scaler = ELMRegressor(hidden_units=3, orthogonal=True, random_state=3)
        rng = np.random.default_rng(3)
        X = rng.normal(size=(10, 4))
        y = rng.normal(size=(10, 1))

        scaler.fit(X, y)
        W = scaler.input_weights_  # (n_features, n_hidden) -> (4, 3)

        # W.T @ W should equal identity matrix of size (3, 3)
        identity_check = W.T @ W
        assert np.allclose(identity_check, np.eye(3), atol=1e-7)

    def test_orthogonal_unit_length_overdetermined_subspace(self):
        """Column elements normalize back to unit norm lines when hidden_units > features."""
        # features=2, hidden_units=5 -> unit norm length fallback loop executes
        scaler = ELMRegressor(hidden_units=5, orthogonal=True, random_state=4)
        rng = np.random.default_rng(4)
        X = rng.normal(size=(10, 2))
        y = rng.normal(size=(10, 1))

        scaler.fit(X, y)
        W = scaler.input_weights_  # (2, 5)

        column_norms = np.linalg.norm(W, axis=0)
        assert np.allclose(column_norms, 1.0, atol=1e-7)