import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

def _to_2d_float(array):
    """Ensure *array* is a 2-D float64 ndarray with shape (samples, features)."""
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    return values


# ---------------------------------------------------------------------------
# Internal min–max scaling  (mirrors mapminmax_apply_ELM.m / mapminmax_reverse_ELM.m)
# ---------------------------------------------------------------------------

def _minmax_scale(x, x_min, x_max):
    """
    Scale *x* (N × D) feature-wise to [−1, 1].

    Mirrors mapminmax_apply_ELM.m:
        rangex = xmax − xmin   (zero ranges clamped to 1 to avoid division by zero)
        x_normalized = 2 * (x − xmin) / rangex − 1
    """
    rangex = x_max - x_min # (D,)
    rangex = np.where(rangex == 0.0, 1.0, rangex) # zero-range protection
    return 2.0 * (x - x_min) / rangex - 1.0


def _minmax_reverse(y_scaled, y_min, y_max):
    """
    Invert min–max scaling from [−1, 1] back to original range.

    Mirrors mapminmax_reverse_ELM.m:
        x_denorm = (ymax − ymin) * (y − (−1)) / 2 + ymin
    """
    return (y_max - y_min) * (y_scaled + 1.0) / 2.0 + y_min


# ---------------------------------------------------------------------------
# Hidden-layer activation matrix
# ---------------------------------------------------------------------------

def _activation_matrix(x_values, weights, bias, activation):
    """
    Compute the hidden-layer output matrix H  (N × n_hidden).

    Parameters
    ----------
    x_values : (N, n_features)  — scaled input samples
    weights  : (n_features, n_hidden) — input weight matrix  (IW' in MATLAB convention)
    bias     : (n_hidden,) — bias / variance vector
    activation : str

    RBF branch mirrors RBFun.m exactly:
        V[:, i] = −‖P[k] − IW[i]‖²   (pairwise squared Euclidean distance)
        V        = V * Bias             (element-wise; Bias broadcasts over N rows)
        H        = exp(V)

    Sigmoid / sin branches mirror SigActFun.m / SinActFun.m:
        V = P * IW'  +  BiasMatrix
    """
    activation_name = str(activation).lower()

    if activation_name in {'rbf', 'radial', 'radial_basis'}:
        # Each column of `weights` (n_features, n_hidden) is a center vector.
        # Transposing gives centers as rows: (n_hidden, n_features).
        # cdist returns the (N, n_hidden) matrix of pairwise squared Euclidean distances.
        centers = weights.T # (n_hidden, n_features)
        dist_sq = cdist(x_values, centers, metric='sqeuclidean') # (N, n_hidden)
        # bias acts as a per-neuron variance scaling factor (RBFun.m: V = V .* BiasMatrix)
        V = -dist_sq * bias # (N, n_hidden), bias broadcast
        return np.exp(V)

    # Linear-projection activations (SigActFun.m / SinActFun.m pattern)
    hidden_input = x_values @ weights + bias # (N, n_hidden)

    if activation_name in {'sin', 'sine', 'sinusoidal'}:
        return np.sin(hidden_input)
    if activation_name in {'tanh', 'hyperbolic_tangent'}:
        return np.tanh(hidden_input)

    # Default: sigmoid  (SigActFun.m — 1 / (1 + exp(−V)))
    clipped = np.clip(hidden_input, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


# ---------------------------------------------------------------------------
# ELM Regressor
# ---------------------------------------------------------------------------

class ELMRegressor(BaseEstimator, RegressorMixin):
    """
    Extreme Learning Machine regressor with scikit-learn compatible API.

    Mirrors buildELM.m (training) and evalELM.m (inference) from the NASA
    ACCEPT framework.

    Parameters
    ----------
    hidden_units : int
        Number of hidden neurons  (``nh`` in MATLAB).
    activation : {'sig', 'rbf', 'sin', 'tanh'}
        Hidden-layer activation function.
    alpha : float
        Tikhonov regularisation weight  (``lam`` in MATLAB; MATLAB default 0.1).
    random_state : int
        RNG seed for reproducibility.
    orthogonal : bool
        Apply QR orthogonalisation to input weights  (``orth_flag_ELM`` in MATLAB).
    """

    def __init__(
        self,
        hidden_units,
        activation='sig',
        alpha=1e-1, # Matches MATLAB fixed lam = 0.1 in buildELM.m
        random_state=0,
        orthogonal=False,
    ):
        self.hidden_units = max(1, int(hidden_units))
        self.activation = activation
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.orthogonal = bool(orthogonal)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, x_values, y_values):
        """
        Train the ELM.

        Implements buildELM.m:
          1. Internal [−1, 1] min–max normalisation of X and Y.
          2. Uniform [−1, 1] weight / bias initialisation  (rand(...)*2−1).
          3. Hidden-layer output matrix H via the selected activation.
          4. Adaptive primal (n_hidden ≤ N) or dual (n_hidden > N) regularised solve.
        """
        x_values = _to_2d_float(x_values)
        y_values = _to_2d_float(y_values)

        n_samples, n_features = x_values.shape
        n_hidden = self.hidden_units

        # ---- 1. Internal min–max scaling (mapminmax_apply_ELM.m) --------
        self.xmin_ = x_values.min(axis=0) # (n_features,)
        self.xmax_ = x_values.max(axis=0)
        self.ymin_ = y_values.min(axis=0) # (n_outputs,)
        self.ymax_ = y_values.max(axis=0)

        x_scaled = _minmax_scale(x_values, self.xmin_, self.xmax_) # (N, n_features)
        y_scaled = _minmax_scale(y_values, self.ymin_, self.ymax_) # (N, n_outputs)

        # ---- 2. Input weight initialisation  (rand(...)*2−1) -------------
        rng = np.random.default_rng(self.random_state)

        if self.orthogonal and n_hidden <= n_features:
            # MATLAB: IW = orth(rand(n_features, n_hidden)*2−1)'
            # QR of a square uniform matrix gives orthonormal columns; take first n_hidden.
            raw = rng.uniform(-1.0, 1.0, size=(n_features, n_features))
            q_matrix, _ = np.linalg.qr(raw)
            weights = q_matrix[:, :n_hidden] # (n_features, n_hidden)
        else:
            # MATLAB: IW = rand(n_hidden, n_features)*2−1  →  Python stores as (n_features, n_hidden)
            weights = rng.uniform(-1.0, 1.0, size=(n_features, n_hidden))
            if self.orthogonal:
                # Normalise each column to unit length when n_hidden > n_features
                norms = np.linalg.norm(weights, axis=0, keepdims=True)
                norms[norms == 0.0] = 1.0
                weights = weights / norms

        # Bias: uniform [−1, 1]  (rand(1, nh)*2−1 for sig/sin; rand(1, nh) for rbf in MATLAB)
        bias = rng.uniform(-1.0, 1.0, size=n_hidden) # (n_hidden,)

        # ---- 3. Hidden-layer output matrix H ----------------------------
        hidden = _activation_matrix(x_scaled, weights, bias, self.activation)
        # hidden: (N, n_hidden)

        # ---- 4. Regularised output weights — adaptive primal / dual ------
        # Primal (n_hidden ≤ N): W = (H'H + λI)^{-1} H' Y  — inverts n_hidden×n_hidden
        # Dual  (n_hidden  > N): W = H' (HH' + λI)^{-1}  Y — inverts      N×N
        # (buildELM.m lines 72-80; MATLAB condition: nh <= size(xtrain, 2) = N)
        lam = self.alpha
        if n_hidden <= n_samples:
            A = hidden.T @ hidden + lam * np.eye(n_hidden) # (n_hidden, n_hidden)
            b = hidden.T @ y_scaled # (n_hidden, n_outputs)
        else:
            A = hidden @ hidden.T + lam * np.eye(n_samples) # (N, N)
            b = y_scaled # (N, n_outputs)

        try:
            x_sol = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x_sol = np.linalg.lstsq(A, b, rcond=None)[0]

        if n_hidden <= n_samples:
            output_weights = x_sol # (n_hidden, n_outputs)
        else:
            output_weights = hidden.T @ x_sol # (n_hidden, n_outputs)

        # ---- Store model state -------------------------------------------
        self.input_weights_ = weights
        self.bias_ = bias
        self.output_weights_ = output_weights
        self.n_features_in_ = n_features
        self.n_outputs_ = y_values.shape[1]
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, x_values):
        """
        Generate predictions, mirroring evalELM.m.

        Steps:
          1. Scale inputs to [−1, 1] using training bounds (xmin_, xmax_).
          2. Compute H via the same activation path used during fit.
          3. Multiply by output weights.
          4. De-normalise using training Y bounds (ymin_, ymax_).
        """
        x_values = _to_2d_float(x_values)

        # Scale inputs with training statistics (mapminmax_apply_ELM.m)
        x_scaled = _minmax_scale(x_values, self.xmin_, self.xmax_)

        # Hidden-layer output
        hidden = _activation_matrix(
            x_scaled, self.input_weights_, self.bias_, self.activation
        )

        # Raw prediction in [−1, 1] space
        y_scaled = hidden @ self.output_weights_ # (N, n_outputs)

        # De-normalise (mapminmax_reverse_ELM.m)
        predictions = _minmax_reverse(y_scaled, self.ymin_, self.ymax_)

        if self.n_outputs_ == 1:
            return predictions.ravel()
        return predictions
