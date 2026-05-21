"""
regressopt/preprocessing.py
----------------------------
Global Z-score normalisation for the ACCEPT framework.

Replaces the streaming mathematics of MATLAB's ``zscoreStream.m`` and
``stdStream.m``.  The scaler is fit once on nominal (baseline) training
flight data and then applied transform-only to every other data split
(validation, test, streaming).  This guarantees a single, consistent geometric
space across all cross-validation folds and optimizer iterations, eliminating
the subtle data-leakage introduced by fitting a local StandardScaler inside
each CV fold.
"""

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


class GlobalDataScaler:
    """
    Handles global Z-score preprocessing for the ACCEPT framework.

    Replaces the streaming mathematics of ``zscoreStream.m`` and
    ``stdStream.m``.  Fits once on nominal training data; all other
    splits are transform-only, preventing data leakage across
    cross-validation folds.

    Attributes
    ----------
    scaler_x : StandardScaler
        Fitted scaler for feature matrices.
    scaler_y : StandardScaler
        Fitted scaler for target vectors.
    is_fitted : bool
        True after ``fit_global_baselines`` has been called successfully.
    """

    def __init__(self):
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_global_baselines(self, raw_train_x, raw_train_y):
        """Fit and transform nominal training / baseline flight data.

        This is the only method that calls ``fit``; all other splits must
        use `transform_evaluation_data`.

        Parameters
        ----------
        raw_train_x : array-like of shape (n_samples, n_features)
            Raw feature matrix for the nominal training set.
        raw_train_y : array-like of shape (n_samples,) or (n_samples, 1)
            Raw target vector for the nominal training set.  1-D arrays are
            handled internally — callers do not need to reshape.

        Returns
        -------
        tr_x_scaled : ndarray of shape (n_samples, n_features)
        tr_y_scaled : ndarray of shape (n_samples,)
        """
        raw_train_x = np.asarray(raw_train_x)
        y_2d = np.asarray(raw_train_y).reshape(-1, 1)

        tr_x_scaled = self.scaler_x.fit_transform(raw_train_x)
        tr_y_scaled = self.scaler_y.fit_transform(y_2d).ravel()

        self.is_fitted = True
        return tr_x_scaled, tr_y_scaled

    def transform_evaluation_data(self, raw_x, raw_y=None):
        """Transform validation, test, or streaming data using training parameters.

        Never calls ``fit``; uses the mean and std computed during `fit_global_baselines`.

        Parameters
        ----------
        raw_x : array-like of shape (n_samples, n_features)
            Feature matrix to scale.
        raw_y : array-like of shape (n_samples,) or (n_samples, 1), optional
            Target vector to scale.  If ``None``, only ``x`` is transformed.

        Returns
        -------
        x_scaled : ndarray of shape (n_samples, n_features)
        y_scaled : ndarray of shape (n_samples,) or None

        Raises
        ------
        ValueError
            If called before `fit_global_baselines` has been run.
        """
        if not self.is_fitted:
            raise ValueError(
                "Global baselines must be fitted before transforming evaluation data. "
                "Call fit_global_baselines() on nominal training data first."
            )

        x_scaled = self.scaler_x.transform(np.asarray(raw_x))

        if raw_y is not None:
            y_scaled = self.scaler_y.transform(
                np.asarray(raw_y).reshape(-1, 1)
            ).ravel()
        else:
            y_scaled = None

        return x_scaled, y_scaled

    # ------------------------------------------------------------------
    # Persistence helpers (for deployment / real-time inference)
    # ------------------------------------------------------------------

    def save_scaler_state(self, filepath):
        """Persist the fitted scaler parameters to disk.

        Parameters
        ----------
        filepath : str or Path
            Destination path for the joblib dump (e.g. ``'scaler.pkl'``).
        """
        joblib.dump(
            {'scaler_x': self.scaler_x, 'scaler_y': self.scaler_y},
            filepath,
        )

    def load_scaler_state(self, filepath):
        """Restore previously saved scaler parameters from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to a file previously written by `save_scaler_state`.
        """
        state = joblib.load(filepath)
        self.scaler_x = state['scaler_x']
        self.scaler_y = state['scaler_y']
        self.is_fitted = True
