"""
tests/ldslearn/test_inverse_covariance_selection.py
-----------------------------------------------------
Pytest suite for ldslearn/inverse_covariance_selection.py.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('user_input_ressarch', MagicMock())

_mod_path = os.path.join(os.path.dirname(__file__), "..", "..", "ldslearn", "inverse_covariance_selection.py")
_spec = importlib.util.spec_from_file_location("ldslearn.inverse_covariance_selection", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

inverse_covariance_selection = _mod.inverse_covariance_selection


def test_already_pd_input_stays_pd():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    S = A @ A.T + np.eye(4)

    Q = inverse_covariance_selection(S)

    assert np.linalg.eigvalsh(Q).min() > 0


def test_rank_deficient_input_becomes_pd():
    rng = np.random.default_rng(1)
    v = rng.standard_normal((4, 1))
    S = v @ v.T  # rank 1, PSD but not PD

    Q = inverse_covariance_selection(S)

    assert np.linalg.eigvalsh(Q).min() > 0


def test_output_is_symmetric():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((4, 4))
    S = A @ A.T + np.eye(4)

    Q = inverse_covariance_selection(S)

    assert np.allclose(Q, Q.T)


def test_epsilon_floor_respected():
    rng = np.random.default_rng(3)
    v = rng.standard_normal((4, 1))
    S = v @ v.T

    Q = inverse_covariance_selection(S, epsilon=0.1)

    # epsilon floors the precision matrix X = Q^-1, not Q itself.
    X = np.linalg.inv(Q)
    assert np.linalg.eigvalsh(X).min() > 0.1 - 1e-4
