"""
tests/ldslearn/test_guess.py
------------------------------
Pytest suite for ldslearn/guess.py.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('user_input_ressarch', MagicMock())

_mod_path = os.path.join(os.path.dirname(__file__), "..", "..", "ldslearn", "guess.py")
_spec = importlib.util.spec_from_file_location("ldslearn.guess", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

guess = _mod.guess


DIMS = [(3, 2, 4), (1, 1, 1), (5, 3, 2)]


def _make_yt(n, T, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, T))


@pytest.mark.parametrize("k,m,n", DIMS)
def test_output_shapes(k, m, n):
    yt = _make_yt(n, T=20, seed=0)
    A_g, B_g, C_g, D_g, Q_g, R_g, x0_g, P0_g = guess(k, m, n, yt)

    assert A_g.shape == (k, k)
    assert B_g.shape == (k, m)
    assert C_g.shape == (n, k)
    assert D_g.shape == (n, m)
    assert Q_g.shape == (k, k)
    assert R_g.shape == (n, n)
    assert x0_g.shape == (k, 1)
    assert P0_g.shape == (k, k)


@pytest.mark.parametrize("k,m,n", DIMS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_A_g_is_stable(k, m, n, seed):
    yt = _make_yt(n, T=20, seed=seed)
    A_g, *_ = guess(k, m, n, yt)

    # A_g must be diagonal with entries in [0, 1).
    assert np.allclose(A_g, np.diag(np.diagonal(A_g)))
    diag_entries = np.diagonal(A_g)
    assert np.all(diag_entries >= 0.0)
    assert np.all(diag_entries < 1.0)

    spectral_radius = np.max(np.abs(np.linalg.eigvals(A_g)))
    assert spectral_radius < 1.0


def test_Q_g_and_P0_g_are_identity():
    k, m, n = 4, 2, 3
    yt = _make_yt(n, T=20, seed=42)
    _, _, _, _, Q_g, _, _, P0_g = guess(k, m, n, yt)

    assert np.array_equal(Q_g, np.eye(k))
    assert np.array_equal(P0_g, np.eye(k))


def test_x0_g_is_zero():
    k, m, n = 4, 2, 3
    yt = _make_yt(n, T=20, seed=42)
    _, _, _, _, _, _, x0_g, _ = guess(k, m, n, yt)

    assert np.array_equal(x0_g, np.zeros((k, 1)))


def test_R_g_matches_sample_covariance_and_is_psd():
    k, m, n = 4, 2, 3
    yt = _make_yt(n, T=20, seed=7)
    _, _, _, _, _, R_g, _, _ = guess(k, m, n, yt)

    assert np.allclose(R_g, np.cov(yt))
    assert np.allclose(R_g, R_g.T)
    eigvals = np.linalg.eigvalsh(R_g)
    assert np.all(eigvals >= -1e-10)
