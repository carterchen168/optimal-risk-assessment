"""
tests/stablelds/test_learnCGModelEM.py
---------------------------------------
Pytest suite for stablelds/learnCGModelEM.py's constraint-generation
stabilizer (learn_cg_model_em).
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import numpy as np

sys.modules.setdefault('user_input_ressarch', MagicMock())

_mod_path = os.path.join(os.path.dirname(__file__), "..", "..", "stablelds", "learnCGModelEM.py")
_spec = importlib.util.spec_from_file_location("stablelds.learnCGModelEM", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

learn_cg_model_em = _mod.learn_cg_model_em
_solve_qp = _mod._solve_qp
_solve_qp_quadprog = _mod._solve_qp_quadprog
_solve_qp_scipy = _mod._solve_qp_scipy

D = 3


def _sufficient_stats(seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((D, D))
    gamma1 = X @ X.T + D * np.eye(D)
    beta = rng.standard_normal((D, D))
    return rng, gamma1, beta


def test_unstable_input_returns_stable_matrix():
    rng, gamma1, beta = _sufficient_stats(seed=1)

    A_raw = rng.standard_normal((D, D))
    spectral_radius = np.max(np.abs(np.linalg.eigvals(A_raw)))
    A = A_raw / spectral_radius * 1.5

    M = learn_cg_model_em(beta, gamma1, A)

    assert M.shape == (D, D)
    assert np.max(np.abs(np.linalg.eigvals(M))) < 1


def test_already_stable_input_returns_stable_matrix():
    rng, gamma1, beta = _sufficient_stats(seed=2)

    A = np.diag(rng.uniform(0.1, 0.5, size=D))

    M = learn_cg_model_em(beta, gamma1, A)

    assert M.shape == (D, D)
    assert np.max(np.abs(np.linalg.eigvals(M))) < 1


def _small_qp(seed):
    """A tiny constrained QP: min m'Pm - 2q'm  s.t. g'm <= h, with P PD."""
    rng = np.random.default_rng(seed)
    d2 = 4
    X = rng.standard_normal((d2, d2))
    P = X @ X.T + d2 * np.eye(d2)
    q = rng.standard_normal(d2)
    g = rng.standard_normal(d2)
    g = g / np.linalg.norm(g)
    G = g[np.newaxis, :]
    h = np.array([0.1])
    return P, q, G, h


def test_solve_qp_quadprog_matches_scipy_solution():
    P, q, G, h = _small_qp(seed=10)

    m_qp, _ = _solve_qp_quadprog(P, q, G, h)
    m_sp, _ = _solve_qp_scipy(P, q, G, h)

    assert np.allclose(m_qp, m_sp, atol=1e-3)


def test_solve_qp_uses_quadprog_when_available(monkeypatch):
    P, q, G, h = _small_qp(seed=11)

    def _boom(*args, **kwargs):
        raise AssertionError("scipy fallback should not run when quadprog is available")

    monkeypatch.setattr(_mod, "_solve_qp_scipy", _boom)

    m, _ = _solve_qp(P, q, G, h, d=2, cvx_flag=False)

    assert m is not None
    assert np.all(np.isfinite(m))


def test_solve_qp_falls_back_to_scipy_when_quadprog_missing(monkeypatch):
    P, q, G, h = _small_qp(seed=12)

    def _raise_import_error(*args, **kwargs):
        raise ImportError("simulated missing quadprog")

    monkeypatch.setattr(_mod, "_solve_qp_quadprog", _raise_import_error)

    m, _ = _solve_qp(P, q, G, h, d=2, cvx_flag=False)

    assert m is not None
    assert np.all(np.isfinite(m))
