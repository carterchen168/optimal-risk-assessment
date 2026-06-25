"""
tests/stablelds/test_learn_lds.py
----------------------------------
Pytest suite for stablelds/learn_lds.py (Hankel-SVD subspace identification
with five dynamics-matrix learning algorithms).
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('user_input_ressarch', MagicMock())

_mod_path = os.path.join(os.path.dirname(__file__), "..", "..", "stablelds", "learn_lds.py")
_spec = importlib.util.spec_from_file_location("stablelds.learn_lds", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

learn_lds = _mod.learn_lds


def _simulate_lds(n, m, t, seed):
    """Simulate a small stable LDS, returning Y in (m, t) orientation."""
    rng = np.random.default_rng(seed)
    A = np.diag(rng.uniform(0.3, 0.7, size=n))
    C = rng.standard_normal((m, n))

    x = np.zeros((n, t + 1))
    y = np.zeros((m, t))
    for k in range(t):
        w = rng.standard_normal(n) * 0.01
        v = rng.standard_normal(m) * 0.01
        x[:, k + 1] = A @ x[:, k] + w
        y[:, k] = C @ x[:, k] + v
    return y


N, M, T, D = 2, 2, 60, 3
TAU = T - D + 1


@pytest.mark.parametrize("algo", [1, 2, 3, 4, 5])
def test_output_shapes(algo):
    Y = _simulate_lds(N, M, T, seed=algo)

    Ahat, Chat, Qhat, Rhat, Xhat, Ymean = learn_lds(Y, N, D, algo=algo)

    assert Ahat.shape == (N, N)
    assert Chat.shape == (M, N)
    assert Qhat.shape == (N, N)
    assert Rhat.shape == (M, M)  # m=2 <= 100, so Rhat is computed
    assert Xhat.shape == (N, TAU)
    assert Ymean.shape == (M, 1)


@pytest.mark.parametrize("algo", [1, 3, 4, 5])
def test_stable_algos_return_stable_A(algo):
    """
    algo 1 (Constraint Generation): learn_cg_model_em's constraint-generation
    loop + binary-search refinement explicitly enforces spectral radius < 1.
    algo 3 (Lacy-Bernstein 1): the CVXPY SDP enforces a norm constraint on
    the block matrix that bounds the dynamics matrix's singular values <= 1.
    algo 4 (Lacy-Bernstein 1 via CG): learn_cg_model_em with simulate_LB1=True
    stops once the top singular value <= 1 + 5e-4; spectral radius is always
    <= the largest singular value, so this also guarantees (near-)stability.
    algo 5 (Lacy-Bernstein 2): the SDP's block PSD constraint
    [[Z, A], [A', P]] >= delta*I guarantees a stable A.
    """
    Y = _simulate_lds(N, M, T, seed=100 + algo)

    Ahat, *_ = learn_lds(Y, N, D, algo=algo)

    spectral_radius = np.max(np.abs(np.linalg.eigvals(Ahat)))
    assert spectral_radius < 1 + 1e-3


def test_least_squares_algo_has_no_stability_guarantee():
    """
    algo 2 is a plain pseudoinverse least-squares fit with no constraint on
    eigenvalues, so unlike the other four branches it does NOT guarantee a
    stable A. Only shape is asserted here (covered by test_output_shapes);
    asserting spectral radius < 1 would encode a property the algorithm
    does not actually provide.
    """
    Y = _simulate_lds(N, M, T, seed=2)

    Ahat, *_ = learn_lds(Y, N, D, algo=2)

    assert Ahat.shape == (N, N)


def test_invalid_algo_raises_value_error():
    Y = _simulate_lds(N, M, T, seed=0)

    with pytest.raises(ValueError):
        learn_lds(Y, N, D, algo=6)
