"""
tests/subfun/test_subid.py
---------------------------
Pytest suite for subfun/subid.py (SIPPY/N4SID subspace identification wrapper).
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.modules.setdefault('user_input_ressarch', MagicMock())

_mod_path = os.path.join(os.path.dirname(__file__), "..", "..", "subfun", "subid.py")
_spec = importlib.util.spec_from_file_location("subfun.subid", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

subid = _mod.subid


def _simulate_lds(n, l, m, T, seed, with_input):
    """Simulate a small stable LDS, returning (y, u) in (channels, T) orientation."""
    rng = np.random.default_rng(seed)
    A = np.diag(rng.uniform(0.3, 0.7, size=n))
    C = rng.standard_normal((l, n))
    B = rng.standard_normal((n, m)) if with_input else None
    D = rng.standard_normal((l, m)) if with_input else None
    u = rng.standard_normal((m, T)) if with_input else None

    x = np.zeros((n, T + 1))
    y = np.zeros((l, T))
    for t in range(T):
        ut = u[:, t] if with_input else 0.0
        w = rng.standard_normal(n) * 0.01
        v = rng.standard_normal(l) * 0.01
        x[:, t + 1] = A @ x[:, t] + (B @ ut if with_input else 0.0) + w
        y[:, t] = C @ x[:, t] + (D @ ut if with_input else 0.0) + v
    return y, u


def test_stochastic_case_shapes_and_none_io():
    n, l = 2, 3
    y, _ = _simulate_lds(n, l, m=1, T=300, seed=0, with_input=False)

    result = subid(y, None, i=10, n=n, sil=1)
    assert len(result) == 12
    A, B, C, D, K, Ro, AUX, ss, Q, S, R, cvx_flag = result

    assert B is None
    assert D is None
    assert AUX is None

    n_id = A.shape[0]
    assert A.shape == (n_id, n_id)
    assert C.shape == (l, n_id)
    assert K.shape == (n_id, l)
    assert Ro.shape == (l, l)
    assert ss.shape == (n_id,)
    assert Q.shape == (n_id, n_id)
    assert S.shape == (n_id, l)
    assert R.shape == (l, l)
    assert isinstance(cvx_flag, bool)


def test_input_driven_case_shapes_and_populated_io():
    n, l, m = 2, 3, 1
    y, u = _simulate_lds(n, l, m, T=300, seed=1, with_input=True)

    result = subid(y, u, i=10, n=n, sil=1)
    A, B, C, D, K, Ro, AUX, ss, Q, S, R, cvx_flag = result

    n_id = A.shape[0]
    assert A.shape == (n_id, n_id)
    assert B.shape == (n_id, m)
    assert C.shape == (l, n_id)
    assert D.shape == (l, m)
    assert K.shape == (n_id, l)
    assert Ro.shape == (l, l)
    assert Q.shape == (n_id, n_id)
    assert S.shape == (n_id, l)
    assert R.shape == (l, l)
    assert isinstance(cvx_flag, bool)


@pytest.mark.parametrize("with_input", [False, True])
def test_cvx_flag_matches_spectral_radius_of_A(with_input):
    n, l, m = 2, 3, 1
    y, u = _simulate_lds(n, l, m, T=300, seed=2, with_input=with_input)

    result = subid(y, u if with_input else None, i=10, n=n, sil=1)
    A = result[0]
    cvx_flag = result[-1]

    expected = bool(np.max(np.abs(np.linalg.eigvals(A))) >= 1)
    assert cvx_flag == expected


def test_sippy_failure_returns_all_none_contract():
    """
    When SIPPY identification raises, subid() must return (None,) * 11 + (False,)
    so that ldslearn/lds_timeseries.py's `params.init.ad is None` check (the first
    element of this tuple) can detect total failure and fall back to guess().
    """
    y = np.zeros((2, 50))

    with patch("sippy_unipi.system_identification", side_effect=RuntimeError("boom")):
        result = subid(y, None, i=5, n=2, sil=1)

    assert result == (None,) * 11 + (False,)
    A = result[0]
    assert A is None  # contract relied on by lds_timeseries.py's guess() fallback
