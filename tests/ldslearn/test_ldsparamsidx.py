"""
tests/ldslearn/test_ldsparamsidx.py
-------------------------------------
Pytest suite for ldslearn/ldsparamsidx.py.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('user_input_ressarch', MagicMock())

_mod_path = os.path.join(os.path.dirname(__file__), "..", "..", "ldslearn", "ldsparamsidx.py")
_spec = importlib.util.spec_from_file_location("ldslearn.ldsparamsidx", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

lds_params_idx = _mod.lds_params_idx


def _make_params(n_iters=3, k=2, p=2, m=1, with_input=True):
    """Build a params/learned pair where each 3rd-axis slice is filled with
    a distinct, recognizable value (iteration index * 10), so that selecting
    the wrong slice is immediately detectable."""

    def stacked(rows, cols):
        return np.stack(
            [np.full((rows, cols), (i + 1) * 10) for i in range(n_iters)],
            axis=2,
        )

    learned = MagicMock()
    learned.ad = stacked(k, k)
    learned.qwd = stacked(k, k)
    learned.cd = stacked(p, k)
    learned.rvd = stacked(p, p)
    learned.xssd = stacked(k, k)
    if with_input:
        learned.bd = stacked(k, m)
        learned.dd = stacked(p, m)
    else:
        del learned.bd
        del learned.dd

    params = MagicMock()
    params.learned = learned
    return params


def test_one_based_to_zero_based_conversion_for_each_bookmark():
    # MATLAB bookmarks are 1-based; for every valid 1-based index, the
    # selected slice must equal the (index - 1)'th Python slice, i.e. the
    # conversion is exactly "subtract one", not an off-by-one in either
    # direction.
    n_iters = 5
    params = _make_params(n_iters=n_iters)
    learned = params.learned

    for one_based in range(1, n_iters + 1):
        params.ka = one_based
        params.kq = one_based
        params.kr = one_based
        params.kxinit = one_based

        a, q, c, r, params_out, b, d = lds_params_idx(params)

        zero_based = one_based - 1
        np.testing.assert_array_equal(a, learned.ad[:, :, zero_based])
        np.testing.assert_array_equal(q, learned.qwd[:, :, zero_based])
        np.testing.assert_array_equal(r, learned.rvd[:, :, zero_based])
        np.testing.assert_array_equal(params_out.xssd, learned.xssd[:, :, zero_based])


def test_bookmark_one_selects_first_iteration_not_second():
    # Regression guard for the classic off-by-one: ka=1 (MATLAB "first
    # element") must select index 0, not index 1.
    params = _make_params(n_iters=3)
    params.ka = params.kq = params.kr = params.kxinit = 1

    a, q, c, r, params_out, b, d = lds_params_idx(params)

    assert a[0, 0] == 10
    assert q[0, 0] == 10
    assert r[0, 0] == 10
    assert params_out.xssd[0, 0] == 10


def test_bookmark_equal_to_n_iters_selects_last_slice():
    # The other edge: ka=n_iters (MATLAB "last element") must select
    # index n_iters - 1, the true last slice, with no out-of-range error.
    n_iters = 4
    params = _make_params(n_iters=n_iters)
    params.ka = params.kq = params.kr = params.kxinit = n_iters

    a, q, c, r, params_out, b, d = lds_params_idx(params)

    expected_value = n_iters * 10
    assert a[0, 0] == expected_value
    assert q[0, 0] == expected_value
    assert r[0, 0] == expected_value
    assert params_out.xssd[0, 0] == expected_value


def test_c_always_taken_from_final_iteration_regardless_of_bookmarks():
    # MATLAB `cd(:,:,end)` has no bookmark of its own; c must always come
    # from the last iteration even when ka/kq/kr/kxinit point elsewhere.
    n_iters = 5
    params = _make_params(n_iters=n_iters)
    params.ka = params.kq = params.kr = params.kxinit = 1

    _, _, c, _, _, _, _ = lds_params_idx(params)

    np.testing.assert_array_equal(c, params.learned.cd[:, :, -1])


def test_input_matrices_returned_from_final_iteration_when_present():
    params = _make_params(n_iters=3, with_input=True)
    params.ka = params.kq = params.kr = params.kxinit = 2

    *_, b, d = lds_params_idx(params)

    np.testing.assert_array_equal(b, params.learned.bd[:, :, -1])
    np.testing.assert_array_equal(d, params.learned.dd[:, :, -1])


def test_input_matrices_are_none_when_no_input_matrices_present():
    params = _make_params(n_iters=3, with_input=False)
    params.ka = params.kq = params.kr = params.kxinit = 1

    *_, b, d = lds_params_idx(params)

    assert b is None
    assert d is None
