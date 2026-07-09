"""
tests/ldslearn/test_learn_kalman.py
-------------------------------------
Pytest suite for ldslearn/learn_kalman.py.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.modules.setdefault('user_input_ressarch', MagicMock())

_ldslearn_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ldslearn")
# learn_kalman.py imports its siblings (e.g. em_converged) as bare modules,
# so ldslearn/ must be importable on sys.path before exec_module runs.
sys.path.insert(0, _ldslearn_dir)

_mod_path = os.path.join(_ldslearn_dir, "learn_kalman.py")
_spec = importlib.util.spec_from_file_location("ldslearn.learn_kalman", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

learn_kalman = _mod.learn_kalman
Struct = _mod.Struct
kalman_smoother = _mod.kalman_smoother
pstruct = _mod.pstruct
pstruct_noinput = _mod.pstruct_noinput
ExactEstep_noinput = _mod.ExactEstep_noinput
ExactEstep = _mod.ExactEstep
_estep = _mod._estep
_estep_input = _mod._estep_input


def _simulate_lds(ss, os_, T, seed, r_scale=0.01, q_scale=0.01):
    """Simulate a small stable no-input LDS, returning (y, A, C, Q, R, initx, initV)."""
    rng = np.random.default_rng(seed)
    A = np.diag(rng.uniform(0.3, 0.7, size=ss))
    C = rng.standard_normal((os_, ss))
    Q = np.eye(ss) * q_scale
    R = np.eye(os_) * r_scale
    initx = np.zeros(ss)
    initV = np.eye(ss)

    x = np.zeros((ss, T))
    y = np.zeros((os_, T))
    x_prev = initx.copy()
    for t in range(T):
        w = rng.multivariate_normal(np.zeros(ss), Q)
        x[:, t] = A @ x_prev + w
        v = rng.multivariate_normal(np.zeros(os_), R)
        y[:, t] = C @ x[:, t] + v
        x_prev = x[:, t]
    return y, A, C, Q, R, initx, initV


def test_kalman_smoother_shapes_no_input():
    ss, os_, T = 3, 2, 40
    y, A, C, Q, R, initx, initV = _simulate_lds(ss, os_, T, seed=1)

    xsmooth, Vsmooth, VVsmooth, loglik, perfect_loglik, aici_bias_cr = kalman_smoother(
        y, A, C, Q, R, initx, initV
    )

    assert xsmooth.shape == (ss, T)
    assert Vsmooth.shape == (ss, ss, T)
    assert VVsmooth.shape == (ss, ss, T)
    assert np.isfinite(loglik)
    assert isinstance(loglik, float)
    assert perfect_loglik == 0.0
    assert aici_bias_cr == 0.0


def test_kalman_smoother_shapes_with_input():
    ss, os_, is_, T = 3, 2, 2, 40
    rng = np.random.default_rng(2)
    y, A, C, Q, R, initx, initV = _simulate_lds(ss, os_, T, seed=2)
    B = rng.standard_normal((ss, is_)) * 0.1
    D = rng.standard_normal((os_, is_)) * 0.1
    u = rng.standard_normal((is_, T))

    xsmooth, Vsmooth, VVsmooth, loglik, perfect_loglik, aici_bias_cr = kalman_smoother(
        y, A, C, Q, R, initx, initV, B=B, D=D, u=u
    )

    assert xsmooth.shape == (ss, T)
    assert Vsmooth.shape == (ss, ss, T)
    assert VVsmooth.shape == (ss, ss, T)
    assert np.isfinite(loglik)
    assert perfect_loglik == 0.0
    assert aici_bias_cr == 0.0


def test_smoothed_covariances_are_symmetric_psd():
    ss, os_, T = 3, 2, 40
    y, A, C, Q, R, initx, initV = _simulate_lds(ss, os_, T, seed=3)

    _, Vsmooth, _, _, _, _ = kalman_smoother(y, A, C, Q, R, initx, initV)

    for t in range(T):
        V = Vsmooth[:, :, t]
        assert np.allclose(V, V.T, atol=1e-8)
        eigvals = np.linalg.eigvalsh(V)
        assert np.all(eigvals >= -1e-8)


def test_loglik_direction_across_noise_levels():
    # Correctly-specified model in both cases (R passed to the smoother matches
    # the R used to generate the data) — asserting direction only, not an exact
    # value: a low-noise correctly-specified model should not score worse than
    # a high-noise correctly-specified model.
    ss, os_, T = 3, 2, 80

    y_low, A, C, Q, R_low, initx, initV = _simulate_lds(
        ss, os_, T, seed=42, r_scale=0.01
    )
    y_high, _, _, _, R_high, _, _ = _simulate_lds(
        ss, os_, T, seed=42, r_scale=1.0
    )

    _, _, _, loglik_low, _, _ = kalman_smoother(y_low, A, C, Q, R_low, initx, initV)
    _, _, _, loglik_high, _, _ = kalman_smoother(y_high, A, C, Q, R_high, initx, initV)

    assert loglik_low > loglik_high


def _assert_symmetric_psd(M, atol=1e-8):
    assert np.allclose(M, M.T, atol=atol)
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals >= -atol)


def test_exact_estep_noinput_shapes_and_invariants():
    ss, os_, T = 3, 2, 40
    y, A, C, Q, R, initx, initV = _simulate_lds(ss, os_, T, seed=5)
    ps = pstruct_noinput(A, C, Q, R, initx, initV)

    expt, loglik_t, err = ExactEstep_noinput(y, ps)

    assert expt.Ex_x_1.shape == (ss, ss)
    assert expt.Ex_x_0.shape == (ss, ss)
    assert expt.Ey_x_0.shape == (os_, ss)
    assert expt.Ex_.shape == (ss, T)
    assert expt.Exx_.end.shape == (ss, ss)
    assert expt.Exx_.start.shape == (ss, ss)
    assert err is None
    assert np.isfinite(loglik_t)

    _assert_symmetric_psd(expt.Ex_x_0)
    _assert_symmetric_psd(expt.Exx_.end)
    _assert_symmetric_psd(expt.Exx_.start)


def test_exact_estep_noinput_matches_kalman_smoother():
    ss, os_, T = 3, 2, 40
    y, A, C, Q, R, initx, initV = _simulate_lds(ss, os_, T, seed=5)
    ps = pstruct_noinput(A, C, Q, R, initx, initV)

    xsmooth, Vsmooth, VVsmooth, loglik, _, _ = kalman_smoother(
        y, A, C, Q, R, initx, initV
    )
    expt, loglik_t, _ = ExactEstep_noinput(y, ps)

    assert np.allclose(expt.Ex_, xsmooth)
    assert np.isclose(loglik_t, loglik)

    end_expected = xsmooth[:, -1:] @ xsmooth[:, -1:].T + Vsmooth[:, :, -1]
    start_expected = xsmooth[:, :1] @ xsmooth[:, :1].T + Vsmooth[:, :, 0]
    assert np.allclose(expt.Exx_.end, end_expected)
    assert np.allclose(expt.Exx_.start, start_expected)

    delta = np.zeros((os_, ss))
    gamma = np.zeros((ss, ss))
    beta = np.zeros((ss, ss))
    for t in range(T):
        delta += y[:, [t]] @ xsmooth[:, [t]].T
        gamma += xsmooth[:, [t]] @ xsmooth[:, [t]].T + Vsmooth[:, :, t]
        if t > 0:
            beta += xsmooth[:, [t]] @ xsmooth[:, [t - 1]].T + VVsmooth[:, :, t]

    assert np.allclose(expt.Ex_x_0, gamma)
    assert np.allclose(expt.Ey_x_0, delta)
    assert np.allclose(expt.Ex_x_1, beta)


def test_exact_estep_with_input_shapes_and_invariants():
    ss, os_, is_, T = 3, 2, 2, 40
    rng = np.random.default_rng(6)
    y, A, C, Q, R, initx, initV = _simulate_lds(ss, os_, T, seed=6)
    B = rng.standard_normal((ss, is_)) * 0.1
    D = rng.standard_normal((os_, is_)) * 0.1
    u = rng.standard_normal((is_, T))
    ps = pstruct(A, B, C, D, Q, R, initx, initV)

    expt, loglik_t, err = ExactEstep(y, u, ps)

    assert expt.Ex_x_1.shape == (ss, ss)
    assert expt.Ex_x_0.shape == (ss, ss)
    assert expt.Ey_x_0.shape == (os_, ss)
    assert expt.Ex_.shape == (ss, T)
    assert expt.Exx_.end.shape == (ss, ss)
    assert expt.Exx_.start.shape == (ss, ss)
    assert expt.Eu_x_0.shape == (is_, ss)
    assert expt.Ex_u_1.shape == (ss, is_)
    assert err is None
    assert np.isfinite(loglik_t)

    _assert_symmetric_psd(expt.Ex_x_0)
    _assert_symmetric_psd(expt.Exx_.end)
    _assert_symmetric_psd(expt.Exx_.start)


def test_exact_estep_with_input_matches_kalman_smoother():
    ss, os_, is_, T = 3, 2, 2, 40
    rng = np.random.default_rng(6)
    y, A, C, Q, R, initx, initV = _simulate_lds(ss, os_, T, seed=6)
    B = rng.standard_normal((ss, is_)) * 0.1
    D = rng.standard_normal((os_, is_)) * 0.1
    u = rng.standard_normal((is_, T))
    ps = pstruct(A, B, C, D, Q, R, initx, initV)

    xsmooth, Vsmooth, VVsmooth, loglik, _, _ = kalman_smoother(
        y, A, C, Q, R, initx, initV, B=B, D=D, u=u
    )
    expt, loglik_t, _ = ExactEstep(y, u, ps)

    assert np.allclose(expt.Ex_, xsmooth)
    assert np.isclose(loglik_t, loglik)

    end_expected = xsmooth[:, -1:] @ xsmooth[:, -1:].T + Vsmooth[:, :, -1]
    start_expected = xsmooth[:, :1] @ xsmooth[:, :1].T + Vsmooth[:, :, 0]
    assert np.allclose(expt.Exx_.end, end_expected)
    assert np.allclose(expt.Exx_.start, start_expected)

    delta = np.zeros((os_, ss))
    gamma = np.zeros((ss, ss))
    beta = np.zeros((ss, ss))
    xi = np.zeros((is_, ss))
    psi = np.zeros((is_, ss))
    for t in range(T):
        delta += y[:, [t]] @ xsmooth[:, [t]].T
        gamma += xsmooth[:, [t]] @ xsmooth[:, [t]].T + Vsmooth[:, :, t]
        if t > 0:
            beta += xsmooth[:, [t]] @ xsmooth[:, [t - 1]].T + VVsmooth[:, :, t]
            psi += u[:, [t - 1]] @ xsmooth[:, [t]].T
        if t < T - 1:
            xi += u[:, [t]] @ xsmooth[:, [t]].T

    assert np.allclose(expt.Ex_x_0, gamma)
    assert np.allclose(expt.Ey_x_0, delta)
    assert np.allclose(expt.Ex_x_1, beta)
    assert np.allclose(expt.Eu_x_0, xi)
    assert np.allclose(expt.Ex_u_1, psi.T)


def test_pstruct_noinput_fields_and_shapes():
    ss, os_, T = 3, 2, 40
    y, A, C, Q, R, initx, initV = _simulate_lds(ss, os_, T, seed=7)

    ps = pstruct_noinput(A, C, Q, R, initx, initV)

    assert isinstance(ps, Struct)
    assert np.array_equal(ps.A, A)
    assert np.array_equal(ps.C, C)
    assert np.array_equal(ps.Q, Q)
    assert np.array_equal(ps.R, R)
    assert ps.initx.shape == (ss,)
    assert np.array_equal(ps.initx, initx)
    assert ps.initV.shape == (ss, ss)
    assert np.array_equal(ps.initV, initV)


def test_pstruct_fields_and_shapes():
    ss, os_, is_, T = 3, 2, 2, 40
    rng = np.random.default_rng(8)
    y, A, C, Q, R, initx, initV = _simulate_lds(ss, os_, T, seed=8)
    B = rng.standard_normal((ss, is_)) * 0.1
    D = rng.standard_normal((os_, is_)) * 0.1

    ps = pstruct(A, B, C, D, Q, R, initx, initV)

    assert isinstance(ps, Struct)
    assert np.array_equal(ps.A, A)
    assert np.array_equal(ps.B, B)
    assert np.array_equal(ps.C, C)
    assert np.array_equal(ps.D, D)
    assert np.array_equal(ps.Q, Q)
    assert np.array_equal(ps.R, R)
    assert ps.initx.shape == (ss,)
    assert np.array_equal(ps.initx, initx)
    assert ps.initV.shape == (ss, ss)
    assert np.array_equal(ps.initV, initV)


def test_pstruct_ravels_column_vector_initx():
    ss, os_, is_ = 3, 2, 2
    _, A, C, Q, R, initx, initV = _simulate_lds(ss, os_, 5, seed=9)
    rng = np.random.default_rng(9)
    B = rng.standard_normal((ss, is_)) * 0.1
    D = rng.standard_normal((os_, is_)) * 0.1
    initx_col = initx.reshape(ss, 1)

    ps = pstruct(A, B, C, D, Q, R, initx_col, initV)
    ps_noinput = pstruct_noinput(A, C, Q, R, initx_col, initV)

    assert ps.initx.shape == (ss,)
    assert np.array_equal(ps.initx, initx)
    assert ps_noinput.initx.shape == (ss,)
    assert np.array_equal(ps_noinput.initx, initx)


def test_estep_ar_mode_bypasses_kalman_smoother():
    ss, T = 3, 10
    rng = np.random.default_rng(10)
    A = np.diag(rng.uniform(0.3, 0.7, size=ss))
    C = np.eye(ss)
    Q = np.eye(ss) * 0.01
    R = np.eye(ss) * 0.01
    initx = np.zeros((ss, 1))
    initV = np.eye(ss)
    y = rng.standard_normal((ss, T))

    (beta, gamma, delta, gamma1, gamma2,
     x1, V1, xsmooth, loglik, perfect_loglik, aici_bias_cr) = _estep(
        y, A, C, Q, R, initx, initV, ar_mode=True
    )

    assert xsmooth is y
    assert np.array_equal(x1, y[:, 0])
    assert np.array_equal(V1, np.zeros((ss, ss)))
    assert loglik == 0.0
    assert perfect_loglik == 0.0
    assert aici_bias_cr == 0.0

    gamma_expected = np.zeros((ss, ss))
    beta_expected = np.zeros((ss, ss))
    delta_expected = np.zeros((ss, ss))
    for t in range(T):
        delta_expected += y[:, [t]] @ y[:, [t]].T
        gamma_expected += y[:, [t]] @ y[:, [t]].T
        if t > 0:
            beta_expected += y[:, [t]] @ y[:, [t - 1]].T
    assert np.allclose(gamma, gamma_expected)
    assert np.allclose(delta, delta_expected)
    assert np.allclose(beta, beta_expected)
    assert np.allclose(gamma1, gamma - y[:, [-1]] @ y[:, [-1]].T)
    assert np.allclose(gamma2, gamma - y[:, [0]] @ y[:, [0]].T)


def test_estep_input_ar_mode_bypasses_kalman_smoother():
    ss, is_, T = 3, 2, 10
    rng = np.random.default_rng(11)
    A = np.diag(rng.uniform(0.3, 0.7, size=ss))
    B = rng.standard_normal((ss, is_)) * 0.1
    C = np.eye(ss)
    D = rng.standard_normal((ss, is_)) * 0.1
    Q = np.eye(ss) * 0.01
    R = np.eye(ss) * 0.01
    initx = np.zeros((ss, 1))
    initV = np.eye(ss)
    y = rng.standard_normal((ss, T))
    u = rng.standard_normal((is_, T))

    (beta, gamma, delta, gamma1, gamma2, xi, psi,
     x1, V1, xsmooth, loglik, perfect_loglik, aici_bias_cr) = _estep_input(
        y, u, A, B, C, D, Q, R, initx, initV, ar_mode=True
    )

    assert xsmooth is y
    assert np.array_equal(x1, y[:, 0])
    assert np.array_equal(V1, np.zeros((ss, ss)))
    assert loglik == 0.0
    assert perfect_loglik == 0.0
    assert aici_bias_cr == 0.0

    gamma_expected = np.zeros((ss, ss))
    beta_expected = np.zeros((ss, ss))
    delta_expected = np.zeros((ss, ss))
    xi_expected = np.zeros((is_, ss))
    psi_expected = np.zeros((is_, ss))
    for t in range(T):
        delta_expected += y[:, [t]] @ y[:, [t]].T
        gamma_expected += y[:, [t]] @ y[:, [t]].T
        if t > 0:
            beta_expected += y[:, [t]] @ y[:, [t - 1]].T
            psi_expected += u[:, [t - 1]] @ y[:, [t]].T
        if t < T - 1:
            xi_expected += u[:, [t]] @ y[:, [t]].T
    assert np.allclose(gamma, gamma_expected)
    assert np.allclose(delta, delta_expected)
    assert np.allclose(beta, beta_expected)
    assert np.allclose(xi, xi_expected)
    assert np.allclose(psi, psi_expected)
    assert np.allclose(gamma1, gamma - y[:, [-1]] @ y[:, [-1]].T)
    assert np.allclose(gamma2, gamma - y[:, [0]] @ y[:, [0]].T)


def test_exact_estep_noinput_dispatches_to_estep_only():
    ss, os_, T = 3, 2, 20
    y, A, C, Q, R, initx, initV = _simulate_lds(ss, os_, T, seed=12)
    ps = pstruct_noinput(A, C, Q, R, initx, initV)

    with patch.object(_mod, "_estep", wraps=_mod._estep) as spy_estep, \
         patch.object(_mod, "_estep_input", wraps=_mod._estep_input) as spy_estep_input:
        ExactEstep_noinput(y, ps)

    spy_estep.assert_called_once()
    assert spy_estep.call_args.kwargs.get("ar_mode", spy_estep.call_args.args[-1]) is False
    spy_estep_input.assert_not_called()


def test_exact_estep_dispatches_to_estep_input_only():
    ss, os_, is_, T = 3, 2, 2, 20
    rng = np.random.default_rng(13)
    y, A, C, Q, R, initx, initV = _simulate_lds(ss, os_, T, seed=13)
    B = rng.standard_normal((ss, is_)) * 0.1
    D = rng.standard_normal((os_, is_)) * 0.1
    u = rng.standard_normal((is_, T))
    ps = pstruct(A, B, C, D, Q, R, initx, initV)

    with patch.object(_mod, "_estep", wraps=_mod._estep) as spy_estep, \
         patch.object(_mod, "_estep_input", wraps=_mod._estep_input) as spy_estep_input:
        ExactEstep(y, u, ps)

    spy_estep_input.assert_called_once()
    assert spy_estep_input.call_args.kwargs.get(
        "ar_mode", spy_estep_input.call_args.args[-1]
    ) is False
    spy_estep.assert_not_called()


def _assert_em_shapes(learned, LL, shapes):
    assert len(LL) >= 1
    n_hist = len(LL) + 1
    for field, shape in shapes.items():
        assert getattr(learned, field).shape == (*shape, n_hist)


def _assert_em_monotone_and_shapes(learned, LL, shapes):
    assert np.all(np.diff(LL) >= -1e-3)
    _assert_em_shapes(learned, LL, shapes)


def test_learn_kalman_main_loop_noinput():
    ss, os_, T = 2, 3, 60
    y, _, C_true, _, _, _, _ = _simulate_lds(
        ss, os_, T, seed=20, r_scale=0.05, q_scale=0.05
    )

    # Perturbed initial guess (distinct from the true params) so EM has room
    # to improve the likelihood across iterations.
    rng = np.random.default_rng(21)
    A0 = np.diag(rng.uniform(0.2, 0.5, size=ss))
    C0 = C_true + 0.1 * rng.standard_normal(C_true.shape)
    Q0 = np.eye(ss) * 0.05
    R0 = np.eye(os_) * 0.05
    initx0 = np.zeros(ss)
    initV0 = np.eye(ss)

    params = Struct(init=Struct(
        ad=A0, cd=C0, qwd=Q0, rvd=R0, initx0=initx0, xssd=initV0,
    ))

    learned, LL, LLp, aici = learn_kalman([y], params, max_iter=5, verbose=False)

    _assert_em_monotone_and_shapes(learned, LL, {
        "ad": (ss, ss), "cd": (os_, ss), "qwd": (ss, ss),
        "rvd": (os_, os_), "xssd": (ss, ss), "initx0": (ss,),
    })
    assert len(aici) == len(LL)


def test_learn_kalman_main_loop_with_input():
    ss, os_, is_, T = 2, 3, 2, 60
    rng_data = np.random.default_rng(22)
    y, _, C_true, _, _, _, _ = _simulate_lds(
        ss, os_, T, seed=22, r_scale=0.05, q_scale=0.05
    )
    B_true = rng_data.standard_normal((ss, is_)) * 0.1
    D_true = rng_data.standard_normal((os_, is_)) * 0.1
    u = rng_data.standard_normal((is_, T))
    y = y + D_true @ u  # fold the feedthrough term into the observations

    rng = np.random.default_rng(23)
    A0 = np.diag(rng.uniform(0.2, 0.5, size=ss))
    B0 = B_true + 0.05 * rng.standard_normal(B_true.shape)
    C0 = C_true + 0.1 * rng.standard_normal(C_true.shape)
    D0 = D_true + 0.05 * rng.standard_normal(D_true.shape)
    Q0 = np.eye(ss) * 0.05
    R0 = np.eye(os_) * 0.05
    initx0 = np.zeros(ss)
    initV0 = np.eye(ss)

    params = Struct(init=Struct(
        ad=A0, bd=B0, cd=C0, dd=D0, qwd=Q0, rvd=R0,
        initx0=initx0, xssd=initV0,
    ))

    learned, LL, LLp, aici = learn_kalman(
        [y], params, max_iter=5, verbose=False, datain=[u]
    )

    _assert_em_monotone_and_shapes(learned, LL, {
        "ad": (ss, ss), "bd": (ss, is_), "cd": (os_, ss), "dd": (os_, is_),
        "qwd": (ss, ss), "rvd": (os_, os_), "xssd": (ss, ss), "initx0": (ss,),
    })
    assert len(aici) == len(LL)


def _make_noinput_params(A0, C0, Q0, R0, initx0, initV0):
    return Struct(init=Struct(
        ad=A0.copy(), cd=C0.copy(), qwd=Q0.copy(), rvd=R0.copy(),
        initx0=initx0.copy(), xssd=initV0.copy(),
    ))


def test_learn_kalman_raw_3d_array_split_matches_list_of_sequences():
    # MATLAB splits a raw (os, T, N) array into N per-sequence cells via
    # num2cell(X, [1 2]); this should behave identically to passing the
    # equivalent list of per-sequence 2-D arrays.
    ss, os_, T = 2, 3, 40
    y1, _, C_true, _, _, _, _ = _simulate_lds(ss, os_, T, seed=30)
    y2, _, _, _, _, _, _ = _simulate_lds(ss, os_, T, seed=31)

    rng = np.random.default_rng(32)
    A0 = np.diag(rng.uniform(0.2, 0.5, size=ss))
    C0 = C_true + 0.1 * rng.standard_normal(C_true.shape)
    Q0 = np.eye(ss) * 0.05
    R0 = np.eye(os_) * 0.05
    initx0 = np.zeros(ss)
    initV0 = np.eye(ss)

    raw_3d = np.stack([y1, y2], axis=2)
    assert raw_3d.shape == (os_, T, 2)

    learned_list, LL_list, _, _ = learn_kalman(
        [y1, y2], _make_noinput_params(A0, C0, Q0, R0, initx0, initV0),
        max_iter=3, verbose=False,
    )
    learned_raw, LL_raw, _, _ = learn_kalman(
        raw_3d, _make_noinput_params(A0, C0, Q0, R0, initx0, initV0),
        max_iter=3, verbose=False,
    )

    assert LL_list == pytest.approx(LL_raw)
    assert np.allclose(learned_list.ad, learned_raw.ad)
    assert np.allclose(learned_list.cd, learned_raw.cd)
    assert np.allclose(learned_list.qwd, learned_raw.qwd)
    assert np.allclose(learned_list.rvd, learned_raw.rvd)


def test_learn_kalman_symmetrizes_q_r_v_exactly():
    # MATLAB's `any(any(X~=X'))` always symmetrizes on any FP asymmetry;
    # the fixed Python check (`np.any(X != X.T)`) should do the same, so
    # every learned Q/R/V0 slice is exactly (not just approximately)
    # symmetric.
    ss, os_, T = 2, 3, 50
    y, _, C_true, _, _, _, _ = _simulate_lds(
        ss, os_, T, seed=40, r_scale=0.05, q_scale=0.05
    )

    rng = np.random.default_rng(41)
    A0 = np.diag(rng.uniform(0.2, 0.5, size=ss))
    C0 = C_true + 0.1 * rng.standard_normal(C_true.shape)
    Q0 = np.eye(ss) * 0.05
    R0 = np.eye(os_) * 0.05
    initx0 = np.zeros(ss)
    initV0 = np.eye(ss)

    learned, LL, LLp, aici = learn_kalman(
        [y], _make_noinput_params(A0, C0, Q0, R0, initx0, initV0),
        max_iter=5, verbose=False,
    )

    for i in range(learned.qwd.shape[2]):
        assert np.array_equal(learned.qwd[:, :, i], learned.qwd[:, :, i].T)
        assert np.array_equal(learned.rvd[:, :, i], learned.rvd[:, :, i].T)
        assert np.array_equal(learned.xssd[:, :, i], learned.xssd[:, :, i].T)


def test_learn_kalman_nan_bailout_returns_four_tuple():
    # Force the NaN-bailout branch (gamma/delta contain NaN) by mocking
    # ExactEstep_noinput's sufficient statistics, and confirm the early
    # return still yields the 4-tuple (params, LL, LLp, aici) with
    # matching lengths, same as the normal-exit convention.
    ss, os_, T = 2, 3, 20
    y, _, C_true, _, _, _, _ = _simulate_lds(
        ss, os_, T, seed=50, r_scale=0.05, q_scale=0.05
    )

    rng = np.random.default_rng(51)
    A0 = np.diag(rng.uniform(0.2, 0.5, size=ss))
    C0 = C_true + 0.1 * rng.standard_normal(C_true.shape)
    Q0 = np.eye(ss) * 0.05
    R0 = np.eye(os_) * 0.05
    initx0 = np.zeros(ss)
    initV0 = np.eye(ss)
    params = _make_noinput_params(A0, C0, Q0, R0, initx0, initV0)

    nan_expt = Struct(
        Ex_x_1=np.full((ss, ss), np.nan),
        Ex_x_0=np.full((ss, ss), np.nan),
        Ey_x_0=np.full((os_, ss), np.nan),
        Ex_=np.zeros((ss, T)),
        Exx_=Struct(end=np.zeros((ss, ss)), start=np.zeros((ss, ss))),
    )

    with patch.object(
        _mod, "ExactEstep_noinput", return_value=(nan_expt, 0.0, None)
    ):
        result = learn_kalman([y], params, max_iter=5, verbose=False)

    assert len(result) == 4
    learned, LL, LLp, aici = result
    assert len(LL) == 1
    assert len(LLp) == len(LL)
    assert len(aici) == len(LL)
    assert learned.ad.shape[-1] == 1  # bailed before any M-step slice was appended


def test_learn_kalman_stops_early_when_converged():
    # Drive the loop's exit condition directly (mock em_converged) rather
    # than tuning data/thresholds to converge numerically — isolates the
    # main loop's wiring from em_converged's own (separately tested) logic.
    ss, os_, T = 2, 3, 30
    y, _, C_true, _, _, _, _ = _simulate_lds(
        ss, os_, T, seed=60, r_scale=0.05, q_scale=0.05
    )

    rng = np.random.default_rng(61)
    A0 = np.diag(rng.uniform(0.2, 0.5, size=ss))
    C0 = C_true + 0.1 * rng.standard_normal(C_true.shape)
    Q0 = np.eye(ss) * 0.05
    R0 = np.eye(os_) * 0.05
    initx0 = np.zeros(ss)
    initV0 = np.eye(ss)
    params = _make_noinput_params(A0, C0, Q0, R0, initx0, initV0)

    with patch.object(_mod, "em_converged", return_value=(True, False)):
        learned, LL, LLp, aici = learn_kalman(
            [y], params, max_iter=10, verbose=False
        )

    assert len(LL) == 1
    assert len(LL) < 10


def test_learn_kalman_exhausts_max_iter_without_converging():
    ss, os_, T = 2, 3, 30
    y, _, C_true, _, _, _, _ = _simulate_lds(
        ss, os_, T, seed=62, r_scale=0.05, q_scale=0.05
    )

    rng = np.random.default_rng(63)
    A0 = np.diag(rng.uniform(0.2, 0.5, size=ss))
    C0 = C_true + 0.1 * rng.standard_normal(C_true.shape)
    Q0 = np.eye(ss) * 0.05
    R0 = np.eye(os_) * 0.05
    initx0 = np.zeros(ss)
    initV0 = np.eye(ss)
    params = _make_noinput_params(A0, C0, Q0, R0, initx0, initV0)

    with patch.object(_mod, "em_converged", return_value=(False, False)):
        learned, LL, LLp, aici = learn_kalman(
            [y], params, max_iter=4, verbose=False
        )

    assert len(LL) == 4


def test_learn_kalman_asosflag_noinput():
    # T must exceed 2*edgesize + klim + 1 = 53 since edgesize=25 is hardcoded
    # at the ApproxEStep call site.
    ss, os_, T = 2, 3, 150
    y, _, C_true, _, _, _, _ = _simulate_lds(
        ss, os_, T, seed=70, r_scale=0.05, q_scale=0.05
    )

    rng = np.random.default_rng(71)
    A0 = np.diag(rng.uniform(0.2, 0.5, size=ss))
    C0 = C_true + 0.1 * rng.standard_normal(C_true.shape)
    Q0 = np.eye(ss) * 0.05
    R0 = np.eye(os_) * 0.05
    initx0 = np.zeros(ss)
    initV0 = np.eye(ss)
    params = _make_noinput_params(A0, C0, Q0, R0, initx0, initV0)
    params.klim = 2

    learned, LL, LLp, aici = learn_kalman(
        [y], params, max_iter=3, verbose=False, asos_flag=True
    )

    _assert_em_shapes(learned, LL, {
        "ad": (ss, ss), "cd": (os_, ss), "qwd": (ss, ss),
        "rvd": (os_, os_), "xssd": (ss, ss), "initx0": (ss,),
    })
    assert np.all(np.isfinite(LL))
    assert len(aici) == len(LL)


def test_learn_kalman_asosflag_with_input():
    ss, os_, is_, T = 2, 3, 2, 150
    rng_data = np.random.default_rng(72)
    y, _, C_true, _, _, _, _ = _simulate_lds(
        ss, os_, T, seed=72, r_scale=0.05, q_scale=0.05
    )
    B_true = rng_data.standard_normal((ss, is_)) * 0.1
    D_true = rng_data.standard_normal((os_, is_)) * 0.1
    u = rng_data.standard_normal((is_, T))
    y = y + D_true @ u  # fold the feedthrough term into the observations

    rng = np.random.default_rng(73)
    A0 = np.diag(rng.uniform(0.2, 0.5, size=ss))
    B0 = B_true + 0.05 * rng.standard_normal(B_true.shape)
    C0 = C_true + 0.1 * rng.standard_normal(C_true.shape)
    D0 = D_true + 0.05 * rng.standard_normal(D_true.shape)
    Q0 = np.eye(ss) * 0.05
    R0 = np.eye(os_) * 0.05
    initx0 = np.zeros(ss)
    initV0 = np.eye(ss)

    params = Struct(init=Struct(
        ad=A0, bd=B0, cd=C0, dd=D0, qwd=Q0, rvd=R0,
        initx0=initx0, xssd=initV0,
    ))
    params.klim = 2

    learned, LL, LLp, aici = learn_kalman(
        [y], params, max_iter=3, verbose=False, datain=[u], asos_flag=True
    )

    _assert_em_shapes(learned, LL, {
        "ad": (ss, ss), "bd": (ss, is_), "cd": (os_, ss), "dd": (os_, is_),
        "qwd": (ss, ss), "rvd": (os_, os_), "xssd": (ss, ss), "initx0": (ss,),
    })
    assert np.all(np.isfinite(LL))
    assert len(aici) == len(LL)
