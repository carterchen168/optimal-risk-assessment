"""
tests/ldslearn/test_asos.py
------------------------------
Pytest suite for ldslearn/asos.py — ASOS Toolbox dependencies
(logdet, pextract, eextract, struct builders, doubling algorithms).
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('user_input_ressarch', MagicMock())

_mod_path = os.path.join(os.path.dirname(__file__), "..", "..", "ldslearn", "asos.py")
_spec = importlib.util.spec_from_file_location("ldslearn.asos", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Struct = _mod.Struct
logdet = _mod.logdet
pextract = _mod.pextract
eextract = _mod.eextract
estruct = _mod.estruct
pstruct = _mod.pstruct
KalmanDoubling = _mod.KalmanDoubling
LyapDoubling = _mod.LyapDoubling
SylvDoubling = _mod.SylvDoubling
Step = _mod.Step
Step_out = _mod.Step_out
ApproxEStep = _mod.ApproxEStep


def _make_pd(n, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


def _make_stable(n, seed, scale=0.5):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    M = M / (np.max(np.abs(np.linalg.eigvals(M))) + 1e-8)
    return scale * M


# ---------------------------------------------------------------------------
# logdet
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,seed", [(2, 0), (3, 1), (5, 2)])
def test_logdet_matches_numpy_for_pd_matrix(n, seed):
    A = _make_pd(n, seed)
    expected = np.log(np.linalg.det(A))
    assert np.isclose(logdet(A), expected, rtol=1e-6)


def test_logdet_identity_is_zero():
    assert np.isclose(logdet(np.eye(4)), 0.0)


# ---------------------------------------------------------------------------
# pstruct / pextract
# ---------------------------------------------------------------------------

def test_pstruct_returns_struct_with_expected_fields():
    rng = np.random.default_rng(0)
    A, B, C, D = rng.standard_normal((3, 3)), rng.standard_normal((3, 2)), \
        rng.standard_normal((4, 3)), rng.standard_normal((4, 2))
    Q, R = np.eye(3), np.eye(4)
    pi_1, V_1 = np.zeros(3), np.eye(3)

    params = pstruct(A, B, C, D, Q, R, pi_1, V_1)

    assert isinstance(params, Struct)
    assert np.array_equal(params.A, A)
    assert np.array_equal(params.B, B)
    assert np.array_equal(params.C, C)
    assert np.array_equal(params.D, D)
    assert np.array_equal(params.Q, Q)
    assert np.array_equal(params.R, R)
    assert np.array_equal(params.pi_1, pi_1)
    assert np.array_equal(params.V_1, V_1)


def test_pextract_round_trips_pstruct_and_derives_sizes():
    rng = np.random.default_rng(1)
    A, B, C, D = rng.standard_normal((3, 3)), rng.standard_normal((3, 2)), \
        rng.standard_normal((4, 3)), rng.standard_normal((4, 2))
    Q, R = np.eye(3), np.eye(4)
    pi_1, V_1 = np.zeros(3), np.eye(3)
    params = pstruct(A, B, C, D, Q, R, pi_1, V_1)

    A_, B_, C_, D_, Q_, R_, insize, outsize, hidsize, pi_1_, V_1_ = pextract(params)

    assert np.array_equal(A_, A) and np.array_equal(B_, B)
    assert np.array_equal(C_, C) and np.array_equal(D_, D)
    assert np.array_equal(Q_, Q) and np.array_equal(R_, R)
    assert insize == B.shape[1]
    assert outsize == C.shape[0]
    assert hidsize == A.shape[0]
    assert np.array_equal(pi_1_, pi_1)
    assert np.array_equal(V_1_, V_1)


# ---------------------------------------------------------------------------
# estruct / eextract
# ---------------------------------------------------------------------------

def test_estruct_returns_struct_with_expected_fields():
    rng = np.random.default_rng(2)
    vals = {k: rng.standard_normal((2, 2)) for k in
            ["Ex_x_0", "Ex_x_1", "Ey_x_0", "Eu_x_0", "Eu_x_1", "Ex_", "Exx_", "x1_"]}

    expt = estruct(**vals)

    assert isinstance(expt, Struct)
    for k, v in vals.items():
        assert np.array_equal(getattr(expt, k), v)


def test_eextract_round_trips_estruct():
    rng = np.random.default_rng(3)
    vals = {k: rng.standard_normal((2, 2)) for k in
            ["Ex_x_0", "Ex_x_1", "Ey_x_0", "Eu_x_0", "Eu_x_1", "Ex_", "Exx_", "x1_"]}
    expt = estruct(**vals)

    Ex_x_0, Ex_x_1, Ey_x_0, Eu_x_0, Eu_x_1, Ex_, Exx_, x1_ = eextract(expt)

    assert np.array_equal(Ex_x_0, vals["Ex_x_0"])
    assert np.array_equal(Ex_x_1, vals["Ex_x_1"])
    assert np.array_equal(Ey_x_0, vals["Ey_x_0"])
    assert np.array_equal(Eu_x_0, vals["Eu_x_0"])
    assert np.array_equal(Eu_x_1, vals["Eu_x_1"])
    assert np.array_equal(Ex_, vals["Ex_"])
    assert np.array_equal(Exx_, vals["Exx_"])
    assert np.array_equal(x1_, vals["x1_"])


# ---------------------------------------------------------------------------
# LyapDoubling — solves Theta = Phi Theta Phi' + M (discrete Lyapunov)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,seed", [(2, 0), (3, 1), (4, 2)])
def test_lyap_doubling_converges_to_fixed_point(n, seed):
    E = _make_stable(n, seed)
    M = _make_pd(n, seed + 100)

    Theta = LyapDoubling(E, M, T=50)

    assert Theta.shape == (n, n)
    residual = Theta - (E @ Theta @ E.T + M)
    assert np.linalg.norm(residual) < 1e-4 * np.linalg.norm(Theta)


def test_lyap_doubling_is_symmetric_for_symmetric_seed():
    n = 3
    E = _make_stable(n, 5)
    M = _make_pd(n, 6)
    Theta = LyapDoubling(E, M, T=50)
    assert np.allclose(Theta, Theta.T, atol=1e-6)


# ---------------------------------------------------------------------------
# SylvDoubling — solves Theta = Phi Theta Upsilon + M (Sylvester-type)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,seed", [(2, 0), (3, 1)])
def test_sylv_doubling_converges_to_fixed_point(n, seed):
    E = _make_stable(n, seed)
    F = _make_stable(n, seed + 10)
    M = np.random.default_rng(seed + 20).standard_normal((n, n))

    Theta = SylvDoubling(E, F, M, T=50)

    assert Theta.shape == (n, n)
    residual = Theta - (E @ Theta @ F + M)
    assert np.linalg.norm(residual) < 1e-4 * (np.linalg.norm(Theta) + 1e-12)


# ---------------------------------------------------------------------------
# KalmanDoubling — steady-state Kalman filter/smoother gain quantities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,m,seed", [(2, 2, 0), (3, 2, 1)])
def test_kalman_doubling_output_shapes_and_psd(n, m, seed):
    A = _make_stable(n, seed)
    Q = _make_pd(n, seed + 1)
    C = np.random.default_rng(seed + 2).standard_normal((m, n))
    R = _make_pd(m, seed + 3)

    V_0_0, V_0_1, V_0_T, V_1_T, K, J, invS = KalmanDoubling(A, Q, C, R, T=50)

    assert V_0_0.shape == (n, n)
    assert V_0_1.shape == (n, n)
    assert V_0_T.shape == (n, n)
    assert V_1_T.shape == (n, n)
    assert K.shape == (n, m)
    assert J.shape == (n, n)
    assert invS.shape == (m, m)

    # Steady-state covariances must be PSD.
    for mat in (V_0_0, V_0_1, V_0_T):
        eigvals = np.linalg.eigvalsh((mat + mat.T) / 2)
        assert np.all(eigvals >= -1e-6)


def test_kalman_doubling_v_0_1_satisfies_steady_state_riccati():
    n, m, seed = 3, 2, 4
    A = _make_stable(n, seed)
    Q = _make_pd(n, seed + 1)
    C = np.random.default_rng(seed + 2).standard_normal((m, n))
    R = _make_pd(m, seed + 3)

    V_0_0, V_0_1, *_ = KalmanDoubling(A, Q, C, R, T=80)

    rhs = A @ V_0_0 @ A.T + Q
    assert np.linalg.norm(V_0_1 - rhs) < 1e-3 * np.linalg.norm(V_0_1)


# ---------------------------------------------------------------------------
# Step / Step_out — the core ASOS approximate E-step
# ---------------------------------------------------------------------------

def _simulate_lds(A, B, C, D, Q, R, pi_1, V_1, u_, seed):
    """Simulate a synthetic LDS trajectory given a fixed input sequence."""
    rng = np.random.default_rng(seed)
    hidsize = A.shape[0]
    outsize = C.shape[0]
    T = u_.shape[1]

    x_ = np.zeros((hidsize, T))
    y_ = np.zeros((outsize, T))
    x_prev = rng.multivariate_normal(pi_1, V_1)
    for t in range(T):
        if t == 0:
            x_[:, t] = x_prev
        else:
            w = rng.multivariate_normal(np.zeros(hidsize), Q)
            x_[:, t] = A @ x_[:, t - 1] + B @ u_[:, t - 1] + w
        v = rng.multivariate_normal(np.zeros(outsize), R)
        y_[:, t] = C @ x_[:, t] + D @ u_[:, t] + v
    return y_, x_


def _synthetic_step_inputs(seed, insize=1):
    hidsize, outsize = 2, 2
    klim, edgesize, T = 2, 2, 30
    klag = 2 * klim + 1
    A = _make_stable(hidsize, seed)
    Q = _make_pd(hidsize, seed + 1) * 0.01
    C = np.random.default_rng(seed + 2).standard_normal((outsize, hidsize))
    R = _make_pd(outsize, seed + 3) * 0.01
    rng = np.random.default_rng(seed + 4)
    B = rng.standard_normal((hidsize, insize)) if insize else np.zeros((hidsize, 0))
    D = rng.standard_normal((outsize, insize)) if insize else np.zeros((outsize, 0))
    pi_1 = np.zeros(hidsize)
    V_1 = _make_pd(hidsize, seed + 5) * 0.1
    u_ = rng.standard_normal((insize, T)) if insize else np.zeros((0, T))

    y_, _ = _simulate_lds(A, B, C, D, Q, R, pi_1, V_1, u_, seed + 6)
    o = ApproxEStep(y_, u_, klim, klag, edgesize, insize, outsize)
    params = pstruct(A, B, C, D, Q, R, pi_1, V_1)
    return o, params


def test_approxestep_output_shapes():
    hidsize, outsize, insize = 2, 2, 1
    klim, edgesize, T = 2, 2, 30
    klag = 2 * klim + 1
    A = _make_stable(hidsize, seed=30)
    Q = _make_pd(hidsize, seed=31) * 0.01
    C = np.random.default_rng(32).standard_normal((outsize, hidsize))
    R = _make_pd(outsize, seed=33) * 0.01
    rng = np.random.default_rng(34)
    B = rng.standard_normal((hidsize, insize))
    D = rng.standard_normal((outsize, insize))
    pi_1 = np.zeros(hidsize)
    V_1 = _make_pd(hidsize, seed=35) * 0.1
    u_ = rng.standard_normal((insize, T))
    y_, _ = _simulate_lds(A, B, C, D, Q, R, pi_1, V_1, u_, seed=36)

    o = ApproxEStep(y_, u_, klim, klag, edgesize, insize, outsize)

    assert o.klim == klim
    assert o.klag == klag
    assert o.edgesize == edgesize
    assert o.T == T
    for field in ("y_y_", "y_u_", "u_y_", "u_u_"):
        arr = getattr(o, field)
        rows = outsize if field[0] == "y" else insize
        cols = outsize if field[2] == "y" else insize
        assert arr.shape == (rows, cols, klim + 2)
    for field in ("y_lead_", "u_lead_", "y_trail_", "u_trail_"):
        rows = outsize if field[0] == "y" else insize
        assert getattr(o, field).shape == (rows, edgesize)
    for field in ("y_pre_", "u_pre_", "y_post_", "u_post_"):
        rows = outsize if field[0] == "y" else insize
        assert getattr(o, field).shape == (rows, klag)


def test_step_output_shapes_and_finite_LL_err():
    o, params = _synthetic_step_inputs(seed=10, insize=1)
    A, B, C, D, Q, R, insize, outsize, hidsize, pi_1, V_1 = pextract(params)

    o_out, expt, err, LL = Step(o, params)

    assert o_out is o
    assert expt.Ex_x_0.shape == (hidsize, hidsize)
    assert expt.Ex_x_1.shape == (hidsize, hidsize)
    assert expt.Ey_x_0.shape == (outsize, hidsize)
    assert expt.Eu_x_0.shape == (insize, hidsize)
    assert expt.Eu_x_1.shape == (hidsize, insize)
    assert expt.Ex_.start.shape == (hidsize,)
    assert expt.Ex_.end.shape == (hidsize,)
    assert expt.Exx_.start.shape == (hidsize, hidsize)
    assert expt.Exx_.end.shape == (hidsize, hidsize)

    assert np.isfinite(err)
    assert np.isfinite(LL)
    assert np.isreal(err)
    assert np.isreal(LL)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_step_sufficient_stats_are_psd(seed):
    o, params = _synthetic_step_inputs(seed=seed, insize=1)
    _, expt, _, _ = Step(o, params)

    for mat in (expt.Ex_x_0, expt.Exx_.start, expt.Exx_.end):
        sym = (mat + mat.T) / 2
        eigvals = np.linalg.eigvalsh(sym)
        assert np.all(eigvals >= -1e-6 * max(1.0, np.max(np.abs(eigvals))))


def test_step_out_is_step_with_zero_insize():
    """Resolves issue #014: Step_out is Step() called with insize=0 (no
    distinct algorithm), matching learn_kalman.py's no-input call site
    which only supplies A, C, Q, R, initx, initV (no B/D)."""
    o, params = _synthetic_step_inputs(seed=20, insize=0)
    A, B, C, D, Q, R, insize, outsize, hidsize, pi_1, V_1 = pextract(params)
    assert insize == 0

    in_struct = dict(A=A, C=C, Q=Q, R=R, initx=pi_1, initV=V_1)
    o_wrap, expt_wrap, err_wrap, LL_wrap = Step_out(o, in_struct)

    o_direct, expt_direct, err_direct, LL_direct = Step(o, params)

    assert o_wrap is o_direct is o
    assert np.allclose(expt_wrap.Ex_x_0, expt_direct.Ex_x_0)
    assert np.allclose(expt_wrap.Ex_x_1, expt_direct.Ex_x_1)
    assert np.allclose(expt_wrap.Ey_x_0, expt_direct.Ey_x_0)
    assert expt_wrap.Eu_x_0.shape == (0, A.shape[0])
    assert np.isclose(err_wrap, err_direct)
    assert np.isclose(LL_wrap, LL_direct)
