"""
tests/test_integration_testoptloop_run.py
-------------------------------------------
Phase 1 integration test for testoptloop_ressarch.run() (the orchestrator
ressarch.py actually calls), as opposed to test_integration_regressopt.py
which calls make_datafiles/modelsearch/mainREGcode_ressarch directly.

Pipeline under test (production path):
  run(params) with params.regressonly=True
    -> make_datafiles -> GlobalDataScaler -> per-algo modelsearch loop
    -> mainREGcode_ressarch (train/val/test)
    -> regressonly short-circuit (skips detectopt/ldslearn Phase 2/3)

All 10 algorithms, optIdx=6 (L-BFGS-B, matches MATLAB fmincon default).

Prerequisites: run boston_housing/boston_housing.ipynb to generate
  boston_housing/Training/train.csv
  boston_housing/Validation/val.csv
  boston_housing/Testing/test.csv

Tests are skipped automatically if the splits are missing.
"""

import sys
import importlib.util
import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Module isolation — mock Phase 2/3 dependencies before any project import
# so testoptloop_ressarch loads without side-effects. Because
# params.regressonly=True means run() never calls truthdata/leveltune/
# detectioncall, these mocks only need to satisfy the module-level imports.
# ---------------------------------------------------------------------------

sys.modules.setdefault("user_input_ressarch", MagicMock())
for _name in [
    "detectopt",
    "detectopt.truthdata",
    "detectopt.leveltune",
    "detectopt.detectioncall",
    "ldslearn",
    "ldslearn.lds_timeseries",
]:
    sys.modules.setdefault(_name, MagicMock())

# ---------------------------------------------------------------------------
# Import run() via spec (testoptloop_ressarch imports detectopt at module
# level; mocks above satisfy those imports).
# ---------------------------------------------------------------------------

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "testoptloop_ressarch",
    os.path.join(_ROOT, "testoptloop_ressarch.py"),
)
_tol_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tol_mod)
run = _tol_mod.run
Struct = _tol_mod.Struct

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOSTON_DIR = os.path.join(_ROOT, "boston_housing")
TRAINING_DIR = os.path.join(BOSTON_DIR, "Training")

COLUMNS = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV",
]
FEATURE_IDX = list(range(13))

ALL_ALGOS = ["gp", "svr", "libsvr", "knn", "btree", "lin", "quad", "bnet", "elm", "ransac"]


# ---------------------------------------------------------------------------
# Fixture: run() once for all 10 algos, cache results
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run_result():
    if not os.path.isdir(TRAINING_DIR):
        pytest.skip(
            "Boston Housing splits not found. "
            "Run boston_housing/boston_housing.ipynb first."
        )

    params = Struct(
        nompath=os.path.join(BOSTON_DIR, "Training"),
        valpath=os.path.join(BOSTON_DIR, "Validation"),
        testpath=os.path.join(BOSTON_DIR, "Testing"),
        header=COLUMNS,
        targetName="MEDV",
        channelContineous=FEATURE_IDX,
        filelength=[354],
        algo=ALL_ALGOS,
        tune=[
            [1e-5, 1e5, 1.0],    # gp
            [1e-5, 1e5, 1.0],    # svr
            [1e-5, 1e5, 100.0],  # libsvr
            [1,    20,  5.0],    # knn
            [1,    50,  5.0],    # btree
            [1e-5, 1e5, 0.01],   # lin
            [1e-5, 1e5, 0.01],   # quad
            [1,    50,  10.0],   # bnet
            [1,    200, 50.0],   # elm
            [1e-5, 1e5, 1.0],    # ransac
        ],
        regress=Struct(flag=1, optIdx=6),
        distrib=2,
        regressonly=True,
    )

    model_select_data, rocarea = run(params)
    return model_select_data, rocarea


# ---------------------------------------------------------------------------
# 1. regressonly short-circuit — Phase 2/3 actually skipped
# ---------------------------------------------------------------------------

class TestRegressonlySkip:

    def test_skip_flag_set(self, run_result):
        model_select_data, _ = run_result
        assert model_select_data.regressonly_skip is True

    def test_skip_reason_mentions_phase_2_3(self, run_result):
        model_select_data, _ = run_result
        assert "Phase 2/3" in model_select_data.regressonly_skip_reason

    def test_rocarea_empty(self, run_result):
        _, rocarea = run_result
        assert rocarea == []


# ---------------------------------------------------------------------------
# 2. Output contract — validation yhat shape and no NaN/Inf, per algo
# ---------------------------------------------------------------------------

class TestRunOutputContract:

    @pytest.mark.parametrize("algo_idx", range(len(ALL_ALGOS)))
    def test_yhat_correct_shape(self, algo_idx, run_result):
        model_select_data, _ = run_result
        yhat = model_select_data.output_val[algo_idx].yhat[0]
        assert yhat.shape == (76,)

    @pytest.mark.parametrize("algo_idx", range(len(ALL_ALGOS)))
    def test_yhat_no_nan_or_inf(self, algo_idx, run_result):
        model_select_data, _ = run_result
        yhat = model_select_data.output_val[algo_idx].yhat[0]
        assert not np.any(np.isnan(yhat)), f"{ALL_ALGOS[algo_idx]}: NaN in yhat"
        assert not np.any(np.isinf(yhat)), f"{ALL_ALGOS[algo_idx]}: Inf in yhat"


# ---------------------------------------------------------------------------
# 3. R² sanity — all 10 algos beat naive baseline on val set
# ---------------------------------------------------------------------------

class TestRunR2:

    @pytest.mark.parametrize("algo_idx", range(len(ALL_ALGOS)))
    def test_r2_above_zero(self, algo_idx, run_result):
        model_select_data, _ = run_result
        yhat = model_select_data.output_val[algo_idx].yhat[0]
        r2 = r2_score(model_select_data.vld_y, yhat)
        assert r2 > 0.0, f"{ALL_ALGOS[algo_idx]}: R²={r2:.3f} <= 0 (worse than predicting mean)"


# ---------------------------------------------------------------------------
# 4. modelsearch convergence — tuneval finite, Jmse beats naive baseline
# ---------------------------------------------------------------------------

class TestRunTuneval:

    @pytest.mark.parametrize("algo_idx", range(len(ALL_ALGOS)))
    def test_tuneval_finite(self, algo_idx, run_result):
        model_select_data, _ = run_result
        tuneval = model_select_data.tuneval[algo_idx]
        assert np.isfinite(tuneval), f"{ALL_ALGOS[algo_idx]}: tuneval={tuneval} not finite"

    @pytest.mark.parametrize("algo_idx", range(len(ALL_ALGOS)))
    def test_jmse_beats_naive_baseline(self, algo_idx, run_result):
        model_select_data, _ = run_result
        jmse = model_select_data.Jmse[algo_idx]
        # NMSE < 1.0: model error smaller than predicting the training mean.
        assert jmse < 1.0, f"{ALL_ALGOS[algo_idx]}: Jmse={jmse:.4f} >= 1.0 (no better than mean)"
