"""
tests/test_integration_regressopt.py
-------------------------------------
Phase 1 integration test for the regressopt toolbox on the Boston Housing dataset.

Lower level than test_integration_testoptloop_run.py: this test calls individual modules 
directly (make_datafiles, modelsearch, mainREGcode_ressarch) instead of the orchestrator testoptloop_ressarch.run().

Pipeline under test (production path):
  make_datafiles → GlobalDataScaler → modelsearch → mainREGcode_ressarch
  All 10 algorithms, optIdx=6 (L-BFGS-B, matches MATLAB fmincon default).

To test other optimizers, change params.regress.optIdx in the boston_data
fixture (see modelsearch.py for the optIdx → scipy solver mapping).

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
# so testoptloop_ressarch loads without side-effects.
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
# Import real regressopt modules
# ---------------------------------------------------------------------------

from regressopt.mainREGcode_ressarch import mainREGcode_ressarch
from regressopt.preprocessing import GlobalDataScaler

# ---------------------------------------------------------------------------
# Import make_datafiles and modelsearch via spec (testoptloop imports detectopt
# at module level; mocks above satisfy those imports).
# ---------------------------------------------------------------------------

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "testoptloop_ressarch",
    os.path.join(_ROOT, "testoptloop_ressarch.py"),
)
_tol_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tol_mod)
make_datafiles = _tol_mod.make_datafiles
modelsearch_fn = _tol_mod.modelsearch


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
# Struct helper
# ---------------------------------------------------------------------------

class Struct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Fixture: load and scale data once for the whole module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def boston_data():
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
            [1e-5, 1e5, 1.0],    # svr  (x0=sigma; was C=100 before parity fix)
            [1e-5, 1e5, 100.0],  # libsvr
            [1,    20,  5.0],    # knn
            [1,    50,  5.0],    # btree
            [1e-5, 1e5, 0.01],   # lin
            [1e-5, 1e5, 0.01],   # quad
            [1,    50,  10.0],   # bnet
            [1,    200, 50.0],   # elm
            [1e-5, 1e5, 1.0],    # ransac
        ],
        # optIdx=6 → L-BFGS-B (fastest local solver, MATLAB fmincon equivalent).
        # Change to test other optimizers: 1=shgo, 2=dual_annealing, 3=diff_evolution,
        # 4=Nelder-Mead, 5=basinhopping (see modelsearch.py).
        regress=Struct(flag=1, optIdx=6),
        distrib=2,
    )

    tr, vld, train_cell, _, _, params, _, _, _ = make_datafiles(params, 2)

    scaler = GlobalDataScaler()
    tr.x, tr.y = scaler.fit_global_baselines(tr.x, tr.y)
    vld.x, vld.y = scaler.transform_evaluation_data(vld.x, vld.y)

    train_cell.x = [tr.x[i : i + 1, :] for i in range(len(tr.y))]
    train_cell.y = [np.array([tr.y[i]]) for i in range(len(tr.y))]

    trtest = Struct(x=[tr.x], y=[tr.y])
    vld_batched = Struct(x=[vld.x], y=[vld.y])

    return params, tr, vld, train_cell, trtest, vld_batched


# ---------------------------------------------------------------------------
# Fixture: run modelsearch for all 10 algos once, cache results
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def modelsearch_results(boston_data):
    params, tr, vld, train_cell, trtest, vld_batched = boston_data
    results = {}
    for i, algo in enumerate(ALL_ALGOS):
        tuneval, fval, _ = modelsearch_fn(
            params.tune[i][2], params, tr, trtest, i
        )
        results[algo] = {"tuneval": tuneval, "fval": fval}
    return results


# ---------------------------------------------------------------------------
# 1. make_datafiles — correct shapes
# ---------------------------------------------------------------------------

class TestMakeDatafilesLoads:

    def test_train_feature_shape(self, boston_data):
        _, tr, _, _, _, _ = boston_data
        assert tr.x.shape == (354, 13)

    def test_train_target_shape(self, boston_data):
        _, tr, _, _, _, _ = boston_data
        assert tr.y.shape == (354,)

    def test_val_feature_shape(self, boston_data):
        _, _, vld, _, _, _ = boston_data
        assert vld.x.shape == (76, 13)

    def test_val_target_shape(self, boston_data):
        _, _, vld, _, _, _ = boston_data
        assert vld.y.shape == (76,)

    def test_train_cell_length(self, boston_data):
        _, tr, _, train_cell, _, _ = boston_data
        assert len(train_cell.x) == 354
        assert len(train_cell.y) == 354


# ---------------------------------------------------------------------------
# 2. GlobalDataScaler — no leakage from val into train fit
# ---------------------------------------------------------------------------

class TestPreprocessing:

    def test_train_features_mean_near_zero(self, boston_data):
        _, tr, _, _, _, _ = boston_data
        assert np.allclose(tr.x.mean(axis=0), 0.0, atol=1e-10)

    def test_train_features_std_near_one(self, boston_data):
        _, tr, _, _, _, _ = boston_data
        assert np.allclose(tr.x.std(axis=0), 1.0, atol=1e-10)

    def test_val_not_refitted(self, boston_data):
        _, _, vld, _, _, _ = boston_data
        assert not np.allclose(vld.x.mean(axis=0), 0.0, atol=1e-2)


# ---------------------------------------------------------------------------
# 3. modelsearch convergence — all 10 algos, fval < 1.0 (beats naive baseline)
# ---------------------------------------------------------------------------

class TestModelsearchConvergence:

    @pytest.mark.parametrize("algo", ALL_ALGOS)
    def test_tuneval_finite(self, algo, modelsearch_results):
        tuneval = modelsearch_results[algo]["tuneval"]
        assert np.isfinite(tuneval), f"{algo}: tuneval={tuneval} not finite"

    @pytest.mark.parametrize("algo", ALL_ALGOS)
    def test_fval_beats_naive_baseline(self, algo, modelsearch_results):
        fval = modelsearch_results[algo]["fval"]
        # NMSE < 1.0: model error smaller than predicting the training mean.
        assert fval < 1.0, f"{algo}: fval={fval:.4f} >= 1.0 (no better than mean)"


# ---------------------------------------------------------------------------
# 4. Output contract — yhat shape and no NaN/Inf using modelsearch tunevals
# ---------------------------------------------------------------------------

class TestAllAlgorithmsOutputContract:

    @pytest.mark.parametrize("algo", ALL_ALGOS)
    def test_yhat_correct_shape(self, algo, boston_data, modelsearch_results):
        _, tr, _, _, _, vld_batched = boston_data
        tuneval = modelsearch_results[algo]["tuneval"]
        output, _ = mainREGcode_ressarch(tuneval, tr, vld_batched, [algo], Struct())
        assert output.yhat[0].shape == (76,)

    @pytest.mark.parametrize("algo", ALL_ALGOS)
    def test_yhat_no_nan_or_inf(self, algo, boston_data, modelsearch_results):
        _, tr, _, _, _, vld_batched = boston_data
        tuneval = modelsearch_results[algo]["tuneval"]
        output, _ = mainREGcode_ressarch(tuneval, tr, vld_batched, [algo], Struct())
        yhat = output.yhat[0]
        assert not np.any(np.isnan(yhat)), f"{algo}: NaN in yhat"
        assert not np.any(np.isinf(yhat)), f"{algo}: Inf in yhat"


# ---------------------------------------------------------------------------
# 5. R² sanity — all 10 algos beat naive baseline on val set
# ---------------------------------------------------------------------------

class TestAllAlgorithmsR2:

    @pytest.mark.parametrize("algo", ALL_ALGOS)
    def test_r2_above_zero(self, algo, boston_data, modelsearch_results):
        _, tr, vld, _, _, vld_batched = boston_data
        tuneval = modelsearch_results[algo]["tuneval"]
        output, _ = mainREGcode_ressarch(tuneval, tr, vld_batched, [algo], Struct())
        r2 = r2_score(vld.y, output.yhat[0])
        assert r2 > 0.0, f"{algo}: R²={r2:.3f} <= 0 (worse than predicting mean)"
