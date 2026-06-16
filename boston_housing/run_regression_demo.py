"""
boston_housing/run_regression_demo.py
--------------------------------------
Phase 1 regression demo on the Boston Housing dataset.

Prerequisites: run export_data.ipynb first to generate
  boston_housing/Training/train.csv
  boston_housing/Validation/val.csv
  boston_housing/Testing/test.csv

Output:
  boston_housing/tuning_curves.png
  boston_housing/predictions.png
  R² summary table printed to stdout
"""

import sys
import os
import importlib.util
from unittest.mock import MagicMock

import numpy as np
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Mock Phase 2/3 deps so testoptloop_ressarch loads cleanly
# ---------------------------------------------------------------------------

sys.modules.setdefault('user_input_ressarch', MagicMock())
for _name in ['detectopt', 'detectopt.truthdata', 'detectopt.leveltune',
              'detectopt.detectioncall', 'ldslearn', 'ldslearn.lds_timeseries']:
    sys.modules.setdefault(_name, MagicMock())

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

_spec = importlib.util.spec_from_file_location(
    'testoptloop_ressarch',
    os.path.join(_ROOT, 'testoptloop_ressarch.py'),
)
_tol_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tol_mod)
make_datafiles = _tol_mod.make_datafiles
Struct = _tol_mod.Struct

from regressopt.preprocessing import GlobalDataScaler
from regressopt.mainREGcode_ressarch import mainREGcode_ressarch
from regressopt.modelopttest import modelopttest
import plotregressresults

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLUMNS = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
           'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
FEATURE_IDX = list(range(13))
ALL_ALGOS = ['gp', 'svr', 'libsvr', 'knn', 'btree', 'lin', 'quad', 'bnet', 'elm', 'ransac']

# ---------------------------------------------------------------------------
# Build params
# ---------------------------------------------------------------------------

params = Struct(
    nompath=os.path.join(_SCRIPT_DIR, 'Training'),
    valpath=os.path.join(_SCRIPT_DIR, 'Validation'),
    testpath=os.path.join(_SCRIPT_DIR, 'Testing'),
    header=COLUMNS,
    targetName='MEDV',
    channelContineous=FEATURE_IDX,
    filelength=[354],
    algo=ALL_ALGOS,
    tune=[
        [1e-5, 1e5, 1.0],    # gp
        [1e-5, 1e5, 100.0],  # svr
        [1e-5, 1e5, 100.0],  # libsvr
        [1,    20,  5.0],    # knn
        [1,    50,  5.0],    # btree
        [1e-5, 1e5, 0.01],   # lin
        [1e-5, 1e5, 0.01],   # quad
        [1,    50,  10.0],   # bnet
        [1,    200, 50.0],   # elm
        [1e-5, 1e5, 1.0],    # ransac
    ],
    regress=Struct(flag=1, optIdx=7),
    distrib=2,
    datapath=_SCRIPT_DIR,
    Ntests=1,
)

# ---------------------------------------------------------------------------
# Load and scale data
# ---------------------------------------------------------------------------

print('Loading data...')
tr, vld, train_cell, _, _, params, _, _, _ = make_datafiles(params, 2)

scaler = GlobalDataScaler()
tr.x, tr.y = scaler.fit_global_baselines(tr.x, tr.y)
vld.x, vld.y = scaler.transform_evaluation_data(vld.x, vld.y)

train_cell.x = [tr.x[i:i+1, :] for i in range(len(tr.y))]
train_cell.y = [np.array([tr.y[i]]) for i in range(len(tr.y))]
trtest = Struct(x=[tr.x], y=[tr.y])
vld_batched = Struct(x=[vld.x], y=[vld.y])

# ---------------------------------------------------------------------------
# Grid search (optIdx=7)
# ---------------------------------------------------------------------------

modelselectdata = Struct()
modelselectdata.hyp_param = []
modelselectdata.Jmse = []
modelselectdata.tuneval = np.array([])
modelselectdata.output_val = []
modelselectdata.vld_y = vld.y

print(f'\nRunning grid search for {len(ALL_ALGOS)} algorithms...')

run_opts = Struct()

for i, algo in enumerate(ALL_ALGOS):
    tune_min, tune_max = params.tune[i][0], params.tune[i][1]

    if algo in ['svr', 'libsvr', 'gp', 'lin', 'quad', 'ransac']:
        hp = np.logspace(np.log10(tune_min), np.log10(tune_max), 50)
    elif algo == 'lasso':
        hp = np.linspace(tune_min, tune_max, 50)
    else:
        step = max(1, int(np.ceil(tune_max / 50)))
        hp = np.arange(tune_min, tune_max + 1, step)

    print(f'  {algo}: evaluating {len(hp)} grid points...', flush=True)
    params.Ntests = len(trtest.y)

    jmse_vals = []
    for x in hp:
        result = modelopttest(x, params, i, tr, trtest)
        jmse_vals.append(result)

    opt_idx = int(np.argmin(jmse_vals))
    tuneval = float(hp[opt_idx])

    modelselectdata.hyp_param.append(hp)
    modelselectdata.Jmse.append(np.array(jmse_vals))
    modelselectdata.tuneval = np.append(modelselectdata.tuneval, tuneval)

    params.Ntests = len(vld_batched.y)
    output_val, run_opts = mainREGcode_ressarch(tuneval, tr, vld_batched, [algo], run_opts)
    modelselectdata.output_val.append(output_val)

    r2 = r2_score(vld.y, output_val.yhat[0])
    print(f'    tuneval={tuneval:.4g}  R²={r2:.3f}')

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

print('\nGenerating figures...')
plotregressresults.run(params, modelselectdata)
print(f'  Saved tuning_curves.png and predictions.png to {_SCRIPT_DIR}/')

# ---------------------------------------------------------------------------
# R² summary table
# ---------------------------------------------------------------------------

print('\n── R² Summary ─────────────────────────────')
print(f'{"Algorithm":<10}  {"R²":>8}')
print('─' * 22)
for i, algo in enumerate(ALL_ALGOS):
    yhat = np.concatenate([np.asarray(b) for b in modelselectdata.output_val[i].yhat])
    r2 = r2_score(vld.y, yhat)
    print(f'{algo:<10}  {r2:>8.3f}')
print('─' * 22)
