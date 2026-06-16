import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

_LOG_SCALE_ALGOS = {'svr', 'libsvr', 'gp', 'lin', 'quad', 'ransac'}


def _grid_shape(n):
    nrows = math.ceil(math.sqrt(n))
    ncols = int(n / math.floor(math.sqrt(n)))
    return nrows, ncols


def run(params, modelselectdata):
    algos = params.algo
    n = len(algos)
    nrows, ncols = _grid_shape(n)
    tunetype = getattr(params, 'tunetype', None)

    # ── Figure 1: tuning curves ───────────────────────────────────────────────
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for i, algo in enumerate(algos):
        ax = axes1[i // ncols][i % ncols]
        hyp = np.asarray(modelselectdata.hyp_param[i])
        jmse = np.asarray(modelselectdata.Jmse[i])
        ax.plot(hyp, jmse, label='NMSE value')
        if algo in _LOG_SCALE_ALGOS:
            ax.set_xscale('log')
        opt_idx = int(np.argmin(jmse))
        min_j = float(jmse[opt_idx])
        tuneval = float(modelselectdata.tuneval[i])
        ax.plot(tuneval, min_j, 'r.', markersize=12,
                label=f'Min NMSE value of {min_j:.4f} found with {algo} = {tuneval:.4f}')
        xlabel = tunetype[i] if tunetype else algo
        ax.set_xlabel(xlabel)
        ax.set_ylabel('NMSE')
        ax.set_title(f'Regression grid search results for {algo}')
        ax.legend(fontsize=7)

    for j in range(n, nrows * ncols):
        axes1[j // ncols][j % ncols].set_visible(False)

    fig1.tight_layout()
    fig1.savefig(os.path.join(params.datapath, 'tuning_curves.png'), dpi=150)
    plt.close(fig1)

    # ── Figure 2: predictions vs actual ──────────────────────────────────────
    if not hasattr(modelselectdata, 'vld_y') or not hasattr(modelselectdata, 'output_val'):
        return

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    y_true = np.asarray(modelselectdata.vld_y)

    for i, algo in enumerate(algos):
        ax = axes2[i // ncols][i % ncols]
        yhat = np.concatenate([np.asarray(b) for b in modelselectdata.output_val[i].yhat])
        r2 = r2_score(y_true, yhat)
        lim_lo = min(y_true.min(), yhat.min())
        lim_hi = max(y_true.max(), yhat.max())
        ax.scatter(y_true, yhat, s=10, alpha=0.6)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], '--', color='grey', linewidth=1)
        ax.set_xlabel('Actual (Z-scored)')
        ax.set_ylabel('Predicted (Z-scored)')
        ax.set_title(f'{algo}  R²={r2:.3f}')

    for j in range(n, nrows * ncols):
        axes2[j // ncols][j % ncols].set_visible(False)

    fig2.tight_layout()
    fig2.savefig(os.path.join(params.datapath, 'predictions.png'), dpi=150)
    plt.close(fig2)
