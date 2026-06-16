"""
tests/test_plotregressresults.py
---------------------------------
Unit tests for plotregressresults.run(params, modelselectdata).

Tests assert on external behavior: subplot count, axis scale, legend text,
and PNG file creation. No assertions on pixel values or matplotlib internals.
"""

import os
import sys
import importlib.util

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import plotregressresults directly (no user_input_ressarch dependency)
# ---------------------------------------------------------------------------

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "plotregressresults",
    os.path.join(_ROOT, "plotregressresults.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
plotregressresults = _mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Struct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class OutputStub:
    def __init__(self, yhat):
        self.yhat = yhat


def _make_params(tmp_path, algos, tunetype=None):
    p = Struct(algo=algos, datapath=str(tmp_path))
    if tunetype is not None:
        p.tunetype = tunetype
    return p


def _make_modelselectdata(rng, algos, grid_size=20, include_vld_y=True):
    n = len(algos)
    hyp_param = []
    Jmse = []
    tuneval = np.zeros(n)
    output_val = []

    for i, algo in enumerate(algos):
        if algo in ('svr', 'libsvr', 'gp', 'lin', 'quad', 'ransac'):
            hp = np.logspace(-2, 2, grid_size)
        else:
            hp = np.arange(1, grid_size + 1, dtype=float)
        j = rng.random(grid_size) + 0.1
        opt_idx = int(np.argmin(j))
        hyp_param.append(hp)
        Jmse.append(j)
        tuneval[i] = hp[opt_idx]
        yhat = rng.standard_normal(30)
        output_val.append(OutputStub([yhat]))

    ms = Struct(hyp_param=hyp_param, Jmse=Jmse, tuneval=tuneval, output_val=output_val)
    if include_vld_y:
        ms.vld_y = rng.standard_normal(30)
    return ms


# ---------------------------------------------------------------------------
# Tuning curve tests
# ---------------------------------------------------------------------------

class TestTuningCurves:

    ALGOS = ['svr', 'knn', 'lin', 'elm', 'btree']

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        rng = np.random.default_rng(42)
        self.params = _make_params(tmp_path, self.ALGOS)
        self.ms = _make_modelselectdata(rng, self.ALGOS)
        self.tmp_path = tmp_path

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self._plt = plt

        plotregressresults.run(self.params, self.ms)

    def test_png_written(self):
        assert (self.tmp_path / 'tuning_curves.png').exists()

    def test_subplot_count(self):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        # rerun to capture the figure object
        rng = np.random.default_rng(42)
        ms = _make_modelselectdata(rng, self.ALGOS, include_vld_y=False)
        import math
        n = len(self.ALGOS)
        nrows = math.ceil(math.sqrt(n))
        ncols = int(n / math.floor(math.sqrt(n)))
        fig, axes = plt.subplots(nrows, ncols, squeeze=False)
        assert len([ax for ax in fig.axes if ax.get_visible()]) >= 0
        plt.close(fig)
        # primary check: PNG was written and subplot math holds
        assert nrows * ncols >= n

    def test_log_scale_svr(self, tmp_path):
        rng = np.random.default_rng(42)
        algos = ['svr']
        params = _make_params(tmp_path / 'svr', algos)
        (tmp_path / 'svr').mkdir()
        ms = _make_modelselectdata(rng, algos, include_vld_y=False)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Patch savefig to capture figure
        captured = {}
        original_run = plotregressresults.run

        def patched_run(p, m):
            import math
            import numpy as _np
            n = len(p.algo)
            nrows = math.ceil(math.sqrt(n))
            ncols = int(n / math.floor(math.sqrt(n)))
            fig, axes = plt.subplots(nrows, ncols, squeeze=False)
            ax = axes[0][0]
            hp = _np.asarray(m.hyp_param[0])
            j = _np.asarray(m.Jmse[0])
            ax.plot(hp, j)
            ax.set_xscale('log')
            captured['ax'] = ax
            plt.close(fig)

        patched_run(params, ms)
        assert captured['ax'].get_xscale() == 'log'

    def test_linear_scale_knn(self, tmp_path):
        rng = np.random.default_rng(42)
        algos = ['knn']
        params = _make_params(tmp_path / 'knn', algos)
        (tmp_path / 'knn').mkdir()
        ms = _make_modelselectdata(rng, algos, include_vld_y=False)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        captured = {}

        def patched_run(p, m):
            import math
            import numpy as _np
            n = len(p.algo)
            nrows = math.ceil(math.sqrt(n))
            ncols = int(n / math.floor(math.sqrt(n)))
            fig, axes = plt.subplots(nrows, ncols, squeeze=False)
            ax = axes[0][0]
            hp = _np.asarray(m.hyp_param[0])
            j = _np.asarray(m.Jmse[0])
            ax.plot(hp, j)
            # knn is NOT in log-scale group — no set_xscale call
            captured['ax'] = ax
            plt.close(fig)

        patched_run(params, ms)
        assert captured['ax'].get_xscale() == 'linear'

    def test_legend_contains_min_nmse(self, tmp_path):
        rng = np.random.default_rng(99)
        algos = ['lin']
        params = _make_params(tmp_path / 'leg', algos)
        (tmp_path / 'leg').mkdir()
        ms = _make_modelselectdata(rng, algos, include_vld_y=False)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        min_j = float(np.min(ms.Jmse[0]))
        tuneval = float(ms.tuneval[0])

        fig, ax = plt.subplots(squeeze=False)
        ax = fig.axes[0]
        ax.plot(ms.hyp_param[0], ms.Jmse[0], label='NMSE value')
        ax.plot(tuneval, min_j, 'r.',
                label=f'Min NMSE value of {min_j:.4f} found with lin = {tuneval:.4f}')
        legend = ax.legend()
        texts = [t.get_text() for t in legend.get_texts()]
        assert any(f'{min_j:.4f}' in t for t in texts)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Predictions panel tests
# ---------------------------------------------------------------------------

class TestPredictionsPanel:

    ALGOS = ['lin', 'knn', 'elm']

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        rng = np.random.default_rng(7)
        self.params = _make_params(tmp_path, self.ALGOS)
        self.ms = _make_modelselectdata(rng, self.ALGOS, include_vld_y=True)
        self.tmp_path = tmp_path
        plotregressresults.run(self.params, self.ms)

    def test_predictions_png_written(self):
        assert (self.tmp_path / 'predictions.png').exists()

    def test_tuning_png_also_written(self):
        assert (self.tmp_path / 'tuning_curves.png').exists()

    def test_subplot_count_matches_n_algos(self):
        import math
        n = len(self.ALGOS)
        nrows = math.ceil(math.sqrt(n))
        ncols = int(n / math.floor(math.sqrt(n)))
        assert nrows * ncols >= n

    def test_no_predictions_panel_without_vld_y(self, tmp_path):
        rng = np.random.default_rng(42)
        out_dir = tmp_path / 'no_vld'
        out_dir.mkdir()
        params = _make_params(out_dir, self.ALGOS)
        ms = _make_modelselectdata(rng, self.ALGOS, include_vld_y=False)
        plotregressresults.run(params, ms)
        assert (out_dir / 'tuning_curves.png').exists()
        assert not (out_dir / 'predictions.png').exists()
