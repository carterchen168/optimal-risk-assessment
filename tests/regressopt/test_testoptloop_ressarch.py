"""
tests/test_testoptloop_ressarch.py
------------------------------------
Pytest suite for helper functions in testoptloop_ressarch.py.

Covers:
  - mergeParams: field copying, snew-wins precedence, empty snew, identity return
  - modelsearch: tune_val → ndarray wrapping, 3-tuple return (globalhistory dropped),
    x[0] scalar extraction, i-forwarding to optimsearch
  - make_datafiles: empty-path no-crash, mode=2 (tr+vld), mode=3 (tr+test),
    Struct field contracts, train_cell batch wrapping, Statistics fields,
    feature/target column selection
"""

import sys
import importlib.util
import os
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Module isolation
# ---------------------------------------------------------------------------

for _name in ['user_input_ressarch', 'regressopt', 'detectopt',
              'ldslearn', 'ldslearn.lds_timeseries']:
    sys.modules.setdefault(_name, MagicMock())

_path = os.path.join(os.path.dirname(__file__), "..", "..", "testoptloop_ressarch.py")
_spec = importlib.util.spec_from_file_location("testoptloop_ressarch", _path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

mergeParams    = _mod.mergeParams
modelsearch    = _mod.modelsearch
make_datafiles = _mod.make_datafiles
Struct         = _mod.Struct


# ---------------------------------------------------------------------------
# 1. mergeParams
# ---------------------------------------------------------------------------

class TestMergeParams:
    def test_copies_all_fields_from_snew(self):
        s    = Struct(a=1)
        snew = Struct(b=2, c=3)
        mergeParams(s, snew)
        assert s.b == 2
        assert s.c == 3

    def test_overlapping_field_snew_wins(self):
        s    = Struct(x=10)
        snew = Struct(x=99)
        mergeParams(s, snew)
        assert s.x == 99

    def test_empty_snew_leaves_s_unchanged(self):
        s    = Struct(a=1)
        snew = Struct()
        mergeParams(s, snew)
        assert vars(s) == {'a': 1}

    def test_returns_same_object_identity(self):
        s    = Struct(a=1)
        snew = Struct(b=2)
        result = mergeParams(s, snew)
        assert result is s

    def test_original_fields_preserved_after_merge(self):
        s    = Struct(keep=True, x=0)
        snew = Struct(x=1)
        mergeParams(s, snew)
        assert s.keep is True


# ---------------------------------------------------------------------------
# 2. modelsearch
# ---------------------------------------------------------------------------

_FAKE_x    = np.array([2.5])
_FAKE_fval = 0.42
_FAKE_lhist = {'x': [np.array([2.5])], 'fval': [0.42]}
_FAKE_ghist = {'exitflag': 1, 'output': {}}
_FAKE_RESULT = (_FAKE_x, _FAKE_fval, _FAKE_lhist, _FAKE_ghist)


@pytest.fixture(autouse=False)
def patch_optimsearch():
    original = _mod.optimsearch
    _mod.optimsearch = MagicMock(return_value=_FAKE_RESULT)
    yield _mod.optimsearch
    _mod.optimsearch = original


def _ms_args():
    rng = np.random.default_rng(42)
    params   = Struct(regress=Struct(optIdx=6), distrib=2, algo=['lin'], filelength=[5])
    tr       = Struct(x=rng.standard_normal((5, 2)), y=rng.standard_normal(5))
    trtest   = Struct(x=[tr.x], y=[tr.y])
    return params, tr, trtest


class TestModelsearch:
    def test_tune_val_wrapped_to_1d_ndarray(self, patch_optimsearch):
        params, tr, trtest = _ms_args()
        modelsearch(3.7, params, tr, trtest, 0)
        x0 = patch_optimsearch.call_args.args[0]
        assert isinstance(x0, np.ndarray)
        assert x0.shape == (1,)
        assert x0[0] == pytest.approx(3.7)

    def test_returns_three_tuple_not_four(self, patch_optimsearch):
        params, tr, trtest = _ms_args()
        result = modelsearch(1.0, params, tr, trtest, 0)
        assert len(result) == 3

    def test_x_returned_as_scalar_not_array(self, patch_optimsearch):
        params, tr, trtest = _ms_args()
        x_val, _, _ = modelsearch(1.0, params, tr, trtest, 0)
        assert np.ndim(x_val) == 0

    def test_fval_and_localhistory_forwarded(self, patch_optimsearch):
        params, tr, trtest = _ms_args()
        _, fval, lhist = modelsearch(1.0, params, tr, trtest, 0)
        assert fval == pytest.approx(_FAKE_fval)
        assert lhist is _FAKE_lhist

    def test_globalhistory_dropped(self, patch_optimsearch):
        params, tr, trtest = _ms_args()
        result = modelsearch(1.0, params, tr, trtest, 0)
        # 3-tuple means globalhistory not present
        assert len(result) == 3

    def test_i_forwarded_verbatim(self, patch_optimsearch):
        params, tr, trtest = _ms_args()
        modelsearch(1.0, params, tr, trtest, 7)
        call_i = patch_optimsearch.call_args.args[4]
        assert call_i == 7


# ---------------------------------------------------------------------------
# 3. make_datafiles — helpers
# ---------------------------------------------------------------------------

def _make_params_with_paths(n_features=3):
    header = [f'feat{k}' for k in range(n_features)] + ['target']
    return Struct(
        nompath='/fake/train',
        valpath='/fake/val',
        testpath='/fake/test',
        header=header,
        targetName='target',
        channelContineous=list(range(n_features)),
        filelength=[5],
    )


def _make_df(n_rows, n_features, seed):
    rng  = np.random.default_rng(seed)
    data = rng.standard_normal((n_rows, n_features + 1))
    cols = [f'feat{k}' for k in range(n_features)] + ['target']
    return pd.DataFrame(data, columns=cols)


def _glob_csv_only(pattern):
    return [pattern.replace('*.csv', 'data.csv')] if '*.csv' in pattern else []


def _run_make_datafiles(mode, n_tr=10, n_secondary=8, n_features=3):
    """Run make_datafiles with fully mocked file I/O."""
    p       = _make_params_with_paths(n_features)
    df_tr   = _make_df(n_tr, n_features, seed=0)
    df_sec  = _make_df(n_secondary, n_features, seed=1)
    calls   = [0]

    def fake_read_csv(path, *a, **kw):
        calls[0] += 1
        return df_tr if calls[0] == 1 else df_sec

    with patch('os.path.isdir', return_value=True), \
         patch('glob.glob', side_effect=_glob_csv_only), \
         patch('pandas.read_csv', side_effect=fake_read_csv):
        return make_datafiles(p, mode)


# ---------------------------------------------------------------------------
# 3a. Empty path — no crash, empty arrays
# ---------------------------------------------------------------------------

class TestMakeDatafilesEmpty:
    def test_no_nompath_attr_returns_empty_tr(self):
        p = Struct()
        tr, *_ = make_datafiles(p, 2)
        assert tr.x.size == 0
        assert tr.y.size == 0

    def test_no_crash_on_empty_data_mode2(self):
        p = Struct()
        result = make_datafiles(p, 2)
        assert len(result) == 9

    def test_no_crash_on_empty_data_mode3(self):
        p = Struct()
        result = make_datafiles(p, 3)
        assert len(result) == 9


# ---------------------------------------------------------------------------
# 3b. mode=2 (training + validation)
# ---------------------------------------------------------------------------

class TestMakeDatafilesMode2:
    def test_returns_9_tuple(self):
        assert len(_run_make_datafiles(2)) == 9

    def test_tr_shape(self):
        result = _run_make_datafiles(2, n_tr=10, n_features=3)
        tr = result[0]
        assert tr.x.shape == (10, 3)
        assert tr.y.shape == (10,)

    def test_vld_shape(self):
        result = _run_make_datafiles(2, n_tr=10, n_secondary=8, n_features=3)
        vld = result[1]
        assert vld.x.shape == (8, 3)
        assert vld.y.shape == (8,)

    def test_train_cell_batch_wrapping(self):
        result = _run_make_datafiles(2, n_tr=5, n_features=3)
        train_cell = result[2]
        assert len(train_cell.x) == 5
        assert len(train_cell.y) == 5
        for row in train_cell.x:
            assert row.shape == (1, 3)
        for elem in train_cell.y:
            assert elem.shape == (1,)

    def test_statistics_has_required_fields(self):
        result = _run_make_datafiles(2, n_tr=10, n_features=3)
        stats = result[6]
        for field in ('mean', 'std', 'min', 'max', 'samples'):
            assert hasattr(stats, field), f"Statistics missing field: {field}"

    def test_statistics_samples_matches_tr_length(self):
        result = _run_make_datafiles(2, n_tr=10, n_features=3)
        tr, stats = result[0], result[6]
        assert stats.samples == len(tr.y)

    def test_rawdata_tr_has_all_columns(self):
        # rawdata_tr (index 4) = full matrix incl. target; tr.x = features only
        result = _run_make_datafiles(2, n_tr=10, n_features=3)
        rawdata_tr = result[4]
        tr         = result[0]
        assert rawdata_tr.shape[1] == tr.x.shape[1] + 1  # features + target

    def test_rawdata_val_has_all_columns(self):
        result = _run_make_datafiles(2, n_tr=10, n_secondary=8, n_features=3)
        rawdata_val = result[3]
        vld         = result[1]
        assert rawdata_val.shape[1] == vld.x.shape[1] + 1


# ---------------------------------------------------------------------------
# 3c. mode=3 (training + test)
# ---------------------------------------------------------------------------

class TestMakeDatafilesMode3:
    def test_returns_9_tuple(self):
        assert len(_run_make_datafiles(3)) == 9

    def test_test_split_shape(self):
        result = _run_make_datafiles(3, n_tr=10, n_secondary=6, n_features=3)
        test_struct = result[1]
        assert test_struct.x.shape == (6, 3)
        assert test_struct.y.shape == (6,)

    def test_train_cell_batch_wrapping_mode3(self):
        result = _run_make_datafiles(3, n_tr=4, n_features=2)
        train_cell = result[2]
        assert len(train_cell.x) == 4
        for row in train_cell.x:
            assert row.shape == (1, 2)

    def test_rawdata_tst_has_all_columns(self):
        result = _run_make_datafiles(3, n_tr=10, n_secondary=6, n_features=3)
        rawdata_tst = result[3]
        test_struct = result[1]
        assert rawdata_tst.shape[1] == test_struct.x.shape[1] + 1
