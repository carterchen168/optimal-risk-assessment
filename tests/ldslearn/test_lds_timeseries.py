"""
tests/ldslearn/test_lds_timeseries.py
---------------------------------------
Pytest suite for ldslearn/lds_timeseries.py.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('user_input_ressarch', MagicMock())

_ldslearn_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ldslearn")
# lds_timeseries.py imports its siblings (guess, ldsparamsidx, learn_kalman) as
# bare modules, so ldslearn/ must be importable on sys.path before exec_module runs.
sys.path.insert(0, _ldslearn_dir)

_mod_path = os.path.join(_ldslearn_dir, "lds_timeseries.py")
_spec = importlib.util.spec_from_file_location("ldslearn.lds_timeseries", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

lds_timeseries = _mod.lds_timeseries
learn_lds = _mod.learn_lds
