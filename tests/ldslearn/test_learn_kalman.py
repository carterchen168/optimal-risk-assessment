"""
tests/ldslearn/test_learn_kalman.py
-------------------------------------
Pytest suite for ldslearn/learn_kalman.py.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

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
