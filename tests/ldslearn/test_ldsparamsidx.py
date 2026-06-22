"""
tests/ldslearn/test_ldsparamsidx.py
-------------------------------------
Pytest suite for ldslearn/ldsparamsidx.py.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('user_input_ressarch', MagicMock())

_mod_path = os.path.join(os.path.dirname(__file__), "..", "..", "ldslearn", "ldsparamsidx.py")
_spec = importlib.util.spec_from_file_location("ldslearn.ldsparamsidx", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

lds_params_idx = _mod.lds_params_idx
