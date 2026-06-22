"""
tests/ldslearn/test_inverse_covariance_selection.py
-----------------------------------------------------
Pytest suite for ldslearn/inverse_covariance_selection.py.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('user_input_ressarch', MagicMock())

_mod_path = os.path.join(os.path.dirname(__file__), "..", "..", "ldslearn", "inverse_covariance_selection.py")
_spec = importlib.util.spec_from_file_location("ldslearn.inverse_covariance_selection", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

inverse_covariance_selection = _mod.inverse_covariance_selection
