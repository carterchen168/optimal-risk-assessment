"""
tests/ldslearn/test_em_converged.py
------------------------------------
Pytest suite for ldslearn/em_converged.py.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('user_input_ressarch', MagicMock())

_mod_path = os.path.join(os.path.dirname(__file__), "..", "..", "ldslearn", "em_converged.py")
_spec = importlib.util.spec_from_file_location("ldslearn.em_converged", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

em_converged = _mod.em_converged
