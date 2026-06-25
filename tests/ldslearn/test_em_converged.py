"""
tests/ldslearn/test_em_converged.py
------------------------------------
Pytest suite for ldslearn/em_converged.py.
"""

import importlib.util
import math
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


def test_converged_small_delta():
    converged, decrease = em_converged(loglik=-100.0, previous_loglik=-100.000001)
    assert converged is True
    assert decrease is False


def test_not_converged_large_delta():
    converged, decrease = em_converged(loglik=-50.0, previous_loglik=-100.0)
    assert converged is False
    assert decrease is False


def test_decrease_detected_when_check_increased():
    # Delta is tiny relative to magnitude (would read as "converged" under
    # the threshold check), but loglik decreased by more than 1e-3, so the
    # early-return decrease path must fire instead.
    converged, decrease = em_converged(
        loglik=-100.01, previous_loglik=-100.0, check_increased=True
    )
    assert converged is False
    assert decrease is True


def test_decrease_ignored_when_check_increased_false():
    converged, decrease = em_converged(
        loglik=-100.01, previous_loglik=-100.0, check_increased=False
    )
    assert decrease is False
    assert converged is True


def test_first_iteration_edge_case():
    converged, decrease = em_converged(loglik=-100.0, previous_loglik=-math.inf)
    assert converged is False
    assert decrease is False


def test_nonfinite_loglik_forces_converged():
    converged, decrease = em_converged(loglik=math.inf, previous_loglik=-100.0)
    assert converged is True

    converged_nan, _ = em_converged(loglik=math.nan, previous_loglik=-100.0)
    assert converged_nan is True


def test_verbose_prints_on_decrease(capsys):
    em_converged(
        loglik=-100.01, previous_loglik=-100.0, check_increased=True, verbose=True
    )
    captured = capsys.readouterr()
    assert "likelihood decreased" in captured.out
