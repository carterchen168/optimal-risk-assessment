import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# 1. Mock user_input_ressarch module dependencies to isolate testing
mock_user_input = MagicMock()
sys.modules['user_input_ressarch'] = mock_user_input

# Import the target module after setting up the system module mock
import regressopt.reg_ranges as reg_ranges


@pytest.fixture(autouse=True)
def setup_mock_params():
    """Resets the mock parameters object before every test run."""
    mock_user_input.params = MagicMock()
    # Default common attributes inside the framework
    mock_user_input.params._algoIdx = [1, 2, 4]  # Example indices (1-based: gp, svr, knn)
    mock_user_input.params._tuneparamtypes = ['width', 'cost', 'cost', 'k', 'nodes', 'lambda']
    mock_user_input.params._detectionIdx = [1]
    mock_user_input.params.algo = ['gp', 'svr', 'knn']
    mock_user_input.params.filelength = [1000]
    mock_user_input.params.avg_thresh = 0.0
    mock_user_input.params.regress = MagicMock()
    mock_user_input.params.regress.flag = 0  # Default to no manual intervention dialog
    yield


# ---------------------------------------------------------------------------
# 1. Parameter Mapping & Core Array Contracts
# ---------------------------------------------------------------------------

class TestHyperparameterArrayContracts:

    def test_tunetype_mapping(self):
        """Validates that 1-based algoIdx maps precisely to correct 0-based tuneparamtypes."""
        mock_user_input.params._algoIdx = [1, 4]  # 'gp' and 'knn'
        mock_user_input.params._tuneparamtypes = ['width_param', 'type2', 'type3', 'k_neighbors']
        
        reg_ranges.run()
        
        assert mock_user_input.params.tunetype == ['width_param', 'k_neighbors']

    def test_knn_max_dynamically_bound_to_filelength(self):
        """Ensures k-NN hyperparameter max cap is exactly the total sum of file lengths."""
        mock_user_input.params._algoIdx = [4]  # knn index
        mock_user_input.params.algo = ['knn']
        mock_user_input.params.filelength = [500, 250, 250]  # Total length = 1000 
        
        reg_ranges.run()
        
        # Check params.tune matrix layout: [min, max, value]
        # knn is 4th index (0-based index 3)
        assert mock_user_input.params.tune[0, 1] == 1000  # max bound must equal 1000 

    def test_fallback_default_filelength(self):
        """Verifies fallback data length assigns [1000] if not explicitly evaluated."""
        if hasattr(mock_user_input.params, 'filelength'):
            delattr(mock_user_input.params, 'filelength')
        delattr(mock_user_input.params, 'nompath')
        delattr(mock_user_input.params, 'nomflightpath')
        delattr(mock_user_input.params, 'fcnpath')

        reg_ranges.run()

        assert mock_user_input.params.filelength == [1000]

    def test_tune_shape_is_N_by_3_for_N_algos(self):
        """tune matrix must be (N, 3) where N equals the number of selected algorithms."""
        mock_user_input.params._algoIdx = [1, 2, 3]
        mock_user_input.params._tuneparamtypes = ['w', 'c', 'c', 'k', 'n', 'l', 'q', 'b', 'e', 'r']
        mock_user_input.params.algo = ['gp', 'svr', 'libsvr']
        mock_user_input.params.avg_thresh = 0.0

        reg_ranges.run()

        assert mock_user_input.params.tune.ndim == 2
        assert mock_user_input.params.tune.shape == (3, 3)

    def test_kernel_algo_min_max_ranges(self):
        """Kernel algorithms (gp, svr, libsvr) use kernel_min=1e-5 and kernel_max=1e5."""
        mock_user_input.params._algoIdx = [1, 2, 3]
        mock_user_input.params._tuneparamtypes = ['w', 'c', 'c', 'k', 'n', 'l', 'q', 'b', 'e', 'r']
        mock_user_input.params.algo = ['gp', 'svr', 'libsvr']
        mock_user_input.params.avg_thresh = 0.0

        reg_ranges.run()

        assert np.allclose(mock_user_input.params.tune[:, 0], 1e-5)
        assert np.allclose(mock_user_input.params.tune[:, 1], 1e5)

    def test_linear_algo_min_max_ranges(self):
        """lin uses lin_min=1e-10, quad uses 1e-13; both max at 1.0; tunevals are hardcoded 0."""
        mock_user_input.params._algoIdx = [6, 7]
        mock_user_input.params._tuneparamtypes = ['w', 'c', 'c', 'k', 'n', 'l', 'q', 'b', 'e', 'r']
        mock_user_input.params.algo = ['lin', 'quad']
        mock_user_input.params.avg_thresh = 0.0

        reg_ranges.run()

        assert mock_user_input.params.tune[0, 0] == 1e-10
        assert mock_user_input.params.tune[1, 0] == 1e-13
        assert np.allclose(mock_user_input.params.tune[:, 1], 1.0)
        assert np.allclose(mock_user_input.params.tune[:, 2], 0.0)

    def test_integer_algo_min_max_ranges(self):
        """Integer algorithms (btree, bnet, elm) use min=1 and max=500."""
        mock_user_input.params._algoIdx = [5, 8, 9]
        mock_user_input.params._tuneparamtypes = ['w', 'c', 'c', 'k', 'n', 'l', 'q', 'b', 'e', 'r']
        mock_user_input.params.algo = ['btree', 'bnet', 'elm']
        mock_user_input.params.avg_thresh = 0.0

        reg_ranges.run()

        assert np.all(mock_user_input.params.tune[:, 0] == 1.0)
        assert np.all(mock_user_input.params.tune[:, 1] == 500.0)


# ---------------------------------------------------------------------------
# 2. Dynamic Threshold Constraints (RANSAC / ELM Baseline Sync)
# ---------------------------------------------------------------------------

class TestDynamicThresholdConstraints:

    def test_ransac_min_range_with_avg_thresh(self):
        """Verifies RANSAC minimum boundary accurately responds to avg_thresh parameters."""
        mock_user_input.params._algoIdx = [10]  # ransac index
        mock_user_input.params._tuneparamtypes = ['w', 'c', 'c', 'k', 'n', 'l', 'q', 'b', 'e', 'r']
        mock_user_input.params.algo = ['ransac']
        mock_user_input.params.avg_thresh = 50.0

        reg_ranges.run()

        # Expected formula: avg_thresh / (lin_max * 10) where lin_max = 1
        expected_min = 50.0 / (1 * 10)
        assert mock_user_input.params.tune[0, 0] == expected_min

    def test_ransac_min_range_without_avg_thresh(self):
        """RANSAC minimum bounds gracefully defaults to 0 when avg_thresh missing or None."""
        mock_user_input.params._algoIdx = [10]
        mock_user_input.params._tuneparamtypes = ['w', 'c', 'c', 'k', 'n', 'l', 'q', 'b', 'e', 'r']
        mock_user_input.params.algo = ['ransac']
        mock_user_input.params.avg_thresh = None

        reg_ranges.run()

        assert mock_user_input.params.tune[0, 0] == 0.0


# ---------------------------------------------------------------------------
# 3. Interactive UI Prompts (klim & Direct Flag Selection Dialogs)
# ---------------------------------------------------------------------------

class TestUserInterfaceDialogs:

    @patch('tkinter.simpledialog.askinteger')
    def test_klim_prompt_triggered_under_conditions(self, mock_askinteger):
        """klim configuration dialogue appears when asos framework flag matches requirements."""
        mock_user_input.params._detectionIdx = [1, 3]  # Neither 2 nor 7
        mock_user_input.params.asos = True
        if hasattr(mock_user_input.params, 'klim'):
            delattr(mock_user_input.params, 'klim')
        mock_user_input.params.filelength = [800, 400]
        mock_askinteger.return_value = 250

        reg_ranges.run()

        mock_askinteger.assert_called_once()
        # Ensure it passed minvalue=1 and contextual prompt max range validation string (< 400)
        assert "400" in mock_askinteger.call_args[0][1]
        assert mock_user_input.params.klim == 250

    @patch('tkinter.simpledialog.askstring')
    def test_manual_hyperparameter_override_flag_2(self, mock_askstring):
        """User interaction string prompt collects explicit manual hyperparameter selections if flag==2."""
        mock_user_input.params._algoIdx = [4, 6]  # knn (idx 4), lin (idx 6)
        mock_user_input.params.algo = ['knn', 'lin']
        mock_user_input.params.regress.flag = 2
        
        # User simulates forcing direct adjustments over randomly assigned baselines
        mock_askstring.return_value = "5, 0.02"

        reg_ranges.run()

        mock_askstring.assert_called_once()
        # Verify targeted numpy storage values sync directly with user choices
        assert mock_user_input.params.tune[0, 2] == 5.0    # Overwritten k-NN choice
        assert mock_user_input.params.tune[1, 2] == 0.02  # Overwritten linear ridge choice

    @patch('tkinter.simpledialog.askstring')
    def test_manual_hyperparameter_parse_error_fallback(self, mock_askstring, capsys):
        """Malformed user entries do not crash framework, defaults gracefully to randomized seeds."""
        mock_user_input.params._algoIdx = [4]
        mock_user_input.params.algo = ['knn']
        mock_user_input.params.regress.flag = 2
        mock_askstring.return_value = "malformed_string_input"

        reg_ranges.run()

        # Should output warning notice console logs safely
        captured = capsys.readouterr()
        assert "Warning: could not parse hyperparameter values" in captured.out

    @patch('tkinter.simpledialog.askinteger')
    def test_klim_suppressed_when_detection_idx_contains_2(self, mock_askinteger):
        """klim prompt is skipped when detectionIdx contains index 2."""
        mock_user_input.params._detectionIdx = [2, 4]
        mock_user_input.params.asos = True
        if hasattr(mock_user_input.params, 'klim'):
            delattr(mock_user_input.params, 'klim')

        reg_ranges.run()

        mock_askinteger.assert_not_called()

    @patch('tkinter.simpledialog.askinteger')
    def test_klim_suppressed_when_detection_idx_contains_7(self, mock_askinteger):
        """klim prompt is skipped when detectionIdx contains index 7."""
        mock_user_input.params._detectionIdx = [7]
        mock_user_input.params.asos = True
        if hasattr(mock_user_input.params, 'klim'):
            delattr(mock_user_input.params, 'klim')

        reg_ranges.run()

        mock_askinteger.assert_not_called()

    @patch('tkinter.simpledialog.askinteger')
    def test_klim_suppressed_when_asos_is_false(self, mock_askinteger):
        """klim prompt is skipped when asos is False regardless of detectionIdx."""
        mock_user_input.params._detectionIdx = [1, 3]
        mock_user_input.params.asos = False
        if hasattr(mock_user_input.params, 'klim'):
            delattr(mock_user_input.params, 'klim')

        reg_ranges.run()

        mock_askinteger.assert_not_called()

    @patch('tkinter.simpledialog.askinteger')
    def test_klim_suppressed_when_klim_already_set(self, mock_askinteger):
        """klim prompt is skipped and existing value preserved when klim already configured."""
        mock_user_input.params._detectionIdx = [1, 3]
        mock_user_input.params.asos = True
        mock_user_input.params.klim = 200

        reg_ranges.run()

        mock_askinteger.assert_not_called()
        assert mock_user_input.params.klim == 200

    @patch('tkinter.simpledialog.askstring', return_value=None)
    @patch('numpy.random.default_rng')
    def test_manual_hyperparameter_cancel_preserves_random_defaults(self, mock_rng, mock_askstring):
        """Cancelling the hyperparameter dialog (None return) preserves randomly generated defaults."""
        mock_rng.return_value = np.random.Generator(np.random.PCG64(42))
        mock_user_input.params._algoIdx = [1]
        mock_user_input.params._tuneparamtypes = ['w', 'c', 'c', 'k', 'n', 'l', 'q', 'b', 'e', 'r']
        mock_user_input.params.algo = ['gp']
        mock_user_input.params.regress.flag = 2
        mock_user_input.params.avg_thresh = 0.0

        reg_ranges.run()

        mock_askstring.assert_called_once()
        expected = 1e-5 + np.random.Generator(np.random.PCG64(42)).random() * (1e5 - 1e-5)
        assert np.isclose(mock_user_input.params.tune[0, 2], expected)


# ---------------------------------------------------------------------------
# 4. Environment & Directory Restoration Guardrails
# ---------------------------------------------------------------------------

class TestEnvironmentGuardrails:

    @patch.dict(os.environ, {"ACCEPT_DIR": "/fake/mock/accept/path"})
    @patch('os.path.isdir', side_effect=lambda p: p == '/fake/mock/accept/path')
    @patch('os.chdir')
    def test_restores_working_directory_to_accept_root(self, mock_chdir, _mock_isdir):
        """Ensures framework preserves consistency by switching execution back to ACCEPT path environment."""
        reg_ranges.run()
        mock_chdir.assert_called_with("/fake/mock/accept/path")


# ---------------------------------------------------------------------------
# 5. Tuning Value Bounds Validation
# ---------------------------------------------------------------------------

class TestTuneValsBoundsValidation:

    @patch('numpy.random.default_rng')
    def test_kernel_algo_tunevals_within_bounds(self, mock_rng):
        """Randomly initialised kernel hyperparameters fall within [kernel_min=1e-5, kernel_max=1e5]."""
        mock_rng.return_value = np.random.Generator(np.random.PCG64(42))
        mock_user_input.params._algoIdx = [1, 2, 3]
        mock_user_input.params._tuneparamtypes = ['w', 'c', 'c', 'k', 'n', 'l', 'q', 'b', 'e', 'r']
        mock_user_input.params.algo = ['gp', 'svr', 'libsvr']
        mock_user_input.params.avg_thresh = 0.0

        reg_ranges.run()

        for val in mock_user_input.params.tune[:, 2]:
            assert 1e-5 <= val <= 1e5

    @patch('numpy.random.default_rng')
    def test_integer_algo_tunevals_within_bounds(self, mock_rng):
        """Randomly initialised integer hyperparameters fall within [1, integer_max=500]."""
        mock_rng.return_value = np.random.Generator(np.random.PCG64(42))
        mock_user_input.params._algoIdx = [5, 8, 9]
        mock_user_input.params._tuneparamtypes = ['w', 'c', 'c', 'k', 'n', 'l', 'q', 'b', 'e', 'r']
        mock_user_input.params.algo = ['btree', 'bnet', 'elm']
        mock_user_input.params.avg_thresh = 0.0

        reg_ranges.run()

        for val in mock_user_input.params.tune[:, 2]:
            assert 1 <= val <= 500


# ---------------------------------------------------------------------------
# 6. Nominal Path File Scanning Branch
# ---------------------------------------------------------------------------

class TestNompathFileScanning:

    def _base_params(self):
        mock_user_input.params.nompath = '/fake/nompath'
        if hasattr(mock_user_input.params, 'filelength'):
            delattr(mock_user_input.params, 'filelength')
        mock_user_input.params.anomalytype = ['typeA']
        mock_user_input.params.anomtype = 'typeA'
        mock_user_input.params.loadfcn = ['myloadmod']
        mock_user_input.params.avg_thresh = 0.0

    @patch('tkinter.Tk')
    @patch('os.path.isfile', return_value=True)
    @patch('os.listdir', return_value=['flight1.csv'])
    @patch('os.path.isdir', side_effect=lambda p: p == '/fake/nompath')
    def test_nompath_scan_filelength_from_shape_attribute(
            self, _mock_isdir, _mock_listdir, _mock_isfile, _mock_tk):
        """Successful load with a shaped array uses sampflight.shape[0] as the file length."""
        self._base_params()
        fake_loadmod = MagicMock()
        fake_sf = MagicMock()
        fake_sf.shape = (250, 5)
        fake_loadmod.run.return_value = (MagicMock(), fake_sf)
        sys.modules['myloadmod'] = fake_loadmod
        try:
            reg_ranges.run()
        finally:
            del sys.modules['myloadmod']
        assert mock_user_input.params.filelength == [250]

    @patch('tkinter.Tk')
    @patch('os.path.isfile', return_value=True)
    @patch('os.listdir', return_value=['flight1.csv'])
    @patch('os.path.isdir', side_effect=lambda p: p == '/fake/nompath')
    def test_nompath_scan_filelength_from_len_fallback(
            self, _mock_isdir, _mock_listdir, _mock_isfile, _mock_tk):
        """Successful load with a plain list uses len() when sampflight has no shape attribute."""
        self._base_params()
        fake_loadmod = MagicMock()
        fake_loadmod.run.return_value = (MagicMock(), list(range(80)))
        sys.modules['myloadmod'] = fake_loadmod
        try:
            reg_ranges.run()
        finally:
            del sys.modules['myloadmod']
        assert mock_user_input.params.filelength == [80]

    @patch('tkinter.Tk')
    @patch('os.path.isfile', return_value=True)
    @patch('os.listdir', return_value=['flight1.csv'])
    @patch('os.path.isdir', side_effect=lambda p: p == '/fake/nompath')
    def test_nompath_scan_import_error_prints_warning_and_uses_fallback(
            self, _mock_isdir, _mock_listdir, _mock_isfile, _mock_tk, capsys):
        """A failed file load prints a warning and falls back to filelength=[1000]."""
        self._base_params()
        fake_loadmod = MagicMock()
        fake_loadmod.run.side_effect = ImportError('no such module')
        sys.modules['myloadmod'] = fake_loadmod
        try:
            reg_ranges.run()
        finally:
            del sys.modules['myloadmod']
        captured = capsys.readouterr()
        assert 'Warning: could not load' in captured.out
        assert mock_user_input.params.filelength == [1000]