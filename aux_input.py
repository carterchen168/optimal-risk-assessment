import tkinter as tk
from tkinter import simpledialog, messagebox
import user_input_ressarch as user_input
from utils import list_dialog


class Params:
    pass


def run():

    params = user_input.params

    root = tk.Tk()
    root.withdraw()

    # Regression algorithm selection
    # feel free to add more algorithms!
    algotypes = ['gp', 'svr', 'libsvr', 'knn', 'btree', 'lin', 'quad', 'bnet', 'elm', 'ransac']
    tuneparamtypes = [
        'sample kernel width',
        'kernel width',
        'cost parameter',
        'number of nearest neighbors',
        'minimal number of observations per tree leaf',
        'regularization coefficient',
        'regularization coefficient',
        'number of hidden units',
        'number of hidden neurons',
        'threshold'
    ]

    algo_selection = list_dialog(
        algotypes,
        'Pick regression methods',
        multiple=True
    )
    # algo_selection is a list of 0-based indices
    algoIdx = [i + 1 for i in algo_selection]  # 1-based to match MATLAB indexing logic
    params.algo = [algotypes[i] for i in algo_selection]

    # Check for Neural Network Toolbox if bnet was selected
    if 'bnet' in params.algo:
        if 'Neural Network Toolbox' in getattr(params, 'toolboxname', []):
            params.maxtime = simpledialog.askfloat(
                "Training Time",
                "Enter the longest training time for a single bagged neural network regression run in seconds:",
                parent=root,
                minvalue=0.0
            )
            params.gui = False
            params.numBaseModels = 1
        else:
            # Remove bnet from selections
            bnet_idx_in_selection = [i for i, a in enumerate(params.algo) if a == 'bnet']
            for idx in sorted(bnet_idx_in_selection, reverse=True):
                del params.algo[idx]
                del algoIdx[idx]

            if not params.algo:
                raise RuntimeError('Neural Network Toolbox not detected, and no other regression methods detected !')
            else:
                print('Neural Network Toolbox not detected, defaulting to the other regression techniques that you selected')

    # Workers for distributed regression optimization
    if getattr(params.regress, 'flag', 0) == 1 and params.distrib == 1:
        params.regress.Nwork = simpledialog.askinteger(
            "Workers",
            "How many workers for regression optimization (limit 128)?",
            parent=root,
            minvalue=1, maxvalue=128
        )

    # SVR-specific parameters
    if 'svr' in params.algo:
        params.gp_pnt = 500
        params.betac = 1    # weight of continuous kernel SVR
        params.betad = 0    # weight of continuous kernel SVR
        params.kernelType = 'rbf'  # Type of kernel: RBF
        params.regress.Ngrid = simpledialog.askinteger(
            "SVR Grid",
            "How many points for additional preliminary SVR regression optimization?",
            parent=root,
            minvalue=1
        )

    # KNN-specific parameters
    if 'knn' in params.algo:
        params.verbose = False
        params.perfectHit = 0  # flag to ignore perfect matches between training and test data

    # GP-specific parameters
    if 'gp' in params.algo:
        params.gp_pnt = 500
        params.nSteps = 10

    # ELM-specific parameters
    if 'elm' in params.algo:
        params.orth_flag_ELM = 0       # 1 if random input parameter matrix is orthogonal, 0 if not
        params.randomSeedELM = 0       # any valid seed state value (default 0)
        params.ActivationFunction = 'sig'  # can choose between 'sig' and 'rbf'

    # libSVR-specific parameters
    if 'libsvr' in params.algo:
        # nu-SVR option, gaussian kernel, nu, cost parameter
        params.lsvm_options = '-s 4 -t 2 -n 0.400000 -q -c 0.100000'

    # RANSAC-specific parameters
    if 'ransac' in params.algo:
        params.epsilon = 1e-6
        params.P_inlier = 0.99
        params.sigma = 1
        params.est_fun = 'estimate_line'   # function reference stored as string
        params.man_fun = 'error_line'      # function reference stored as string
        params.mode = 'RANSAC'
        params.Ps = []
        params.notify_iters = []
        params.min_iters = 100
        params.fix_seed = False
        params.reestimate = True
        params.stabilize = False
        # avg_thresh would normally come from calling man_fun; set placeholder
        # In practice, the Python equivalent of feval(man_fun,...) should be called here
        try:
            from ransac_utils import error_line
            _, params.avg_thresh, _ = error_line(None, None, params.sigma, params.P_inlier)
        except ImportError:
            params.avg_thresh = 0.0
            print("Warning: ransac_utils not found, avg_thresh set to 0.0")
    else:
        params.avg_thresh = []

    # Detection configuration (only if not regression-only mode)
    if not getattr(params, 'regressonly', False):
        detectiontypes = [
            'Redline - Training',
            'Redline - Validation',
            'Predictive - Training',
            'Predictive - Validation',
            'Optimal - Training',
            'Optimal - Validation',
            'SPRT'
        ]
        detecttuneparamtypes = ['Mpos', 'Mneg', 'Vnom', 'Vinv', 'dstepmin', 'p-value', 'nmin']

        if 'Control System Toolbox' in getattr(params, 'toolboxname', []):
            detect_selection = list_dialog(
                detectiontypes,
                'Pick detection methods',
                multiple=True
            )
            detectionIdx = [i + 1 for i in detect_selection]  # 1-based
        else:
            detectionIdx = [1]
            print('Control System Toolbox not detected, ACCEPT will use only standard exceedances for detection')

        if not detectionIdx:
            raise RuntimeError('No valid detection methods can be run for your configuration....')

        params.detect = [detectiontypes[i - 1] for i in detectionIdx]
        params.tunetypes = [tuneparamtypes[i - 1] for i in algoIdx]

        # Determine which are training vs validation methods
        train_set = {detectiontypes[0], detectiontypes[2], detectiontypes[4]}  # indices 1, 3, 5
        val_set = {detectiontypes[1], detectiontypes[3], detectiontypes[5]}    # indices 2, 4, 6

        trainstring = [d in train_set for d in params.detect]
        valstring = [d in val_set for d in params.detect]

        # Design alarm system constraints
        if any(idx < 7 for idx in detectionIdx):
            params.consttype = simpledialog.askinteger(
                "Alarm Constraint",
                "Design Alarm System by constraint on:\n1 - False Alarm Rate\n2 - Missed Detection Rate\n3 - Equal Tradeoff",
                parent=root,
                minvalue=1, maxvalue=3
            )
            if params.consttype == 1:
                params.maxfprate = simpledialog.askfloat(
                    "False Alarm Rate",
                    "Enter max allowable false alarm rate:",
                    parent=root,
                    minvalue=0.0
                )
            elif params.consttype == 2:
                params.maxpmd = simpledialog.askfloat(
                    "Missed Detection Rate",
                    "Enter max allowable missed detection rate:",
                    parent=root,
                    minvalue=0.0
                )

        # ASOS approximation
        # valstring indices 1,3:end in MATLAB correspond to indices 0, 2+ in 0-based
        val_subset = [valstring[0]] + valstring[2:] if len(valstring) > 2 else [valstring[0]] if valstring else []
        if any(trainstring) or any(val_subset) or ('SPRT' in params.detect):
            asos = simpledialog.askinteger(
                "ASOS",
                "Use ASOS approximation for LDS learning?\n1 - Yes, 2 - No",
                parent=root,
                minvalue=1, maxvalue=2
            )
            params.asos = (asos == 1)

            if not any(idx in (2, 7) for idx in detectionIdx):
                if params.asos:
                    min_filelength = min(getattr(params, 'filelength', [999]))
                    params.klim = simpledialog.askinteger(
                        "klim",
                        f"Enter in a value for klim (< {min_filelength}):",
                        parent=root,
                        minvalue=1
                    )

            params.nmin = 2
            params.nmax = simpledialog.askinteger(
                "State Order",
                "Maximum state order =",
                parent=root,
                minvalue=1
            )
            params.inittype = 2

        # SPRT parameters
        if any(idx > 6 for idx in detectionIdx):
            Mpos = 0.01
            Mneg = 0.01
            Vnom = 1.5
            Vinv = 1.5
            params.sprt = [Mpos, Mneg, Vnom, Vinv]
            params.maxfprate = simpledialog.askfloat(
                "SPRT False Positive",
                "Enter in max allowable false positive rate for SPRT alarm system design:",
                parent=root,
                minvalue=0.0
            )
            params.maxpmd = simpledialog.askfloat(
                "SPRT Missed Detection",
                "Enter in max allowable missed detection rate for SPRT alarm system design:",
                parent=root,
                minvalue=0.0
            )

        # Prediction horizon
        params.dstepmin = 1
        params.dstepmax = simpledialog.askinteger(
            "Prediction Horizon",
            "What is the maximum design (and validation) prediction horizon?",
            parent=root,
            minvalue=1
        )

        # Monte Carlo integration settings for training methods
        if any(trainstring):
            params.N = simpledialog.askinteger(
                "Monte Carlo",
                "Enter resolution (number of points) for Monte Carlo-based integration (smoothness factor):",
                parent=root,
                minvalue=1
            )
            params.tol = simpledialog.askfloat(
                "ROC Resolution",
                "Resolution of ROC curve (bits):",
                parent=root,
                minvalue=0.0
            )

        # Optimal detection approximation type
        if 5 in detectionIdx:
            params.flag = simpledialog.askinteger(
                "Approximation",
                "1 - Closed form approximation\n2 - Root-Finding Approximation",
                parent=root,
                minvalue=1, maxvalue=2
            )

        # Workers for distributed detection
        if any(idx > 1 for idx in detectionIdx) and params.distrib == 1:
            params.detection.Nwork = simpledialog.askinteger(
                "Detection Workers",
                "How many workers for detection optimization (limit 128)?",
                parent=root,
                minvalue=1, maxvalue=128
            )

    # Store algoIdx and tuneparamtypes on params for use by reg_ranges
    params._algoIdx = algoIdx
    params._tuneparamtypes = tuneparamtypes
    params._detectionIdx = detectionIdx if not getattr(params, 'regressonly', False) else []

    try:
        root.destroy()
    except Exception:
        pass