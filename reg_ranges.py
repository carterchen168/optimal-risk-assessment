import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import user_input_ressarch as user_input
from utils import list_dialog


def run():

    params = user_input.params

    root = tk.Tk()
    root.withdraw()

    # Retrieve indices and tuneparam types set by aux_input
    algoIdx = getattr(params, '_algoIdx', [])
    tuneparamtypes = getattr(params, '_tuneparamtypes', [])
    detectionIdx = getattr(params, '_detectionIdx', [])

    # Map algoIdx to tuneparamtypes
    params.tunetype = [tuneparamtypes[i - 1] for i in algoIdx]

    # Gather file lengths from nominal flight data
    nompath = getattr(params, 'nompath', None)
    nomflightpath = getattr(params, 'nomflightpath', None)
    fcnpath = getattr(params, 'fcnpath', None)

    if nompath is not None and fcnpath is not None:
        data_dir = fcnpath
    elif nomflightpath is not None:
        data_dir = os.path.dirname(nomflightpath)
    else:
        data_dir = os.getcwd()

    # Scan for data files and compute file lengths
    filelength_list = []
    if nompath is not None and os.path.isdir(nompath):
        file_list = [f for f in os.listdir(nompath) if os.path.isfile(os.path.join(nompath, f))]
        for fname in file_list:
            filepath = os.path.join(nompath, fname)
            for j, atype in enumerate(getattr(params, 'anomalytype', [])):
                if getattr(params, 'anomtype', '') == atype:
                    # Call the appropriate load function
                    loadfcn_name = params.loadfcn[j] if hasattr(params, 'loadfcn') else None
                    if loadfcn_name is not None:
                        try:
                            loadmod = __import__(loadfcn_name)
                            header, sampflight = loadmod.run(filepath)
                            filelength_list.append(sampflight.shape[0] if hasattr(sampflight, 'shape') else len(sampflight))
                        except (ImportError, Exception) as e:
                            print(f"Warning: could not load {filepath} with {loadfcn_name}: {e}")

    if filelength_list:
        params.filelength = filelength_list
    elif not hasattr(params, 'filelength'):
        params.filelength = [1000]  # fallback default

    # klim prompt (if needed and not already set)
    if not any(idx in (2, 7) for idx in detectionIdx):
        if getattr(params, 'asos', False):
            if not hasattr(params, 'klim'):
                min_fl = min(params.filelength)
                params.klim = simpledialog.askinteger(
                    "klim",
                    f"Enter in a value for klim (< {min_fl}):",
                    parent=root,
                    minvalue=1
                )

    # Regression hyperparameter ranges
    # Algo order: 'gp' 'svr' 'libsvr' 'knn' 'btree' 'lin' 'quad' 'bnet' 'elm' 'ransac'
    integer_max = 500
    lin_min = 1e-10
    lin_max = 1
    kernel_min = 1e-5
    kernel_max = 1e5
    knn_max = sum(params.filelength)

    tuneminrange = np.array([
        kernel_min,   # gp
        kernel_min,   # svr
        kernel_min,   # libsvr
        1,            # knn
        1,            # btree
        lin_min,      # lin
        lin_min * 1e-3,  # quad
        1,            # bnet
        1,            # elm (placeholder, uses avg_thresh below)
        getattr(params, 'avg_thresh', 0) / (lin_max * 10) if getattr(params, 'avg_thresh', None) else 0  # ransac
    ])

    # Correct elm entry: MATLAB has ones(2,1) for bnet and elm, then avg_thresh/(lin_max*10)
    # Replicating exact MATLAB layout: [kernel_min*3; 1,1; lin_min,lin_min*1e-3; 1,1; avg_thresh/(lin_max*10)]
    avg_thresh = getattr(params, 'avg_thresh', 0) if getattr(params, 'avg_thresh', None) is not None else 0
    tuneminrange = np.array([
        kernel_min, kernel_min, kernel_min,           # gp, svr, libsvr
        1, 1,                                          # knn, btree
        lin_min, lin_min * 1e-3,                       # lin, quad
        1, 1,                                          # bnet, elm
        avg_thresh / (lin_max * 10) if avg_thresh else 0  # ransac
    ])

    tunemaxrange = np.array([
        kernel_max, kernel_max, kernel_max,           # gp, svr, libsvr
        knn_max, integer_max,                          # knn, btree
        lin_max, lin_max,                              # lin, quad
        integer_max, integer_max,                      # bnet, elm
        lin_max * 10                                   # ransac
    ])

    # Default tuning values (random initialization)
    rng = np.random.default_rng()
    rand_kernel = kernel_min + rng.random() * (kernel_max - kernel_min)
    rand_int = rng.integers(1, integer_max + 1)

    tunevals = np.array([
        rand_kernel, rand_kernel, rand_kernel,         # gp, svr, libsvr
        rng.integers(1, knn_max + 1), rng.integers(1, integer_max + 1),  # knn, btree
        0, 0,                                          # lin, quad
        rand_int, rand_int,                            # bnet, elm
        avg_thresh if avg_thresh else 0                # ransac
    ], dtype=float)

    # Build labels for hyperparameter dialog
    tuneparams_labels = []
    for i, idx in enumerate(algoIdx):
        algo_name = params.algo[i] if i < len(params.algo) else ''
        ttype = tuneparamtypes[idx - 1] if idx <= len(tuneparamtypes) else ''
        tuneparams_labels.append(f"{algo_name} - {ttype}")

    # Convert algoIdx to 0-based for numpy indexing
    algo_0based = [i - 1 for i in algoIdx]

    # If flag==2, ask user to provide hyperparameter values directly
    if getattr(params.regress, 'flag', 0) == 2:
        default_vals = tunevals[algo_0based]
        # Build a multi-field dialog
        val_strs = [str(v) for v in default_vals]
        prompt_lines = '\n'.join([f"{label}: (default {val})" for label, val in zip(tuneparams_labels, val_strs)])
        user_vals_str = simpledialog.askstring(
            "Regression Hyperparameter selection",
            f"Enter comma-separated values for:\n{prompt_lines}",
            initialvalue=', '.join(val_strs),
            parent=root
        )
        if user_vals_str:
            try:
                user_vals = [float(x.strip()) for x in user_vals_str.split(',')]
                tunevals[algo_0based] = user_vals
            except ValueError:
                print("Warning: could not parse hyperparameter values, using defaults")

    # Assemble final tune matrix: [min, max, value] for each selected algorithm
    params.tune = np.column_stack([
        tuneminrange[algo_0based],
        tunemaxrange[algo_0based],
        tunevals[algo_0based]
    ])

    # Change back to ACCEPT directory
    accept_dir = os.environ.get('ACCEPT_DIR', '')
    if accept_dir and os.path.isdir(accept_dir):
        os.chdir(accept_dir)

    try:
        root.destroy()
    except Exception:
        pass