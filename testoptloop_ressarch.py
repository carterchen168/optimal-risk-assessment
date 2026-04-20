import os
import time
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# External project modules — these must exist elsewhere in the codebase.
# They are imported at call time inside run() where possible so that
# import-order issues don't block loading this file.


class Params:
    """Lightweight attribute bag, mirrors the pattern used across the project."""
    pass


def _merge_params(target, source):
    """Merge all attributes of *source* into *target*, skipping 'fieldinit'."""
    for key in vars(source) if hasattr(source, '__dict__') else {}:
        if key == 'fieldinit':
            continue
        setattr(target, key, getattr(source, key))
    return target


# Distributed / parallel helper

def _run_parallel(func, arg_list, n_workers):
    """Execute *func* over *arg_list* using a process pool of *n_workers*.

    Returns a list of results aligned with *arg_list*.  Failed tasks
    produce ``None`` in the corresponding slot.
    """
    results = [None] * len(arg_list)
    with ProcessPoolExecutor(max_workers=min(n_workers, len(arg_list))) as pool:
        future_to_idx = {
            pool.submit(func, *args): idx
            for idx, args in enumerate(arg_list)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                print(f"Warning: parallel task {idx} raised {exc}")
                results[idx] = None
    return results


# Main entry point

def run(params):
    """Port of ``testoptloop.m``.

    Parameters
    ----------
    params : Params
        Configuration structure populated by the ACCEPT config pipeline.

    Returns
    -------
    modelselectdata : Params
        Collected regression / detection model-selection results.
    rocarea : list
        Per-algorithm detection ROC results.
    """
    # Late imports of project-specific modules
    from make_datafiles import make_datafiles
    from mainREGcode_ressarch import mainREGcode_ressarch
    from truthdata import truthdata
    from leveltune import leveltune
    from detectioncall import detectioncall

    # Try optional modules — they may not be needed for every config
    try:
        from modelopttest import modelopttest
    except ImportError:
        modelopttest = None
    try:
        from modelsearch import modelsearch
    except ImportError:
        modelsearch = None
    try:
        from lds_timeseries import lds_timeseries
    except ImportError:
        lds_timeseries = None

    detectiontypes = [
        'Redline - Training',
        'Redline - Validation',
        'Predictive - Training',
        'Predictive - Validation',
        'Optimal - Training',
        'Optimal - Validation',
        'SPRT',
    ]

    params.jobrecordID = []

    # Build data splits
    tr, vld, train_cell, rawdata_val, rawdata_tr, params, Statistics, features, fullfeatures = \
        make_datafiles(params, 2)

    modelselectdata = Params()
    modelselectdata.rawdata_tr = rawdata_tr
    modelselectdata.Statistics = Statistics

    train, test, train_cell, rawdata_tst, rawdata_tr = make_datafiles(params, 3)

    trtest = Params()
    trtest.x = [tr.x]
    trtest.y = [tr.y]

    modelselectdata.rawdata_val = rawdata_val
    modelselectdata.rawdata_tst = rawdata_tst
    header = params.header

    # Map selected detection methods to indices in detectiontypes (1-based)
    loc = []
    for d in params.detect:
        if d in detectiontypes:
            loc.append(detectiontypes.index(d) + 1)  # 1-based

    # Pre-allocate result containers
    n_algos = params.tune.shape[0] if hasattr(params.tune, 'shape') else len(params.tune)
    modelselectdata.hyp_param = [None] * n_algos
    modelselectdata.Jmse = [None] * n_algos
    modelselectdata.localhistory = [None] * n_algos
    modelselectdata.tuneval = np.zeros(n_algos)
    modelselectdata.output_tr = [None] * n_algos
    modelselectdata.obstrain = [None] * n_algos
    modelselectdata.output_val = [None] * n_algos
    modelselectdata.output_tst = [None] * n_algos
    modelselectdata.obsval = [None] * n_algos
    modelselectdata.obstest = [None] * n_algos
    modelselectdata.params = [None] * n_algos
    modelselectdata.ord = [None] * n_algos
    modelselectdata.lds_params = [None] * n_algos

    rocarea = [None] * n_algos

    # Main loop over selected algorithms
    for i in range(n_algos):
        t_start = time.time()
        tune_row = params.tune[i]  # [min, max, value]

        # 1. Regression hyper-parameter optimisation
        if params.regress.flag == 1:
            params.Ntests = len(trtest.y)

            if params.regress.optIdx == 7:
                # --- Grid search ---
                algo_name = params.algo[i]
                if algo_name in ('svr', 'libsvr', 'gp', 'lin', 'quad', 'ransac'):
                    modelselectdata.hyp_param[i] = np.logspace(
                        np.log10(tune_row[0]), np.log10(tune_row[1]), 100
                    )
                elif algo_name == 'lasso':
                    modelselectdata.hyp_param[i] = np.linspace(
                        tune_row[0], tune_row[1], 100
                    )
                elif algo_name in ('knn', 'btree', 'bnet', 'elm'):
                    if tune_row[1] > 100:
                        step = math.ceil(tune_row[1] / 100)
                        modelselectdata.hyp_param[i] = np.arange(
                            tune_row[0], tune_row[1] + 1, step
                        )
                    else:
                        modelselectdata.hyp_param[i] = np.arange(
                            tune_row[0], tune_row[1] + 1
                        )

                hp = modelselectdata.hyp_param[i]
                jmse = np.full(len(hp), np.nan)

                if params.distrib == 1:
                    # Distributed grid search
                    n_workers = getattr(params.regress, 'Nwork', 4)
                    arg_list = [(hp[j], params, i, tr, trtest) for j in range(len(hp))]
                    results = _run_parallel(modelopttest, arg_list, n_workers)
                    for j, res in enumerate(results):
                        jmse[j] = res if res is not None else np.nan
                else:
                    # Sequential grid search
                    for j in range(len(hp)):
                        jmse[j] = modelopttest(hp[j], params, i, tr, trtest)

                modelselectdata.Jmse[i] = jmse
                modelselectdata.tuneval[i] = hp[int(np.nanargmin(jmse))]
            else:
                # --- Optimisation via modelsearch ---
                tuneval_i, jmse_i, hist_i = modelsearch(
                    tune_row[2], params, tr, trtest, i
                )
                modelselectdata.tuneval[i] = tuneval_i
                modelselectdata.Jmse[i] = jmse_i
                modelselectdata.localhistory[i] = hist_i
        else:
            # No optimisation — use the fixed point directly
            modelselectdata.tuneval[i] = tune_row[2]

        t_elapsed = time.time() - t_start
        print(f"Finished {params.algo[i]} regression optimization in {t_elapsed:.2f} sec")

        # 2. Evaluate regression on train / val / test splits
        et = time.process_time()

        # --- Training ---
        params.Ntests = len(train_cell.y)
        output_tr = mainREGcode_ressarch(
            modelselectdata.tuneval[i], tr, train_cell, [params.algo[i]], params
        )
        modelselectdata.output_tr[i] = output_tr

        obstrain = []
        obstr_data = []
        for k in range(params.Ntests):
            residual = (np.asarray(train_cell.y[k]) - np.asarray(output_tr.yhat[k])).T
            obstr_data.append(residual)
            obs_k = Params()
            obs_k.data = residual
            obstrain.append(obs_k)
        modelselectdata.obstrain[i] = obstrain

        obstr = Params()
        obstr.data = obstr_data

        # --- Validation ---
        params.Ntests = len(vld.y)
        output_val = mainREGcode_ressarch(
            modelselectdata.tuneval[i], tr, vld, [params.algo[i]], params
        )
        yvalPredicted = output_val.yhat

        # Change directory context (mirrors the MATLAB cd calls)
        if hasattr(params, 's2flightpath'):
            target = os.path.dirname(params.s2flightpath)
        else:
            target = getattr(params, 'fcnpath', None)
        if target and os.path.isdir(target):
            os.chdir(target)

        dstepvals = list(range(params.dstepmin, params.dstepmax + 1))
        modelselectdata.output_val[i] = output_val

        # --- Test ---
        params.Ntests = len(test.y)
        output_tst = mainREGcode_ressarch(
            modelselectdata.tuneval[i], train, test, [params.algo[i]], params
        )
        ytestPredicted = output_tst.yhat
        modelselectdata.output_tst[i] = output_tst

        # --- Observation structures per prediction horizon ---
        obsval_steps = [None] * len(dstepvals)
        obstest_steps = [None] * len(dstepvals)
        for ds_idx, dstep in enumerate(dstepvals):
            obsval_steps[ds_idx] = truthdata(
                vld, yvalPredicted, params, rawdata_val, obstrain, ds_idx + 1
            )
            obstest_steps[ds_idx] = truthdata(
                test, ytestPredicted, params, rawdata_tst, obstrain, ds_idx + 1
            )
        modelselectdata.obsval[i] = obsval_steps
        modelselectdata.obstest[i] = obstest_steps

        header = params.header
        modelselectdata.header = header
        et = time.process_time() - et

        # 3. LDS time-series learning (if detection requires it)
        need_lds = (len(loc) > 1) or any(l != 2 for l in loc)
        if need_lds:
            ord_range = list(range(params.nmin, params.nmax + 1))
            print(f"Now training LDS w/ model orders ranging from {min(ord_range)} to {max(ord_range)}")
            et = time.process_time()

            accept_dir = os.environ.get('ACCEPT_DIR', '')
            if accept_dir and os.path.isdir(accept_dir):
                os.chdir(accept_dir)

            if params.distrib == 1 and len(ord_range) > 1:
                # Distributed LDS learning
                n_workers = len(ord_range)
                arg_list = [
                    (params, order, obstr.data, None, True)
                    for order in ord_range
                ]
                results = _run_parallel(lds_timeseries, arg_list, n_workers)
                good_idx = [j for j, r in enumerate(results) if r is not None]
                modelselectdata.ord[i] = [ord_range[j] for j in good_idx]
                if good_idx:
                    modelselectdata.lds_params[i] = [results[j] for j in good_idx]
                else:
                    raise RuntimeError('No valid LDS models found !')
            else:
                # Sequential LDS learning
                lds_results = []
                for order in ord_range:
                    lds_results.append(
                        lds_timeseries(params, order, obstr.data, None, True)
                    )
                modelselectdata.ord[i] = ord_range
                modelselectdata.lds_params[i] = lds_results

            et = time.process_time() - et
            print(f"Finished learning LDS parameters for all model orders in {et:.2f} sec")

        # 4. Detection threshold optimisation (leveltune)
        s = Params()
        s.fieldinit = None

        # Identify which detection methods are NOT in detectiontypes (i.e. custom)
        detectionstring = [
            d not in detectiontypes for d in params.detect
        ]
        # Check if any matched detection index is < 7 (non-SPRT)
        run_leveltune = False
        for j, d in enumerate(params.detect):
            if d in detectiontypes:
                idx_1based = detectiontypes.index(d) + 1
                if idx_1based < 7:
                    run_leveltune = True
                    break

        if run_leveltune:
            print('Now optimizing critical threshold to match ground truth, for relevant detection techniques')
            et = time.process_time()

            # Initialise per-algorithm detection params container
            modelselectdata.params[i] = Params()
            modelselectdata.params[i].auctemp = [None] * len(dstepvals)
            modelselectdata.params[i].rocdatatemp = [None] * len(dstepvals)

            if params.distrib == 1:
                n_workers = getattr(params.detection, 'Nwork', 4)
                arg_list = [(obsval_steps[ds],) for ds in range(len(dstepvals))]
                results = _run_parallel(leveltune, arg_list, n_workers)
                for ds in range(len(dstepvals)):
                    if results[ds] is not None:
                        auc_val, rocdata_val = results[ds]
                        modelselectdata.params[i].auctemp[ds] = auc_val
                        modelselectdata.params[i].rocdatatemp[ds] = rocdata_val
                    else:
                        raise RuntimeError('Empty result for distributed job !')
            else:
                for ds in range(len(dstepvals)):
                    auc_val, rocdata_val = leveltune(obsval_steps[ds])
                    modelselectdata.params[i].auctemp[ds] = auc_val
                    modelselectdata.params[i].rocdatatemp[ds] = rocdata_val

            # Pick the prediction horizon that maximises AUC
            auc_arr = np.array(modelselectdata.params[i].auctemp, dtype=float)
            dstepIdx = int(np.nanargmax(auc_arr))

            modelselectdata.params[i].auc = modelselectdata.params[i].auctemp[dstepIdx]
            modelselectdata.params[i].rocdata = modelselectdata.params[i].rocdatatemp[dstepIdx]
            modelselectdata.params[i].dstep = dstepvals[dstepIdx]
            print(
                f"Optimal prediction horizon is {modelselectdata.params[i].dstep} steps, "
                f"with AUC = {modelselectdata.params[i].auc}"
            )

            # Select threshold L based on constraint type
            rocdata = modelselectdata.params[i].rocdata
            avg_stats = rocdata.avg_stats
            if params.consttype == 1:
                LvalIdx = int(np.argmin(np.abs(params.maxfprate - np.asarray(avg_stats.fprate))))
            elif params.consttype == 2:
                LvalIdx = int(np.argmin(np.abs(params.maxpmd - np.asarray(avg_stats.pmd))))
            else:
                LvalIdx = int(np.argmin(
                    np.abs(np.asarray(avg_stats.pmd) - np.asarray(avg_stats.fprate))
                ))
            modelselectdata.params[i].L = avg_stats.thresh[LvalIdx]

            et = time.process_time() - et
            print(f"Finished optimizing critical threshold to match ground truth in {et:.2f} sec")

        # 5. Detection calls
        for j, detect_method in enumerate(params.detect):
            if detect_method in detectiontypes:
                det_idx = detectiontypes.index(detect_method) + 1  # 1-based

                if det_idx < 7:
                    # Non-SPRT detection
                    if need_lds:
                        snew, params = detectioncall(
                            obsval_steps[dstepIdx], obstest_steps[dstepIdx],
                            params, det_idx,
                            modelselectdata.params[i], modelselectdata.lds_params[i]
                        )
                    else:
                        snew, params = detectioncall(
                            obsval_steps[dstepIdx], obstest_steps[dstepIdx],
                            params, det_idx,
                            modelselectdata.params[i]
                        )
                else:
                    # SPRT detection
                    if need_lds:
                        snew, params = detectioncall(
                            obsval_steps[dstepIdx], obstest_steps[dstepIdx],
                            params, det_idx,
                            None, modelselectdata.lds_params[i]
                        )
                    else:
                        snew, params = detectioncall(
                            obsval_steps[dstepIdx], obstest_steps[dstepIdx],
                            params, det_idx
                        )

                s = _merge_params(s, snew)

        # Strip the fieldinit sentinel and store
        rocarea_i = Params()
        for key, val in vars(s).items():
            if key != 'fieldinit':
                setattr(rocarea_i, key, val)
        rocarea[i] = rocarea_i

        # Clean up distributed job records
        if params.distrib == 1 and getattr(params, 'mcr', 2) == 1:
            params.jobrecordID = []

    return modelselectdata, rocarea