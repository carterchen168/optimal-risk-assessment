import numpy as np
import time
import os
import glob
import pandas as pd
import scipy.io as sio
from typing import Dict, List, Tuple, Any
from regressopt import mainREGcode_ressarch, modelopttest, optimsearch
from detectopt import truthdata, leveltune, detectioncall
from ldslearn.lds_timeseries import lds_timeseries

class Struct:
    """A lightweight class to replicate MATLAB struct dot-notation behavior."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def run(params: Any) -> Tuple[Any, List]:
    """
    Convert of MATLAB testoptloop_ressarch.m to Python.
    Renamed to 'run' to match the call from ressarch.py.
    """
    detection_types = [
        'Redline - Training', 
        'Redline - Validation', 
        'Predictive - Training', 
        'Predictive - Validation', 
        'Optimal - Training', 
        'Optimal - Validation', 
        'SPRT'
    ]
    
    params.jobrecordID = []
    
    # make_datafiles calls
    tr, vld, train_cell, rawdata_val, rawdata_tr, params, Statistics, features, fullfeatures = make_datafiles(params, 2)
    
    model_select_data = Struct()
    model_select_data.rawdata_tr = rawdata_tr
    model_select_data.Statistics = Statistics
    
    train, test, train_cell, rawdata_tst, rawdata_tr, params, _, _, _ = make_datafiles(params, 3)
    
    trtest = Struct()
    trtest.x = [tr.x]
    trtest.y = [tr.y]
    
    model_select_data.rawdata_val = rawdata_val
    model_select_data.rawdata_tst = rawdata_tst
    
    header = params.header
    
    # ismember equivalent - find location of params.detect in detection_types
    loc = []
    detect_list = params.detect if isinstance(params.detect, list) else [params.detect]
    for detect in detect_list:
        if detect in detection_types:
            loc.append(detection_types.index(detect) + 1)  # MATLAB uses 1-based indexing
        else:
            loc.append(-1)
    
    # Main loop over tune parameters
    for i in range(len(params.tune)):
        t_start = time.time()
        
        if params.regress.flag == 1:
            params.Ntests = len(trtest.y)
            
            if params.regress.optIdx == 7:
                # Generate hyperparameter values based on algorithm type
                algo = params.algo[i]
                tune_min = params.tune[i][0]
                tune_max = params.tune[i][1]
                
                if not hasattr(model_select_data, 'hyp_param'):
                    model_select_data.hyp_param = []
                
                if i >= len(model_select_data.hyp_param):
                    model_select_data.hyp_param.append(None)
                
                if algo in ['svr', 'libsvr', 'gp', 'lin', 'quad', 'ransac']:
                    model_select_data.hyp_param[i] = np.logspace(np.log10(tune_min), np.log10(tune_max), 100)
                    
                elif algo == 'lasso':
                    model_select_data.hyp_param[i] = np.linspace(tune_min, tune_max, 100)
                    
                elif algo in ['knn', 'btree', 'bnet', 'elm']:
                    if tune_max > 100:
                        step = int(np.ceil(tune_max / 100))
                        model_select_data.hyp_param[i] = np.arange(tune_min, tune_max + 1, step)
                    else:
                        model_select_data.hyp_param[i] = np.arange(tune_min, tune_max + 1)
                
                # Handle distributed computing
                if params.distrib == 1:
                    # Distributed job submission - TODO: implement based on your job scheduler
                    pass
                else:
                    # Serial execution
                    if not hasattr(model_select_data, 'Jmse'):
                        model_select_data.Jmse = []
                    if i >= len(model_select_data.Jmse):
                        model_select_data.Jmse.append([])
                    
                    for j in range(len(model_select_data.hyp_param[i])):
                        result = modelopttest(model_select_data.hyp_param[i][j], params, i, tr, trtest)
                        model_select_data.Jmse[i].append(result)
                
                # Find optimal hyperparameter (argmin)
                if not hasattr(model_select_data, 'tuneval'):
                    model_select_data.tuneval = np.array([])
                    
                optimal_idx = np.argmin(model_select_data.Jmse[i])
                model_select_data.tuneval = np.append(model_select_data.tuneval, 
                                                      model_select_data.hyp_param[i][optimal_idx])
            else:
                # modelsearch alternative
                tune_val, jmse, local_history = modelsearch(params.tune[i][2], params, tr, trtest, i)
                
                if not hasattr(model_select_data, 'tuneval'):
                    model_select_data.tuneval = np.array([])
                model_select_data.tuneval = np.append(model_select_data.tuneval, tune_val)
                
                if not hasattr(model_select_data, 'Jmse'):
                    model_select_data.Jmse = []
                if i >= len(model_select_data.Jmse):
                    model_select_data.Jmse.append(jmse)
                else:
                    model_select_data.Jmse[i] = jmse
                    
                if not hasattr(model_select_data, 'localhistory'):
                    model_select_data.localhistory = []
                if i >= len(model_select_data.localhistory):
                    model_select_data.localhistory.append(local_history)
        else:
            if not hasattr(model_select_data, 'tuneval'):
                model_select_data.tuneval = np.array([])
            model_select_data.tuneval = np.append(model_select_data.tuneval, params.tune[i][2])
        
        t_end = time.time() - t_start
        print(f"Finished {params.algo[i]} regression optimization in {t_end:.2f} sec")
        
        # Training phase
        et = time.time()
        params.Ntests = len(train_cell.y)
        # output_tr = mainREGcode_ressarch(model_select_data.tuneval[i], tr, train_cell, 
        #                                  params.algo[i], params)
        output_tr, params = mainREGcode_ressarch(model_select_data.tuneval[i], tr, train_cell, 
                                                [params.algo[i]], params)
                                         
        if not hasattr(model_select_data, 'output_tr'):
            model_select_data.output_tr = []
        if i >= len(model_select_data.output_tr):
            model_select_data.output_tr.append(output_tr)
        else:
            model_select_data.output_tr[i] = output_tr
        
        # Calculate training residuals
        obstr = Struct(data=[])
        obstrain = []
        for k in range(params.Ntests):
            residual = train_cell.y[k] - output_tr.yhat[k]
            obstr.data.append(residual)
            obstrain.append(Struct(data=residual))
        
        if not hasattr(model_select_data, 'obstrain'):
            model_select_data.obstrain = []
        if i >= len(model_select_data.obstrain):
            model_select_data.obstrain.append(obstrain)
        else:
            model_select_data.obstrain[i] = obstrain
        
        # Validation phase
        params.Ntests = len(vld.y)
        output_val, params = mainREGcode_ressarch(model_select_data.tuneval[i], tr, vld, 
                                                 [params.algo[i]], params)
        yval_predicted = output_val.yhat
        
        if hasattr(params, 's2flightpath'):
            os.chdir(params.s2flightpath)
            os.chdir('..')
        else:
            os.chdir(getattr(params, 'fcnpath', '.'))
        
        dstepvals = np.arange(params.dstepmin, params.dstepmax + 1)
        
        if not hasattr(model_select_data, 'output_val'):
            model_select_data.output_val = []
        if i >= len(model_select_data.output_val):
            model_select_data.output_val.append(output_val)
        else:
            model_select_data.output_val[i] = output_val
        
        # Testing phase
        params.Ntests = len(test.y)
        output_tst, params = mainREGcode_ressarch(model_select_data.tuneval[i], train, test, 
                             [params.algo[i]], params)
        ytest_predicted = output_tst.yhat
        
        if not hasattr(model_select_data, 'output_tst'):
            model_select_data.output_tst = []
        if i >= len(model_select_data.output_tst):
            model_select_data.output_tst.append(output_tst)
        else:
            model_select_data.output_tst[i] = output_tst
        
        # Calculate observation metrics for different detection steps
        obsval = {}
        obstest = {}
        for dstep_idx, dstep in enumerate(dstepvals):
            obsval[dstep_idx] = truthdata(vld, yval_predicted, params, rawdata_val, obstrain, dstep)
            obstest[dstep_idx] = truthdata(test, ytest_predicted, params, rawdata_tst, obstrain, dstep)
        
        if not hasattr(model_select_data, 'obsval'):
            model_select_data.obsval = []
            model_select_data.obstest = []
            
        if i >= len(model_select_data.obsval):
            model_select_data.obsval.append(obsval)
            model_select_data.obstest.append(obstest)
        else:
            model_select_data.obsval[i] = obsval
            model_select_data.obstest[i] = obstest
        
        model_select_data.header = params.header
        
        et = time.time() - et
        
        # LDS model training
        if len(loc) > 1 or any(l != 2 for l in loc):
            ord_vals = np.arange(params.nmin, params.nmax + 1)
            print(f"Now training LDS w/ model orders ranging from {ord_vals.min()} to {ord_vals.max()}")
            
            et = time.time()
            os.chdir(os.getenv('ACCEPT_DIR', '.'))
            
            if params.distrib == 1 and len(ord_vals) > 1:
                # Distributed LDS training - TODO: implement
                pass
            else:
                # Serial LDS training
                if not hasattr(model_select_data, 'lds_params'):
                    model_select_data.lds_params = []
                if i >= len(model_select_data.lds_params):
                    model_select_data.lds_params.append([])
                
                for j, ord_val in enumerate(ord_vals):
                    result = lds_timeseries(params, ord_val, obstr.data, None, True)
                    model_select_data.lds_params[i].append(result)
            
            et = time.time() - et
            print(f"Finished learning LDS parameters for all model orders in {et:.2f} sec")
        
        # Detection optimization
        s = Struct(fieldinit=[])
        
        run_leveltune = False
        for detect in detect_list:
            if detect in detection_types:
                # 1-based index to match MATLAB logic (SPRT is 7)
                idx = detection_types.index(detect) + 1 
                if idx < 7:
                    run_leveltune = True
                    break
        
        if run_leveltune:
            print('Now optimizing critical threshold to match ground truth, for relevant detection techniques')
            et = time.time()
            
            if params.distrib == 1:
                # Distributed threshold optimization - TODO: implement
                pass
            else:
                # Serial threshold optimization
                for dstep_idx in obsval.keys():
                    auc_val, rocdata = leveltune(obsval[dstep_idx])
                    
                    if not hasattr(model_select_data, 'params'):
                        model_select_data.params = []
                    if i >= len(model_select_data.params):
                        model_select_data.params.append(Struct())
                    
                    if not hasattr(model_select_data.params[i], 'auctemp'):
                        model_select_data.params[i].auctemp = {}
                        model_select_data.params[i].rocdatatemp = {}
                    
                    model_select_data.params[i].auctemp[dstep_idx] = auc_val
                    model_select_data.params[i].rocdatatemp[dstep_idx] = rocdata
            
            # Find optimal decision step (argmax of AUC)
            auctemp_vals = list(model_select_data.params[i].auctemp.values())
            dstep_idx_optimal = list(model_select_data.params[i].auctemp.keys())[np.argmax(auctemp_vals)]
            
            model_select_data.params[i].auc = model_select_data.params[i].auctemp[dstep_idx_optimal]
            model_select_data.params[i].rocdata = model_select_data.params[i].rocdatatemp[dstep_idx_optimal]
            model_select_data.params[i].dstep = dstepvals[dstep_idx_optimal]
            
            print(f"Optimal prediction horizon is {model_select_data.params[i].dstep} steps, with AUC = {model_select_data.params[i].auc:.4f}")
            
            # Find optimal likelihood threshold
            rocdata_stats = model_select_data.params[i].rocdata.avg_stats
            
            if params.consttype == 1:
                fprate_vals = rocdata_stats.fprate
                lval_idx = np.argmin(np.abs(params.maxfprate - fprate_vals))
            elif params.consttype == 2:
                pmd_vals = rocdata_stats.pmd
                lval_idx = np.argmin(np.abs(params.maxpmd - pmd_vals))
            else:
                pmd_vals = rocdata_stats.pmd
                fprate_vals = rocdata_stats.fprate
                lval_idx = np.argmin(np.abs(pmd_vals - fprate_vals))
            
            model_select_data.params[i].L = rocdata_stats.thresh[lval_idx]
            
            et = time.time() - et
            print(f"Finished optimizing critical threshold to match ground truth in {et:.2f} sec")
        
        # Run detection algorithms
        for j, detect in enumerate(detect_list):
            if detect in detection_types:
                detect_idx = detection_types.index(detect) + 1
                
                if detect_idx < 7:
                    if len(loc) > 1 or any(l != 2 for l in loc):
                        snew, params = detectioncall(obsval[dstep_idx_optimal], obstest[dstep_idx_optimal], 
                                                    params, detect_idx, 
                                                    model_select_data.params[i],
                                                    model_select_data.lds_params[i])
                    else:
                        snew, params = detectioncall(obsval[dstep_idx_optimal], obstest[dstep_idx_optimal], 
                                                    params, detect_idx, 
                                                    model_select_data.params[i])
                else:
                    if len(loc) > 1 or any(l != 2 for l in loc):
                        snew, params = detectioncall(obsval[dstep_idx_optimal], obstest[dstep_idx_optimal], 
                                                    params, detect_idx, None,
                                                    model_select_data.lds_params[i])
                    else:
                        snew, params = detectioncall(obsval[dstep_idx_optimal], obstest[dstep_idx_optimal], 
                                                    params, detect_idx)
                
                s = mergeParams(s, snew)
        
        # Create rocarea
        rocarea = []
        if i >= len(rocarea):
            rocarea.append(None)
        rocarea[i] = {k: v for k, v in vars(s).items() if k != 'fieldinit'}
        
        # Clean up distributed jobs
        if params.distrib == 1 and getattr(params, 'mcr', 0) == 1:
            for j in params.jobrecordID:
                # TODO: destroy jobs - destroyJobNew(job)
                pass
    
    return model_select_data, rocarea


# Placeholder functions updated to use and return Struct objects
def make_datafiles(params: Any, mode: int) -> Tuple:
    def load_data_from_path(path: str) -> Tuple[np.ndarray, List[str]]:
        if not os.path.isdir(path):
            return np.array([]), []
        data_list = []
        csv_files = glob.glob(os.path.join(path, '*.csv'))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            data_list.append(df.values)
        mat_files = glob.glob(os.path.join(path, '*.mat'))
        for mat_file in mat_files:
            try:
                mat_data = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
                for key in mat_data.keys():
                    if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                        if mat_data[key].ndim == 2:
                            data_list.append(mat_data[key])
            except Exception as e:
                print(f"Warning: Could not load {mat_file}: {e}")
        concatenated = np.vstack(data_list) if data_list else np.array([])
        return concatenated, []
    
    print("Loading training data...")
    train_data, _ = load_data_from_path(getattr(params, 'nompath', ''))
    rawdata_tr = train_data.copy()
    
    header = getattr(params, 'header', [])
    target_name = getattr(params, 'targetName', '')
    target_idx = 0
    if target_name and header:
        try:
            target_idx = header.index(target_name)
        except ValueError:
            target_idx = 0 
    
    feature_indices = getattr(params, 'channelContineous', list(range(len(header))))
    
    if train_data.size > 0:
        tr_y = train_data[:, target_idx] if target_idx < train_data.shape[1] else np.zeros(len(train_data))
        tr_x = train_data[:, feature_indices] if feature_indices else train_data
    else:
        tr_x = np.array([])
        tr_y = np.array([])
    
    train_cell = Struct(
        x=[tr_x[i:i+1, :] for i in range(len(tr_y))] if len(tr_y) > 0 else [],
        y=[np.array([tr_y[i]]) for i in range(len(tr_y))] if len(tr_y) > 0 else []
    )
    
    tr = Struct(
        x=tr_x,
        y=tr_y,
        header=[header[i] for i in feature_indices] if header else []
    )
    
    Statistics = Struct(
        mean=np.mean(tr_x, axis=0) if tr_x.size > 0 else np.array([]),
        std=np.std(tr_x, axis=0) if tr_x.size > 0 else np.array([]),
        min=np.min(tr_x, axis=0) if tr_x.size > 0 else np.array([]),
        max=np.max(tr_x, axis=0) if tr_x.size > 0 else np.array([]),
        samples=len(tr_y)
    )
    
    if mode == 2: 
        print("Loading validation data...")
        val_data, _ = load_data_from_path(getattr(params, 'valpath', ''))
        rawdata_val = val_data.copy()
        
        if val_data.size > 0:
            val_y = val_data[:, target_idx] if target_idx < val_data.shape[1] else np.zeros(len(val_data))
            val_x = val_data[:, feature_indices] if feature_indices else val_data
        else:
            val_x = np.array([])
            val_y = np.array([])
        
        vld = Struct(x=val_x, y=val_y, header=tr.header)
        return tr, vld, train_cell, rawdata_val, rawdata_tr, params, Statistics, feature_indices, np.arange(train_data.shape[1]) if train_data.size > 0 else np.array([])
        
    else:
        print("Loading test data...")
        test_data, _ = load_data_from_path(getattr(params, 'testpath', ''))
        rawdata_tst = test_data.copy()
        
        if test_data.size > 0:
            test_y = test_data[:, target_idx] if target_idx < test_data.shape[1] else np.zeros(len(test_data))
            test_x = test_data[:, feature_indices] if feature_indices else test_data
        else:
            test_x = np.array([])
            test_y = np.array([])
        
        test = Struct(x=test_x, y=test_y, header=tr.header)
        return tr, test, train_cell, rawdata_tst, rawdata_tr, params, Statistics, feature_indices, np.arange(train_data.shape[1]) if train_data.size > 0 else np.array([])

def modelsearch(tune_val: float, params: Any, tr: Any, trtest: Any, i: int) -> Tuple[float, float, Any]:
    x0 = np.array([tune_val])
    x, fval, localhistory, globalhistory = optimsearch(x0, params, tr, trtest, i)
    return x[0], fval, localhistory

def mergeParams(s: Any, snew: Any) -> Any:
    for k, v in vars(snew).items():
        setattr(s, k, v)
    return s