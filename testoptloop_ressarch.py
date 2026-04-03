import numpy as np
import time
import os
import glob
import pandas as pd
import scipy.io as sio
from typing import Dict, List, Tuple, Any
from modelsearch import optimsearch

def testoptloop(params: Dict) -> Tuple[Dict, List]:
    """
    Convert of MATLAB testoptloop_ressarch.m to Python
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
    
    params['jobrecordID'] = []
    
    # make_datafiles calls
    tr, vld, train_cell, rawdata_val, rawdata_tr, params, Statistics, features, fullfeatures = make_datafiles(params, 2)
    
    model_select_data = {}
    model_select_data['rawdata_tr'] = rawdata_tr
    model_select_data['Statistics'] = Statistics
    
    train, test, train_cell, rawdata_tst, rawdata_tr = make_datafiles(params, 3)
    
    trtest = {}
    trtest['x'] = [tr['x']]
    trtest['y'] = [tr['y']]
    
    model_select_data['rawdata_val'] = rawdata_val
    model_select_data['rawdata_tst'] = rawdata_tst
    
    header = params['header']
    
    # ismember equivalent - find location of params.detect in detection_types
    loc = []
    for detect in params['detect'] if isinstance(params['detect'], list) else [params['detect']]:
        if detect in detection_types:
            loc.append(detection_types.index(detect) + 1)  # MATLAB uses 1-based indexing
        else:
            loc.append(-1)
    
    # Main loop over tune parameters
    for i in range(len(params['tune'])):
        t_start = time.time()
        
        if params['regress']['flag'] == 1:
            params['Ntests'] = len(trtest['y'])
            
            if params['regress']['optIdx'] == 7:
                # Generate hyperparameter values based on algorithm type
                algo = params['algo'][i]
                tune_min = params['tune'][i, 0]
                tune_max = params['tune'][i, 1]
                
                if algo in ['svr', 'libsvr', 'gp', 'lin', 'quad', 'ransac']:
                    model_select_data['hyp_param'] = model_select_data.get('hyp_param', [])
                    if i >= len(model_select_data['hyp_param']):
                        model_select_data['hyp_param'].append(None)
                    model_select_data['hyp_param'][i] = np.logspace(np.log10(tune_min), np.log10(tune_max), 100)
                    
                elif algo == 'lasso':
                    if i >= len(model_select_data.get('hyp_param', [])):
                        model_select_data['hyp_param'] = model_select_data.get('hyp_param', [])
                        model_select_data['hyp_param'].append(None)
                    model_select_data['hyp_param'][i] = np.linspace(tune_min, tune_max, 100)
                    
                elif algo in ['knn', 'btree', 'bnet', 'elm']:
                    if tune_max > 100:
                        step = int(np.ceil(tune_max / 100))
                        model_select_data['hyp_param'][i] = np.arange(tune_min, tune_max + 1, step)
                    else:
                        model_select_data['hyp_param'][i] = np.arange(tune_min, tune_max + 1)
                
                # Handle distributed computing
                if params['distrib'] == 1:
                    # Distributed job submission - TODO: implement based on your job scheduler
                    pass
                else:
                    # Serial execution
                    model_select_data['Jmse'] = model_select_data.get('Jmse', [])
                    if i >= len(model_select_data['Jmse']):
                        model_select_data['Jmse'].append([])
                    
                    for j in range(len(model_select_data['hyp_param'][i])):
                        result = modelopttest(model_select_data['hyp_param'][i][j], params, i, tr, trtest)
                        model_select_data['Jmse'][i].append(result)
                
                # Find optimal hyperparameter (argmin)
                model_select_data['tuneval'] = model_select_data.get('tuneval', np.array([]))
                optimal_idx = np.argmin(model_select_data['Jmse'][i])
                model_select_data['tuneval'] = np.append(model_select_data['tuneval'], 
                                                         model_select_data['hyp_param'][i][optimal_idx])
            else:
                # modelsearch alternative
                tune_val, jmse, local_history = modelsearch(params['tune'][i, 2], params, tr, trtest, i)
                model_select_data['tuneval'] = model_select_data.get('tuneval', np.array([]))
                model_select_data['tuneval'] = np.append(model_select_data['tuneval'], tune_val)
                model_select_data['Jmse'] = model_select_data.get('Jmse', [])
                if i >= len(model_select_data['Jmse']):
                    model_select_data['Jmse'].append(jmse)
                else:
                    model_select_data['Jmse'][i] = jmse
                model_select_data['localhistory'] = model_select_data.get('localhistory', [])
                if i >= len(model_select_data['localhistory']):
                    model_select_data['localhistory'].append(local_history)
        else:
            model_select_data['tuneval'] = model_select_data.get('tuneval', np.array([]))
            model_select_data['tuneval'] = np.append(model_select_data['tuneval'], params['tune'][i, 2])
        
        t_end = time.time() - t_start
        print(f"Finished {params['algo'][i]} regression optimization in {t_end:.2f} sec")
        
        # Training phase
        et = time.time()
        params['Ntests'] = len(train_cell['y'])
        output_tr = mainREGcode_ressarch(model_select_data['tuneval'][i], tr, train_cell, 
                                         params['algo'][i], params)
        model_select_data['output_tr'] = model_select_data.get('output_tr', [])
        if i >= len(model_select_data['output_tr']):
            model_select_data['output_tr'].append(output_tr)
        else:
            model_select_data['output_tr'][i] = output_tr
        
        # Calculate training residuals
        obstr = {}
        obstr['data'] = []
        obstrain = []
        for k in range(params['Ntests']):
            residual = train_cell['y'][k] - output_tr['yhat'][k]
            obstr['data'].append(residual)
            obstrain.append({'data': residual})
        
        model_select_data['obstrain'] = model_select_data.get('obstrain', [])
        if i >= len(model_select_data['obstrain']):
            model_select_data['obstrain'].append(obstrain)
        else:
            model_select_data['obstrain'][i] = obstrain
        
        # Validation phase
        params['Ntests'] = len(vld['y'])
        output_val = mainREGcode_ressarch(model_select_data['tuneval'][i], tr, vld, 
                                         params['algo'][i], params)
        yval_predicted = output_val['yhat']
        
        if 's2flightpath' in params:
            os.chdir(params['s2flightpath'])
            os.chdir('..')
        else:
            os.chdir(params['fcnpath'])
        
        dstepvals = np.arange(params['dstepmin'], params['dstepmax'] + 1)
        model_select_data['output_val'] = model_select_data.get('output_val', [])
        if i >= len(model_select_data['output_val']):
            model_select_data['output_val'].append(output_val)
        else:
            model_select_data['output_val'][i] = output_val
        
        # Testing phase
        params['Ntests'] = len(test['y'])
        output_tst = mainREGcode_ressarch(model_select_data['tuneval'][i], train, test, 
                                         params['algo'][i], params)
        ytest_predicted = output_tst['yhat']
        model_select_data['output_tst'] = model_select_data.get('output_tst', [])
        if i >= len(model_select_data['output_tst']):
            model_select_data['output_tst'].append(output_tst)
        else:
            model_select_data['output_tst'][i] = output_tst
        
        # Calculate observation metrics for different detection steps
        obsval = {}
        obstest = {}
        for dstep_idx, dstep in enumerate(dstepvals):
            obsval[dstep_idx] = truthdata(vld, yval_predicted, params, rawdata_val, obstrain, dstep)
            obstest[dstep_idx] = truthdata(test, ytest_predicted, params, rawdata_tst, obstrain, dstep)
        
        model_select_data['obsval'] = model_select_data.get('obsval', [])
        model_select_data['obstest'] = model_select_data.get('obstest', [])
        if i >= len(model_select_data['obsval']):
            model_select_data['obsval'].append(obsval)
            model_select_data['obstest'].append(obstest)
        else:
            model_select_data['obsval'][i] = obsval
            model_select_data['obstest'][i] = obstest
        
        header = params['header']
        model_select_data['header'] = header
        
        et = time.time() - et
        
        # LDS model training
        if len(loc) > 1 or any(l != 2 for l in loc):
            ord_vals = np.arange(params['nmin'], params['nmax'] + 1)
            print(f"Now training LDS w/ model orders ranging from {ord_vals.min()} to {ord_vals.max()}")
            
            et = time.time()
            os.chdir(os.getenv('ACCEPT_DIR', '.'))
            
            if params['distrib'] == 1 and len(ord_vals) > 1:
                # Distributed LDS training - TODO: implement
                pass
            else:
                # Serial LDS training
                model_select_data['lds_params'] = model_select_data.get('lds_params', [])
                if i >= len(model_select_data['lds_params']):
                    model_select_data['lds_params'].append([])
                
                for j, ord_val in enumerate(ord_vals):
                    result = lds_timeseries(params, ord_val, obstr['data'], None, True)
                    model_select_data['lds_params'][i].append(result)
            
            et = time.time() - et
            print(f"Finished learning LDS parameters for all model orders in {et:.2f} sec")
        
        # Detection optimization
        s = {'fieldinit': []}
        
        detection_string = []
        for detect in params['detect'] if isinstance(params['detect'], list) else [params['detect']]:
            detection_string.append(detect not in detection_types)
        
        if any(not d for d in detection_string):
            print('Now optimizing critical threshold to match ground truth, for relevant detection techniques')
            et = time.time()
            
            if params['distrib'] == 1:
                # Distributed threshold optimization - TODO: implement
                pass
            else:
                # Serial threshold optimization
                for dstep_idx in obsval.keys():
                    auc_val, rocdata = leveltune(obsval[dstep_idx])
                    
                    model_select_data['params'] = model_select_data.get('params', [])
                    if i >= len(model_select_data['params']):
                        model_select_data['params'].append({})
                    
                    if 'auctemp' not in model_select_data['params'][i]:
                        model_select_data['params'][i]['auctemp'] = {}
                        model_select_data['params'][i]['rocdatatemp'] = {}
                    
                    model_select_data['params'][i]['auctemp'][dstep_idx] = auc_val
                    model_select_data['params'][i]['rocdatatemp'][dstep_idx] = rocdata
            
            # Find optimal decision step (argmax of AUC)
            auctemp_vals = list(model_select_data['params'][i]['auctemp'].values())
            dstep_idx_optimal = list(model_select_data['params'][i]['auctemp'].keys())[np.argmax(auctemp_vals)]
            
            model_select_data['params'][i]['auc'] = model_select_data['params'][i]['auctemp'][dstep_idx_optimal]
            model_select_data['params'][i]['rocdata'] = model_select_data['params'][i]['rocdatatemp'][dstep_idx_optimal]
            model_select_data['params'][i]['dstep'] = dstepvals[dstep_idx_optimal]
            
            print(f"Optimal prediction horizon is {model_select_data['params'][i]['dstep']} steps, with AUC = {model_select_data['params'][i]['auc']:.4f}")
            
            # Find optimal likelihood threshold
            rocdata_stats = model_select_data['params'][i]['rocdata']['avg_stats']
            
            if params['consttype'] == 1:
                fprate_vals = rocdata_stats['fprate']
                lval_idx = np.argmin(np.abs(params['maxfprate'] - fprate_vals))
            elif params['consttype'] == 2:
                pmd_vals = rocdata_stats['pmd']
                lval_idx = np.argmin(np.abs(params['maxpmd'] - pmd_vals))
            else:
                pmd_vals = rocdata_stats['pmd']
                fprate_vals = rocdata_stats['fprate']
                lval_idx = np.argmin(np.abs(pmd_vals - fprate_vals))
            
            model_select_data['params'][i]['L'] = rocdata_stats['thresh'][lval_idx]
            
            et = time.time() - et
            print(f"Finished optimizing critical threshold to match ground truth in {et:.2f} sec")
        
        # Run detection algorithms
        for j, detect in enumerate(params['detect'] if isinstance(params['detect'], list) else [params['detect']]):
            if detect in detection_types:
                detect_idx = detection_types.index(detect) + 1
                
                if detect_idx < 7:
                    if len(loc) > 1 or any(l != 2 for l in loc):
                        snew, params = detectioncall(obsval[dstep_idx_optimal], obstest[dstep_idx_optimal], 
                                                    params, detect_idx, 
                                                    model_select_data['params'][i],
                                                    model_select_data['lds_params'][i])
                    else:
                        snew, params = detectioncall(obsval[dstep_idx_optimal], obstest[dstep_idx_optimal], 
                                                    params, detect_idx, 
                                                    model_select_data['params'][i])
                else:
                    if len(loc) > 1 or any(l != 2 for l in loc):
                        snew, params = detectioncall(obsval[dstep_idx_optimal], obstest[dstep_idx_optimal], 
                                                    params, detect_idx, None,
                                                    model_select_data['lds_params'][i])
                    else:
                        snew, params = detectioncall(obsval[dstep_idx_optimal], obstest[dstep_idx_optimal], 
                                                    params, detect_idx)
                
                s = mergeParams(s, snew)
        
        # Create rocarea
        rocarea = []
        if i >= len(rocarea):
            rocarea.append(None)
        rocarea[i] = {k: v for k, v in s.items() if k != 'fieldinit'}
        
        # Clean up distributed jobs
        if params['distrib'] == 1 and params['mcr'] == 1:
            for j in params['jobrecordID']:
                # TODO: destroy jobs - destroyJobNew(job)
                pass
    
    return model_select_data, rocarea


# Placeholder functions - these need to be implemented based on your actual code
def make_datafiles(params: Dict, mode: int) -> Tuple:
    """
    Prepare and split data into training, validation, and testing sets.
    
    Loads data from CSV or MAT files in the Training, Validation, Testing directories.
    Extracts features (X) and target variable (y) for regression.
    
    Args:
        params: Parameters dictionary containing:
            - nompath: Path to Training data directory
            - valpath: Path to Validation data directory  
            - testpath: Path to Testing data directory
            - targetName: Name of target variable to predict
            - header: Column names
            - channelContineous: Indices of continuous parameters
        mode: 2 = train/validation split, 3 = train/test split
                
    Returns:
        Tuple of (data1, data2, train_cell, rawdata_2, rawdata_tr, params, Statistics, features, fullfeatures)
        - mode 2: (tr, vld, train_cell, rawdata_val, rawdata_tr, ...)
        - mode 3: (train, test, train_cell, rawdata_tst, rawdata_tr, ...)
        
        Where each data dict has structure:
            - 'x': Features array (nsamples x nfeatures)
            - 'y': Target variable array (nsamples,)
            - 'header': Feature names
            
        train_cell is list of individual samples from training set
        rawdata_* contains original unprocessed data
        Statistics contains mean, std, etc. of training data
        features/fullfeatures contain feature indices/names
    """
    def load_data_from_path(path: str) -> Tuple[np.ndarray, List[str]]:
        """Load all CSV/MAT files from a directory and concatenate them."""
        if not os.path.isdir(path):
            return np.array([]), []
            
        data_list = []
        
        # Try CSV files first
        csv_files = glob.glob(os.path.join(path, '*.csv'))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            data_list.append(df.values)
        
        # Try MAT files
        mat_files = glob.glob(os.path.join(path, '*.mat'))
        for mat_file in mat_files:
            try:
                mat_data = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
                # Look for data arrays in the mat file
                for key in mat_data.keys():
                    if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                        if mat_data[key].ndim == 2:
                            data_list.append(mat_data[key])
            except Exception as e:
                print(f"Warning: Could not load {mat_file}: {e}")
        
        if not data_list:
            return np.array([]), []
            
        # Concatenate all data
        concatenated = np.vstack(data_list) if data_list else np.array([])
        return concatenated, []
    
    # Load training data
    print("Loading training data...")
    train_data, _ = load_data_from_path(params.get('nompath', ''))
    rawdata_tr = train_data.copy()
    
    # Find target column index
    header = params.get('header', [])
    target_name = params.get('targetName', '')
    target_idx = None
    
    if target_name and header:
        try:
            target_idx = header.index(target_name)
        except ValueError:
            print(f"Warning: Target '{target_name}' not found in header")
            target_idx = 0  # Default to first column
    else:
        target_idx = 0
    
    # Get feature indices (continuous parameters)
    feature_indices = params.get('channelContineous', list(range(len(header))))
    
    # Extract training features and targets
    if train_data.size > 0:
        tr_y = train_data[:, target_idx] if target_idx < train_data.shape[1] else np.zeros(len(train_data))
        tr_x = train_data[:, feature_indices] if feature_indices else train_data
    else:
        tr_x = np.array([])
        tr_y = np.array([])
    
    # Create train_cell: list of individual training samples
    train_cell = {
        'x': [tr_x[i:i+1, :] for i in range(len(tr_y))] if len(tr_y) > 0 else [],
        'y': [np.array([tr_y[i]]) for i in range(len(tr_y))] if len(tr_y) > 0 else []
    }
    
    # Training data dict
    tr = {
        'x': tr_x,
        'y': tr_y,
        'header': [header[i] for i in feature_indices] if header else []
    }
    
    # Calculate training statistics
    Statistics = {
        'mean': np.mean(tr_x, axis=0) if tr_x.size > 0 else np.array([]),
        'std': np.std(tr_x, axis=0) if tr_x.size > 0 else np.array([]),
        'min': np.min(tr_x, axis=0) if tr_x.size > 0 else np.array([]),
        'max': np.max(tr_x, axis=0) if tr_x.size > 0 else np.array([]),
        'samples': len(tr_y)
    }
    
    # Load validation or test data based on mode
    if mode == 2:  # Train/Validation split
        print("Loading validation data...")
        val_data, _ = load_data_from_path(params.get('valpath', ''))
        rawdata_val = val_data.copy()
        
        if val_data.size > 0:
            val_y = val_data[:, target_idx] if target_idx < val_data.shape[1] else np.zeros(len(val_data))
            val_x = val_data[:, feature_indices] if feature_indices else val_data
        else:
            val_x = np.array([])
            val_y = np.array([])
        
        vld = {
            'x': val_x,
            'y': val_y,
            'header': tr['header']
        }
        
        return tr, vld, train_cell, rawdata_val, rawdata_tr, params, Statistics, feature_indices, np.arange(train_data.shape[1]) if train_data.size > 0 else np.array([])
        
    else:  # mode == 3: Train/Test split
        print("Loading test data...")
        test_data, _ = load_data_from_path(params.get('testpath', ''))
        rawdata_tst = test_data.copy()
        
        if test_data.size > 0:
            test_y = test_data[:, target_idx] if target_idx < test_data.shape[1] else np.zeros(len(test_data))
            test_x = test_data[:, feature_indices] if feature_indices else test_data
        else:
            test_x = np.array([])
            test_y = np.array([])
        
        test = {
            'x': test_x,
            'y': test_y,
            'header': tr['header']
        }
        
        return tr, test, train_cell, rawdata_tst, rawdata_tr, params, Statistics, feature_indices, np.arange(train_data.shape[1]) if train_data.size > 0 else np.array([])


def modelopttest(hyp_param: float, params: Dict, i: int, tr: Dict, trtest: Dict) -> float:
    """Placeholder for modelopttest function"""
    # TODO: Implement based on your MATLAB modelopttest.m
    pass


def modelsearch(tune_val: float, params: Dict, tr: Dict, trtest: Dict, i: int) -> Tuple[float, float, Dict]:
    """
    Model search using optimization search methods from modelsearch.optimsearch.
    Searches for optimal hyperparameter value to minimize cost function.
    
    Args:
        tune_val: Initial tuning value (x0)
        params: Parameters dictionary containing optIdx, distrib, etc.
        tr: Training data dictionary
        trtest: Train/test data dictionary
        i: Algorithm index
        
    Returns:
        Tuple of (optimal_x, fval, localhistory)
    """
    x0 = np.array([tune_val])
    x, fval, localhistory, globalhistory = optimsearch(x0, params, tr, trtest, i)
    return x[0], fval, localhistory


def mainREGcode_ressarch(tune_val: float, tr: Dict, data: Dict, algo: str, params: Dict) -> Dict:
    """
    Train regression model with given algorithm and hyperparameter.
    
    Args:
        tune_val: Hyperparameter tuning value
        tr: Training data dict with 'x' (features) and 'y' (targets)
        data: Data dict with 'x' (features) and 'y' (targets) - train_cell, vld, or test
        algo: Algorithm name ('gp', 'svr', 'libsvr', 'knn', 'btree', 'lin', 'quad', 'bnet', 'elm', 'ransac')
        params: Parameters dictionary
        
    Returns:
        Dict with keys:
            - 'yhat': Predicted y values
            - 'model': Trained model object (algorithm-specific)
            - Other algorithm-specific outputs
            
    Supported algorithms:
        - 'gp': Gaussian Process Regression
        - 'svr': Support Vector Regression
        - 'libsvr': libSVM wrapper
        - 'knn': k-Nearest Neighbors
        - 'btree': Bagged Decision Trees
        - 'lin': Linear Regression
        - 'quad': Quadratic Regression
        - 'bnet': Bayesian Neural Network
        - 'elm': Extreme Learning Machine
        - 'ransac': RANSAC Regression
    """
    # TODO: Implement based on your MATLAB mainREGcode_ressarch.m
    pass


def truthdata(data: Dict, y_predicted: np.ndarray, params: Dict, rawdata: Dict, 
              obstrain: List, dstep: int) -> Dict:
    """
    Calculate performance metrics comparing predictions against ground truth.
    
    Args:
        data: Data dict with 'y' (actual targets)
        y_predicted: Array of predicted values
        params: Parameters dictionary
        rawdata: Raw data with ground truth anomaly information
        obstrain: Training residuals/observations
        dstep: Detection step (prediction horizon)
        
    Returns:
        Dict with observation/detection metrics:
            - 'event': Boolean array indicating anomaly events
            - 'data': Time series or residual data
            - 'stats': Performance statistics
    """
    # TODO: Implement based on your MATLAB truthdata.m
    pass


def lds_timeseries(params: Dict, order: int, data: List, additional_arg, flag: bool) -> Dict:
    """
    Learn Linear Dynamical System (LDS) parameters from time series data.
    
    Args:
        params: Parameters dictionary with LDS settings (nmin, nmax, asos, klim, inittype)
        order: Model order/state dimension
        data: Training residual/observation data
        additional_arg: Not used (kept for MATLAB compatibility)
        flag: Boolean flag (typically True for learning)
        
    Returns:
        Dict with LDS model parameters:
            - 'A': State transition matrix
            - 'C': Observation matrix
            - 'Q': State noise covariance
            - 'R': Observation noise covariance
            - 'x0': Initial state
            - Other model-specific parameters
    """
    # TODO: Implement LDS learning based on your MATLAB lds_timeseries.m
    # Typically uses algorithms like EM (Expectation-Maximization) or subspace identification
    pass


def leveltune(obs_data: Dict) -> Tuple[float, Dict]:
    """
    Optimize detection threshold/level to maximize AUC against ground truth.
    
    Args:
        obs_data: Observation data dict containing:
            - 'event': Boolean array of true anomaly events
            - 'data': Detection statistics/scores
            - 'threshold_range': Optional range to search
            
    Returns:
        Tuple of (auc_value, rocdata_dict) where:
            - auc_value: Area Under Curve (AUC) at optimal threshold
            - rocdata_dict: ROC curve data with keys:
                - 'thresh': Threshold values tested
                - 'tpr': True positive rates
                - 'fpr': False positive rates
                - 'pmd': Probability of missed detection
                - 'fprate': False positive rate
                - 'avg_stats': Average statistics
    """
    # TODO: Implement threshold optimization based on your MATLAB leveltune.m
    pass


def detectioncall(obsval: Dict, obstest: Dict, params: Dict, detect_idx: int, 
                  model_params: Dict = None, lds_params: List = None) -> Tuple[Dict, Dict]:
    """
    Run specified detection algorithm on validation and test data.
    
    Args:
        obsval: Validation observation data
        obstest: Test observation data  
        params: Parameters dictionary with detection config
        detect_idx: Detection method index (1-7):
            1 = Redline - Training
            2 = Redline - Validation
            3 = Predictive - Training
            4 = Predictive - Validation
            5 = Optimal - Training
            6 = Optimal - Validation
            7 = SPRT (Sequential Probability Ratio Test)
        model_params: Optimal model parameters (for detection methods 5-6), optional
        lds_params: LDS model parameters (for prediction-based methods), optional
        
    Returns:
        Tuple of (snew, params) where:
            - snew: Detection results dict with fields like 'tptest', 'fptest', 'tdsamp', etc.
            - params: Updated parameters dict
    """
    # TODO: Implement based on your MATLAB detectioncall.m
    # Different detection algorithms available depending on detect_idx
    pass


def mergeParams(s: Dict, snew: Dict) -> Dict:
    """Merge two parameter dictionaries"""
    result = s.copy()
    result.update(snew)
    return result
