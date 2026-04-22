import numpy as np
import importlib

class Struct:
    """Lightweight class for dot-notation, matching the rest of the framework."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def truthdata(vld, yval_predicted, params, rawdata_val, obstrain, dstep: int):
    """
    Calculates observation residuals and builds the ground truth 'event' arrays
    based on a dynamic prediction horizon (dstep).
    
    Equivalent to truthdata.m
    """
    # Handle dict vs object accessing for vld.y
    vld_y = vld.y if hasattr(vld, 'y') else vld['y']
    n_batches = len(vld_y)
    
    # Pre-allocate obsval array. 
    # MATLAB structure: indices 0 to N-1 are val/test, indices N to 2N-1 are training.
    obsval = [Struct() for _ in range(2 * n_batches)]
    
    for i in range(n_batches):
        # 1. Calculate Validation/Test Residuals
        # MATLAB: obsval(i).data = (vld.y{i} - yvalPredicted{i});
        y_true = np.asarray(vld_y[i])
        y_pred = np.asarray(yval_predicted[i])
        obsval[i].data = y_true - y_pred
        
        # 2. Extract Training Residuals
        # MATLAB: obsval(i+length(vld.y)).data = obstrain(i).data';
        train_data = obstrain[i].data if hasattr(obstrain[i], 'data') else obstrain[i]['data']
        obsval[i + n_batches].data = np.asarray(train_data).T
        
        # 3. Dynamic Truth Function Evaluation
        # MATLAB uses eval(['subevent = ' params.truthfcn{j} '(...)'])
        subevent = np.zeros(len(obsval[i].data), dtype=bool) # Default fallback
        
        if hasattr(params, 'anomalytype') and hasattr(params, 'anomtype'):
            anomaly_types = params.anomalytype if isinstance(params.anomalytype, list) else [params.anomalytype]
            truth_fcns = params.truthfcn if isinstance(params.truthfcn, list) else [params.truthfcn]
            
            for j, a_type in enumerate(anomaly_types):
                if params.anomtype == a_type:
                    fcn_name = truth_fcns[j]
                    
                    # Safely dispatch the function call by string name
                    if fcn_name in globals():
                        subevent = globals()[fcn_name](i, params, obsval, rawdata_val)
                    else:
                        print(f"Warning: Truth function '{fcn_name}' not found. Defaulting to nominal.")
        
        # 4. Pad the subevent arrays by the prediction horizon (dstep)
        subevent = np.asarray(subevent).flatten()
        
        # For validation, pad the end with the last known value
        last_val = subevent[-1] if len(subevent) > 0 else 0
        pad_val = np.full(dstep, last_val)
        obsval[i].subevent = np.concatenate((subevent, pad_val))
        
        # For training, pad the end with zeros
        train_len = len(obsval[i + n_batches].data)
        obsval[i + n_batches].subevent = np.zeros(train_len + dstep)
        
        # 5. Look-ahead Event Calculation (Sliding Window)
        # If ANY subevent occurs within the next 'dstep' steps, flag the current step as an event.
        n_pts_val = len(obsval[i].data)
        obsval[i].event = np.zeros((n_pts_val, 1), dtype=bool)
        
        for k in range(n_pts_val):
            # Python slicing is exclusive at the end, so k+1 to k+1+dstep maps to MATLAB's k+1:k+dstep
            window = obsval[i].subevent[k+1 : k+1+dstep]
            obsval[i].event[k, 0] = np.any(window)
            
        n_pts_train = len(obsval[i + n_batches].data)
        obsval[i + n_batches].event = np.zeros((n_pts_train, 1), dtype=bool)
        
        for k in range(n_pts_train):
            window = obsval[i + n_batches].subevent[k+1 : k+1+dstep]
            obsval[i + n_batches].event[k, 0] = np.any(window)

    return obsval