import numpy as np
from regressopt import mainREGcode_ressarch

class Struct:
    """Lightweight class for dot-notation, matching the rest of the framework."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def modelopttest(x: float, params, algIdx: int, tr, tst) -> float:
    """
    Evaluates a hyperparameter tuning value (x) using K-Fold Cross Validation.
    Matches the data-splicing and normalization logic of the original MATLAB modelopttest.m.
    """
    nrmse = 0.0

    # Extract the full continuous array from the batch wrapper
    full_x = tst.x[0]
    full_y = tst.y[0]
    total_samples = len(full_y)

    # 1. Handle Chunking (File lengths)
    # The MATLAB code specifically splits cross-validation folds based on the original 
    # file lengths to prevent data leakage across contiguous time-series datasets.
    if hasattr(params, 'filelength') and params.filelength is not None and len(params.filelength) > 0:
        file_lengths = np.array(params.filelength).flatten()
        num_folds = len(file_lengths)
    else:
        # Fallback: If no file lengths are provided, evaluate on the whole set
        file_lengths = np.array([total_samples])
        num_folds = 1

    # Calculate exact start and end indices for each fold
    fold_ends = np.cumsum(file_lengths)
    fold_starts = np.concatenate(([0], fold_ends[:-1]))

    # 2. Cross-Validation Loop
    for i in range(num_folds):
        start_idx = fold_starts[i]
        end_idx = fold_ends[i]

        # Create a boolean mask to isolate the current fold as the Test set
        test_mask = np.zeros(total_samples, dtype=bool)
        test_mask[start_idx:end_idx] = True

        # Invert the mask to use all other folds as the Training set
        train_mask = ~test_mask
        
        if num_folds == 1:
            train_mask = test_mask  # Prevent empty training arrays if fallback is triggered

        # Build the trpart and tstpart objects
        trpart = Struct()
        trpart.x = full_x[train_mask]
        trpart.y = full_y[train_mask]

        tstpart = Struct()
        # tstpart expects a list of batches, so we wrap the array in []
        tstpart.x = [full_x[test_mask]]  
        tstpart.y = [full_y[test_mask]]

        # 3. Call the Execution Engine
        # We must wrap the algorithm string in a list so mainREGcode parses it correctly
        algo_list = [params.algo[algIdx]]
        
        output, _ = mainREGcode_ressarch(x, trpart, tstpart, algo_list, params)

        # Calculate Sum of Squared Errors for this fold
        y_true = np.squeeze(tstpart.y[0])
        y_pred = np.squeeze(output.yhat[0])

        residuals = y_true - y_pred

        if len(residuals) == 0 or np.any(np.isnan(residuals)):
            print(f"Warning: NaNs found in predictions during cross-validation fold {i+1}")

        sse = np.sum(residuals ** 2)
        nrmse += sse

    # 4. Error Normalization
    # Replicating MATLAB's denominator: ((sum(filelength) - length(filelength)) * var(tst))
    if num_folds > 1:
        degrees_of_freedom = total_samples - num_folds
    else:
        degrees_of_freedom = total_samples

    # MATLAB's var() function normalizes by N-1 by default. 
    # We force numpy to do the same by setting ddof=1 (Delta Degrees of Freedom)
    variance = np.var(full_y, ddof=1)

    # Prevent division by zero
    if degrees_of_freedom == 0 or variance == 0:
        return float('inf')

    # Return the normalized error score
    out = nrmse / (degrees_of_freedom * variance)
    
    return out