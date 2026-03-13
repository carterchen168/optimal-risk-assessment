import numpy as np

def get_val(obj, field):
    if obj is None: return np.nan
    if hasattr(obj, field): return getattr(obj, field)
    if isinstance(obj, dict) and field in obj: return obj[field]
    return np.nan

# receiver operating curve

def run(loadedmat_rocarea, rocarea):
    """
    Translates the extraction of metric figures to Python lists/arrays.
    Accepts the rocarea array loaded from the MAT file, and the actively computed rocarea.
    """
    groups = ['redtrain', 'redval', 'predtrain', 'predval', 'opttrain', 'optval', 'sprt']
    metrics = {g: {'Laopt': [], 'recallsamp': [], 'pmdsamp': [], 'fpratesamp': [], 'tdsamp': [], 'fixed': []} for g in groups}

    if not isinstance(loadedmat_rocarea, (list, np.ndarray)):
        loadedmat_rocarea = [loadedmat_rocarea]
    if not isinstance(rocarea, (list, np.ndarray)):
        rocarea = [rocarea]

    for j in range(len(loadedmat_rocarea)):
        loaded_roc = loadedmat_rocarea[j]
        roc = rocarea[j] if j < len(rocarea) else {}

        for g in groups:
            loaded_group = get_val(loaded_roc, g)
            roc_group = get_val(roc, g)
            
            # If the loaded mat file has this subgroup
            if loaded_group is not np.nan:
                metrics[g]['Laopt'].append(get_val(loaded_group, 'Laopt'))
                
                # Check the actively calculated rocarea fields
                recallsamp = get_val(roc_group, 'recallsamp')
                metrics[g]['recallsamp'].append(recallsamp)
                metrics[g]['pmdsamp'].append(get_val(roc_group, 'pmdsamp') if recallsamp is not np.nan else np.nan)
                
                metrics[g]['fpratesamp'].append(get_val(roc_group, 'fpratesamp'))
                metrics[g]['tdsamp'].append(get_val(roc_group, 'tdsamp'))
                
                fixed_val = get_val(roc_group, 'fixed')
                if fixed_val is not np.nan:
                    metrics[g]['fixed'].append(fixed_val)
                    
            if g == 'sprt' and loaded_group is not np.nan:
                fval = get_val(loaded_group, 'fval')
                if fval is not np.nan:
                    modordIdx = np.argmin(fval)

    print("Finished extracting figure parameters!")
    # Return metrics if other scripts need them
    return metrics