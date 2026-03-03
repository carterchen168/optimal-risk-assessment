import warnings
import glob
import os
import numpy as np
import scipy.io as sio

warnings.filterwarnings('ignore')

import user_input_accept

from utils import list_dialog 

params = user_input_accept.params

# Find result files
search_path = os.path.join(params.datapath, '*-results*.mat')
config_files = glob.glob(search_path)

if config_files:
    # Extract filenames for the UI dialog
    file_names = [os.path.basename(f) for f in config_files]
    configIdx = list_dialog(file_names, 'Pick previously saved result files to combine:', multiple=True)
    
    # Initialize dictionaries/lists to hold aggregated data across the loop
    ptest = {}
    ntest = {}
    rocarea_aggregated = {}
    
    for i, idx in enumerate(configIdx):
        loadedconfig = config_files[idx]
        
        # scipy.io.loadmat
        loadedmat = sio.loadmat(loadedconfig, struct_as_record=False, squeeze_me=True)
        modelselectdata = loadedmat['modelselectdata']
        
        if i == 0:
            for j, param_set in enumerate(modelselectdata.params):
                dstepIdx = np.argmax(param_set.auctemp)
                
                obstest_data = modelselectdata.obstest[0][dstepIdx]
                for l, obs in enumerate(obstest_data):
                    ptest[(l, j)] = np.sum(obs.event)
                    ntest[(l, j)] = len(obs.data) - ptest[(l, j)]
        
        # Gather rocarea dynamically
        rocarea_list = []
        if 'rocarea' not in loadedmat:
            rocarea_keys = [k for k in loadedmat.keys() if k.startswith('rocarea')]
            for key in sorted(rocarea_keys):
                rocarea_list.append(loadedmat[key])
        else:
            rocarea_list = np.atleast_1d(loadedmat['rocarea'])
            
        # Extract and aggregate data fields
        for j, roc in enumerate(rocarea_list):
            if j not in rocarea_aggregated:
                rocarea_aggregated[j] = {
                    'redtrain': {'tptest': [], 'fptest': [], 'tdsampraw': []},
                    'redval': {'tptest': [], 'fptest': [], 'tdsampraw': []},
                    'predtrain': {'tptest': [], 'fptest': [], 'tdsampraw': []},
                    'predval': {'tptest': [], 'fptest': [], 'tdsampraw': []},
                    'opttrain': {'tptest': [], 'fptest': [], 'tdsampraw': []},
                    'optval': {'tptest': [], 'fptest': [], 'tdsampraw': []},
                    'sprt': {'tptest': [], 'fptest': [], 'tdsampraw': []}
                }
            
            def extract_metrics(roc_obj, field_name):
                if hasattr(roc_obj, field_name):
                    field_data = getattr(roc_obj, field_name)
                    # Get sum of ptest and ntest for this specific 'j' column
                    p_sum = sum(v for k, v in ptest.items() if k[1] == j)
                    n_sum = sum(v for k, v in ntest.items() if k[1] == j)
                    
                    tptest_val = field_data.recallsamp * p_sum
                    fptest_val = field_data.fpratesamp * n_sum
                    
                    rocarea_aggregated[j][field_name]['tptest'].append(tptest_val)
                    rocarea_aggregated[j][field_name]['fptest'].append(fptest_val)
                    rocarea_aggregated[j][field_name]['tdsampraw'].append(field_data.tdsamp)

            # Apply extraction across all potential fields
            fields_to_check = ['redtrain', 'redval', 'predtrain', 'predval', 'opttrain', 'optval', 'sprt']
            for field in fields_to_check:
                extract_metrics(roc, field)
                
        print(f"Finished loading mat file # {i + 1} of {len(configIdx)}")

    # Calculate final averages/sums
    rocarea = []
    num_configs = len(configIdx)
    for j in range(len(rocarea_list)):
        roc_final = {}
        p_sum = sum(v for k, v in ptest.items() if k[1] == j)
        n_sum = sum(v for k, v in ntest.items() if k[1] == j)
        
        for field, metrics in rocarea_aggregated[j].items():
            if metrics['tptest']: # If data was appended
                roc_final[field] = {
                    'recallsamp': sum(metrics['tptest']) / (p_sum * num_configs) if p_sum else 0,
                    'fpratesamp': sum(metrics['fptest']) / (n_sum * num_configs) if n_sum else 0,
                    'tdsamp': np.mean(metrics['tdsampraw']),
                }
                roc_final[field]['pmdsamp'] = 1 - roc_final[field]['recallsamp']
        rocarea.append(roc_final)
        
    resflag = True
else:
    print('No existing results...')
    resflag = False

if resflag:
    import combine_ressarch_figs
    import combine_ressarch_tables
    
    combine_ressarch_figs.run(rocarea)
    combine_ressarch_tables.run(rocarea)

    