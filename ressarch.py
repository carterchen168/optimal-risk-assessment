import warnings
import glob
import os
import tkinter as tk
from tkinter import messagebox
import scipy.io as sio
from datetime import datetime

# Ignore warnings
warnings.filterwarnings('ignore')

root = tk.Tk()
root.withdraw()

# Load user inputs and utilities
import user_input_ressarch as user_input
from utils import list_dialog

params = user_input.params

oldresults = messagebox.askquestion('Question', 'Load old results?')

if oldresults == 'no':
    configStatus = messagebox.askquestion('Question', 'Create new configuration file?')
    
    if configStatus == 'no':
        allmatfiles = glob.glob(os.path.join(params.datapath, '*.mat'))
        resultmatfiles = glob.glob(os.path.join(params.datapath, '*results*.mat'))
        
        # Get only the files that are NOT result files
        config_files = list(set(allmatfiles) - set(resultmatfiles))
        
        if config_files:
            file_names = [os.path.basename(f) for f in config_files]
            configIdx = list_dialog(file_names, 'Pick one of the following previously saved configurations:', multiple=False)
            
            try:
                import extractacceptmetadata
                extractacceptmetadata.run()
            except ImportError:
                pass
                
            print('Loading old configuration...')
            selected_config = config_files[configIdx]
            loaded_config = sio.loadmat(selected_config, struct_as_record=False, squeeze_me=True)
            
            # Reconstruct params from loaded configuration
            if 'params' in loaded_config:
                loaded_params = loaded_config['params']
                if not hasattr(loaded_params, 'regressonly'):
                    params.regressonly = False
            
            if 'mcr' in loaded_config:
                params.mcr = loaded_config['mcr']
                
            newconfigflag = False
            configfilenames = config_files
        else:
            print('No existing configurations, must create a new one...')
            try:
                import newacceptconfig
                newacceptconfig.run()
            except ImportError:
                print("Missing newacceptconfig module.")
            newconfigflag = True
    else:
        try:
            import newacceptconfig
            newacceptconfig.run()
        except ImportError:
            print("Missing newacceptconfig module.")
        newconfigflag = True

    import testoptloop_ressarch
    modelselectdata, rocarea = testoptloop_ressarch.run(params)
    resflag = True
    
    if not hasattr(params, 'datapath') and hasattr(params, 'aiaapath'):
        params.datapath = params.aiaapath

    # Prepare data for saving
    save_dict = {
        'params': params.__dict__ if hasattr(params, '__dict__') else params,
        'modelselectdata': modelselectdata,
        'rocarea': rocarea
    }

    if newconfigflag:
        # Assuming configfilename is defined somewhere in newacceptconfig
        configfilename = getattr(params, 'configfilename', ['default_config'])[0]
        save_path = os.path.join(params.datapath, f"{configfilename}-results.mat")
    else:
        timestamp = datetime.now().strftime("%d-%b-%Y_%H_%M_%S")
        base_name = os.path.basename(configfilenames[configIdx]).replace('.mat', '')
        save_path = os.path.join(params.datapath, f"{base_name}-results-from-{timestamp}.mat")
        
    sio.savemat(save_path, save_dict)
    print(f"Results saved to: {save_path}")

else:
    config_files = glob.glob(os.path.join(params.datapath, '*results*.mat'))
    
    if config_files:
        file_names = [os.path.basename(f) for f in config_files]
        configIdx = list_dialog(file_names, 'Pick one of the following previously saved result files:', multiple=False)
        
        loadedconfig = config_files[configIdx]
        loadedmat = sio.loadmat(loadedconfig, struct_as_record=False, squeeze_me=True)
        
        # Load variables into local scope
        params = loadedmat.get('params', params)
        modelselectdata = loadedmat.get('modelselectdata', None)
        
        if 'rocarea' in loadedmat:
            rocarea = loadedmat['rocarea']
        else:
            rocarea_keys = sorted([k for k in loadedmat.keys() if k.startswith('rocarea')])
            rocarea = [loadedmat[k] for k in rocarea_keys]
            
        resflag = True
    else:
        print('No existing results...')
        resflag = False

if resflag:
    import ressarch_tables
    
    if not hasattr(params, 'regressonly') or not params.regressonly:
        import process_ressarch_figs
        process_ressarch_figs.run()
        
    ressarch_tables.run()
    
    # Check if regression was flagged
    if hasattr(params, 'regress') and getattr(params.regress, 'flag', 0) == 1:
        if hasattr(modelselectdata, 'hyp_param'):
            import plotregressresults
            plotregressresults.run()
            
    if not hasattr(params, 'regressonly') or not params.regressonly:
        import plotdetectresults
        plotdetectresults.run()