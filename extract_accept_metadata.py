import tkinter as tk
from tkinter import messagebox, simpledialog

def extract_accept_metadata(config_data, config_idx, config_files, params):
    # Hide the root tkinter window
    root = tk.Tk()
    root.withdraw()
    
    cont = 'No'
    
    while cont == 'No':
        # Safely extract current config dictionary
        current_config = config_data[config_idx]['params']
        eventname = current_config['anomalytype'][current_config['anomalytypeidx']]
        
        # 1. Determine Distribution Status
        if current_config.get('distrib') == 1:
            distrib_status = 'Using parallel computing (multiprocessing)'
        else:
            distrib_status = 'Using local computing resources'
            
        # 2. Determine Regression Optimization Status
        regressflag = current_config.get('regress', {}).get('globaltoolboxflag', False)
        regress_status = 'Using global optimization for regression' if regressflag else 'Local optimization for regression'
        
        # 3. Determine Detection Optimization Status
        detectflag = current_config.get('detection', {}).get('globaltoolboxflag', False)
        detect_status = 'Using global optimization for detection' if detectflag else 'Local optimization for detection'
        
        # 4. Format Regression and Detection Lists
        regress_types = [f"{i+1}) {algo}" for i, algo in enumerate(current_config.get('algo', []))]
        detect_types = [f"{i+1}) {det}" for i, det in enumerate(current_config.get('detect', []))]
        
        # Build the summary message
        msg = (
            f"Adverse Event: {eventname}\n\n"
            f"{distrib_status}\n\n"
            f"{regress_status}\n"
            f"Selected regression types:\n" + "\n".join(regress_types) + "\n\n"
            f"{detect_status}\n"
            f"Selected detection types:\n" + "\n".join(detect_types) + "\n\n"
            "Correct configuration?"
        )
        
        # Pop up the Yes/No Dialog (Equivalent to questdlg)
        is_correct = messagebox.askyesno("Configuration Check", msg)
        
        if is_correct:
            cont = 'Yes'
        else:
            # Replicate listdlg: Ask user to pick a new configuration
            # Note: simpledialog is basic; for a true dropdown list, you'd build a custom tk.Toplevel window.
            file_list = "\n".join([f"{i}: {f}" for i, f in enumerate(config_files)])
            choice = simpledialog.askinteger(
                "Pick Configuration", 
                f"Pick the index of a previously saved configuration:\n\n{file_list}"
            )
            if choice is not None and 0 <= choice < len(config_files):
                config_idx = choice
                
    return config_idx