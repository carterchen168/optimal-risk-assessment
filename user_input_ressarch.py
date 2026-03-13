import os
import glob
import random
import sys
import importlib
import tkinter as tk
import pandas as pd
from tkinter import simpledialog
from user_input_accept import params

# first-pass replacement for User_input_ressarch_generic.m MATLAB file

def listdlg(list_string, title="Select an option", single_selection=True):
    root = tk.Tk()
    root.withdraw()
    top = tk.Toplevel(root)
    top.title(title)

    mode = tk.SINGLE if single_selection else tk.MULTIPLE
    listbox = tk.Listbox(top, selectmode=mode, width=80, height=15)
    listbox.pack(padx=10, pady=10)

    for item in list_string:
        listbox.insert(tk.END, item)

    selected_indices = []
    def on_select():
        nonlocal selected_indices
        selected_indices = listbox.curselection()
        top.destroy()

    tk.Button(top, text="OK", command=on_select).pack(pady=5)
    root.wait_window(top)
    root.destroy()
    
    if single_selection:
        return selected_indices[0] if selected_indices else None
    return list(selected_indices)

params.anomaly_typeidx = listdlg(params.anomaly_type, title='Pick an anomaly candidate to analyze', single_selection=True)

params.fcnpath = params.data_path
params.nompath = os.path.join(params.data_path, 'Training')
params.valpath = os.path.join(params.data_path, 'Validation')
params.testpath = os.path.join(params.data_path, 'Testing')

# -------------------------------------------------------------------------

params_file = os.path.join(os.environ.get('DATA_DIR', ''), 'params.txt')
with open(params_file, 'r') as fid:
    allparams = fid.read().splitlines()
allparams = [line.replace('"', '') for line in allparams]

# -------------------------------------------------------------------------
params.anomtype = params.anomalytype[params.anomalytypeidx]

if not hasattr(params, 'valpath'):
    print('Currently in test only mode, no validation directory has been set')

params.targetNameIdx = listdlg(allparams, title='Pick a target parameter to be predicted', single_selection=True)
params.targetName = allparams[params.targetNameIdx]

# Look for BOTH .csv and .mat files
valid_exts = ('.mat', '.csv')
data_files = [f for f in os.listdir(params.nompath) if f.endswith(valid_exts) and os.path.isfile(os.path.join(params.nompath, f))]

selected_file = random.choice(data_files)
selected_file_path = os.path.join(params.nompath, selected_file)

os.chdir(params.fcnpath)
if params.fcnpath not in sys.path:
    sys.path.insert(0, params.fcnpath)

header, m, message = None, None, None

if selected_file_path.endswith('.csv'):
    df = pd.read_csv(selected_file_path)
    header = df.columns.tolist()
    m = df.values # Converts the dataframe into a 2D numpy array
    message = "CSV loaded successfully."
else:
    # Find the matching function and execute it for .mat files
    for j, anomaly in enumerate(params.anomalytype):
        if params.anomtype == anomaly:
            func_name = params.loadfcn[j]
            if func_name in globals():
                header, m, message = globals()[func_name](selected_file_path)
            else:
                print(f"Warning: Function {func_name} not found in the global scope.")
            break

# add exceptions here

os.chdir(params.acceptpath)

if not m:
    # Raising an error if message is provided
    raise ValueError(message if message else "Warning: m is empty or not loaded.")

params.header = [h.replace('"', '') for h in header] if isinstance(header, list) else header.replace('"', '')
discreteIdx = listdlg(allparams, title='Pick discrete parameters to be used only for ground truth purposes', single_selection=False)
if discreteIdx is None: discreteIdx = []

# Get continuous parameters (Equivalent to setdiff)
contparams = [p for i, p in enumerate(allparams) if i not in discreteIdx]

params.channelContineous = []
for p in contparams:
    if params in params.header:
        params.channelContineous.append(params.header.index(p))