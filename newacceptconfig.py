import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog
import user_input_ressarch as user_input
import scipy.io as sio

def run():

    params = user_input.params

    # ressarch.py might already handle the tkinter root and message boxes.
    root = tk.Tk()
    root.withdraw()

    # Collect toolbox names if provided by user_input_ressarch
    toolboxdata = getattr(user_input, 'toolboxdata', [])
    toolbox_names = []
    for td in toolboxdata:
        # td could be an object with attribute Name or a dict
        name = None
        if hasattr(td, 'Name'):
            name = getattr(td, 'Name')
        elif isinstance(td, dict) and 'Name' in td:
            name = td['Name']
        if name is not None:
            toolbox_names.append(name)
    params.toolboxname = toolbox_names

    # Ask distribution choice
    distrib = simpledialog.askinteger(
        "Distribution",
        "1 - Use cluster scheduler\n2 - Use local resources",
        parent=root,
        minvalue=1, maxvalue=2
    )
    if distrib is None:
        distrib = 2
    params.distrib = distrib

    if params.distrib == 1:
        # try change to accept path if present
        acceptpath = getattr(params, 'acceptpath', None)
        if acceptpath:
            try:
                os.chdir(acceptpath)
            except Exception:
                pass

        mcr = simpledialog.askinteger(
            "MCR/Parallel",
            "1 - Use MCR for cluster scheduling\n2 - Use Parallel Computing Toolbox",
            parent=root,
            minvalue=1, maxvalue=2
        )
        if mcr is None:
            mcr = 2
        params.mcr = mcr

        if params.mcr == 1:
            if 'MATLAB Compiler' in toolbox_names:
                messagebox.showinfo("Info", "MATLAB Compiler detected, ACCEPT will use MCR scripts for distributed computing")
            else:
                messagebox.showwarning("Warning", "MATLAB Compiler not detected, ACCEPT will use local resources only")
                params.distrib = 2
        else:
            if 'Parallel Computing Toolbox' in toolbox_names:
                messagebox.showinfo("Info", "Parallel Computing Toolbox detected, ACCEPT will use these resources for distributed computing")
            else:
                messagebox.showwarning("Warning", "Parallel Computing Toolbox not detected, ACCEPT will use local resources only")
                params.distrib = 2
    else:
        params.mcr = 2

    # Run supplementary configuration modules if present
    for modname in ('regress_input', 'aux_input', 'reg_ranges', 'detect_input'):
        try:
            mod = __import__(modname)
            if hasattr(mod, 'run'):
                mod.run()
        except ImportError:
            # missing module is acceptable; continue
            pass

    # Ask for configuration filename and save params
    messagebox.showinfo("Save", "Saving current configuration for future loads...")
    default_name = f"filename-config_{datetime.now().strftime('%d-%b-%Y_%H_%M_%S')}.mat"
    configfilename = simpledialog.askstring(
        "Configuration Filename",
        "Select prefix:",
        initialvalue=default_name,
        parent=root
    )
    if not configfilename:
        configfilename = default_name

    # Ensure datapath exists or fallback to current working dir
    datapath = getattr(params, 'datapath', os.getcwd())
    if not os.path.isdir(datapath):
        try:
            os.makedirs(datapath, exist_ok=True)
        except Exception:
            datapath = os.getcwd()
    params.datapath = datapath

    # store configfilename as list to be compatible with existing code expectations
    params.configfilename = [configfilename]

    save_path = os.path.join(datapath, configfilename)
    save_dict = {'params': params.__dict__ if hasattr(params, '__dict__') else params}
    try:
        sio.savemat(save_path, save_dict)
        messagebox.showinfo("Saved", f"Configuration saved to:\n{save_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save configuration:\n{e}")

    try:
        root.destroy()
    except Exception:
        pass