import tkinter as tk
from tkinter import simpledialog
import user_input_ressarch as user_input
from utils import list_dialog


class Params:
    pass


def run():

    params = user_input.params

    root = tk.Tk()
    root.withdraw()

    # Ensure params.detection sub-structure exists
    if not hasattr(params, 'detection'):
        params.detection = Params()

    # Retrieve detectionIdx set by aux_input
    detectionIdx = getattr(params, '_detectionIdx', [])

    if any(idx > 6 for idx in detectionIdx):
        # Check for Global Optimization Toolbox
        has_global_toolbox = 'Global Optimization Toolbox' in getattr(params, 'toolboxname', [])

        if has_global_toolbox:
            use_global = simpledialog.askinteger(
                "Global Optimization Toolbox",
                "Global Optimization Toolbox detected, use it for SPRT optimizations?\n1 - Yes, 2 - No",
                parent=root,
                minvalue=1, maxvalue=2
            )
            params.detection.globaltoolboxflag = (use_global == 1)
        else:
            params.detection.globaltoolboxflag = False

        if has_global_toolbox and params.detection.globaltoolboxflag:
            opttypes = [
                'Global search',
                'Simulated Annealing',
                'Genetic Algorithm',
                'Pattern Search',
                'Multistart'
            ]
            opt_selection = list_dialog(
                opttypes,
                'Pick optimization method(s) for detection',
                multiple=True
            )
            # Store as 1-based indices to match MATLAB convention
            params.detection.optIdx = [i + 1 for i in opt_selection]

            if any(idx == 5 for idx in params.detection.optIdx):
                params.detection.optinitpts = simpledialog.askinteger(
                    "Multistart",
                    "Using Multistart optimization, how many initial starting points should be used for optimizations?",
                    parent=root,
                    minvalue=1
                )

            params.detection.maxtime = simpledialog.askfloat(
                "Max Time",
                "Enter the longest run time for optimizations in seconds:",
                parent=root,
                minvalue=0.0
            )
        else:
            if has_global_toolbox:
                print('Global Optimization Toolbox not selected for use')
            else:
                print('Global Optimization Toolbox not detected')
    else:
        params.detection.globaltoolboxflag = False
        params.detection.optIdx = 1

    try:
        root.destroy()
    except Exception:
        pass