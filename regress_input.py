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

    # Ensure params.regress sub-structure exists
    if not hasattr(params, 'regress'):
        params.regress = Params()

    # Check for Global Optimization Toolbox
    has_global_toolbox = 'Global Optimization Toolbox' in getattr(params, 'toolboxname', [])

    if has_global_toolbox:
        use_global = simpledialog.askinteger(
            "Global Optimization Toolbox",
            "Global Optimization Toolbox detected, use it for regression optimizations?\n1 - Yes, 2 - No",
            parent=root,
            minvalue=1, maxvalue=2
        )
        params.regress.globaltoolboxflag = (use_global == 1)
    else:
        params.regress.globaltoolboxflag = False

    if params.distrib == 1:
        # Distributed / cluster mode
        if has_global_toolbox and params.regress.globaltoolboxflag:
            opttypes = [
                'Genetic Algorithm',
                'Pattern Search',
                'Multistart',
                'Grid search',
                'No optimization, use fixed point'
            ]
            opt_choice = list_dialog(
                opttypes,
                'Pick optimization method for regression',
                multiple=False
            )
            # list_dialog returns 0-based index
            params.regress.optIdx = opt_choice + 1  # store as 1-based to match MATLAB

            if params.regress.optIdx == 3:
                params.regress.optinitpts = simpledialog.askinteger(
                    "Multistart",
                    "Using Multistart optimization, how many initial starting points should be used for optimizations?",
                    parent=root,
                    minvalue=1
                )

            if params.regress.optIdx == 4:
                params.regress.optIdx = 7
                params.regress.flag = 1
            elif params.regress.optIdx == 5:
                params.regress.flag = 2
            else:
                params.regress.flag = 1
                params.regress.maxtime = simpledialog.askfloat(
                    "Max Time",
                    "Enter the longest run time for optimizations in seconds:",
                    parent=root,
                    minvalue=0.0
                )
        else:
            opttypes = [
                'Grid search',
                'No optimization, use fixed point'
            ]
            if has_global_toolbox:
                print('Global Optimization Toolbox not selected for use, ACCEPT will use either a grid search for optimizing the regression hyperparameters, or you can provide the hyperparameter values directly')
            else:
                print('Global Optimization Toolbox not detected, ACCEPT will use either a grid search for optimizing the regression hyperparameters, or you can provide the hyperparameter values directly')

            opt_choice = list_dialog(
                opttypes,
                'Pick optimization method for regression',
                multiple=False
            )
            params.regress.optIdx = opt_choice + 1

            if params.regress.optIdx == 1:
                params.regress.optIdx = 7
                params.regress.flag = 1
            else:
                params.regress.flag = 2
    else:
        # Local mode
        if has_global_toolbox and params.regress.globaltoolboxflag:
            opttypes = [
                'Global search',
                'Simulated Annealing',
                'Genetic Algorithm',
                'Pattern Search',
                'Multistart',
                'Local search only',
                'Grid search',
                'No optimization, use fixed point'
            ]
            opt_choice = list_dialog(
                opttypes,
                'Pick optimization method for regression',
                multiple=False
            )
            params.regress.optIdx = opt_choice + 1

            if params.regress.optIdx != 6:
                params.regress.maxtime = simpledialog.askfloat(
                    "Max Time",
                    "Enter the longest run time for optimizations in seconds:",
                    parent=root,
                    minvalue=0.0
                )

            if params.regress.optIdx == 5:
                params.regress.optinitpts = simpledialog.askinteger(
                    "Multistart",
                    "Using Multistart optimization, how many initial starting points should be used for optimizations?",
                    parent=root,
                    minvalue=1
                )

            if params.regress.optIdx == 8:
                params.regress.flag = 2
            else:
                params.regress.flag = 1
        else:
            opttypes = [
                'Local search only',
                'Grid search',
                'No optimization, use fixed point'
            ]
            if has_global_toolbox:
                print('Global Optimization Toolbox not selected for use, ACCEPT will use either a grid search for optimizing the regression hyperparameters, or you can provide the hyperparameter values directly')
            else:
                print('Global Optimization Toolbox not detected, ACCEPT will use either a grid search for optimizing the regression hyperparameters, or you can provide the hyperparameter values directly')

            opt_choice = list_dialog(
                opttypes,
                'Pick optimization method for regression',
                multiple=False
            )
            params.regress.optIdx = opt_choice + 1

            if params.regress.optIdx == 2:
                params.regress.optIdx = 7
                params.regress.flag = 1
            elif params.regress.optIdx == 1:
                params.regress.flag = 1
            else:
                params.regress.flag = 2

    try:
        root.destroy()
    except Exception:
        pass