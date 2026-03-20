import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing, shgo

# Placeholder for the external objective function expected by the script
def modelopttest(x, params, i, tr, trtest):
    pass # Replace with actual objective logic

def optimsearch(x0, params, tr, trtest, i):
    x0 = np.array(x0, dtype=float)
    
    localhistory = {'x': [], 'fval': []}
    globalhistory = {}
    
    # Calculate bounds based on MATLAB logic: 10^(log10(x0)-5) to 10^(log10(x0)+5)
    # Adding a small epsilon to x0 to avoid log10(0) if x0 contains zeros
    eps = 1e-10
    lower_bounds = 10**(np.log10(np.abs(x0) + eps) - 5)
    upper_bounds = 10**(np.log10(np.abs(x0) + eps) + 5)
    bounds = list(zip(lower_bounds, upper_bounds))

    # Callback function to track history (equivalent to MATLAB OutputFcn)
    def callback_tracker(xk, *args, **kwargs):
        localhistory['x'].append(xk.copy())
        # To record fval in callbacks reliably, we evaluate it or extract from kwargs if available
        # SciPy callbacks vary by solver. This is a generic implementation.
        fval = modelopttest(xk, params, i, tr, trtest)
        localhistory['fval'].append(fval)

    # Lambda for the objective to pass to SciPy
    objective = lambda x: modelopttest(x, params, i, tr, trtest)

    # Mapping the optimization choices based on params['optIdx']
    opt_idx = params.get('optIdx', 6)
    max_time = params.get('maxtime', None) # Note: SciPy handles time limits differently (usually maxiter)
    
    if params.get('distrib') == 1:
        # Python parallel optimization requires setting 'workers' args where available
        if opt_idx == 1: # Genetic Algorithm -> Differential Evolution
            res = differential_evolution(objective, bounds, callback=callback_tracker, workers=-1, disp=True)
        else:
            # Fallback for distributed/multistart
            res = shgo(objective, bounds, options={'disp': False})
            
    else:
        if opt_idx == 1: # Global Search
            res = shgo(objective, bounds, options={'disp': False})
            
        elif opt_idx == 2: # Simulated Annealing -> Dual Annealing
            res = dual_annealing(objective, bounds, callback=callback_tracker, x0=x0)
            
        elif opt_idx == 3: # Genetic Algorithm -> Differential Evolution
            res = differential_evolution(objective, bounds, callback=callback_tracker, popsize=100)
            
        elif opt_idx == 4: # Pattern Search -> Nelder-Mead (closest local derivative-free analog in base SciPy)
            res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, callback=callback_tracker)
            
        elif opt_idx == 5: # Multistart -> Basin-hopping
            res = basinhopping(objective, x0, minimizer_kwargs={'bounds': bounds}, callback=callback_tracker)
            
        else: # Default local solver (fmincon) -> L-BFGS-B or SLSQP
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, callback=callback_tracker)

    # Map SciPy's OptimizeResult back to the expected MATLAB return format
    x = res.x
    fval = res.fun
    globalhistory['exitflag'] = res.success
    globalhistory['output'] = res.message

    return x, fval, localhistory, globalhistory