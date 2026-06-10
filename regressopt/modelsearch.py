import time
import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing, shgo, basinhopping
from regressopt import modelopttest

def _make_time_guard(max_time, base_cb=None):
    """Wraps base_cb and stops solver after max_time seconds by returning True."""
    if max_time is None:
        return base_cb
    start = time.monotonic()
    def cb(*args, **kwargs):
        if base_cb:
            base_cb(*args, **kwargs)
        return (time.monotonic() - start) > max_time
    return cb

def optimsearch(x0, params, tr, trtest, i):
    x0 = np.array(x0, dtype=float)

    localhistory = {'x': [], 'fval': []}
    globalhistory = {}

    eps = 1e-10
    lower_bounds = 10**(np.log10(np.abs(x0) + eps) - 5)
    upper_bounds = 10**(np.log10(np.abs(x0) + eps) + 5)
    bounds = list(zip(lower_bounds, upper_bounds))

    def callback_tracker(xk, *args, **kwargs):
        localhistory['x'].append(xk.copy())
        fval = modelopttest(xk, params, i, tr, trtest)
        localhistory['fval'].append(fval)

    objective = lambda x: modelopttest(x, params, i, tr, trtest)

    opt_idx = getattr(params.regress, 'optIdx', 6)
    max_time = getattr(params.regress, 'maxtime', None)

    if getattr(params, 'distrib', 2) == 1:
        if opt_idx == 1:  # Genetic Algorithm -> Differential Evolution
            res = differential_evolution(
                objective, bounds,
                callback=_make_time_guard(max_time, callback_tracker),
                workers=-1, disp=True,
            )
        else:
            # Fallback for distributed/multistart
            # TODO: scipy shgo has no callback-based stop; maxtime not enforced here
            res = shgo(objective, bounds, options={'disp': False})

    else:
        if opt_idx == 1:  # Global Search
            # TODO: scipy shgo has no callback-based stop; maxtime not enforced here
            res = shgo(objective, bounds, options={'disp': False})

        elif opt_idx == 2:  # Simulated Annealing -> Dual Annealing
            res = dual_annealing(
                objective, bounds,
                callback=_make_time_guard(max_time, callback_tracker),
                x0=x0,
            )

        elif opt_idx == 3:  # Genetic Algorithm -> Differential Evolution
            res = differential_evolution(
                objective, bounds,
                callback=_make_time_guard(max_time, callback_tracker),
                popsize=100,
            )

        elif opt_idx == 4:  # Pattern Search -> Nelder-Mead
            # TODO: scipy minimize/Nelder-Mead callback is monitor-only; maxtime not enforced here
            res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, callback=callback_tracker)

        elif opt_idx == 5:  # Multistart -> Basin-hopping
            res = basinhopping(
                objective, x0,
                minimizer_kwargs={'bounds': bounds},
                callback=_make_time_guard(max_time, callback_tracker),
            )

        else:  # Default local solver (fmincon) -> L-BFGS-B
            # TODO: scipy minimize/L-BFGS-B callback is monitor-only; maxtime not enforced here
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, callback=callback_tracker)

    x = res.x
    fval = res.fun
    globalhistory['exitflag'] = res.success
    globalhistory['output'] = res.message

    return x, fval, localhistory, globalhistory
