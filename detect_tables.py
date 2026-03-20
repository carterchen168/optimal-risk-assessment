import numpy as np
from create_latex_detect_tables import run

def detect_tables(params, pmdtable, pfatable, tdtable, output_type):
    # Mapping definitions
    opt_types = ['Global search', 'Simulated Annealing', 'Genetic Algorithm', 'Pattern Search', 'Multistart', 'Local search only', 'Grid search', 'No optimization, use fixed point']
    algo_types = ['gp', 'svr', 'libsvr', 'knn', 'btree', 'lin', 'quad', 'bnet', 'elm', 'ransac', 'rbm']
    reg_types = ['GPR', 'SVR', 'SVR', 'k-NN', 'Bagged Decision Trees', 'LR1', 'LR2', 'BNN', 'ELM', 'RANSAC', 'RBM']
    detect_types = ['Redline - Training', 'Redline - Validation', 'Predictive - Training', 'Predictive - Validation', 'Optimal - Training', 'Optimal - Validation', 'SPRT']
    
    # Tool/SPRT check (Assuming params is a dictionary/object)
    if params.get('globaltoolboxflag') and 'Global Optimization Toolbox' in params.get('toolboxname', ''):
        if 'SPRT' not in params['detect']:
            raise ValueError("Configuration obsolete: Global optimization selected without SPRT as a detection method")
        else:
            sprt_idx = params['detect'].index('SPRT')
            sprt_detection_types = ['SPRT - Global search', 'SPRT - Simulated Annealing', 'SPRT - Genetic Algorithm', 'SPRT - Pattern Search', 'SPRT - Multistart']
            
            # Update detect lists based on optimization indices
            params['detect'][sprt_idx:sprt_idx+len(params['optIdx'])] = [sprt_detection_types[i] for i in params['optIdx']]
            detect_types = detect_types[:-1] + sprt_detection_types
            
            if 'dstepmin' not in params:
                params['dstepmin'] = 1
                params['dstepmax'] = float(input("What is the maximum design (and validation) prediction horizon ?: "))
                params['nmin'] = 2
                params['nmax'] = float(input("Maximum state order = : "))
                params['inittype'] = 2

    # Output generation
    algo_header = "\t".join([f"{a:>8}" for a in params['algo']])
    
    if output_type == 2:
        def print_table(title, table):
            print(f"{title}...")
            print(f"{'':>25}\t{algo_header}")
            for i in range(table.shape[0]):
                row_str = ""
                for j in range(table.shape[1]):
                    val = table[i, j]
                    if np.isnan(val):
                        row_str += f"{'-':>8}\t"
                    else:
                        row_str += f"{val:8.4f}\t"
                print(f"{params['detect'][i]:>25}\t{row_str}")
            print("\n")

        print_table("Missed detection results", pmdtable)
        print_table("False alarm results", pfatable)
        print_table("Detection time results", tdtable)
    else:
        print("LaTeX table generation would go here (requires porting create_latex_detect_tables)")
        # TODO: add create_latex_detect_tables functionality here with run()
        # TODO: research possible latex libraries for better conversion

    # Find absolute best or rank top performers
    # np.argmin/argmax flattens the array, so we use unravel_index to get 2D coordinates
    pmd_idx = np.unravel_index(np.nanargmin(pmdtable), pmdtable.shape)
    pfa_idx = np.unravel_index(np.nanargmin(pfatable), pfatable.shape)
    td_idx = np.unravel_index(np.nanargmax(tdtable), tdtable.shape)

    # Check if there's a unanimous best performer
    if pmd_idx == pfa_idx == td_idx:
        det_best, reg_best = pmd_idx
        print(f"{params['detect'][det_best]} is the best performing detection method, yielding a missed detection rate of {pmdtable[det_best, reg_best]*100:.2f} %, a false alarm rate of {pfatable[det_best, reg_best]*100:.2f} %, and an early detection time of {tdtable[det_best, reg_best]:.4f} sec, using regression method {params['algo'][reg_best]}")
    else:
        print("No clear best detection method")
        num_top = int(input('Enter the approx. number of top performers to compare: '))
        
        # Flatten and rank arrays (handling NaNs by putting them at the end)
        flat_pmd = pmdtable.flatten()
        flat_pfa = pfatable.flatten()
        flat_td = -tdtable.flatten() # Negative for descending sort
        
        # Argsort gives us the flattened indices in ranked order
        rank_pmd = np.argsort(flat_pmd)
        rank_pfa = np.argsort(flat_pfa)
        rank_td = np.argsort(flat_td)
        
        # Create a combined score based on average rank position
        combined_scores = {}
        for flat_idx in range(len(flat_pmd)):
            rank_p = np.where(rank_pmd == flat_idx)[0][0]
            rank_f = np.where(rank_pfa == flat_idx)[0][0]
            rank_t = np.where(rank_td == flat_idx)[0][0]
            combined_scores[flat_idx] = np.mean([rank_p, rank_f, rank_t])
            
        # Sort by best combined rank
        sorted_indices = sorted(combined_scores, key=combined_scores.get)
        
        print('The best regression/detection combinations are as follows (in ranked order)...')
        for idx in sorted_indices[:num_top]:
            det_idx, reg_idx = np.unravel_index(idx, pmdtable.shape)
            print(f"Regression method {params['algo'][reg_idx]} and detection method {params['detect'][det_idx]} yields a missed detection rate of {pmdtable[det_idx, reg_idx]*100:.2f} %, a false alarm rate of {pfatable[det_idx, reg_idx]*100:.2f} %, and a time of first alarm appearing {tdtable[det_idx, reg_idx]:.4f} sec before the event occured")