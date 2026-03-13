import numpy as np

def run(mat, params, regtypes, opttypes, detectypelatex, detectIdx, algoIdx):
    """
    Generates and prints a LaTeX table string from the processed matrices.
    """
    mat = np.atleast_2d(np.array(mat)) 
    rows, cols = mat.shape
    
    # Use raw strings r"..." to prevent Python from interpreting backslashes
    tablestr = r"\begin{tabular}{||l|l|"
    headerstr = r"\multicolumn{2}{||c|}{\textbf{Detection Method}}"
    
    for j in range(cols):
        tablestr += "c|"
        headerstr += r"&\textbf{" + str(regtypes[algoIdx[j]]) + r"}"
        
    tablestr += r"}\hline"
    headerstr += r"\\ \hline"
    
    active_detection_labels = [detectypelatex[idx] for idx in detectIdx]
    sprtIdx = any("SPRT" in d for d in active_detection_labels)
    
    k = 0 # 0-based indexing for params.detection.optIdx iterator
    rowstr = [""] * rows
    
    for i in range(rows):
        current_detect = detectypelatex[detectIdx[i]]
        
        if current_detect == 'Standard Exceedance':
            rowstr[i] = r"\multicolumn{2}{||c|}{Standard Exceedance}"
            
        elif current_detect == 'Redline':
            rowstr[i] = r"\multicolumn{2}{||c|}{Redline}"
            
        elif current_detect == 'Predictive (Numerical Integration)':
            if 'Predictive (Monte Carlo Simulation)' in active_detection_labels:
                rowstr[i] = r"\multirow{2}{*}{Predictive}&Numerical Integration"
            else:
                rowstr[i] = r"Predictive&Numerical Integration"
                
        elif current_detect == 'Predictive (Monte Carlo Simulation)':
            if 'Predictive (Numerical Integration)' in active_detection_labels:
                rowstr[i] = r"&Monte Carlo Simulation"
            else:
                rowstr[i] = r"Predictive&Monte Carlo Simulation"
                
        elif current_detect == 'Optimal (Numerical Integration)':
            if 'Optimal (Monte Carlo Simulation)' in active_detection_labels:
                rowstr[i] = r"\multirow{2}{*}{Optimal}&Numerical Integration"
            else:
                rowstr[i] = r"Optimal&Numerical Integration"
                
        elif current_detect == 'Optimal (Monte Carlo Simulation)':
            if 'Optimal (Numerical Integration)' in active_detection_labels:
                rowstr[i] = r"&Monte Carlo Simulation"
            else:
                rowstr[i] = r"Optimal&Monte Carlo Simulation"
                
        elif "SPRT" in current_detect:
            params_detection = getattr(params, 'detection', params)
            opt_val = opttypes[params_detection.optIdx[k]]
            
            if k > 0:
                rowstr[i] = r"&" + str(opt_val)
            else:
                rowstr[i] = r"\multirow{5}{*}{SPRT}&" + str(opt_val)
            k += 1
            
        for j in range(cols):
            if j < cols - 1:
                rowstr[i] += f"&{mat[i, j]:.4f}"
            else:
                algo_len = len(getattr(params, 'algo', [])) + 2 
                cline_str = r"\\ \cline{2-" + str(algo_len) + r"}"
                
                if r"\multirow" in rowstr[i]:
                    rowstr[i] += f"&{mat[i, j]:.4f}" + cline_str
                else:
                    if "SPRT" in current_detect:
                        # Check if row starts with '&' and is not the last row
                        if rowstr[i].startswith("&") and i < rows - 1:
                            rowstr[i] += f"&{mat[i, j]:.4f}" + cline_str
                        else:
                            rowstr[i] += f"&{mat[i, j]:.4f}" + r"\\ \hline"
                    else:
                        rowstr[i] += f"&{mat[i, j]:.4f}" + r"\\ \hline"
                        
    # Output the final printed LaTeX table 
    print(tablestr)
    print(headerstr)
    for r in rowstr:
        print(r)
    print(r"\end{tabular}")
    print(r"\end{center}")
    print(r"\normalsize")
    print(r"\end{table}")
    print("\n")