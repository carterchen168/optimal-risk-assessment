import numpy as np
from scipy import stats as sp_stats
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _argmin_2d(arr: np.ndarray):
    """Global argmin returning (row, col) — mirrors MATLAB argmin on matrices."""
    return np.unravel_index(np.nanargmin(arr), arr.shape)


def _argmax_2d(arr: np.ndarray):
    """Global argmax returning (row, col) — mirrors MATLAB argmax on matrices."""
    return np.unravel_index(np.nanargmax(arr), arr.shape)


def _argmin_1d(arr: np.ndarray) -> int:
    """Column-wise argmin (0-indexed)."""
    return int(np.nanargmin(arr))


def _argmax_1d(arr: np.ndarray) -> int:
    """Column-wise argmax (0-indexed)."""
    return int(np.nanargmax(arr))


def _intersect_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Rows present in both arrays (MATLAB intersect(...,'rows')).
    Preserves the order from `a`.
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    if a.size == 0 or b.size == 0:
        cols = a.shape[1] if a.ndim > 1 else 0
        return np.empty((0, cols), dtype=int)
    set_b = set(map(tuple, b.astype(int)))
    result = [row for row in a.astype(int) if tuple(row) in set_b]
    return np.array(result, dtype=int) if result else np.empty((0, a.shape[1]), dtype=int)


def _sortrows(arr: np.ndarray, col: int) -> np.ndarray:
    """
    Sort 2-D array by column `col` (1-indexed, negative = descending),
    mirroring MATLAB sortrows().
    """
    ascending = col > 0
    col_idx   = abs(col) - 1
    order     = np.argsort(arr[:, col_idx], kind='stable')
    return arr[order] if ascending else arr[order[::-1]]


def _nan_cell_fmt(row: np.ndarray):
    """
    Return a list of formatted strings, replacing NaN values with '-'.
    Mirrors the MATLAB NaN-handling logic inside each table-printing loop.
    """
    cells = []
    for v in row:
        cells.append("-" if np.isnan(v) else f"{v:.4f}")
    return cells


# ---------------------------------------------------------------------------
# LaTeX table builders
# ---------------------------------------------------------------------------

_OPT_TYPES = [
    "Global search",
    "Simulated Annealing",
    "Genetic Algorithm",
    "Pattern Search",
    "Multistart",
    "Local search only",
    "Grid search",
    "No optimization, use fixed point",
]
_ALGO_TYPES   = ["gp", "svr", "svrm", "knn", "btree", "lin", "quad", "bnet", "elm"]
_REG_TYPES    = ["GPR", "SVR", "Multivariate SVR", "k-NN",
                 "Bagged Decision Trees", "LR1", "LR2", "BNN", "ELM"]
_DET_TYPES    = [
    "Redline - Training", "Redline - Validation",
    "Predictive - Training", "Predictive - Validation",
    "Optimal - Training",  "Optimal - Validation",
    "SPRT",
]
_DET_LATEX    = [
    "Standard Exceedance", "Redline",
    "Predictive (Numerical Integration)", "Predictive (Monte Carlo Simulation)",
    "Optimal (Numerical Integration)",   "Optimal (Monte Carlo Simulation)",
    "SPRT",
]


def _build_latex_table(
    table: np.ndarray,
    caption: str,
    label: str,
    detect_latex_names,      # ordered list of LaTeX display names for rows
    reg_names,               # ordered list of regression display names for cols
    has_sprt: bool,
    sprt_opt_type: str,
    n_total_detect: int,
):
    """
    Emit a LaTeX tabular block to stdout, mirroring the MATLAB LaTeX branch.
    """
    n_rows, n_cols = table.shape

    # --- table environment header ---
    if has_sprt:
        tab_spec  = r"\begin{tabular}{||l|l|" + "c|" * n_cols + r"|}\hline"
        hdr       = r"\textbf{SPRT Global Optimization Method}&\textbf{Detection Method}"
    else:
        tab_spec  = r"\begin{tabular}{||l|" + "c|" * n_cols + r"|}\hline"
        hdr       = r"\textbf{Detection Method}"

    for rname in reg_names:
        hdr += rf"&\textbf{{{rname}}}"
    hdr += r"\\\hline"

    print(r"\begin{table}[h!]")
    print(r"\tiny")
    print(r"\begin{center}")
    print(rf"\caption{{{caption}}}\label{{{label}}}")
    print(tab_spec)
    print(hdr)

    for i, det_latex in enumerate(detect_latex_names):
        # Left multirow cell (only for SPRT tables)
        if has_sprt:
            if i == 0:
                prefix = rf"\multirow{{{n_total_detect}}}{{*}}{{{sprt_opt_type}}}&"
            else:
                prefix = "&"
        else:
            prefix = ""

        # Build data columns
        row_parts = [det_latex]
        for j in range(n_cols):
            row_parts.append(f"{table[i, j]:.4f}")

        row_str = prefix + row_parts[0] + "".join(
            "&" + v for v in row_parts[1:]
        )

        # Closing rule
        if i < n_rows - 1 and has_sprt:
            row_str += rf"\\\cline{{2-7}}"
        else:
            row_str += r"\\\hline"

        print(row_str)

    print(r"\end{tabular}")
    print(r"\end{center}")
    print(r"\normalsize")
    print(r"\end{table}")
    print()


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def combine_ressarch_tables(
    loadedmat,
    regressoptval: np.ndarray,
    regresstuneval: np.ndarray,
    redtrain=None,
    redval=None,
    predtrain=None,
    predval=None,
    opttrain=None,
    optval=None,
    sprt=None,
):
    """
    Aggregate and display detection-performance results.

    Parameters
    ----------
    loadedmat : object
        Loaded experiment data. Must expose:
          .params.algo   : list[str]  — regression algorithm names
          .params.detect : list[str]  — detection method names
          .params.detection.optIdx : int  — index into _OPT_TYPES (1-indexed)
          .modelselectdata.tuneval : np.ndarray
    regressoptval : np.ndarray, shape (n_runs, n_algos)
        Optimised regression hyperparameter values per run.
    regresstuneval : np.ndarray, shape (n_runs, n_algos)
        Global tuning values per run.
    redtrain … sprt : object, optional
        Detection result objects with attributes .fpratesamp, .pmdsamp, .tdsamp.
    """
    params = loadedmat.params
    num_algos = len(params.algo)

    # -------------------------------------------------------------------------
    # 1. Assemble tables
    # -------------------------------------------------------------------------
    pfa_rows, pmd_rows, td_rows = [], [], []
    for obj in [redtrain, redval, predtrain, predval, opttrain, optval, sprt]:
        if obj is not None:
            pfa_rows.append(np.atleast_2d(obj.fpratesamp))
            pmd_rows.append(np.atleast_2d(obj.pmdsamp))
            td_rows.append(np.atleast_2d(obj.tdsamp))

    if not pmd_rows:
        print("No detection result objects provided. Nothing to display.")
        return

    pfatable = np.vstack(pfa_rows)
    pmdtable = np.vstack(pmd_rows)
    tdtable  = np.vstack(td_rows)

    n_det, n_alg = pmdtable.shape
    col_width    = max(len(d) for d in params.detect)

    # -------------------------------------------------------------------------
    # 2. Output mode selection
    # -------------------------------------------------------------------------
    output_type = int(input("Format results for: 1 - Latex or 2 - Screen : "))

    # -------------------------------------------------------------------------
    # 3. Regression summary
    # -------------------------------------------------------------------------
    print("Regression results...")
    print(
        f"{'':>{col_width}}\t"
        + "\t".join(f"  {a}" for a in params.algo)
    )

    # Format strings per column
    regress_fmts = []
    std_fmts     = []
    for j, algo in enumerate(params.algo):
        std_fmts.append("{:.4f}")
        tune_j = loadedmat.modelselectdata.tuneval[j]
        if tune_j < 0.001:
            regress_fmts.append("{:.1e}")
        elif j in (1, 4):        # MATLAB indices 2 and 5 → 0-indexed 1 and 4
            regress_fmts.append("{:4d}")
        else:
            regress_fmts.append("{:.4f}")

    # Drop rows where regressoptval is NaN
    nan_rows = np.any(np.isnan(regressoptval), axis=1)
    regressoptval_clean  = regressoptval[~nan_rows]
    regresstuneval_clean = regresstuneval[~nan_rows]

    algo_list = list(params.algo)
    has_svr   = "svr" in algo_list

    if has_svr:
        svr_j = algo_list.index("svr")
        other_j = [j for j in range(n_alg) if j != svr_j]

        svr_tunevals  = regresstuneval_clean[:, svr_j]
        svr_mode      = float(sp_stats.mode(svr_tunevals, keepdims=True).mode[0])
        svr_mode_mask = svr_tunevals == svr_mode

        global_opt_vals = [None] * n_alg
        global_opt_vals[svr_j] = svr_mode
        for j in other_j:
            global_opt_vals[j] = regresstuneval_clean[0, j]

        opt_vals = [None] * n_alg
        opt_vals[svr_j] = float(np.mean(regressoptval_clean[svr_mode_mask, svr_j]))
        for j in other_j:
            opt_vals[j] = regressoptval_clean[0, j]

        tune_str = "\t".join(
            regress_fmts[j].format(global_opt_vals[j]) for j in range(n_alg)
        )
        opt_str = "\t".join(
            std_fmts[j].format(opt_vals[j]) for j in range(n_alg)
        )
        print(f"Global Optimum \t\t{tune_str}")
        print(f"Optimized Values \t{opt_str}")
        print()

        # SVR-specific statistics
        print("SVR Regression Statistics...")
        print("\t\t\t\tMin\tMode\tMax")
        svr_min  = float(np.min(svr_tunevals))
        svr_max  = float(np.max(svr_tunevals))

        svr_optval_min  = float(np.mean(regressoptval_clean[svr_tunevals == svr_min,  svr_j]))
        svr_optval_mode = float(np.mean(regressoptval_clean[svr_mode_mask,             svr_j]))
        svr_optval_max  = float(np.mean(regressoptval_clean[svr_tunevals == svr_max,  svr_j]))

        print(
            f"SVR Global Optimum Statistics \t"
            f"{svr_min:.4f}\t{svr_mode:.4f}\t{svr_max:.4f}"
        )
        print(
            f"SVR Optimized Value Statistics \t"
            f"{svr_optval_min:.4f}\t{svr_optval_mode:.4f}\t{svr_optval_max:.4f}"
        )
        print()
    else:
        tune_str = "\t".join(
            regress_fmts[j].format(regresstuneval_clean[0, j]) for j in range(n_alg)
        )
        opt_str = "\t".join(
            std_fmts[j].format(regressoptval_clean[0, j]) for j in range(n_alg)
        )
        print(f"Global Optimum \t\t{tune_str}")
        print(f"Optimized Values \t{opt_str}")
        print()

    # -------------------------------------------------------------------------
    # 4. Resolve display-name mappings for detection methods
    # -------------------------------------------------------------------------
    algo_idx_map   = {a: i for i, a in enumerate(_ALGO_TYPES)}
    detect_idx_map = {d: i for i, d in enumerate(_DET_TYPES)}

    algo_display_idx   = [algo_idx_map[a]   for a in params.algo]
    detect_display_idx = [detect_idx_map[d] for d in params.detect]

    detect_latex_names = [_DET_LATEX[i] for i in detect_display_idx]
    reg_display_names  = [_REG_TYPES[i] for i in algo_display_idx]

    has_sprt     = "SPRT" in list(params.detect)
    sprt_opt_str = (
        _OPT_TYPES[params.detection.optIdx - 1]  # convert 1-indexed
        if has_sprt and hasattr(params, "detection")
        else ""
    )

    # -------------------------------------------------------------------------
    # 5. Print / emit the three tables
    # -------------------------------------------------------------------------

    def _screen_table(table, title):
        print(title)
        print(
            f"{'':>{col_width}}\t"
            + "\t".join(f"  {a}" for a in params.algo)
        )
        for i, det_latex in enumerate(detect_latex_names):
            if i >= table.shape[0]:
                break
            cells = _nan_cell_fmt(table[i, :])
            print(f"{det_latex:>{col_width}}\t" + "\t".join(cells))
        print()

    if output_type == 2:   # ---- Screen output ----
        _screen_table(pmdtable, "Missed detection results...")
        _screen_table(pfatable, "False alarm results...")
        _screen_table(tdtable,  "Detection time results...")

    else:                  # ---- LaTeX output ----
        _build_latex_table(
            pmdtable, r"$P_{md}$ Test Results", "testtable-pmd",
            detect_latex_names, reg_display_names,
            has_sprt, sprt_opt_str, len(params.detect),
        )
        _build_latex_table(
            pfatable, r"$P_{fa}$ Test Results", "testtable-pfa",
            detect_latex_names, reg_display_names,
            has_sprt, sprt_opt_str, len(params.detect),
        )
        _build_latex_table(
            tdtable, r"$T_d$ Test Results", "testtable-td",
            detect_latex_names, reg_display_names,
            has_sprt, sprt_opt_str, len(params.detect),
        )

    # -------------------------------------------------------------------------
    # 6. Per-metric best pairings
    # -------------------------------------------------------------------------
    pfa_pairs_r, pfa_pairs_c = np.where(pfatable == np.nanmin(pfatable))
    pmd_pairs_r, pmd_pairs_c = np.where(pmdtable == np.nanmin(pmdtable))
    td_pairs_r,  td_pairs_c  = np.where(tdtable  == np.nanmax(tdtable))

    print("Best Detection/Regression Pairings for False alarm rate")
    print("_" * 55)
    for r, c in zip(pfa_pairs_r, pfa_pairs_c):
        print(
            f"{params.detect[r]} and {params.algo[c]} "
            f"yield a false alarm rate of {pfatable[r, c] * 100:.4f} %"
        )
    print()

    print("Best Detection/Regression Pairings for Missed detection rate")
    print("_" * 60)
    for r, c in zip(pmd_pairs_r, pmd_pairs_c):
        print(
            f"{params.detect[r]} and {params.algo[c]} "
            f"yield a missed detection rate of {pmdtable[r, c] * 100:.4f} %"
        )
    print()

    print("Best Detection/Regression Pairings for Early detection time")
    print("_" * 61)
    for r, c in zip(td_pairs_r, td_pairs_c):
        print(
            f"{params.detect[r]} and {params.algo[c]} "
            f"yield an early detection time of {tdtable[r, c]:.4f} sec"
        )
    print()

    # -------------------------------------------------------------------------
    # 7. Overall best pairing (intersection)
    # -------------------------------------------------------------------------
    print("Best Overall Detection/Regression Pairing")
    print("_" * 41)
    pfa_pairs_2d = np.column_stack([pfa_pairs_c, pfa_pairs_r])  # [reg, det] to match MATLAB
    pmd_pairs_2d = np.column_stack([pmd_pairs_c, pmd_pairs_r])
    td_pairs_2d  = np.column_stack([td_pairs_c,  td_pairs_r])

    best_pairs = _intersect_rows(
        _intersect_rows(pfa_pairs_2d, pmd_pairs_2d),
        td_pairs_2d
    )

    if best_pairs.size > 0:
        for row in best_pairs:
            alg_j, det_i = int(row[0]), int(row[1])
            print(
                f"{params.detect[det_i]} and {params.algo[alg_j]} "
                f"yield an early detection time of {tdtable[det_i, alg_j]:.4f} sec, "
                f"a false alarm rate of {pfatable[det_i, alg_j] * 100:.4f} %, "
                f"and a missed detection rate of {pmdtable[det_i, alg_j] * 100:.4f} %"
            )
    else:
        print("No clear best pairing")
    print()

    # -------------------------------------------------------------------------
    # 8. Best detection methods (averaged across regression axis)
    # -------------------------------------------------------------------------
    print("Best Detection Methods")
    print("_" * 22)
    mean_pfa_det = np.nanmean(pfatable, axis=1)
    mean_pmd_det = np.nanmean(pmdtable, axis=1)
    mean_td_det  = np.nanmean(tdtable,  axis=1)

    best_pfa_det = _argmin_1d(mean_pfa_det)
    best_pmd_det = _argmin_1d(mean_pmd_det)
    best_td_det  = _argmax_1d(mean_td_det)

    print(
        f"The best mean false alarm rate of "
        f"{np.min(mean_pfa_det) * 100:.4f} % was achieved by the "
        f"{params.detect[best_pfa_det]} detection method, averaged across all regression methods."
    )
    print(
        f"The corresponding mean missed detection rate is "
        f"{(mean_pmd_det[best_pfa_det] - np.min(mean_pmd_det)) * 100:.4f} % over the best, "
        f"and the corresponding mean early detection time is "
        f"{np.max(mean_td_det) - mean_td_det[best_pfa_det]:.4f} sec under the best."
    )
    print()

    print(
        f"The best mean missed detection rate of "
        f"{np.min(mean_pmd_det) * 100:.4f} % was achieved by the "
        f"{params.detect[best_pmd_det]} detection method, averaged across all regression methods."
    )
    print(
        f"The corresponding mean false alarm rate is "
        f"{(mean_pfa_det[best_pmd_det] - np.min(mean_pfa_det)) * 100:.4f} % over the best, "
        f"and the corresponding mean early detection time is "
        f"{np.max(mean_td_det) - mean_td_det[best_pmd_det]:.4f} sec under the best."
    )
    print()

    print(
        f"The best mean early detection time of "
        f"{np.max(mean_td_det):.4f} sec was achieved by the "
        f"{params.detect[best_td_det]} detection method, averaged across all regression methods."
    )
    print(
        f"The corresponding mean false alarm rate is "
        f"{(mean_pfa_det[best_td_det] - np.min(mean_pfa_det)) * 100:.4f} % over the best, "
        f"and the corresponding mean missed detection rate is "
        f"{(mean_pmd_det[best_td_det] - np.min(mean_pmd_det)) * 100:.4f} % over the best."
    )
    print()

    # -------------------------------------------------------------------------
    # 9. Best regression methods (averaged across detection axis)
    # -------------------------------------------------------------------------
    print("Best Regression Methods")
    print("_" * 23)
    mean_pfa_alg = np.nanmean(pfatable, axis=0)
    mean_pmd_alg = np.nanmean(pmdtable, axis=0)
    mean_td_alg  = np.nanmean(tdtable,  axis=0)

    best_pfa_alg = _argmin_1d(mean_pfa_alg)
    best_pmd_alg = _argmin_1d(mean_pmd_alg)
    best_td_alg  = _argmax_1d(mean_td_alg)

    print(
        f"The best mean false alarm rate of "
        f"{np.min(mean_pfa_alg) * 100:.4f} % was achieved by the "
        f"{params.algo[best_pfa_alg]} regression method, averaged across all detection methods."
    )
    print(
        f"The corresponding mean missed detection rate is "
        f"{(mean_pmd_alg[best_pfa_alg] - np.min(mean_pmd_alg)) * 100:.4f} % over the best, "
        f"and the corresponding mean early detection time is "
        f"{np.max(mean_td_alg) - mean_td_alg[best_pfa_alg]:.4f} sec under the best."
    )
    print()

    print(
        f"The best mean missed detection rate of "
        f"{np.min(mean_pmd_alg) * 100:.4f} % was achieved by the "
        f"{params.algo[best_pmd_alg]} regression method, averaged across all detection methods."
    )
    print(
        f"The corresponding mean false alarm rate is "
        f"{(mean_pfa_alg[best_pmd_alg] - np.min(mean_pfa_alg)) * 100:.4f} % over the best, "
        f"and the corresponding mean early detection time is "
        f"{np.max(mean_td_alg) - mean_td_alg[best_pmd_alg]:.4f} sec under the best."
    )
    print()

    print(
        f"The best mean early detection time of "
        f"{np.max(mean_td_alg):.4f} sec was achieved by the "
        f"{params.algo[best_td_alg]} regression method, averaged across all detection methods."
    )
    print(
        f"The corresponding mean false alarm rate is "
        f"{(mean_pfa_alg[best_td_alg] - np.min(mean_pfa_alg)) * 100:.4f} % over the best, "
        f"and the corresponding mean missed detection rate is "
        f"{(mean_pmd_alg[best_td_alg] - np.min(mean_pmd_alg)) * 100:.4f} % over the best."
    )
    print()

    # -------------------------------------------------------------------------
    # 10. Ranked top-N list (interactive)
    # -------------------------------------------------------------------------
    print("Ranked List of Top Detection/Regression Pairings")
    print("_" * 48)

    n_total = pmdtable.size
    det_flat = np.tile(np.arange(n_det), n_alg)
    alg_flat = np.repeat(np.arange(n_alg), n_det)

    pmd_flat = pmdtable.T.ravel()
    pfa_flat = pfatable.T.ravel()
    td_flat  = tdtable.T.ravel()

    pmd_aug = np.column_stack([pmd_flat, det_flat, alg_flat])
    pfa_aug = np.column_stack([pfa_flat, det_flat, alg_flat])
    td_aug  = np.column_stack([td_flat,  det_flat, alg_flat])

    pmd_sorted = _sortrows(pmd_aug, 1)           # ascending by value
    pfa_sorted = _sortrows(pfa_aug, 1)           # ascending by value
    td_sorted  = _sortrows(td_aug,  -1)          # descending by value

    num_top = int(input("Enter the approx. number of top pairings to compare: "))

    top_k   = 1
    top_idx = np.empty((0, 2), dtype=int)
    while top_idx.shape[0] < num_top:
        cand_pmd = pmd_sorted[:top_k, 1:3].astype(int)
        cand_pfa = pfa_sorted[:top_k, 1:3].astype(int)
        cand_td  = td_sorted[:top_k,  1:3].astype(int)
        top_idx  = _intersect_rows(_intersect_rows(cand_pmd, cand_pfa), cand_td)
        top_k   += 1
    top_k -= 1

    cand_pmd = pmd_sorted[:top_k, 1:3].astype(int)
    cand_pfa = pfa_sorted[:top_k, 1:3].astype(int)
    cand_td  = td_sorted[:top_k,  1:3].astype(int)

    def _rank_in(candidates, universe):
        uni_tuples = [tuple(r) for r in universe]
        return np.array([
            uni_tuples.index(tuple(r)) if tuple(r) in uni_tuples else len(universe)
            for r in candidates
        ])

    td_rank  = _rank_in(top_idx, cand_td)
    pfa_rank = _rank_in(top_idx, cand_pfa)
    pmd_rank = _rank_in(top_idx, cand_pmd)

    mean_rank   = np.mean(np.column_stack([td_rank, pfa_rank, pmd_rank]), axis=1)
    ranked      = top_idx[np.argsort(mean_rank)]

    print("The best regression/detection combinations are as follows (in ranked order)...")
    for row in ranked:
        det_i, alg_j = int(row[0]), int(row[1])
        print(
            f"Regression method {params.algo[alg_j]} and detection method "
            f"{params.detect[det_i]} yields a missed detection rate of "
            f"{pmdtable[det_i, alg_j] * 100:.4f} %, "
            f"a false alarm rate of {pfatable[det_i, alg_j] * 100:.4f} %, "
            f"and a time of first alarm appearing "
            f"{tdtable[det_i, alg_j]:.4f} sec before the event occurred"
        )