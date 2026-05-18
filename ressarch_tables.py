import numpy as np
from typing import Any, Optional
from make_datafiles import make_datafiles

# helpers
def argmin(arr: np.ndarray) -> tuple:
    """Return (row, col) index of the global minimum (MATLAB-style argmin)."""
    idx = np.unravel_index(np.nanargmin(arr), arr.shape)
    return idx


def argmax(arr: np.ndarray) -> tuple:
    """Return (row, col) index of the global maximum (MATLAB-style argmax)."""
    idx = np.unravel_index(np.nanargmax(arr), arr.shape)
    return idx


def intersect_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return rows that appear in both 2-D arrays (MATLAB intersect(...,'rows')).
    Preserves order from `a`.
    """
    if a.size == 0 or b.size == 0:
        return np.empty((0, a.shape[1] if a.ndim > 1 else 0), dtype=a.dtype)
    set_b = set(map(tuple, b))
    result = [row for row in a if tuple(row) in set_b]
    return np.array(result) if result else np.empty((0, a.shape[1]), dtype=a.dtype)


def sortrows(arr: np.ndarray, col: int = 0) -> np.ndarray:
    """Sort 2-D array by a given column (col > 0 asc, col < 0 desc — 1-indexed)."""
    ascending = col >= 0
    col_idx = abs(col) - 1          # convert 1-indexed → 0-indexed
    order = np.argsort(arr[:, col_idx], kind='stable')
    if not ascending:
        order = order[::-1]
    return arr[order]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def ressarch_tables(
    params,
    modelselectdata,
    redtrain=None,
    redval=None,
    predtrain=None,
    predval=None,
    opttrain=None,
    optval=None,
    sprt=None,
):
    """
    Build and print summary result tables for missed-detection rate (P_md),
    false-alarm rate (P_fa), and early-detection time (T_d).

    Parameters
    ----------
    params : object
        Must have .algo (list of str) and .detect (list of str).
    modelselectdata : object
        Must have .tuneval (array) and .Jmse (array or list of arrays).
    redtrain, redval, predtrain, predval, opttrain, optval, sprt : object, optional
        Detection-result objects with attributes .fpratesamp, .pmdsamp, .tdsamp
        (each a 1-D or 2-D numpy array, shape = [n_detectors, n_algos]).
    """
    try:
        train, test, train_cell, rawdata_tst, rawdata_tr = make_datafiles(params, 3)
    except NotImplementedError as exc:
        print(f"[WARNING] {exc}\nContinuing without datafiles.")

    num_algos = len(params.algo)

    # -------------------------------------------------------------------------
    # 1. Assemble the three tables from whichever result objects are present
    # -------------------------------------------------------------------------
    pfa_rows, pmd_rows, td_rows = [], [], []

    for obj_name, obj in [
        ("redtrain",  redtrain),
        ("redval",    redval),
        ("predtrain", predtrain),
        ("predval",   predval),
        ("opttrain",  opttrain),
        ("optval",    optval),
        ("sprt",      sprt),
    ]:
        if obj is not None:
            pfa_rows.append(np.atleast_2d(obj.fpratesamp))
            pmd_rows.append(np.atleast_2d(obj.pmdsamp))
            td_rows.append(np.atleast_2d(obj.tdsamp))

    if not pmd_rows:
        print("No detection result objects provided. Nothing to display.")
        return

    pfatable = np.vstack(pfa_rows)   # shape: (n_detectors, n_algos)
    pmdtable = np.vstack(pmd_rows)
    tdtable  = np.vstack(td_rows)

    n_det, n_alg = pmdtable.shape
    col_width = max(len(d) for d in params.detect)  # for aligned printing

    # -------------------------------------------------------------------------
    # 2. Regression tuning / optimised-value summary
    # -------------------------------------------------------------------------
    print("Regression results...")
    header = f"{'':>{col_width}}\t" + "\t".join(f"  {a}" for a in params.algo)
    print(header)

    # Build format strings per algorithm column
    regress_fmts = []
    std_fmts     = []
    regress_optval = np.zeros(n_alg)

    for j, algo in enumerate(params.algo):
        std_fmts.append("{:.4f}")
        tune_j = modelselectdata.tuneval[j]
        if tune_j < 0.001:
            regress_fmts.append("{:.1e}")
        elif algo in ("knn", "bnet"):
            regress_fmts.append("{:4d}")
        else:
            regress_fmts.append("{:.4f}")

        # Replicate MATLAB: Jmse may be a list of arrays or a 2-D array
        if isinstance(modelselectdata.Jmse, list):
            jmse_j = np.array(modelselectdata.Jmse[j])
            regress_optval[j] = jmse_j[np.nanargmin(jmse_j)]
        else:
            jmse_j = modelselectdata.Jmse[j, :]
            regress_optval[j] = jmse_j[np.nanargmin(jmse_j)]

    # cross validation
    tune_str = "\t".join(
        fmt.format(v) for fmt, v in zip(regress_fmts, modelselectdata.tuneval)
    )
    opt_str = "\t".join(
        fmt.format(v) for fmt, v in zip(std_fmts, regress_optval)
    )
    print(f"Global Optimum \t\t{tune_str}")
    print(f"Optimized Values \t{opt_str}")
    print()

    # -------------------------------------------------------------------------
    # 3. Helper to print one table (pmd, pfa, or td)
    # -------------------------------------------------------------------------
    def print_table(table: np.ndarray, title: str):
        print(title)
        header_row = (
            f"{'':>{col_width}}\t"
            + "\t".join(f"  {a}" for a in params.algo)
        )
        print(header_row)
        for i, det_name in enumerate(params.detect):
            if i >= table.shape[0]:
                break
            row = table[i, :]
            cells = []
            for val in row:
                if np.isnan(val):
                    cells.append("-")
                else:
                    cells.append(f"{val:.4f}")
            row_str = "\t".join(cells)
            print(f"{det_name:>{col_width}}\t{row_str}")
        print()

    print_table(pmdtable, "Missed detection results...")
    print_table(pfatable, "False alarm results...")
    print_table(tdtable,  "Detection time results...")

    # -------------------------------------------------------------------------
    # 4. Find best detection × regression combination
    # -------------------------------------------------------------------------
    pmd_idx = argmin(pmdtable)   # (row, col) tuple
    pfa_idx = argmin(pfatable)
    td_idx  = argmax(tdtable)

    # Collect all (det_row, alg_col) pairs that share the global optimum
    def find_ties(table, idx, use_max=False):
        opt_val = table[idx]
        rows, cols = np.where(table == opt_val)
        return np.column_stack([rows, cols])  # shape (n_ties, 2)

    pfa_pairs = find_ties(pfatable, pfa_idx)
    pmd_pairs = find_ties(pmdtable, pmd_idx)
    td_pairs  = find_ties(tdtable,  td_idx, use_max=True)

    best_pairs = intersect_rows(
        intersect_rows(pfa_pairs, pmd_pairs),
        td_pairs
    )

    if best_pairs.size > 0:
        for row in best_pairs:
            det_i, alg_j = int(row[0]), int(row[1])
            print(
                f"{params.detect[det_i]} is the best performing detection method, "
                f"yielding a missed detection rate of "
                f"{pmdtable[pmd_idx] * 100:.4f} %, "
                f"a false alarm rate of {pfatable[pfa_idx] * 100:.4f} %, "
                f"and an early detection time of {tdtable[td_idx]:.4f} sec, "
                f"using regression method {params.algo[alg_j]}"
            )
    else:
        # -----------------------------------------------------------------
        # No clear winner → interactive ranked multi-objective search
        # -----------------------------------------------------------------
        print("No clear best detection method")

        n_total  = pmdtable.size
        det_flat = np.tile(np.arange(n_det), n_alg)           # det index per cell
        alg_flat = np.repeat(np.arange(n_alg), n_det)         # alg index per cell

        # Flatten and augment: [value, det_idx, alg_idx]
        pmd_flat  = pmdtable.T.ravel()
        pfa_flat  = pfatable.T.ravel()
        td_flat   = tdtable.T.ravel()

        pmd_aug = np.column_stack([pmd_flat, det_flat, alg_flat])
        pfa_aug = np.column_stack([pfa_flat, det_flat, alg_flat])
        td_aug  = np.column_stack([td_flat,  det_flat, alg_flat])

        pmd_sorted = pmd_aug[np.argsort(pmd_aug[:, 0])]          # ascending
        pfa_sorted = pfa_aug[np.argsort(pfa_aug[:, 0])]          # ascending
        td_sorted  = td_aug[np.argsort(td_aug[:, 0])[::-1]]      # descending (best = highest)

        num_top = int(input("Enter the approx. number of top performers to compare: "))

        top_k = 1
        top_idx = np.empty((0, 2), dtype=int)
        while top_idx.shape[0] < num_top:
            cand_pmd = pmd_sorted[:top_k, 1:3].astype(int)
            cand_pfa = pfa_sorted[:top_k, 1:3].astype(int)
            cand_td  = td_sorted[:top_k,  1:3].astype(int)
            top_idx  = intersect_rows(intersect_rows(cand_pmd, cand_pfa), cand_td)
            top_k   += 1
        top_k -= 1  # revert the final over-increment

        # Compute ranks within the intersection set
        cand_pmd = pmd_sorted[:top_k, 1:3].astype(int)
        cand_pfa = pfa_sorted[:top_k, 1:3].astype(int)
        cand_td  = td_sorted[:top_k,  1:3].astype(int)

        def rank_in(candidates, universe):
            """Return rank (0-indexed position) of each row in `candidates` within `universe`."""
            uni_tuples = [tuple(r) for r in universe]
            ranks = []
            for row in candidates:
                try:
                    ranks.append(uni_tuples.index(tuple(row)))
                except ValueError:
                    ranks.append(len(universe))
            return np.array(ranks)

        td_rank  = rank_in(top_idx, cand_td)
        pfa_rank = rank_in(top_idx, cand_pfa)
        pmd_rank = rank_in(top_idx, cand_pmd)

        mean_rank = np.mean(np.column_stack([td_rank, pfa_rank, pmd_rank]), axis=1)
        ranked = top_idx[np.argsort(mean_rank)]

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