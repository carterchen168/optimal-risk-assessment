import numpy as np


def lds_params_idx(params):
    """
    Extract the correct learned LDS parameter slices using index bookmarks.

    Parameters
    ----------
    params : object
        Must have a `.learned` attribute (SimpleNamespace or similar) with:
          .ad     : np.ndarray (k, k, n_iters) — system matrices A
          .qwd    : np.ndarray (k, k, n_iters) — process noise Q
          .cd     : np.ndarray (p, k, n_iters) — observation matrix C
          .rvd    : np.ndarray (p, p, n_iters) — observation noise R
          .xssd   : np.ndarray (k, k, n_iters) — steady-state covariance P
          .bd     : np.ndarray (k, m, n_iters) or absent — input matrix B
          .dd     : np.ndarray (p, m, n_iters) or absent — feedthrough D
        And the index bookmarks on params itself:
          .ka, .kq, .kr, .kxinit : int (1-indexed, converted to 0-indexed here)

    Returns
    -------
    a      : np.ndarray (k, k)
    q      : np.ndarray (k, k)
    c      : np.ndarray (p, k)
    r      : np.ndarray (p, p)
    params : object     — updated with params.xssd set to the selected slice
    b      : np.ndarray (k, m) or None
    d      : np.ndarray (p, m) or None
    """
    learned = params.learned

    # MATLAB uses 1-based indexing for the 3rd dimension slice;
    # Python uses 0-based, so subtract 1 when indexing.
    a = learned.ad[:, :, params.ka - 1]
    q = learned.qwd[:, :, params.kq - 1]
    c = learned.cd[:, :, -1]               # MATLAB `end`  →  Python -1
    r = learned.rvd[:, :, params.kr - 1]

    if hasattr(learned, 'bd') and learned.bd is not None:
        b = learned.bd[:, :, -1]
        d = learned.dd[:, :, -1]
    else:
        b = None
        d = None

    params.xssd = learned.xssd[:, :, params.kxinit - 1]

    return a, q, c, r, params, b, d