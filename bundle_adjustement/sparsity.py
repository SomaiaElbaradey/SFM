import numpy as np

from scipy.sparse import lil_matrix

def build_ba_sparsity(
    n_cams: int,
    n_pts: int,
    cam_idxs: np.ndarray,
    pt_idxs: np.ndarray,
) -> lil_matrix:
    """Jacobian sparsity for BA: each observation affects its camera’s 12 params and point’s 3 coords."""
    n_obs = cam_idxs.size
    n_var = n_cams * 12 + n_pts * 3
    A = lil_matrix((2 * n_obs, n_var), dtype=int)
    obs = np.arange(n_obs)

    # Camera blocks
    for k in range(12):
        cols = cam_idxs * 12 + k
        A[2 * obs, cols] = 1
        A[2 * obs + 1, cols] = 1

    # Point blocks
    base = n_cams * 12
    for k in range(3):
        cols = base + pt_idxs * 3 + k
        A[2 * obs, cols] = 1
        A[2 * obs + 1, cols] = 1

    return A