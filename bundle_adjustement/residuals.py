import numpy as np

from .projection import project_obs

def ba_residuals(
    x: np.ndarray,
    n_cams: int,
    n_pts: int,
    cam_idxs: np.ndarray,
    pt_idxs: np.ndarray,
    obs2d: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Compute reprojection residuals for all observations."""
    cams = x[: n_cams * 12].reshape(n_cams, 12)
    pts = x[n_cams * 12 :].reshape(n_pts, 3)
    proj2d = project_obs(pts, cams, K, cam_idxs, pt_idxs)
    return (proj2d - obs2d).ravel()
