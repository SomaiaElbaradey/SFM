import numpy as np
import cv2

def project_obs(
    pts3d: np.ndarray,
    cam_params: np.ndarray,
    K: np.ndarray,
    cam_idxs: np.ndarray,
    pt_idxs: np.ndarray,
) -> np.ndarray:
    """Project each 3D point through its camera; returns (n_obs,2) array."""
    n_obs = cam_idxs.size
    proj = np.zeros((n_obs, 2), dtype=float)

    for i in range(n_obs):
        c = cam_idxs[i]
        p = pt_idxs[i]
        R_flat = cam_params[c, :9].reshape(3,3)
        t = cam_params[c, 9:].reshape(3,1)
        rvec, _ = cv2.Rodrigues(R_flat)
        X = pts3d[p].reshape(1,3)
        p2d, _ = cv2.projectPoints(X, rvec, t, K, distCoeffs=None)
        proj[i] = p2d.ravel()

    return proj
