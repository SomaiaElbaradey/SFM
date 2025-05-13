import numpy as np
import cv2

from scipy.optimize import least_squares

from typing import List, Dict, Tuple

from .residuals import ba_residuals
from .sparsity import build_ba_sparsity

from reconstructioning.data_structure import Point3DWithViews

def bundle_adjust(
    points3d_with_views: List[Point3DWithViews],
    R_mats: Dict[int, np.ndarray],
    t_vecs: Dict[int, np.ndarray],
    resected_imgs: List[int],
    keypoints: List[List[cv2.KeyPoint]],
    K: np.ndarray,
    ftol: float,
) -> Tuple[List[Point3DWithViews], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Perform bundle adjustment on the current set of resected images & triangulated points.

    Returns updated points3d_with_views, R_mats, t_vecs.
    """
    # Map each resected image to a normalized index 0..n_cams-1
    cam_map = {img: idx for idx, img in enumerate(resected_imgs)}
    n_cams = len(resected_imgs)

    # Initial camera-params array (n_cams x 12)
    cam_params = np.vstack([
        np.hstack((R_mats[img].ravel(), t_vecs[img].ravel()))
        for img in resected_imgs
    ])

    # Gather observations
    cam_idxs, pt_idxs, obs2d = [], [], []
    pts3d_arr = np.vstack([p.coords.flatten() for p in points3d_with_views])
    n_pts = pts3d_arr.shape[0]

    for p_idx, p in enumerate(points3d_with_views):
        for img, kpt_idx in p.view_indices.items():
            if img not in cam_map:
                continue
            cam_idxs.append(cam_map[img])
            pt_idxs.append(p_idx)
            obs2d.append(keypoints[img][kpt_idx].pt)

    cam_idxs = np.array(cam_idxs, int)
    pt_idxs = np.array(pt_idxs, int)
    obs2d = np.array(obs2d, float)

    # Build sparsity pattern and initial parameter vector
    A = build_ba_sparsity(n_cams, n_pts, cam_idxs, pt_idxs)
    x0 = np.hstack((cam_params.ravel(), pts3d_arr.ravel()))

    # Run sparse least-squares
    res = least_squares(
        ba_residuals, x0,
        jac_sparsity=A,
        ftol=ftol, xtol=1e-12,
        x_scale='jac', verbose=2, method='trf',
        args=(n_cams, n_pts, cam_idxs, pt_idxs, obs2d, K)
    )

    # Unpack optimized parameters
    opt = res.x
    cams_opt = opt[: n_cams * 12].reshape(n_cams, 12)
    pts_opt = opt[n_cams * 12 :].reshape(n_pts, 3)

    # Build back R_mats & t_vecs in original image indices
    new_R, new_t = {}, {}
    for img, idx in cam_map.items():
        flat = cams_opt[idx]
        new_R[img] = flat[:9].reshape(3, 3)
        new_t[img] = flat[9:].reshape(3, 1)

    # Update 3D points in place
    for i, p in enumerate(points3d_with_views):
        p.coords = pts_opt[i].reshape(1, 3)

    return points3d_with_views, new_R, new_t
