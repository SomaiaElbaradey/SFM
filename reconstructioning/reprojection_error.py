from typing import List, Optional, Tuple

import cv2
import numpy as np

from .data_structure import Point3DWithViews

# ---------- Reprojection Errors ----------

def prepare_reprojection_data(
    img_idx: int,
    points3d: List[Point3DWithViews],
    keypoints: List[List[cv2.KeyPoint]],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Gather all (X, x) pairs for a single image to compute reprojection errors.

    Returns:
        - pts3d_world: array (M, 3)
        - pts2d_img: array (M, 2)
        - indices: which 3D points were used
    """
    pts3d_list, pts2d_list, idx_list = [], [], []

    for i, pt3d in enumerate(points3d):
        if img_idx in pt3d.view_indices:
            idx_list.append(i)
            pts3d_list.append(pt3d.coords.flatten())
            kpt_idx = pt3d.view_indices[img_idx]
            pts2d_list.append(keypoints[img_idx][kpt_idx].pt)

    return np.vstack(pts3d_list), np.array(pts2d_list, dtype=float), idx_list


def compute_reprojection_errors(
    proj_pts: np.ndarray,
    true_pts: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    L1 reprojection error per point and the mean over both coords.

    Returns:
        (mean_error, per_point_errors)
    """
    deltas = np.abs(proj_pts - true_pts)            # shape (M,2)
    per_point = deltas.sum(axis=1) / 2.0            # average of x & y
    mean_err = per_point.mean()
    return mean_err, deltas

def get_reprojection_errors(
    img_idx: int,
    points3d: List[Point3DWithViews],
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    keypoints: List[List[cv2.KeyPoint]],
    distCoeffs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Project all known 3D points into `img_idx` with pose (R, t), compute errors.

    Returns:
        (world_pts, image_pts, mean_error, per_point_errors)
    """
    world_pts, image_pts, _ = prepare_reprojection_data(img_idx, points3d, keypoints)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(world_pts, rvec, t, K, distCoeffs=distCoeffs or np.zeros(5))
    proj_pts = proj.reshape(-1, 2)
    mean_err, errors = compute_reprojection_errors(proj_pts, image_pts)
    return world_pts, image_pts, mean_err, errors
