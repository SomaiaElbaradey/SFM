from typing import List, Tuple

import numpy as np
import cv2

from .utils import (
    check_and_get_unresected_point,
    get_idxs_in_correct_order,
    map_point_to_unplaced,
    order_indices
)
from .data_structure import Point3DWithViews


# ---------- Correspondence Matching ----------

def match_unresected_point(
    res_idx: int,
    match: cv2.DMatch,
    ref_idx: int,
    new_idx: int
) -> Tuple[int, bool]:
    """
    Checks if a 3D point from res_idx matches new_idx and returns its keypoint index.

    :returns: (unresected_keypoint_index, success_flag)
    """
    if res_idx < new_idx and match.queryIdx == res_idx:
        return match.trainIdx, True
    if new_idx < res_idx and match.trainIdx == res_idx:
        return match.queryIdx, True
    return None, False


# ---------- PnP Correspondences ----------

def fetch_pnp_correspondences(
    resected_idx: int,
    unresected_idx: int,
    pts3d: List[Point3DWithViews],
    matches: List[List[List[cv2.DMatch]]],
    keypoints: List[List[cv2.KeyPoint]]
) -> Tuple[
    List[Point3DWithViews],
    List[np.ndarray],
    List[Tuple[float, float]],
    np.ndarray
]:
    """
    Gathers 3D-2D correspondences for PnP and flags points to triangulate.

    :returns:
      - pts3d: original list of 3D points
      - pts3d_for_pnp: aligned 3D coordinates for PnP
      - pts2d_for_pnp: aligned 2D image points for PnP
      - triangulation_status: array flagging unmatched keypoints
    """
    idx1, idx2 = get_idxs_in_correct_order(resected_idx, unresected_idx)
    triangulation_status = np.ones(len(matches[idx1][idx2]), dtype=np.uint8)
    pts3d_for_pnp: List[np.ndarray] = []
    pts2d_for_pnp: List[Tuple[float, float]] = []

    for pt3d in pts3d:
        if resected_idx not in pt3d.view_indices:
            continue

        ref_kpt_idx = pt3d.view_indices[resected_idx]
        for m_idx, match in enumerate(matches[idx1][idx2]):
            unresected_kpt_idx, ok = check_and_get_unresected_point(
                ref_kpt_idx, match, resected_idx, unresected_idx
            )
            if not ok:
                continue

            # update view and collect for PnP
            pt3d.view_indices[unresected_idx] = unresected_kpt_idx
            pts3d_for_pnp.append(pt3d.coords)
            pts2d_for_pnp.append(keypoints[unresected_idx][unresected_kpt_idx].pt)
            triangulation_status[m_idx] = 0

    return pts3d, pts3d_for_pnp, pts2d_for_pnp, triangulation_status


# ---------- PnP Pose Estimation ----------

def estimate_pose_pnp(
    pts3d_for_pnp: List[np.ndarray],
    pts2d_for_pnp: List[Tuple[float, float]],
    K: np.ndarray,
    iterations: int = 200,
    reproj_thresh: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates camera pose using PnP+RANSAC by maximizing inliers.

    :returns: (R, tvec)
    """
    # prepare arrays
    obj_pts = np.squeeze(np.array(pts3d_for_pnp))
    img_pts = np.expand_dims(np.squeeze(np.array(pts2d_for_pnp)), axis=1)
    num_pts = len(obj_pts)

    best_inliers = 0
    best_R, best_t = None, None

    for _ in range(iterations):
        idxs = np.random.choice(num_pts, 6, replace=False)
        sample_obj = obj_pts[idxs]
        sample_img = img_pts[idxs]

        _, rvec, tvec = cv2.solvePnP(
            sample_obj, sample_img, K, distCoeffs=np.array([]),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        R, _ = cv2.Rodrigues(rvec)

        _, _, _, inlier_ratio = evaluate_pnp_reprojection(
            pts3d_for_pnp, pts2d_for_pnp, R, tvec, K, rep_thresh=reproj_thresh
        )

        if inlier_ratio > best_inliers:
            best_inliers = inlier_ratio
            best_R, best_t = R, tvec

    if best_R is None:
        raise RuntimeError("PnP failed to find a valid pose")

    return best_R, best_t


# ---------- PnP Reprojection Evaluation ----------

def evaluate_pnp_reprojection(
    pts3d_for_pnp: List[np.ndarray],
    pts2d_for_pnp: List[Tuple[float, float]],
    R_new: np.ndarray,
    t_new: np.ndarray,
    K: np.ndarray,
    rep_thresh: float = 5.0
) -> Tuple[List[List[float]], List[np.ndarray], float, float]:
    """
    Computes reprojection errors and inlier ratio for PnP estimate.

    :returns:
      - errors: per-point reprojection errors (dx, dy)
      - projpts: projected homogeneous coordinates
      - avg_err: average pixel error
      - inlier_ratio: fraction within threshold
    """
    errors, projpts, inliers = [], [], []

    for pt3, pt2 in zip(pts3d_for_pnp, pts2d_for_pnp):
        X_cam = R_new.dot(pt3.reshape(3, 1)) + t_new
        x = K.dot(X_cam)
        x /= x[2]

        dx = float(x[0] - pt2[0])
        dy = float(x[1] - pt2[1])
        errors.append([dx, dy])
        projpts.append(x)

        inliers.append(int(abs(dx) <= rep_thresh and abs(dy) <= rep_thresh))

    avg_err = sum(abs(e[0]) + abs(e[1]) for e in errors) / (2 * len(errors))
    inlier_ratio = sum(inliers) / len(inliers)

    return errors, projpts, avg_err, inlier_ratio