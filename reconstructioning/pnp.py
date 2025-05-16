from typing import List, Tuple

import numpy as np
import cv2

from .utils import (
    check_and_get_unresected_point,
    get_idxs_in_correct_order,
)
from .data_structure import Point3DWithViews


# ---------- Correspondence Matching ----------

def match_unresected_point(
    unresected_idx: int,
    match: cv2.DMatch,
    reference_idx: int,
    new_image_idx: int
) -> Tuple[int, bool]:
    """
    Checks if a 3D point from the unresected image (unresected_idx) matches a keypoint
    in the new image (new_image_idx) based on the provided cv2.DMatch object.

    Args:
        unresected_idx (int): Index of the unresected image.
        match (cv2.DMatch): The match object containing indices of matched keypoints.
        reference_idx (int): Index of the reference image.
        new_image_idx (int): Index of the new image being matched against.

    Returns:
        Tuple[int, bool]:
            - Matched keypoint index in the new image (int).
            - Success flag (True if a match is found, False otherwise).
    """
    # Case when unresected image index is smaller than new image index
    if unresected_idx < new_image_idx and match.queryIdx == unresected_idx:
        return match.trainIdx, True

    # Case when new image index is smaller than unresected image index
    if new_image_idx < unresected_idx and match.trainIdx == unresected_idx:
        return match.queryIdx, True

    # No valid match found
    return None, False


# ---------- PnP Correspondences ----------

def fetch_pnp_correspondences(
    resected_cam_idx: int,
    unresected_cam_idx: int,
    points_3d: List[Point3DWithViews],
    match_list: List[List[List[cv2.DMatch]]],
    keypoints_list: List[List[cv2.KeyPoint]]
) -> Tuple[
    List[Point3DWithViews],
    List[np.ndarray],
    List[Tuple[float, float]],
    np.ndarray
]:
    """
    Retrieves 3D-2D point correspondences for solving PnP and determines
    points requiring triangulation.

    Parameters:
        resected_cam_idx (int): Index of camera with known pose.
        unresected_cam_idx (int): Index of camera to estimate pose for.
        points_3d (List[Point3DWithViews]): Existing 3D points with associated camera views.
        match_list (List[List[List[cv2.DMatch]]]): Matches between all camera pairs.
        keypoints_list (List[List[cv2.KeyPoint]]): Keypoints detected for each camera.

    Returns:
        Tuple containing:
            - points_3d (List[Point3DWithViews]): Original list of 3D points.
            - pnp_points_3d (List[np.ndarray]): 3D coordinates corresponding to matched 2D points for PnP.
            - pnp_points_2d (List[Tuple[float, float]]): Matched 2D keypoints in the unresected camera.
            - triangulation_flags (np.ndarray): Array indicating unmatched points requiring triangulation (1 = requires triangulation).
    """

    # Determine consistent indexing order for camera pair
    idx_a, idx_b = get_idxs_in_correct_order(resected_cam_idx, unresected_cam_idx)

    # Initialize flags indicating points to triangulate (default: all require triangulation)
    triangulation_flags = np.ones(len(match_list[idx_a][idx_b]), dtype=np.uint8)

    pnp_points_3d: List[np.ndarray] = []
    pnp_points_2d: List[Tuple[float, float]] = []

    # Iterate over existing 3D points to find correspondences
    for point_3d in points_3d:
        # Skip points that are not observed in the resected camera
        if resected_cam_idx not in point_3d.view_indices:
            continue

        # Reference keypoint index in the resected camera
        ref_kpt_idx = point_3d.view_indices[resected_cam_idx]

        # Check matches between the current camera pair
        for match_idx, match in enumerate(match_list[idx_a][idx_b]):
            unresected_kpt_idx, is_match_valid = check_and_get_unresected_point(
                ref_kpt_idx, match, resected_cam_idx, unresected_cam_idx
            )

            if not is_match_valid:
                continue

            # Record the matched point in unresected camera views
            point_3d.view_indices[unresected_cam_idx] = unresected_kpt_idx

            # Append valid correspondences for PnP
            pnp_points_3d.append(point_3d.coords)
            pnp_points_2d.append(keypoints_list[unresected_cam_idx][unresected_kpt_idx].pt)

            # Mark this match as already having a correspondence (no triangulation needed)
            triangulation_flags[match_idx] = 0

    return points_3d, pnp_points_3d, pnp_points_2d, triangulation_flags


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

        # enhnce: use solvePnPRansac.
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