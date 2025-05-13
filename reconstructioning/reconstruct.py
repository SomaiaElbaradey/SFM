import numpy as np
import cv2

from typing import List, Tuple

from .geometry import extract_matched_keypoints
from .triangulation import triangulate_and_reproject

from .data_structure import Point3DWithViews

# ---------- Reconstruction Initialization ----------

def initialize_reconstruction(
    keypoints: List[List[cv2.KeyPoint]],
    matches: List[List[List[cv2.DMatch]]],
    K: np.ndarray,
    idx1: int,
    idx2: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Point3DWithViews]]:
    """
    From one image pair, recover relative pose and build an initial sparse point cloud.
    
    Returns:
        (R0, t0, R1, t1, points3d_with_views)
    """
    pts1, pts2, idxs1, idxs2 = extract_matched_keypoints(idx1, idx2, keypoints, matches)
    E, _ = cv2.findEssentialMat(pts1, pts2, K, cv2.FM_RANSAC, 0.999, 1.0)
    _, R1, t1, _ = cv2.recoverPose(E, pts1, pts2, K)
    assert abs(np.linalg.det(R1) - 1.0) < 1e-7

    R0 = np.eye(3)
    t0 = np.zeros((3, 1))

    points3d: List[Point3DWithViews] = []
    points3d = triangulate_and_reproject(
        R0, t0, R1, t1, K, points3d, idx1, idx2,
        pts1, pts2, idxs1, idxs2, reproject=False
    )
    return R0, t0, R1, t1, points3d

