from typing import List, Optional, Tuple

import numpy as np
import cv2

from .geometry import extract_matched_keypoints

# ---------- Initial Pair Selection ----------

def select_initial_image_pair(
    adjacency: np.ndarray,
    matches: List[List[List[cv2.DMatch]]],
    keypoints: List[List[cv2.KeyPoint]],
    K: np.ndarray,
    top_percent: float = 0.2
) -> Optional[Tuple[int, int]]:
    """
    Select the initial image pair that has both a high number of feature matches
    (top `top_percent`) and the largest relative rotation angle.

    Args:
        adjacency: Binary connectivity matrix (NxN).
        matches: Nested list of cv2.DMatch between image pairs.
        keypoints: List of keypoint lists per image.
        K: Camera intrinsic matrix.
        top_percent: Fraction of top-matched pairs to consider as threshold.

    Returns:
        Tuple of image indices (i, j) if a suitable pair is found; otherwise None.
    """
    num_matches = []

    # Count matches for each connected pair
    for i in range(adjacency.shape[0]):
        for j in range(adjacency.shape[1]):
            if adjacency[i, j] == 1:
                num_matches.append(len(matches[i][j]))

    if not num_matches:
        return None

    # Determine match threshold from top percent
    num_matches_sorted = sorted(num_matches, reverse=True)
    threshold_index = int(len(num_matches_sorted) * top_percent)
    min_matches_threshold = num_matches_sorted[threshold_index]

    best_rotation_angle = 0
    best_pair = None

    # Evaluate pairs above threshold
    for i in range(adjacency.shape[0]):
        for j in range(adjacency.shape[1]):
            if adjacency[i, j] == 1 and len(matches[i][j]) > min_matches_threshold:
                pts_i, pts_j, _, _ = extract_matched_keypoints(i, j, keypoints, matches)
                E, _ = cv2.findEssentialMat(pts_i, pts_j, K, cv2.FM_RANSAC, 0.999, 1.0)
                num_points, R_est, t_est, mask = cv2.recoverPose(E, pts_i, pts_j, K)

                rotation_vector, _ = cv2.Rodrigues(R_est)
                rotation_angle = np.sum(np.abs(rotation_vector))

                if (rotation_angle > best_rotation_angle or best_pair is None) and num_points == len(pts_i):
                    best_rotation_angle = rotation_angle
                    best_pair = (i, j)

    return best_pair
