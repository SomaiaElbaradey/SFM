from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------- Geometry Utilities ----------

def extract_matched_keypoints(
    i: int,
    j: int,
    keypoints: List[List[cv2.KeyPoint]],
    matches: List[List[List[cv2.DMatch]]],
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Retrieve corresponding keypoint coordinates and their indices for two images.

    Args:
        i: Index of the first image.
        j: Index of the second image.
        keypoints: List of keypoint lists per image.
        matches: Nested list of cv2.DMatch objects for each image pair.
        mask: Optional array indicating which matches to include (1=use, 0=skip).

    Returns:
        pts_i, pts_j: Arrays of shape (N, 1, 2) with 2D coordinates for cv2.
        idxs_i, idxs_j: Corresponding original keypoint indices in each image.
    """
    pair_matches = matches[i][j]
    total_matches = len(pair_matches)
    valid_mask = mask if mask is not None else np.ones(total_matches, dtype=bool)

    pts_i_list, pts_j_list = [], []
    idxs_i, idxs_j = [], []

    for k, match in enumerate(pair_matches):
        if not valid_mask[k]:
            continue
        kp_i = keypoints[i][match.queryIdx].pt
        kp_j = keypoints[j][match.trainIdx].pt

        pts_i_list.append(kp_i)
        idxs_i.append(match.queryIdx)
        pts_j_list.append(kp_j)
        idxs_j.append(match.trainIdx)

    pts_i = np.expand_dims(np.array(pts_i_list, dtype=float), axis=1)
    pts_j = np.expand_dims(np.array(pts_j_list, dtype=float), axis=1)

    return pts_i, pts_j, idxs_i, idxs_j
