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
    Pull out corresponding keypoint coordinates for two images.

    Returns:
        pts_i, pts_j : np.ndarray of shape (N,1,2) for cv2 functions
        idxs_i, idxs_j : original keypoint indices
    """
    if mask is None:
        # all matches used. This is helpful if we only want to triangulate certain matches.
        mask = np.ones(len(matches[i][j])) 

    kpts_i, kpts_i_idxs, kpts_j, kpts_j_idxs = [], [], [], []

    for k in range(len(matches[i][j])):
        if mask[k] == 0: 
            continue
        kpts_i.append(keypoints[i][matches[i][j][k].queryIdx].pt)
        kpts_i_idxs.append(matches[i][j][k].queryIdx)
        kpts_j.append(keypoints[j][matches[i][j][k].trainIdx].pt)
        kpts_j_idxs.append(matches[i][j][k].trainIdx)

    kpts_i = np.array(kpts_i)
    kpts_j = np.array(kpts_j)
    kpts_i = np.expand_dims(kpts_i, axis=1)
    kpts_j = np.expand_dims(kpts_j, axis=1)

    return kpts_i, kpts_j, kpts_i_idxs, kpts_j_idxs

