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
    Choose the pair of images with both many matches (in the top `top_percent`) and
    the largest relative rotation.

    Returns:
        A tuple (i, j) of image indices, or None if no suitable pair is found.
    """
    num_matches = []

    for i in range(adjacency.shape[0]):
        for j in range(adjacency.shape[1]):
            if adjacency[i][j] == 1:
                num_matches.append(len(matches[i][j]))

    num_matches = sorted(num_matches, reverse=True)
    min_match_idx = int(len(num_matches)*top_percent)
    min_matches = num_matches[min_match_idx]
    best_R = 0
    best_pair = None

    for i in range(adjacency.shape[0]):
        for j in range(adjacency.shape[1]):

            if adjacency[i][j] == 1:
                if len(matches[i][j]) > min_matches:
                    kpts_i, kpts_j, _, _ = extract_matched_keypoints(i, j, keypoints, matches)
                    E, _ = cv2.findEssentialMat(kpts_i, kpts_j, K, cv2.FM_RANSAC, 0.999, 1.0)

                    points, R1, t1, mask = cv2.recoverPose(E, kpts_i, kpts_j, K)
                    rvec, _ = cv2.Rodrigues(R1)
                    rot_angle = abs(rvec[0]) +abs(rvec[1]) + abs(rvec[2])

                    if (rot_angle > best_R or best_pair == None) and points == len(kpts_i): 
                        best_R = rot_angle
                        best_pair = (i,j)

    return best_pair

