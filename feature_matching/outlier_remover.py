
import cv2
import numpy as np
from typing import List

# --- Outlier Removal ---------------------------------------------------------

def filter_correspondences(
    matches: List[List[List[cv2.DMatch]]],
    keypoints: List[List[cv2.KeyPoint]],
    min_inliers: int = 20
) -> None:
    """
    For each image pair, estimate the fundamental matrix and keep only inliers.
    Discard pairs with fewer than `min_inliers` in the end.
    Modifies `matches` in place.
    """
    for i, row in enumerate(matches):
        for j, m in enumerate(row):
            if j <= i or not m or len(m) < min_inliers:
                matches[i][j] = []
                continue

            pts_i = np.float32([keypoints[i][d.queryIdx].pt for d in m])
            pts_j = np.float32([keypoints[j][d.trainIdx].pt for d in m])

            F, mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, ransacReprojThreshold=3.0)
            if mask is None:
                matches[i][j] = []
                continue

            inliers = mask.ravel().astype(bool)
            filtered = [d for k, d in enumerate(m) if inliers[k]]
            matches[i][j] = filtered if len(filtered) >= min_inliers else []
