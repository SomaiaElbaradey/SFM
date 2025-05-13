import cv2
import numpy as np
from typing import List

# --- Feature Matching --------------------------------------------------------

def match_sift_descriptors(
    descriptors: List[np.ndarray],
    matcher: cv2.DescriptorMatcher,
    ratio_thresh: float = 0.7
) -> List[List[List[cv2.DMatch]]]:
    """
    Perform kNN matching (k=2) and apply Lowe's ratio test between all image pairs.
    Returns an upper-triangular matrix of good matches.
    """
    n = len(descriptors)
    matches: List[List[List[cv2.DMatch]]] = [ [ [] for _ in range(n) ] for _ in range(n) ]

    for i in range(n):
        for j in range(i + 1, n):
            raw = matcher.knnMatch(descriptors[i], descriptors[j], k=2)
            good = [m[0] for m in raw if len(m) == 2 and m[0].distance < ratio_thresh * m[1].distance]
            matches[i][j] = good
    return matches