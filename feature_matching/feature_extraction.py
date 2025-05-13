import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

# --- Feature Extraction ------------------------------------------------------

def extract_sift_features(
    images: List[np.ndarray]
) -> Tuple[List[List[cv2.KeyPoint]], List[np.ndarray]]:
    """
    Compute SIFT keypoints and descriptors for each image.
    Returns lists of keypoints and descriptor arrays.
    """
    sift = cv2.SIFT_create()
    all_keypoints: List[List[cv2.KeyPoint]] = []
    all_descriptors: List[np.ndarray] = []
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        all_keypoints.append(kp)
        all_descriptors.append(des)
    return all_keypoints, all_descriptors
