from typing import List, Optional, Tuple

import cv2
import numpy as np

def order_indices(a: int, b: int) -> Tuple[int, int]:
    """Return (min(a,b), max(a,b))."""
    return (a, b) if a < b else (b, a)

def are_images_adjacent(i: int, j: int, adjacency: np.ndarray) -> bool:
    """True if images i and j share at least one match."""
    return bool(adjacency[i, j] or adjacency[j, i])

def has_link_to_group(
    target: int,
    group: List[int],
    adjacency: np.ndarray
) -> bool:
    """
    True if `target` image has at least one match
    with any image in `group`.
    """
    return any(adjacency[target, g] or adjacency[g, target] for g in group)

def map_point_to_unplaced(
    kpt_idx: int,
    match: cv2.DMatch,
    placed_idx: int,
    new_idx: int
) -> Tuple[Optional[int], bool]:
    """
    Given a 3D point seen in `placed_idx`, check if it corresponds via `match`
    to a keypoint in `new_idx`, for PnP seeding.
    """
    if placed_idx < new_idx:
        if kpt_idx == match.queryIdx:
            return match.trainIdx, True
    else:
        if kpt_idx == match.trainIdx:
            return match.queryIdx, True
    return None, False

def get_idxs_in_correct_order(idx1, idx2):
    """First idx must be smaller than second when using upper-triangular arrays (matches, keypoints)"""
    if idx1 < idx2: return idx1, idx2
    else: return idx2, idx1

def check_and_get_unresected_point(resected_kpt_idx, match, resected_idx, unresected_idx):
    """
    Check if a 3D point seen by the given resected image is involved in a match to the unresected image
    and is therefore usable for Pnp.

    :param resected_kpt_idx: Index of keypoint in keypoints list for resected image
    :param match: cv2.Dmatch object
    :resected_idx: Index of the resected image
    :unresected_idx: Index of the unresected image
    """
    if resected_idx < unresected_idx:
        if resected_kpt_idx == match.queryIdx:
            unresected_kpt_idx = match.trainIdx
            success = True
            return unresected_kpt_idx, success
        else:
            return None, False
    elif unresected_idx < resected_idx:
        if resected_kpt_idx == match.trainIdx:
            unresected_kpt_idx = match.queryIdx
            success = True
            return unresected_kpt_idx, success
        else:
            return None, False