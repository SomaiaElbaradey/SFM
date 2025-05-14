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

def get_idxs_in_correct_order(idx1: int, idx2: int) -> Tuple[int, int]:
    """
    Return a tuple of indices in ascending order for use with upper-triangular structures.

    Args:
        idx1: First index.
        idx2: Second index.

    Returns:
        A tuple (min(idx1, idx2), max(idx1, idx2)).
    """
    return (idx1, idx2) if idx1 < idx2 else (idx2, idx1)


def check_and_get_unresected_point(
    resected_kpt_idx: int,
    match: cv2.DMatch,
    resected_idx: int,
    unresected_idx: int
) -> Tuple[Optional[int], bool]:
    """
    Determine if a 3D point from the resected image participates in a match to the unresected image.

    Args:
        resected_kpt_idx: Keypoint index in the resected image.
        match:      OpenCV DMatch object linking two keypoints.
        resected_idx:    Index of the resected image in pair ordering.
        unresected_idx:  Index of the unresected image in pair ordering.

    Returns:
        A tuple (unresected_kpt_idx, True) if matched,
        otherwise (None, False).
    """
    # If the resected image is the 'query' in the match
    if resected_idx < unresected_idx:
        if resected_kpt_idx == match.queryIdx:
            return match.trainIdx, True
        return None, False

    # If the resected image is the 'train' in the match
    if resected_kpt_idx == match.trainIdx:
        return match.queryIdx, True
    return None, False
