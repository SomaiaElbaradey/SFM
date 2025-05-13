import cv2
import numpy as np
from typing import List, Tuple

# --- Utilities ---------------------------------------------------------------

def count_total_matches(
    matches: List[List[List[cv2.DMatch]]]
) -> int:
    """Return total number of matches in the upper triangle."""
    return sum(len(matches[i][j]) for i in range(len(matches)) for j in range(i+1, len(matches)))


def display_connected_pairs(
    matches: List[List[List[cv2.DMatch]]]
) -> None:
    """Print number of image pairs that have at least one match."""
    total_pairs = sum(1 for i in range(len(matches)) for j in range(i+1, len(matches)))
    connected = sum(1 for i in range(len(matches)) for j in range(i+1, len(matches)) if matches[i][j])
    print(f"Connected pairs: {connected} / {total_pairs}")


def build_adjacency_matrix(
    num_images: int,
    matches: List[List[List[cv2.DMatch]]]
) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    """
    Build binary adjacency matrix and list of connected pairs based on matches.
    """
    adj = np.zeros((num_images, num_images), dtype=int)
    pairs = []
    for i in range(num_images):
        for j in range(i+1, num_images):
            if matches[i][j]:
                adj[i, j] = 1
                pairs.append((i, j))
    return adj, pairs