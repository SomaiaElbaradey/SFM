from typing import List, Tuple

import numpy as np
import random

# ---------- Growth Pair Selection ----------

def select_next_image_pair(
    n_images: int,
    init_pair: Tuple[int, int],
    placed: List[int],
    remaining: List[int],
) -> Tuple[int, int, bool]:
    """
    Decide which (already-placed, not-yet-placed) image pair to use next,
    growing the reconstruction both forward and backward in the circular sequence.
    
    Returns:
        (placed_idx, to_place_idx, prepend_flag)
    """
    if not remaining:
        raise ValueError("No remaining images to place.")

    i0, i1 = init_pair
    # detect wrap-around
    wrap = (i1 - i0) > (n_images / 2)
    arc_length = (i1 - i0 + 1) % n_images or n_images

    # Fill in the initial arc first
    if len(placed) < arc_length:
        # Always take the next slot after the second-to-last placed
        base = placed[-2] if not wrap else placed[-1]
        candidate = (base + 1) % n_images
        while candidate in placed:
            candidate = (candidate + 1) % n_images
        # Pair with a random existing image
        partner = random.choice(placed)
        return partner, candidate, True

    # Then alternate extending both ends
    ext = len(placed) - arc_length
    if wrap:
        # even ext → extend low end, odd → high end
        if ext % 2 == 0:
            new_idx = (i0 + ext//2 + 1) % n_images
            paired = (new_idx - 1) % n_images
        else:
            new_idx = (i1 - ext//2 - 1) % n_images
            paired = (new_idx + 1) % n_images
    else:
        if ext % 2 == 0:
            new_idx = (i1 + ext//2 + 1) % n_images
            paired = (new_idx - 1) % n_images
        else:
            new_idx = (i0 - ext//2 - 1) % n_images
            paired = (new_idx + 1) % n_images

    return paired, new_idx, False
