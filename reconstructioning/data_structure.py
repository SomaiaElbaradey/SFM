from typing import Dict

import numpy as np


class Point3DWithViews:
    """
    Associates a single 3D point with its originating 2D observations.

    Attributes:
        coords: The 3D coordinates of the point as a (1×3) array.
        view_indices: Mapping from image index → keypoint index in that image.
    """

    def __init__(self, coords: np.ndarray, view_indices: Dict[int, int]):
        self.coords = coords
        self.view_indices = view_indices
