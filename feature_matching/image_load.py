import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

# --- Constants ----------------------------------------------------------------
CAMERA_MATRICES: Dict[str, np.ndarray] = {
    'templering': np.array([
        [1520.40, 0.00, 302.32],
        [0.00, 1525.90, 246.87],
        [0.00, 0.00, 1.00]
    ]),
    'dingoring': np.array([
        [3310.40, 0.00, 316.73],
        [0.00, 3325.50, 200.55],
        [0.00, 0.00, 1.00]
    ]),
    "eglise": np.array([
        [2.3940, 0.0, 0.9324],
        [0.0, -2.3981, 0.6283],
        [0.0, 0.0, 0.0010]
    ]),
    "custom": np.array([
        [2.68010484e+03, 0.00000000e+00, 1.20555833e+03],
        [0.00000000e+00, 2.69887481e+03, 1.60406955e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    "wazgha": np.array([
        [7.96599496e+03, 0.00e+00, 4.43831810e+03],
        [0.00000000e+00, 7.89500643e+03, 1.94117547e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])}

# --- Image Loading ------------------------------------------------------------
def load_images(
    dataset: str,
    count: int,
    base_dir: Path = Path('./datasets')
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load `count` grayscale images from `dataset` folder and return images + camera matrix.
    Raises for invalid dataset or missing files.
    """
    if dataset not in CAMERA_MATRICES:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if dataset == 'eglise' or dataset == 'custom' or dataset == 'wazgha':
        ext = 'JPG'
    else:
        ext = 'png'
    
    if dataset == 'templering':
        subdir = 'templeRing' 
    elif dataset == 'dingoring':
        subdir = 'dinoRing'
    elif dataset == 'eglise':
        subdir = 'eglise'
    elif dataset == 'custom':
        subdir = 'custom'
    else:
        raise ValueError(f"Unknown dataset: {dataset}") 

    images: List[np.ndarray] = []
    for idx in range(count):
        path = base_dir / subdir / f"{idx:02d}.{ext}"
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        images.append(img)

    return images, CAMERA_MATRICES[dataset]

