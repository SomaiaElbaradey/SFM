# Structure from Motion Implementation 

## Overview

This project implements a complete Structure from Motion (SfM) pipeline that reconstructs 3D scenes from sequences of 2D images. The implementation builds a sparse 3D point cloud through feature detection, matching, camera pose estimation, triangulation, and bundle adjustment.

## Theory

### Structure from Motion Pipeline

Structure from Motion is a photogrammetric technique that estimates 3D structures from 2D image sequences. The process works by:

1. **Feature Detection & Matching**: Identifying distinctive points in each image and finding correspondences between them.
2. **Initial Pair Selection**: Choosing a good first pair of images with sufficient parallax and feature overlap.
3. **Relative Pose Estimation**: Computing the essential matrix and recovering camera pose.
4. **Triangulation**: Reconstructing 3D points from matched 2D points and known camera poses.
5. **Incremental Reconstruction**: Adding more images to the reconstruction through PnP (Perspective-n-Point).
6. **Bundle Adjustment**: Refining both camera poses and 3D points to minimize reprojection error.

### Camera Model

We use the pinhole camera model represented by its intrinsic matrix:

$$
K = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

Where:
- $(f_x, f_y)$ are the focal lengths in pixel units
- $(c_x, c_y)$ is the principal point (optical center) in pixel coordinates

### Epipolar Geometry

For two images, the Essential matrix $E$ encodes the epipolar constraints:

$$
x_2^T E x_1 = 0
$$

Where $x_1$ and $x_2$ are normalized image coordinates of corresponding points. From $E$, we can extract the relative rotation $R$ and translation $t$ between cameras.

### Perspective-n-Point (PnP)

PnP solves for camera pose given 3D-to-2D correspondences. For a 3D point $X$ and its 2D projection $x$:

$$
x = K[R|t]X
$$

We use RANSAC-PnP to robustly estimate camera pose in the presence of outliers.

### Bundle Adjustment

Bundle adjustment is a large sparse geometric optimization problem that minimizes reprojection error:

$$
\min_{R_i, t_i, X_j} \sum_{i,j} d(P_i X_j, x_{ij})^2
$$

Where:
- $P_i = K[R_i|t_i]$ is the projection matrix for camera $i$
- $X_j$ is the 3D point $j$
- $x_{ij}$ is the observed 2D point in image $i$ corresponding to 3D point $j$
- $d(\cdot,\cdot)$ is the reprojection error (typically Euclidean distance)

## Methodology

### Feature Detection & Matching

We use SIFT (Scale-Invariant Feature Transform) for feature detection and description due to its invariance to scale, rotation, and illumination changes. Features are matched between image pairs using a brute-force matcher with L1 norm and Lowe's ratio test to filter ambiguous matches. We further remove outliers using RANSAC with the fundamental matrix constraint.

### Reconstruction Initialization

1. We select an initial image pair with many inlier matches and significant rotation.
2. The essential matrix is computed and decomposed to obtain the relative pose.
3. Initial 3D points are triangulated from matched 2D points.

### Incremental Reconstruction

The system grows the reconstruction by:
1. Selecting the next image to add using a sequential selection strategy that follows circular paths in image sequences.
2. Finding 3D-2D correspondences between already reconstructed points and keypoints in the new image.
3. Estimating the new camera pose using robust PnP.
4. Triangulating new 3D points between the newly added image and already reconstructed images.
5. Periodically running bundle adjustment to refine the reconstruction.

### Bundle Adjustment

We implement sparse bundle adjustment using the Levenberg-Marquardt algorithm with a carefully designed sparsity pattern that exploits the structure of the problem. This optimization refines both camera poses and 3D point positions to minimize the overall reprojection error.

## Results

The implementation successfully reconstructs 3D scenes from both the Temple Ring and Dino Ring datasets:

- The system successfully triangulates points with low reprojection errors (typically <2 pixels).
- Camera poses are estimated accurately, allowing consistent growth of the reconstruction.
- Bundle adjustment significantly improves reconstruction quality, reducing mean reprojection error.
- The final point cloud captures the structure of the scene with good accuracy.

## How to Run the Code

### Prerequisites

- Python 3.6 or higher
- Required packages: 
  - OpenCV (cv2)
  - NumPy
  - SciPy
  - Open3D (for visualization)
  - Matplotlib (for plotting)

Install dependencies:

```bash
pip install numpy opencv-python scipy open3d matplotlib
```
# Structure from Motion Implementation

## Directory Structure
Ensure your project has the following structure:

```
project/
├── datasets/
│   ├── templeRing/
│   │   ├── 00.png
│   │   ├── 01.png
│   │   └── ...
│   └── dinoRing/
│       ├── 00.png
│       ├── 01.png
│       └── ...
├── feature_matching/
│   ├── __init__.py
│   ├── feature_extraction.py
│   ├── feature_matching.py
│   ├── image_load.py
│   ├── outlier_remover.py
│   └── utilies.py
├── reconstructioning/
│   ├── __init__.py
│   ├── data_structure.py
│   ├── geometry.py
│   ├── initialization.py
│   ├── pnp.py
│   ├── reconstruct.py
│   ├── reprojection_error.py
│   ├── selection.py
│   ├── triangulation.py
│   └── utils.py
├── bundle_adjustement/
│   ├── __init__.py
│   ├── projection.py
│   ├── residuals.py
│   ├── solver.py
│   └── sparsity.py
└── main.py # Script to run the reconstruction
```

## Running the Reconstruction
1. Make sure you have the dataset images in the appropriate folders.
2. Run the main script:

```bash
python main.py
```

3. You can modify the parameters in `main.py` to change:
   * The dataset (templeRing or dinoRing)
   * Number of images to use
   * Bundle adjustment frequency
   * Visualization options

## Results

The reconstruction results will be saved in the `results` folder. Here are some sample visualizations:

![Temple Reconstruction](./results/result.gif)

*Note: Replace the file paths with the actual paths to your GIF files in the results folder.*

## Dataset Preparation
The code expects images to be in the following location:
* Temple dataset: `./datasets/templeRing/00.png`, `./datasets/templeRing/01.png`, etc.
* Dino dataset: `./datasets/dinoRing/00.png`, `./datasets/dinoRing/01.png`, etc.

## Implementation Details

### Key Classes and Functions
* **Feature Extraction and Matching**
   * `extract_sift_features()`: Detects SIFT keypoints and computes descriptors.
   * `match_sift_descriptors()`: Performs kNN matching with ratio test.
   * `filter_correspondences()`: Removes outliers using RANSAC and the fundamental matrix.
* **Reconstruction**
   * `select_initial_image_pair()`: Chooses the best pair to start reconstruction.
   * `initialize_reconstruction()`: Sets up the initial two cameras and 3D points.
   * `select_next_image_pair()`: Determines which image to add next.
   * `fetch_pnp_correspondences()`: Prepares 3D-2D correspondences for PnP.
   * `estimate_pose_pnp()`: Computes the pose of a new camera.
   * `triangulate_and_reproject()`: Creates new 3D points from image pairs.
* **Bundle Adjustment**
   * `bundle_adjust()`: Performs sparse bundle adjustment optimization.
   * `ba_residuals()`: Computes reprojection errors for bundle adjustment.
   * `build_ba_sparsity()`: Constructs the Jacobian sparsity pattern.
* **Data Structures**
   * `Point3DWithViews`: Associates a 3D point with its 2D observations.

### Algorithmic Decisions
1. **Initial Pair Selection**: We choose image pairs with many matches and significant rotation to ensure good triangulation.
2. **Incremental Image Addition**: Images are added in a pattern that follows the ring-like sequence of the datasets.
3. **RANSAC Parameters**: We use a 5.0 pixel threshold for PnP RANSAC to handle noisy keypoints.
4. **Bundle Adjustment Scheduling**: We run bundle adjustment at increasing intervals as the reconstruction grows to balance performance and accuracy.
5. **Point Filtering**: We filter out triangulated points that are too far from the scene to remove outliers.

## Discussion

### Strengths
* The implementation successfully handles ring-like camera paths, common in object-centered photography.
* Incremental reconstruction with periodic bundle adjustment provides robust results
* The use of SIFT features provides good invariance to viewing conditions.
* The RANSAC-based outlier removal produces clean reconstructions.

### Limitations
* Dense reconstruction is not implemented; the result is a sparse point cloud.
* No texture mapping is applied to the reconstructed points.
* Performance could be improved with GPU acceleration for feature extraction and matching.
* The system assumes calibrated cameras with known intrinsics.

### Future Work
* Implement dense reconstruction to create complete surface models.
* Add texture mapping to produce photorealistic 3D models.
* Include automatic camera calibration for uncalibrated image sets.
* Optimize for larger datasets using hierarchical approaches.
* Add support for unordered image collections.

## Acknowledgments
This implementation is based on multiple computer vision and structure from motion techniques. The datasets (Temple Ring and Dino Ring) are from the Middlebury Multi-View Stereo dataset.

## References
