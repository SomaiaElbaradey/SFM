import numpy as np
import cv2

from .data_structure import Point3DWithViews


def triangulate_and_reproject(
    R_left: np.ndarray,
    t_left: np.ndarray,
    R_right: np.ndarray,
    t_right: np.ndarray,
    K: np.ndarray,
    points3d_with_views: list,
    img_idx_left: int,
    img_idx_right: int,
    keypoints_left: np.ndarray,
    keypoints_right: np.ndarray,
    left_indices: list,
    right_indices: list,
    reproject: bool = True
):
    """
    Triangulate 3D points from two camera views and optionally compute reprojection errors.

    Args:
        R_lefteft, t_left: Rotation (3x3) and translation (3x1) of the left camera.
        R_right, t_right: Rotation (3x3) and translation (3x1) of the right camera.
        K: Intrinsic camera matrix.
        points3d_with_views: List to append Point3DWithViews instances.
        img_idx_left, img_idx_right: Identifiers for the two images.
        keypoints_left, keypoints_right: Arrays of matched 2D points of shape (N, 2).
        left_indices, right_indices: Original indices of the 2D points in each image.
        reproject: If True, compute and return reprojection errors.

    Returns:
        If reproject is False:
            List[Point3DWithViews]
        If reproject is True:
            Tuple(List[Point3DWithViews], List[Tuple[float, float]], float, float)
    """

    # Compute projection matrices for each camera
    print(f"Triangulating: {len(keypoints_left)} points.")
    P_left = np.dot(K, np.hstack((R_left, t_left)))
    P_right = np.dot(K, np.hstack((R_right, t_right)))

    # Prepare points for cv2 triangulation: shape (2, N)
    keypoints_left = np.squeeze(keypoints_left)

    keypoints_left = keypoints_left.transpose()
    keypoints_left = keypoints_left.reshape(2,-1)

    keypoints_right = np.squeeze(keypoints_right)

    keypoints_right = keypoints_right.transpose()
    keypoints_right = keypoints_right.reshape(2,-1)

    # Triangulate homogeneous coordinates and convert to 3D
    hom_points = cv2.triangulatePoints(P_left, P_right, keypoints_left, keypoints_right)
    points_3D = cv2.convertPointsFromHomogeneous(hom_points.transpose())

    for i in range(keypoints_left.shape[1]):
        source_2dpt_idxs = {img_idx_left:left_indices[i], img_idx_right:right_indices[i]}
        pt = Point3DWithViews(points_3D[i], source_2dpt_idxs)
        points3d_with_views.append(pt)

    if not reproject:
        return points3d_with_views

    keypoints_left = keypoints_left.transpose()
    keypoints_right = keypoints_right.transpose()

    # Convert rotations to Rodrigues vectors for projection
    rvec_left, _ = cv2.Rodrigues(R_left)
    rvec_right, _ = cv2.Rodrigues(R_right)

    # Project 3D points back to each image plane
    proj_left, _ = cv2.projectPoints(points_3D, rvec_left, t_left, K, distCoeffs=np.array([]))
    proj_right, _ = cv2.projectPoints(points_3D, rvec_right, t_right, K, distCoeffs=np.array([]))

    # Compute reprojection errors per point for each image
    errors_left = [
        np.linalg.norm(proj_left[i, 0] - keypoints_left[i])
        for i in range(points_3D.shape[0])
    ]
    errors_right = [
        np.linalg.norm(proj_right[i, 0] - keypoints_right[i])
        for i in range(points_3D.shape[0])
    ]
    
    avg_error_left = float(np.mean(errors_left))
    avg_error_right = float(np.mean(errors_right))
    print(f"Avg reprojection error on image {img_idx_left}: {avg_error_left:.3f} px")
    print(f"Avg reprojection error on image {img_idx_right}: {avg_error_right:.3f} px")

    # Pair per-point errors
    paired_errors = list(zip(errors_left, errors_right))
    return points3d_with_views, paired_errors, avg_error_left, avg_error_right

    # for i in range(len(projPoints_l)):
    #     delta_l.append(abs(projPoints_l[i][0][0] - keypoints_left[i][0]))
    #     delta_l.append(abs(projPoints_l[i][0][1] - keypoints_left[i][1]))
    #     delta_r.append(abs(projPoints_r[i][0][0] - keypoints_right[i][0]))
    #     delta_r.append(abs(projPoints_r[i][0][1] - keypoints_right[i][1]))
    # avg_erroR_left = sum(delta_l)/len(delta_l)
    # avg_error_r = sum(delta_r)/len(delta_r)
    # print(f"Average reprojection error for just-triangulated points on image {img_idx_left} is:", avg_erroR_left, "pixels.")
    # print(f"Average reprojection error for just-triangulated points on image {img_idx_right} is:", avg_error_r, "pixels.")
    # errors = list(zip(delta_l, delta_r))
    # return points3d_with_views, errors, avg_erroR_left, avg_error_r

