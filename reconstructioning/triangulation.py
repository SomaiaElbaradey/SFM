import numpy as np
import cv2

from .data_structure import Point3DWithViews

# ---------- Triangulation ----------

def triangulate_and_reproject(R_l, t_l, R_r, t_r, K, 
                              points3d_with_views, 
                              img_idx1, img_idx2, kpts_i, 
                              kpts_j, kpts_i_idxs, 
                              kpts_j_idxs, 
                              reproject=True):
    """
    Triangulate points between two calibrated views and optionally compute reprojection errors.

    Returns:
        If reproject: (updated_points, list_of_errors, error_img1, error_img2)
        Otherwise: updated_points
    """

    print(f"Triangulating: {len(kpts_i)} points.")
    P_l = np.dot(K, np.hstack((R_l, t_l)))
    P_r = np.dot(K, np.hstack((R_r, t_r)))

    kpts_i = np.squeeze(kpts_i)

    kpts_i = kpts_i.transpose()
    kpts_i = kpts_i.reshape(2,-1)

    kpts_j = np.squeeze(kpts_j)

    kpts_j = kpts_j.transpose()
    kpts_j = kpts_j.reshape(2,-1)

    point_4d_hom = cv2.triangulatePoints(P_l, P_r, kpts_i, kpts_j)
    points_3D = cv2.convertPointsFromHomogeneous(point_4d_hom.transpose())
    for i in range(kpts_i.shape[1]):
        source_2dpt_idxs = {img_idx1:kpts_i_idxs[i], img_idx2:kpts_j_idxs[i]}
        pt = Point3DWithViews(points_3D[i], source_2dpt_idxs)
        points3d_with_views.append(pt)

    if reproject:
        kpts_i = kpts_i.transpose()
        kpts_j = kpts_j.transpose()

        rvec_l, _ = cv2.Rodrigues(R_l)
        rvec_r, _ = cv2.Rodrigues(R_r)

        projPoints_l, _ = cv2.projectPoints(points_3D, rvec_l, t_l, K, distCoeffs=np.array([]))
        projPoints_r, _ = cv2.projectPoints(points_3D, rvec_r, t_r, K, distCoeffs=np.array([]))

        delta_l , delta_r = [], []

        for i in range(len(projPoints_l)):
            delta_l.append(abs(projPoints_l[i][0][0] - kpts_i[i][0]))
            delta_l.append(abs(projPoints_l[i][0][1] - kpts_i[i][1]))
            delta_r.append(abs(projPoints_r[i][0][0] - kpts_j[i][0]))
            delta_r.append(abs(projPoints_r[i][0][1] - kpts_j[i][1]))

        avg_error_l = sum(delta_l)/len(delta_l)
        avg_error_r = sum(delta_r)/len(delta_r)

        print(f"Average reprojection error for just-triangulated points on image {img_idx1} is:", avg_error_l, "pixels.")
        print(f"Average reprojection error for just-triangulated points on image {img_idx2} is:", avg_error_r, "pixels.")

        errors = list(zip(delta_l, delta_r))

        return points3d_with_views, errors, avg_error_l, avg_error_r

    return points3d_with_views