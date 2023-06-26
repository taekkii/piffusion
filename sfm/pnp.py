import numpy as np

from utils import Rotation2Quaternion
from utils import Quaternion2Rotation


def PnP(X, x):
    """
    Implement the linear perspective-n-point algorithm

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    """
    def get_PnP_matrix_segment(x,y,X,Y,Z):
        """
        given 2d coord (x,y) matching 3d coord (X,Y,Z) (all float scalars),
        build  2x12 matrix for DLT.
        Returns
        -------
        matrix_segment : ndarray of shape (2, 12)
        """
        return np.array([[  X,   Y,   Z, 1.0, 0.0, 0.0, 0.0, 0.0, -x*X, -x*Y,  -x*Z, -x],
                         [0.0, 0.0, 0.0, 0.0,   X,   Y,   Z, 1.0, -y*X, -y*Y , -y*Z, -y]])
    
    # [Jeongtaek Oh] We can do better with P3P, 
    # but I feel too tired to implement that.
    # Just building DLT method as we have PnP_nl() anyways.

    # Build matrix "A" (a stack of equations) of Ap=0.
    A = np.concatenate([get_PnP_matrix_segment(x_i[0], x_i[1], X_i[0], X_i[1], X_i[2]) for x_i, X_i in zip(x, X)],axis=0)
    
    # Do SVD to compute p
    _, _, Vh = np.linalg.svd(A, full_matrices=True)

    # Assuming A is full rank, pick p = last column of V.     
    p = Vh[-1,:]
    P = p.reshape(3,4)
    
    # SO(3) constraint.
    U, sigma, Vh = np.linalg.svd(P[:,:3], full_matrices=True)
    s1 = sigma[0]
    R = U@Vh
    t = P[:,3]/s1
    C = -R.T @ t    
    return R, C



def PnP_RANSAC(X, x, ransac_n_iter, ransac_thr):
    """
    Estimate pose using PnP with RANSAC

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    inlier : ndarray of shape (n,)
        The indicator of inliers, i.e., the entry is 1 if the point is a inlier,
        and 0 otherwise
    """
    n = X.shape[0]
    R, C, inlier = None, None, np.array([])
    reproj_error = 1e9
    X_homog = np.concatenate([X, np.ones((n,1))], axis=1) # (n,4)
    for _ in range(ransac_n_iter):
        
        # Choose 6 candidate points.
        idx = np.random.choice(n, 6, replace=False)
        X_chosen, x_chosen = X[idx], x[idx]
        
        # Estimate R, C using 6 points.
        R_cand, C_cand = PnP(X_chosen, x_chosen)

        # Project X to normalized image coordinate.
        P_cand = np.concatenate([R_cand,(-R_cand @ C_cand).reshape(3,1)],axis=1) # (3,4)
        X_cam = ( P_cand @ X_homog.reshape(n,4,1) ).reshape(n,3) # (n,3)
        x_proj = X_cam[:,:-1] / X_cam[:,-1].reshape(n,1) # (n,2)
        
        # Inlier filter based on reprojection Error CONSIDERING cheirlality
        inlier_cand = np.arange(n)[ (X_cam[:,2] >= 0.0 ) * (np.linalg.norm(x_proj-x, axis=1)  < ransac_thr**2) ]
        if len(inlier) < len(inlier_cand):
            R, C, inlier = R_cand, C_cand, inlier_cand
            reproj_error = np.linalg.norm(x_proj[inlier]-x[inlier], axis=1).sum()
        
        
    print(f"[LINEAR RANSAC] Reproj_error:{reproj_error:9.5f}, #Inlier:{len(inlier)}/{n} \n")
    
    return R, C, inlier



def ComputePoseJacobian(p, X):
    """
    Compute the pose Jacobian

    Parameters
    ----------
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion
    X : ndarray of shape (3,)
        3D point

    Returns
    -------
    dfdp : ndarray of shape (2, 7)
        The pose Jacobian
    """
    C = p[:3]
    q = p[3:]
    R = Quaternion2Rotation(q)
    x = R @ (X - C)

    u = x[0]
    v = x[1]
    w = x[2]
    du_dc = -R[0,:]
    dv_dc = -R[1,:]
    dw_dc = -R[2,:]
    # df_dc is in shape (2, 3)
    df_dc = np.stack([
        (w * du_dc - u * dw_dc) / (w**2),
        (w * dv_dc - v * dw_dc) / (w**2)
    ], axis=0)

    # du_dR = np.concatenate([X-C, np.zeros(3), X-C])
    # dv_dR = np.concatenate([np.zeros(3), X-C, X-C])
    # dw_dR = np.concatenate([np.zeros(3), np.zeros(3), X-C])
    du_dR = np.concatenate([X-C, np.zeros(3), np.zeros(3)])
    dv_dR = np.concatenate([np.zeros(3), X-C, np.zeros(3)])
    dw_dR = np.concatenate([np.zeros(3), np.zeros(3), X-C])
    # df_dR is in shape (2, 9)
    df_dR = np.stack([
        (w * du_dR - u * dw_dR) / (w**2),
        (w * dv_dR - v * dw_dR) / (w**2)
    ], axis=0)


    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    # dR_dq is in shape (9, 4)
    dR_dq = np.asarray([
        [0, 0, -4*qy, -4*qz],
        [-2*qz, 2*qy, 2*qx, -2*qw],
        [2*qy, 2*qz, 2*qw, 2*qx],
        [2*qz, 2*qy, 2*qx, 2*qw],
        [0, -4*qx, 0, -4*qz],
        [-2*qx, -2*qw, 2*qz, 2*qy],
        [-2*qy, 2*qz, -2*qw, 2*qx],
        [2*qx, 2*qw, 2*qz, 2*qy],
        [0, -4*qx, -4*qy, 0],
    ])

    dfdp = np.hstack([df_dc, df_dR @ dR_dq])

    return dfdp


def PnP_nl(R, C, X, x):
    """
    Update the pose using the pose Jacobian

    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix refined by PnP
    c : ndarray of shape (3,)
        Camera center refined by PnP
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R_refined : ndarray of shape (3, 3)
        The rotation matrix refined by nonlinear optimization
    C_refined : ndarray of shape (3,)
        The camera center refined by nonlinear optimization
    """
    n = X.shape[0]
    q = Rotation2Quaternion(R)

    p = np.concatenate([C, q])
    n_iters = 20
    lamb = 1
    error = np.empty((n_iters,))
    for i in range(n_iters):
        R_i = Quaternion2Rotation(p[3:])
        C_i = p[:3]

        proj = (X - C_i[np.newaxis,:]) @ R_i.T
        proj = proj[:,:2] / proj[:,2,np.newaxis]

        H = np.zeros((7,7))
        J = np.zeros(7)
        for j in range(n):
            dfdp = ComputePoseJacobian(p, X[j,:])
            H = H + dfdp.T @ dfdp
            J = J + dfdp.T @ (x[j,:] - proj[j,:])
        
        delta_p = np.linalg.inv(H + lamb*np.eye(7)) @ J
        p += delta_p
        p[3:] /= np.linalg.norm(p[3:])

        error[i] = np.linalg.norm(proj - x)


    R_refined = Quaternion2Rotation(p[3:])
    C_refined = p[:3]
    return R_refined, C_refined