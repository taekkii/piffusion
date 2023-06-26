
import numpy as np

from feature import EstimateE_RANSAC
import taek_debug

def GetCameraPoseFromE(E):
    """
    Find four conﬁgurations of rotation and camera center from E

    Parameters
    ----------
    E : ndarray of shape (3, 3)
        Essential matrix

    Returns
    -------
    R_set : ndarray of shape (4, 3, 3)
        The set of four rotation matrices
    C_set : ndarray of shape (4, 3)
        The set of four camera centers
    """
    # SVD of E.
    U, _, Vh = np.linalg.svd(E)
    
    # t is third column of U (plus or minus)
    t1 = U[:,2]
    t2 = -t1

    # Two possible
    W1 = np.array([[0.0, -1.0, 0.0],
                   [1.0,  0.0, 0.0],
                   [0.0,  0.0, 1.0]])
    W2 = W1.T

    R_set, C_set = [], []
    for W,t in [(W1, t1), (W1, t2), (W2, t1), (W2, t2)]:
        R = U @ (W @ Vh)
        C = -R.T @ t
        
        R_set.append(R)
        C_set.append(C)
    R_set = np.array(R_set)
    C_set = np.array(C_set)

    return R_set, C_set



def Triangulation(P1, P2, track1, track2):
    """
    Use the linear triangulation method to triangulation the point

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    track1 : ndarray of shape (F, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (F, 2)
        Point correspondences from pose 2

    Returns
    -------
    X : ndarray of shape (F, 3)
        The set of 3D points
    """
    X = []
    for pt1, pt2 in zip(track1, track2):
        
        # Set -1 for invalid points
        if pt1[0] == -1.0 or pt1[1] == -1.0 or pt2[0] == -1.0 or pt2[0] == -1.0:
            X.append(np.ones(3) * -1.0)
            continue

        x1, y1 = pt1[0], pt1[1] # scalar each.
        x2, y2 = pt2[0], pt2[1] # scalar each.
        p1_1, p1_2, p1_3 = P1[0,:], P1[1,:], P1[2,:] # (4) each.
        p2_1, p2_2, p2_3 = P2[0,:], P2[1,:], P2[2,:] # (4) each.
        
        # Formulate AX = 0.
        A = np.stack([x1*p1_3 - p1_1,
                      y1*p1_3 - p1_2,
                      x2*p2_3 - p2_1,
                      y2*p2_3 - p2_2])

        # Do SVD, last column of V is answer (again).
        _, _, Vh = np.linalg.svd(A)
        
        X_homog = Vh[-1,:] # (4) 
        X_inhomog = X_homog[:-1] / X_homog[-1] # (3)
        X.append(X_inhomog)
    
    X = np.array(X)

    return X



def EvaluateCheirality(P1, P2, X):
    """
    Evaluate the cheirality condition for the 3D points

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    X : ndarray of shape (n, 3)
        Set of 3D points

    Returns
    -------
    valid_index : ndarray of shape (n,)
        The binary vector indicating the cheirality condition, i.e., the entry 
        is 1 if the point is in front of both cameras, and 0 otherwise
    """
    n = X.shape[0]

    # Filter out unmatched.
    mask1 = X[:,0] != -1
    
    # Cheirality Test.
    masks = [mask1]
    for P in [P1, P2]:
        X_homog = np.concatenate([X, np.ones((n,1))], axis=1) # (n,4)
        
        # World 2 Camera coordinates.
        X_cam = ( P @ X_homog.reshape(n,4,1) ).reshape(n,3)

        # Filter out "behind-the-camera" points.
        masks.append(X_cam[:,2] >= 0.0)
    
    # Pass 3 tests: valids.
    valid_index = masks[0] * masks[1] * masks[2]
    return valid_index



def EstimateCameraPose(track1, track2):
    """
    Return the best pose conﬁguration

    Parameters
    ----------
    track1 : ndarray of shape (F, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (F, 2)
        Point correspondences from pose 2

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    X : ndarray of shape (F, 3)
        The set of reconstructed 3D points
    """
    
    F = track1.shape[0]
    # Build corresponding 2d point arrays from track
    # NOTE TO SELF: x1, x2 are normalized coordinates.
    x1, x2 = [], []
    for p1, p2 in zip(track1, track2):
        if p1[0] == -1.0 or p1[1] == -1.0 or p2[0] == -1.0 or p2[0] == -1.0: continue
        x1.append(p1)
        x2.append(p2)

    x1, x2 = np.stack(x1), np.stack(x2)

    # Compute Essential Matrix.        
    E, _ = EstimateE_RANSAC(x1,x2,ransac_n_iter=2000,ransac_thr=1e-1)
    
    R_set, C_set = GetCameraPoseFromE(E)
    
    ans = -1
    X = np.ones((F,3)) * -1
    print("="*50)
    print("Finding correct configuration...")
    print("="*50, "\n")
    
    for i, (R,C) in enumerate(zip(R_set, C_set)):
        t = -R@C
        P2 = np.concatenate([R,t.reshape(3,1)],axis=1) # (3,4)

        # Without loss of generality
        P1 = np.concatenate([np.eye(3),np.zeros((3,1))],axis=1) # (3,4)

        X_candidate = Triangulation(P1, P2, track1, track2)
        valid = EvaluateCheirality(P1, P2, X_candidate)
        
        print(f"config[{i}] : {valid.sum()}")
        if (X[:,0] != -1).sum() < valid.sum():
            ans = i
            X = np.ones((F,3)) * -1
            X[valid] = X_candidate[valid]
            
    
    assert ans != -1
    R, C = R_set[ans], C_set[ans]
    
    if 0:
        taek_debug.debug_triangulation(R, C, X)
    
    return R, C, X