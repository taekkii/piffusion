import cv2
import numpy as np
import scipy
import taek_debug

def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
    """

    def get_pairs(des_ref, des_target, threshold=0.8):
        # For each descriptors in image #1, find a nearest neighbors
        dist_matrix = scipy.spatial.distance.cdist(des_ref, des_target, metric='euclidean') # [n1 x n2]
        nearest_2 = np.argpartition(dist_matrix, 2 ,axis=1)[:,:2] #[n1 x 2]
        good_pairs = []
        for ref_i, (target_i1, target_i2) in enumerate(nearest_2):
            if dist_matrix[ref_i,target_i1]/dist_matrix[ref_i,target_i2] < threshold:
                good_pairs.append((ref_i, target_i1))
        
        return good_pairs
    
    
    # bidirectionally finding good matches
    pairs_1_2 = get_pairs(des1, des2)
    pairs_2_1 = get_pairs(des2, des1)
    
    pairs = [(i1,i2) for i1,i2 in pairs_1_2 if (i2,i1) in pairs_2_1]
    
    pairs = list(set(pairs))

    x1 = []
    x2 = []
    ind1 = []

    for i1,i2 in pairs:
        x1.append(loc1[i1])
        x2.append(loc2[i2])
        ind1.append(i1)

    x1 = np.array(x1)
    x2 = np.array(x2)
    ind1 = np.array(ind1)

    return x1, x2, ind1



def EstimateE(x1, x2):
    """
    Estimate the essential matrix, which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    """
    
    def get_essential_matrix_segment(x1,y1,x2,y2):
        """
        format: p2Ep1 = 0
        x1,y1, x2,y2 are single floats
        returns: 1x9 matrix segment
        """
        return np.array([x2*x1 , x2*y1  , x2, y2*x1, y2*y1, y2, x1, y1, 1.0])
    # Build matrix "A" (a stack of equations) of Af=0.
    A = np.stack([get_essential_matrix_segment(p1[0],p1[1],p2[0],p2[1]) for p1,p2 in zip(x1,x2)])
    
    # Do SVD to compute f
    _, _, Vh = np.linalg.svd(A, full_matrices=True)

    # Assuming A is full rank, pick f = last column of V.     
    f = Vh[-1,:]
    E_fullrank = f.reshape(3,3)

    # Force rank(E)=2 by SVD
    U, _ , Vh = np.linalg.svd(E_fullrank, full_matrices=True)
    Sigma = np.array([[1.0, 0.0, 0.0], 
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0]])
    E = U @ (Sigma @ Vh)
    return E



def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the essential matrix robustly using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    inlier : ndarray of shape (k,)
        The inlier indices
    """
     # Guard
    n = x1.shape[0]
    assert n == x2.shape[0] and x1.shape[1] == x2.shape[1] == 2
    
    
    E, inlier = None, np.array([])
    
    for _ in range(ransac_n_iter):
        
        # Choose 8 candidate points.
        idx = np.random.choice(n,8,replace=False)
        x1_chosen, x2_chosen = x1[idx], x2[idx]
        
        # Estimate E using 8 points.
        E_cand = EstimateE(x1_chosen, x2_chosen)

        x1_homogeneous = np.pad(x1,((0,0),(0,1)),constant_values=1.0) # [nx3]
        x2_homogeneous = np.pad(x2,((0,0),(0,1)),constant_values=1.0) # [nx3]
        
        l2 = (E_cand @ x1_homogeneous.reshape(n,3,1)).reshape(n,3) # [nx3]
        inlier_cand = np.arange(n)[ np.abs((l2*x2_homogeneous).sum(axis=1)) / np.linalg.norm(l2[:,:2],axis=1)  < ransac_thr**2 ]
        if len(inlier) < len(inlier_cand):
            E, inlier = E_cand, inlier_cand
    
    return E, inlier



def BuildFeatureTrack(Im, K, ransac_iters, ransac_threshold):
    """
    Build feature track

    Parameters
    ----------
    Im : ndarray of shape (N, H, W, 3)
        Set of N images with height H and width W
    K : ndarray of shape (3, 3)
        Intrinsic parameters
    ransac_iters: int
        RANSAC iters
    ransac_threshold: float
        RANSAC threshold
    Returns
    -------
    track : ndarray of shape (N, F, 2)
        The feature tensor, where F is the number of total features
    """
    loc, des = [], []

    # numerically accurate 3x3 inverse of K (rather than nparray.inv())
    K_inv = np.array([[1.0 / K[0,0] , 0.0        , -K[0,2]/K[0,0]],
                      [0.0          , 1.0/K[1,1] , -K[1,2]/K[1,1]],
                      [0.0          , 0.0        , 1.0           ]])

    # Extract features
    for img in Im:    
        sift = cv2.SIFT_create()
        
        loc_i, des_i = sift.detectAndCompute(img, None)
        loc_i = np.array([p.pt for p in loc_i])
        loc.append(loc_i)
        des.append(des_i)

    N = Im.shape[0]
    track = np.ones((N,0,2))
    
    def normalize_coordinates(x):
        """
        returns noormalize coordinate of x: ndarray(n,2)
        """
        nonlocal K_inv
        x_nml = np.zeros_like(x)
        x_nml[:,0] = x[:,0]*K_inv[0,0] + K_inv[0,2] 
        x_nml[:,1] = x[:,1]*K_inv[1,1] + K_inv[1,2]
        return x_nml
    print(f"# of image = {N}")
    for i, (loc_i, des_i) in enumerate(zip(loc, des)):
        track_i = np.ones((N,loc_i.shape[0],2)) * -1 # (n,F_i,2)
        for j, (loc_j, des_j) in enumerate(zip(loc[i+1:],des[i+1:]),start=i+1):
            print(f"Matching Img[{i+1}] - Img[{j+1}]...")
            # Match features between the ith and jth images ▷ MatchSIFT
            x_i, x_j, ind_i = MatchSIFT(loc_i,des_i,loc_j,des_j)

            # Normalize coordinate by multiplying the inverse of intrinsics.
            x_i_nml = normalize_coordinates(x_i)
            x_j_nml = normalize_coordinates(x_j)
                        
            # Find inliner matches using essential matrix ▷ EstimateE RANSAC
            E, inlier = EstimateE_RANSAC(x_i_nml, x_j_nml, ransac_n_iter=ransac_iters, ransac_thr=ransac_threshold)

            # Update track_i using the inlier matches.
            track_i[j,ind_i[inlier]] = x_j_nml[inlier]

            print(f"# of inliers = {inlier.shape[0]}\n")
            if 1:
                taek_debug.debug_essential_matrix(Im[i], Im[j], x_i, x_j, K, E, percentile_print=0.1)
            
        # Remove features in track_i that have not been matched for i + 1, · · · , N.
        mask_i = (track_i >= 0.0).sum(axis=(0,2)) != 0
        track_i[i,:,:] = normalize_coordinates(loc_i)
        track_i = track_i[:,mask_i,:]
        
        # track = track ∪ track_i
        # NOTE: there could be possibly better method: probably DFS?
        track = np.concatenate((track,track_i),axis=1)
        
        mask = track[i,:,0] != -1.0
        _,unique_index = np.unique(track[i,mask], return_index=True, axis=0)
        track = np.concatenate( ( track[:,~mask],track[:,mask][:,unique_index]), axis=1)
        print(f"Finished until [Im#{i+1}] : track.shape={track.shape}")
        print("==================================\n")

    if 0:
        taek_debug.debug_track_matrix(Im, track, K)
    

    return track
