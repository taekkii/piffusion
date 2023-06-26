
import cv2
import numpy as np
import open3d as o3d
from scipy.interpolate import RectBivariateSpline

def debug_essential_matrix(Im_i, Im_j, x_i, x_j, K, E, percentile_print=1.0):
    """
    Im_i, Im_j: ndarray(h,w,3)
    x_i, x_j: ndarray(#inliers,2) (NORMALIZED!!)
    K: ndarray(3,3)
    E: ndarray(3,3)
    """
    # numerically accurate 3x3 inverse of K (rather than nparray.inv())
    K_inv = np.array([[1.0 / K[0,0] , 0.0        , -K[0,2]/K[0,0]],
                      [0.0          , 1.0/K[1,1] , -K[1,2]/K[1,1]],
                      [0.0          , 0.0        , 1.0           ]])

    # calc fundamental matrix F (makes my life easier)
    F = K_inv.T @ (E @ K_inv)
    
    for p1, p2 in zip(x_i,x_j):
        p1x = int(p1[0])
        p1y = int(p1[1])
        p2x = int(p2[0])
        p2y = int(p2[1])
        
        if np.random.random()<=percentile_print:
            color = np.random.randint((0,0,0),(255,255,255))
            r,g,b = map(int,color)
            color = (r,g,b)
            sq=3
            cv2.rectangle(Im_i,(p1x-sq,p1y-sq),(p1x+sq,p1y+sq),color,2)
            cv2.rectangle(Im_j,(p2x-sq,p2y-sq),(p2x+sq,p2y+sq),color,2)

            # epipolar line on Im_j
            l2 = F@np.concatenate([p1,np.ones(1)])
            a,b,c = l2[0], l2[1], l2[2]
            cv2.line(Im_j,(-10,int((10*a-c)/b)),(1000,int((-1000*a-c)/b)),color,1)
            
            # epipolar line on Im_i
            l1 = F.T@np.concatenate([p2,np.ones(1)])
            a,b,c = l1[0], l1[1], l1[2]
            cv2.line(Im_i,(-10,int((10*a-c)/b)),(1000,int((-1000*a-c)/b)),color,1)



    cv2.imwrite("./dbg_essential.png",np.concatenate([cv2.cvtColor(Im_i,cv2.COLOR_RGB2BGR),cv2.cvtColor(Im_j,cv2.COLOR_RGB2BGR)],axis=1))
    exit()
def debug_track_matrix(Im, track, K):
    """
    Im: ndarray(n,h,w,3)
    track: ndarray(n,F,2)
    K: ndarray(3,3)
    
    visualize track matrix matching for every pair and exits.
    """
    import shutil
    import os

    dir_dbg = "./track_dbg/"
    if os.path.exists(dir_dbg):
        shutil.rmtree(dir_dbg)
    os.makedirs(dir_dbg, exist_ok=True)
    N = Im.shape[0]

    for i in range(N):
        for j in range(i+1,N):
            image_i = Im[i].copy()
            image_j = Im[j].copy()

            for p1, p2 in zip(track[i],track[j]):
                if p1[0] == -1.0 or p1[1] == -1.0 or p2[0] == -1.0 or p2[0] == -1.0: continue
                # p1x = int(p1[0])
                # p1y = int(p1[1])
                # p2x = int(p2[0])
                # p2y = int(p2[1])
                p1x = int(K[0,0]*p1[0] + K[0,2])
                p1y = int(K[1,1]*p1[1] + K[1,2])
                p2x = int(K[0,0]*p2[0] + K[0,2])
                p2y = int(K[1,1]*p2[1] + K[1,2])
                if np.random.random()<=1.00:
                    color = np.random.randint((0,0,0),(255,255,255))
                    r,g,b = map(int,color)
                    color = (r,g,b)
                    sq=3
                    cv2.rectangle(image_i,(p1x-sq,p1y-sq),(p1x+sq,p1y+sq),color,2)
                    cv2.rectangle(image_j,(p2x-sq,p2y-sq),(p2x+sq,p2y+sq),color,2)

            cv2.imwrite(os.path.join(dir_dbg,f"{i}_{j}.png"),np.concatenate([cv2.cvtColor(image_i,cv2.COLOR_RGB2BGR),cv2.cvtColor(image_j,cv2.COLOR_RGB2BGR)],axis=1))
    exit()


def debug_triangulation(R, C, X , colors=None):
    """
    Visualize Camera Poses (R,C) / Reconstructed Points (X) to .ply file, and exits.
    Parameters
    ----------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    X : ndarray of shape (F, 3)
        The set of reconstructed 3D points
    colors: (optional) ndarray of shape (F, 3)
        RGB color of 3D points.
    Returns
        Nothing
    -------
    """

    # [Cam 1]: Just identity.
    m_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
    
    # [Cam 2]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = C
    m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
    m.transform(T)
    m_cam += m
    import os, shutil
    
    output_dir = "./triangulation_dbg"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    o3d.io.write_triangle_mesh(os.path.join(output_dir,"cameras.ply"), m_cam)

    X = X[X[:,0]!=-1,:]
    if colors is None:
        colors = np.zeros_like(X)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(output_dir,'points.ply'), pcd)

    exit()