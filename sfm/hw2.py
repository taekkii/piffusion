import os
import cv2
import numpy as np

import open3d as o3d
from scipy.interpolate import RectBivariateSpline

from feature import BuildFeatureTrack
from camera_pose import EstimateCameraPose
from camera_pose import Triangulation
from camera_pose import EvaluateCheirality
from pnp import PnP_RANSAC
from pnp import PnP_nl
from reconstruction import FindMissingReconstruction
from reconstruction import Triangulation_nl
from reconstruction import RunBundleAdjustment

import os, shutil
import taek_debug

import yaml

if __name__ == '__main__':
    # [Jeongtaek Oh] For my convenience.
    np.set_printoptions(precision=4, suppress=True, threshold=np.inf) # , linewidth = np.inf)
    config = {'ransac_E_iters': 200,
              'ransac_E_threshold': 1e-1,
              'ransac_PnP_iters': 200,
              'ransac_PnP_threshold': 5e-1,
              'ply_export_type': 'default',
              'fx': 463.1,
              'fy': 463.1,
              'cx': 333.2,
              'cy': 187.5}
    np.random.seed(100)

    # [Jeongtaek Oh Edit] We will configure K later.
    # K = np.asarray([
    #     [463.1, 0, 333.2],
    #     [0, 463.1, 187.5],
    #     [0, 0, 1]
    # ])

    # Load input images [Jeongtaek Oh] EDITED to indicate different input directories.
    Im = []
    img_dir = "./my_data2"
    # img_dir = "./images"
    # img_dir = "./images_mid"

    # Load config from file, use default if not exists.
    try:
        with open(os.path.join(img_dir,"config.yaml"),'r') as fp:
            config.update( yaml.load(fp,Loader=yaml.FullLoader) )
    except FileNotFoundError:
        print(f"Cannot find: {os.path.join(img_dir, 'config.yaml')}. Using default settings.")
    K = np.eye(3)
    K[0,0] = config['fx']
    K[1,1] = config['fy']
    K[0,2] = config['cx']
    K[1,2] = config['cy']
        
    for im_file in sorted(os.listdir(img_dir)):
        if im_file[-4:].lower() in [".jpg",".png"]:
            full_path = os.path.join(img_dir, im_file)
            im = cv2.imread(full_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            Im.append(im)
    Im = np.stack(Im)
    num_images, h_im, w_im, _ = Im.shape

    # Build feature track
    track = BuildFeatureTrack(Im, K, config["ransac_E_iters"], config["ransac_E_threshold"])

    track1 = track[0,:,:]
    track2 = track[1,:,:]

    # Estimate ﬁrst two camera poses
    R, C, X = EstimateCameraPose(track1, track2)
    
    output_dir = 'output'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir)

    # Set of camera poses
    P = np.zeros((num_images, 3, 4))

    # Set first two camera poses
    P[0] = np.eye(4)[:3,:] # Without loss of generality
    P[1] = np.concatenate([R,(-R@C).reshape(3,1)],axis=1) # (3,4)

    # [Jeongtaek Oh] Debug
    if 0:
        # x = (K @ X.reshape(-1,3,1)).reshape(-1,3) # (F,3)
        # x = x[:,:2] / x[:,2].reshape(-1,1)
        # colors = np.stack( [Im[0,int(coord[0]),int(coord[1])]/255 if 0<=int(coord[0])<=Im.shape[1]-1 and 0<=int(coord[1])<=Im.shape[2]-1 else np.zeros(3) for coord in x ] ) # (F,3) : RGB
        
        R, C, X = EstimateCameraPose(track[0], track[2])
        taek_debug.debug_triangulation(R, C, X) #, colors)
    
    # ransac_n_iter = 200
    # ransac_thr = 0.5
    
    print("="*30)
    print("[Starting SFM]")
    print("="*30)
    print()
        
    for i in range(1, num_images):
        print(f"[Im #{i+1}]\n")
        if i>1:
            mask = (track[i,:,0] != -1) * (X[:,0] != -1)
            
            # Estimate new camera pose
            # R,C,_ = EstimateCameraPose(track[0],track[i])
            R, C, inlier = PnP_RANSAC(X[mask], track[i, mask], config["ransac_PnP_iters"], config["ransac_PnP_threshold"])
            # TODO use inliers only
            R, C = PnP_nl(R, C, X[mask][inlier], track[i, mask][inlier])
            
            # Add new camera pose to the set
            P[i] = np.concatenate([R,(-R@C).reshape(3,1)],axis=1) # (3,4)
            print(f"Registered Img[{i+1}]")
            for j in range(i):
                print(f" Reconstructing from img[{j+1}]-img[{i+1}]...")
                # Fine new points to reconstruct
                idx = FindMissingReconstruction(X, track[i])
                if idx.sum() == 0:
                    break

                # Triangulate points
                # FIXME: Use all point? or missing points only?
                X_add = Triangulation(P[i], P[j], track[i], track[j]) # (F,3)
                
                # Filter out points based on cheirality
                valid = EvaluateCheirality(P[i], P[j], X_add)
                X_add[~valid] = -1.

                # I think refining Triangulation belongs here, after cheirality test (unlike pseudo code as written in PDF)
                # because the function only deals with valid 3d points.                    
                X_add[valid] = Triangulation_nl(X_add[valid],P[i], P[j], track[i,valid], track[j,valid]) 
    
                # Update 3D points            
                X[idx] = X_add[idx]

        # Run bundle adjustment
        valid_ind = X[:, 0] != -1
        X_ba = X[valid_ind, :]
        track_ba = track[:i + 1, valid_ind, :]
        P_new, X_new = RunBundleAdjustment(P[:i + 1, :, :], X_ba, track_ba)
        P[:i + 1, :, :] = P_new
        X[valid_ind, :] = X_new

        ###############################################################
        # Save the camera coordinate frames as meshes for visualization
        m_cam = None
        for j in range(i+1):
            R_d = P[j, :, :3]
            C_d = -R_d.T @ P[j, :, 3]
            T = np.eye(4)
            T[:3, :3] = R_d
            T[:3, 3] = C_d
            m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            m.transform(T)
            if m_cam is None:
                m_cam = m
            else:
                m_cam += m
        o3d.io.write_triangle_mesh('{}/cameras_{}.ply'.format(output_dir, i+1), m_cam)

        # Save the reconstructed points as point cloud for visualization
        X_new_h = np.hstack([X_new, np.ones((X_new.shape[0],1))])
        colors = np.zeros_like(X_new)
        for j in range(i, -1, -1):
            x = X_new_h @ P[j,:,:].T
            x = x / x[:, 2, np.newaxis]
            mask_valid = (x[:,0] >= -1) * (x[:,0] <= 1) * (x[:,1] >= -1) * (x[:,1] <= 1)
            uv = x[mask_valid,:] @ K.T
            for k in range(3):
                interp_fun = RectBivariateSpline(np.arange(h_im), np.arange(w_im), Im[j,:,:,k].astype(float)/255, kx=1, ky=1)
                colors[mask_valid, k] = interp_fun(uv[:,1], uv[:,0], grid=False)
        ind = np.sqrt(np.sum(X_ba ** 2, axis=1)) < 200
        
        if config['ply_export_type'] == 'default':
            # Use default export by given code.
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X_new[ind]))
            pcd.colors = o3d.utility.Vector3dVector(colors[ind])
            o3d.io.write_point_cloud('{}/points_{}.ply'.format(output_dir, i+1), pcd)
        elif config['ply_export_type'] == 'forward_facing':
            # [Jeongtaek Oh Edition] Plausible visualization for extra credit with custom images:
            # Meaning all images are oriented toward z axis 
            # i.e. [0,0,1].T @ cam_z_axis is small for all cameras.
            # In this case, we clip between z=n plane to z=f plane (determined by certain portion of reconstructed points.)
            # And normalize the remainings.

            n_percentile, f_percentile = 0.01, 0.06
            X_output = X_new[ind]
            
            # Sort by Z_order
            sort_ind = np.argsort(X_output[:,2])
            X_output = X_output[sort_ind]
            colors = colors[sort_ind]
            
            # near-far clip 
            nb, fb = int(X_output.shape[0]*n_percentile),  int(X_output.shape[0]*(1.0 - f_percentile))
            X_output = X_output[nb:fb]
            colors = colors[nb:fb]

            # normalize
            scale = np.abs(X_output).max()
            X_output /= scale
            
            # recalculate cams too.
            m_cam = None
            for j in range(i+1):
                R_d = P[j, :, :3]
                C_d = -R_d.T @ P[j, :, 3]
                C_d /= scale
                T = np.eye(4)
                T[:3, :3] = R_d
                T[:3, 3] = C_d
                m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
                m.transform(T)
                if m_cam is None:
                    m_cam = m
                else:
                    m_cam += m

            o3d.io.write_triangle_mesh('{}/cameras_{}.ply'.format(output_dir, i+1), m_cam)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X_output))
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud('{}/points_{}.ply'.format(output_dir, i+1), pcd)

            
        else:
            raise NotImplementedError("Unknown ply_export type.")