

import piffusion_data
import piffusion_utils
import pallete_inference

import numpy as np
import os
import cv2

def save_result(args, imgs_arr, extrinsic_arr, result_path, model, skip=16):
    
    OUTPUT_DIR = os.path.join(result_path, "project_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model.eval()    
    
    s, n, h, w, _ = imgs_arr.shape
    

    for idx_s in range(s):
        
        scene_name = f"{idx_s:03d}"
        OUTPUT_DIR_SCENE = os.path.join(OUTPUT_DIR, scene_name)
        
        idx1_n = np.arange(0,n-skip,skip)
        idx2_n = np.arange(skip,n,skip)

        img1 = imgs_arr[idx_s, idx1_n]
        img2 = imgs_arr[idx_s, idx2_n]
        pose1 = extrinsic_arr[idx_s, idx1_n]
        pose2 = extrinsic_arr[idx_s, idx2_n]
        
        prepared_data = piffusion_data.sample_helper(img1, img2, pose1, pose2, bbox=(-4.1,4.1), return_type="tensor", noise=0.0)
        piffusion_utils.save_13channel_image(OUTPUT_DIR_SCENE, prepared_data)
        
        img1_path = os.path.join(OUTPUT_DIR_SCENE, "img1")
        img2_path = os.path.join(OUTPUT_DIR_SCENE, "img2")
        os.makedirs(img1_path, exist_ok=True)
        os.makedirs(img2_path, exist_ok=True)
        
        for i, (x1, x2) in enumerate(zip(img1,img2)):
            cv2.imwrite(os.path.join(img1_path,f"{i:03d}.png"), cv2.cvtColor(x1, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(img2_path,f"{i:03d}.png"), cv2.cvtColor(x2, cv2.COLOR_RGB2BGR))
            
        x = pallete_inference.ddpm_sampling(prepared_data[:, :6], model, prepared_data.shape[0], 1000)
        
        pose_img = (x[:,6:].permute(0,2,3,1)).cpu().numpy()
        relative_poses = piffusion_utils.get_pose_from_pose_img(pose_img)

        result_poses = [pose1[0]]
        cumul = False
        for i, relative_pose in enumerate(relative_poses):
            if cumul:
                result_poses.append( piffusion_data.back_to_absolute_pose(relative_pose.reshape(1,3,4), (-4.1,4.1), result_poses[-1]).reshape((3,4)) )
            else:
                result_poses.append(piffusion_data.back_to_absolute_pose(relative_pose.reshape(1,3,4), (-4.1,4.1), pose1[i]).reshape((3,4)) )
        result_poses = np.stack(result_poses)

        piffusion_utils.visualize_pose(pose=result_poses, output_dir=OUTPUT_DIR_SCENE)
        
        np.save(os.path.join(OUTPUT_DIR_SCENE, "result.npy"), result_poses)
        
            

# # DBG
# if __name__ == "__main__":
    
#     save_result(None, "./data/nerf_synthetic_merge", result_path="./results/merge_pallete")
    