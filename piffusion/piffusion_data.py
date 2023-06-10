
import numpy as np
import os
import torch

from scipy.spatial.transform import Rotation


def prepare_data(data_path:str):
    imgs = np.load(os.path.join(data_path,"imgs.npy"))
    intrinsic = np.load(os.path.join(data_path,"intrinsic.npy"))
    extrinsic = np.load(os.path.join(data_path,"extrinsic.npy"))
    print(f"[LOAD DATA] Successfully loaded data from {data_path}")
    return imgs, intrinsic, extrinsic


def sample_helper(img1, img2, pose1, pose2, bbox=(-1.0,1.0), return_type="numpy", noise=0.04):
    """
    TODO: Rename terrible function name.
    """
    
    img_cat = np.concatenate([img1,img2],axis=-1) # (b,h,w,6)
    img_cat = img_cat.astype(float) / 255.0 # (b,h,w,6)
    img_cat = img_cat * 2.0 - 1.0
    
    b, h, w, _ = img_cat.shape
    # Relative Rotations.
    R1 = Rotation.from_matrix(pose1[:,:,:3])
    R2 = Rotation.from_matrix(pose2[:,:,:3])
    R = R2.inv() * R1
    
    # Relative Translations.
    t1 = pose1[:,:,3] # (b,3)
    t2 = pose2[:,:,3] # (b,3)
    t = t2 - t1 # (b,3)

    t -= bbox[0]
    t /= (bbox[1] - bbox[0])
    t  = (t - 0.5) * 2.0
    
    # Make quarternion image.
    rot_q = R.as_quat() #(b,4)
    q_img = rot_q.reshape((b,1,1,4)) # (b,1,1,4)
    q_img = np.broadcast_to(q_img, (b,h,w,4) ).copy()

    # Make translation image.
    t_img = np.broadcast_to(t.reshape((b,1,1,3)), (b,h,w,3) ).copy()
    
    # Concatenate
    result = np.concatenate([img_cat, q_img, t_img],-1) # (b,h,w,13) 
    result[:,:,:,6:] += np.random.normal(0, noise,size=(b,h,w,7))

    if return_type == "numpy":
        return result
    elif return_type == "tensor":
         return torch.from_numpy( np.transpose(result, axes=(0, 3, 1, 2)) ).float().cuda() # [b,13,h,w]
    else:
        raise NotImplementedError(f"Unseen return type: '{return_type}'")

def sample(batch_size, imgs, extrinsic, bbox=(-1.0,1.0), train_val_ratio=0.95, data_type="train", return_type="numpy"):
    partition = int(imgs.shape[0]*train_val_ratio)
    if data_type == "train":
        imgs = imgs[:partition]
    elif data_type == "val":
        imgs = imgs[partition:]
    else:
        raise NotImplementedError()
    n = imgs.shape[0]
    idx1 = np.random.choice(n,batch_size) # (b)
    idx2 = np.random.choice(n,batch_size) # (b)
    
    # Prepare Images, extrinsic

    img1 = imgs[idx1] # (b,h,w,3)
    img2 = imgs[idx2] # (b,h,w,3)
    pose1 = extrinsic[idx1] # (b,3,4)
    pose2 = extrinsic[idx2] # (b,3,4)

    return sample_helper(img1, img2, pose1, pose2, bbox, return_type)
    

def prepare_data_multiple_scenes(data_paths:list):
    
    imgs_arr, intrinsic_arr, extrinsic_arr = [], [], []
    for data_path in data_paths:
        imgs, intrinsic, extrinsic = prepare_data(data_path)
        imgs_arr.append(imgs)
        intrinsic_arr.append(intrinsic)
        extrinsic_arr.append(extrinsic)

    imgs_arr = np.stack(imgs_arr, axis=0) # (num_scenes, n_imgs, h, w, 3)
    intrinsic_arr = np.stack(intrinsic_arr, axis=0) # (num_scenes, n_imgs, 3, 3)
    extrinsic_arr = np.stack(extrinsic_arr, axis=0) # (num_scenes, n_imgs, 3, 4)

    return imgs_arr, intrinsic_arr, extrinsic_arr 

def sample_multiple_scenes(batch_size, imgs_arr, extrinsic_arr, bbox=(-1.0,1.0), train_val_ratio=0.95, data_type="train", return_type="numpy"):

    partition = int(imgs_arr.shape[1]*train_val_ratio)
    if data_type == "train":
        imgs_arr = imgs_arr[:,:partition]
    elif data_type == "val":
        imgs_arr = imgs_arr[:,partition:]
    else:
        raise NotImplementedError()
    
    s, n, h, w, _ = imgs_arr.shape
    
    idx_s = np.random.choice(s, batch_size)
    
    idx1_n = np.random.choice(n, batch_size)
    idx2_n = np.random.choice(n, batch_size)

    img1 = imgs_arr[idx_s, idx1_n]
    img2 = imgs_arr[idx_s, idx2_n]
    pose1 = extrinsic_arr[idx_s, idx1_n]
    pose2 = extrinsic_arr[idx_s, idx2_n]
    
    return sample_helper(img1, img2, pose1, pose2, bbox, return_type)

def back_to_absolute_pose(relative_pose, bbox, pivot_pose):
    """
    given relative poses, denormalize and get absolute pose by transform from pivot.
    ALL POSES ARE c2w !!
    ----------------------
    relative_pose: (b,3,4)
    bbox: tuple of range
    pivot_pose: (3,4) or (b,3,4)
    """

    b = relative_pose.shape[0]

    if len(pivot_pose.shape) == 2:
        pivot_pose = pivot_pose.reshape((1,3,4))
    
    ### [TRANSLATION] ###
    t = relative_pose[:,:,3] # (b,3)
    t1 = pivot_pose[:,:,3] # (1,3) or (b,3)

    # Denormalize.
    t = t / 2.0 + 0.5 
    t *= (bbox[1] - bbox[0])
    t += bbox[0]

    t2 = t + t1 # (b,3)

    ### [ROTATION] ###
    r = Rotation.from_matrix(relative_pose[:,:3,:3])
    r1 = Rotation.from_matrix(pivot_pose[:,:3,:3]) 
    r2 = r1 * r.inv()
    
    result = np.zeros((b,3,4))
    result[:,:3,:3] = r2.as_matrix()
    result[:,:3, 3] = t2

    return result


# import piffusion_utils

# if __name__ == "__main__":
#     DATA_PATH="./data/lego_sphere"
#     np.set_printoptions(precision=4, suppress=True)
#     img_arr, _ , extrinsic_arr = prepare_data_multiple_scenes(data_paths=[os.path.join(DATA_PATH, directory) for directory in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, directory))])
#     # import pdb;pdb.set_trace()
#     data = sample_multiple_scenes(64, img_arr, extrinsic_arr, bbox=(-4,4), return_type="tensor")
    
#     piffusion_utils.save_13channel_image("./results/tmp", data)
# if __name__ == "__main__":
#     DATA_PATH="./data/nerf_synthetic_merge"
#     np.set_printoptions(precision=4, suppress=True)
#     img_arr, _ , extrinsic_arr = prepare_data_multiple_scenes(data_paths=[os.path.join(DATA_PATH, directory) for directory in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, directory))])
#     # import pdb;pdb.set_trace()
#     data, pose1, pose2 = sample_multiple_scenes(2, img_arr, extrinsic_arr, bbox=(-4,4))
#     pose_img = data[:,:,:,6:]
#     poses = piffusion_utils.get_pose_from_pose_img(pose_img)
    
#     pose_back = back_to_absolute_pose(poses,(-4,4),pose1)
#     print(pose2)
#     print(pose_back)
