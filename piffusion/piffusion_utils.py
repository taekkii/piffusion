
import torch
import torchvision

import numpy as np

import os
import shutil

from scipy.spatial.transform import Rotation
import open3d as o3d

@torch.no_grad()
def save_13channel_image(path, img):
    """
    img: [b x 13 x h x w]
    """
    b,_,h,w = img.shape
    img1 = img[:,:3]
    img2 = img[:,3:6]
    
    q = img[:,6:10]
    t = img[:,10:]
        
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path,exist_ok=True)

    torchvision.utils.save_image(img1/2 + 0.5, os.path.join(path,"img1.png"))
    torchvision.utils.save_image(img2/2 + 0.5, os.path.join(path,"img2.png"))
    
    torchvision.utils.save_image(q[:,0].view(-1,1,h,w)/2 + 0.5, os.path.join(path,"img_q0.png"))
    torchvision.utils.save_image(q[:,1].view(-1,1,h,w)/2 + 0.5, os.path.join(path,"img_q1.png"))
    torchvision.utils.save_image(q[:,2].view(-1,1,h,w)/2 + 0.5, os.path.join(path,"img_q2.png"))
    torchvision.utils.save_image(q[:,3].view(-1,1,h,w)/2 + 0.5, os.path.join(path,"img_q3.png"))
    
    torchvision.utils.save_image(t/2 + 0.5, os.path.join(path,"img_t.png"))

def get_pose_img(poses,h,w):
    """
    Generate  pose image from poses
    Args
        poses: ndarray(b,3,4)
        h: int
        w: int
    Return
        pose_image: ndarray(b,h,w,7)
    """
    b = poses.shape[0]
    R, t = np.split(poses,[3],axis=-1)     #[b x 3 x 3] [b x 3 x 1]

    # rotation
    R = Rotation.from_matrix(R).as_quat() #(b,4)
    t = t.reshape((-1,3)) # (b,3)

    R = np.broadcast_to(R.reshape((b,1,1,4)),(b,h,w,4)).copy() # (b,h,w,4)
    t = np.broadcast_to(t.reshape((b,1,1,3)),(b,h,w,3)).copy() # (b,h,w,3)
    return np.concatenate([R,t],axis=-1) # (b,h,w,7)

def get_pose_from_pose_img(pose_img):
    """
    Get pose from pose image

    Parameters
    ----------
    pose_img: ndarray(b, h, w, 7)
        pose image.
    Returns
    -------
    pose: ndarray(b, 3, 4)
        pose matrix.
    """
    pose_vec = np.mean(pose_img,axis=(1,2)) # (b,7)
    R = Rotation.from_quat(pose_vec[:,:4]).as_matrix() # (b, 3, 3)
    t = pose_vec[:,4:].reshape((-1,3,1)) # (b,3,1)

    return np.concatenate([R,t],axis=-1) # (b,3,4)



def gen_render_path(c2ws, N_views=30):
    N = len(c2ws)
    rotvec, positions = [], []
    rotvec_inteplat, positions_inteplat = [], []
    weight = np.linspace(1.0, .0, N_views//2, endpoint=False).reshape(-1, 1)
    for i in range(N):
        r = Rotation.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0])>180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
        positions.append(c2ws[i, :3, 3:].reshape(1, 3))

        if i:
            rotvec_inteplat.append(weight * rotvec[i - 1] + (1.0 - weight) * rotvec[i])
            positions_inteplat.append(weight * positions[i - 1] + (1.0 - weight) * positions[i])

    rotvec_inteplat.append(weight * rotvec[-1] + (1.0 - weight) * rotvec[0])
    positions_inteplat.append(weight * positions[-1] + (1.0 - weight) * positions[0])

    c2ws_render = []
    angles_inteplat, positions_inteplat = np.concatenate(rotvec_inteplat), np.concatenate(positions_inteplat)
    for rotvec, position in zip(angles_inteplat, positions_inteplat):
        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_euler('xyz', rotvec, degrees=True).as_matrix()
        c2w[:3, 3:] = position.reshape(3, 1)
        c2ws_render.append(c2w.copy())
    c2ws_render = np.stack(c2ws_render).astype(np.float32)
    return c2ws_render



def get_betas(beta_start, beta_end, T):
    '''
    get betas (noise scedule)
    Args:
        beta_start: start value of beta (0.0001)
        beta_end: end value of beta (0.02)
        T: number of pre-defined time stamps (NO NEED TO CHANGE)
    Returns:
        betas: noise schedule (torch.Tensor [T,])
    '''
    return torch.linspace(beta_start, beta_end, T)


def visualize_pose(pose: np.array, output_dir:str, filename:str="pose_visulization.ply"):
    """
    Visualize pose, save to .ply file

    Parameters
    ----------
    output_dir : str
        path to output
    pose:
        ndarray (b, h, w, 7) or ndarray(b, 3, 4) or scipy.spatial.transform.Rotation (b)
        pose image or pose matrix or batch of rotations (translation will be (0,0,0))
    Returns
    -------
        None
    """
    if isinstance(pose, Rotation):
        pose = Rotation.as_matrix(pose) # (b,3,3)
        b = pose.shape[0]
        pose = np.concatenate([pose, np.zeros((b,3,1))], axis=-1)
    if pose.shape[-1] == 7:
        pose = get_pose_from_pose_img(pose)
    
    m_cam = None
    for i, P in enumerate(pose):
        if i >= 5:
            break
        R_d = P[:, :3]
        C_d = P[:, 3]
        T = np.eye(4)
        T[:3, :3] = R_d
        T[:3, 3] = C_d
        m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        m.transform(T)
        if m_cam is None:
            m_cam = m
        else:
            m_cam += m
    output_path = os.path.join(output_dir, filename)
    o3d.io.write_triangle_mesh(output_path, m_cam)



# DEBUG
# if __name__ == "__main__":
#     import piffusion_data
#     _, _, pose = piffusion_data.prepare_data(data_path="./data/lego")
#     visualize_pose(pose[:500], output_dir="./",filename="lego.ply")

