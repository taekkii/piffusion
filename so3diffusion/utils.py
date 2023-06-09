
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import os
import torch

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
    if isinstance(pose, R):
        pose = R.as_matrix(pose) # (b,3,3)
        b = pose.shape[0]
        pose = np.concatenate([pose, np.zeros((b,3,1))], axis=-1)
    # if pose.shape[-1] == 7:
    #     pose = get_pose_from_pose_img(pose)
    
    m_cam = None
    for P in pose:
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

