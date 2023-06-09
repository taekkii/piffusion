
from scipy.spatial.transform import Rotation as R
import numpy as np
from utils import visualize_pose
import os


if __name__ == "__main__":
    r1 = R.random().as_quat().reshape(1,4) # (1,4)
    r2 = R.random().as_quat().reshape(1,4) # (1,4)
    w = np.linspace(0,1,512).reshape(512,1)  # (512,1)
    r = R.from_quat(r1*(1-w) + r2*w)
    
    path = os.path.join("./test_data/")
    os.makedirs(path, exist_ok=True)
    
    visualize_pose(r, path, "rotation.ply")
    np.save(os.path.join(path,"matrices.npy"), R.as_matrix(r))
