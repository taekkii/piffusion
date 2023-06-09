
import torch
import torchvision
import numpy as np

import tqdm

from scipy.spatial.transform import Rotation

import piffusion_utils

import os

@torch.no_grad()
def ddpm_sampling(model, betas, b, device, T=1000):
    '''
    Generate batched image using DDPM sampling 
    Args:
        model: DDPM model
        betas: noise schedule
        b: batch size
        T (optional): number of pre-defined time stamps (NO NEED TO CHANGE)
    Returns:
        x: generated images
    '''

    # send all tensors to 'device' argument
    model = model.to(device)
    betas = betas.to(device)
    
    # Reparametrizations.
    alphas = 1.0 - betas
    alpha_bars = alphas.cumprod(dim=0)
    sigmas = torch.sqrt(betas)
    
    # X_T ~ N(0,I).
    x = torch.randn(b, 13, 64, 64, device=device)
    
    for t in tqdm.trange(T, 0,-1):
        alpha, alpha_bar, sigma = alphas[t-1], alpha_bars[t-1], sigmas[t-1]
        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        x = alpha**-0.5 * (x - (1.0 - alpha)/(1.0 - alpha_bar)**0.5 * model(x,torch.tensor([t for _ in range(b)],device=device)) ) + sigma*z
    
    return x

@torch.no_grad()
def validate(result_path,model, betas, b, device, T=1000):
    
    torch.cuda.empty_cache()
    model.eval()

    x0 = ddpm_sampling(model, betas, b, device, T)
    piffusion_utils.save_13channel_image(path=result_path, img=x0)

    return x0

@torch.no_grad()
def repaint(x_gt, mask, model, betas, device, T=1000):
    """
    mask = True is known.
    """
    # send all tensors to 'device' argument
    model = model.to(device)
    betas = betas.to(device)
    
    b = x_gt.shape[0]

    # Reparametrizations.
    alphas = 1.0 - betas
    alpha_bars = alphas.cumprod(dim=0)
    sigmas = torch.sqrt(betas)
    
    # X_T ~ N(0,I).
    x = torch.randn(b, 13, 64, 64, device=device)
    for t in tqdm.trange(T, 0,-1):
        alpha, alpha_bar, sigma = alphas[t-1], alpha_bars[t-1], sigmas[t-1]
        
        # Noise "Known" part.
        noise = torch.randn_like(x_gt[mask])
        x_known_t = alpha_bar**0.5 * x_gt[mask]  +  (1.0 - alpha_bar)**0.5 * noise 

        # Noise "Unknown" part.
        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        x = alpha**-0.5 * (x - (1.0 - alpha)/(1.0 - alpha_bar)**0.5 * model(x,torch.tensor([t for _ in range(b)],device=device)) ) + sigma*z

        x[mask] = x_known_t
        if 1 and (t%100 == 0 or t==1):
            piffusion_utils.save_13channel_image(f"./results/piffusion/debug/{t:07d}",img=x)
    return x



@torch.no_grad()
def validate_repaint(result_path, model, betas, device, T=1000, frames=64):
    """
    frames: int (interpolation frames)
    """
    torch.cuda.empty_cache()
    model.eval()
    x0 = ddpm_sampling(model, betas, 1, device, T) # [1 x 13 x h x w]
    
    img1 = x0[:, :3] # [1 x 3 x h x w]
    img2 = x0[:,3:6] # [1 x 3 x h x w]
    _, _, h, w = img1.shape
    

    # Compute render trajectory.
    pose1 = np.eye(3,4).reshape((1,3,4))
    
    q2 = x0[:,6:10].mean(dim=(-1,-2)) # [1 x 4]
    r2 = Rotation.from_quat(q2.cpu().numpy()).as_matrix() # (1,3,3)
    t2 = x0[:,10: ].mean(dim=(-1,-2)).view(1,3,1).cpu().numpy() # (1,3,1)
    # pose2 = np.concatenate([r2,t2],axis=-1) # (1,3,4)
    pose2 = np.eye(3,4).reshape(1,3,4)

    c2ws = np.concatenate([pose1,pose2],axis=0) # (2,3,4)
    poses = piffusion_utils.gen_render_path(c2ws=c2ws, N_views=frames)[:,:3,:] # (b,3,4)

    pose_image = piffusion_utils.get_pose_img(poses, 64, 64) # (b,h,w,7)
    pose_image = torch.from_numpy(pose_image).cuda()
    pose_image = pose_image.permute(0, 3, 1, 2) # [b x 7 x h x w]
    
    # Make x_gt using above information.
    b = pose_image.shape[0]
    img1 = img1.expand(b,-1,-1,-1).clone() # [b x 3 x h x w]
    img2 = img2.expand(b,-1,-1,-1).clone() # [b x 3 x h x w]
    x_gt = torch.cat([img1,img2,pose_image],dim=1).float() #[b, 13, h, w]

    mask = torch.ones_like(x_gt).bool()
    mask[:,3:6] = False
    gen_img = repaint(x_gt, mask, model, betas, 'cuda')[:,3:6] # [b x 3 x h x w]

    os.makedirs(result_path,exist_ok=True)
    
    img1 = img1
    img2 = img2
    gen_img = gen_img
    torchvision.utils.save_image(img1/2 + 0.5, os.path.join(result_path,"img1.png"))
    torchvision.utils.save_image(img2/2 + 0.5, os.path.join(result_path,"img2.png"))
    torchvision.utils.save_image(gen_img/2+0.5, os.path.join(result_path,"gen_img.png"))
    # torchvision.io.write_video(filename=result_path, video_array=gen_img.permute(0,2,3,1)/2+0.5,fps=16)
    return gen_img

