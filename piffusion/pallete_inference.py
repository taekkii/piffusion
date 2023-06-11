
import torch
import tqdm
import piffusion_utils

import piffusion_data
import os

@torch.no_grad()
def ddpm_sampling(condition, model, b, T=1000):
    '''
    Generate batched image using DDPM sampling     
    Args:
        condition: [img1,img2] tensor [6,h,w]
        model: DDPM model
        b: batch size
        T (optional): number of pre-defined time stamps (NO NEED TO CHANGE)
    Returns:
        x: generated images
    '''

    
    betas = piffusion_utils.get_betas(0.0001, 0.02, T).cuda()

    # Reparametrizations.
    alphas = 1.0 - betas
    alpha_bars = alphas.cumprod(dim=0)
    sigmas = torch.sqrt(betas)
    
    # X_T ~ N(0,I).
    x = torch.randn(b, 7, 64, 64).cuda() # [b, 7, 64, 64]
    x = torch.cat([condition, x], dim=1) # [b, 13, 64, 64]
    timestamp = torch.ones(b).long().cuda() * T

    for t in tqdm.trange(T, 0,-1):
        alpha, alpha_bar, sigma = alphas[t-1], alpha_bars[t-1], sigmas[t-1]
        z = torch.randn_like(x[:, 6:]) if t > 1 else torch.zeros_like(x[:, 6:])
        x[:, 6:] = alpha**-0.5 * (x[:, 6:] - (1.0 - alpha)/(1.0 - alpha_bar)**0.5 * model(x, timestamp) ) + sigma*z
        
        timestamp = timestamp - 1
        
    return x

@torch.no_grad()
def validate(args, imgs, extrinsic, result_path, model, b, T=1000):
    DATA_TYPE = args.validate_on
    print(f"DATA_TYPE = {DATA_TYPE}")
    torch.cuda.empty_cache()
    model.eval()

    # val_sample = piffusion_data.sample(b, imgs, extrinsic, bbox=(-4.1,4.1), data_type=DATA_TYPE, return_type="tensor") # [B, 13, h, w]       
    val_sample = piffusion_data.sample_multiple_scenes(b, imgs, extrinsic, bbox=(-4.1,4.1), data_type=DATA_TYPE, train_val_ratio=args.train_val_ratio, return_type="tensor") # [B, 13, h, w]       
    
    x = ddpm_sampling(val_sample[:, :6], model, b, 1000)

    piffusion_utils.save_13channel_image(path=os.path.join(result_path, "sample"), img=x)
    piffusion_utils.save_13channel_image(path=os.path.join(result_path, "gt"), img=val_sample)
    piffusion_utils.visualize_pose(pose=x[:,6:].permute(0,2,3,1).cpu().numpy(), output_dir=result_path, filename="sample.ply")
    piffusion_utils.visualize_pose(pose=val_sample[:,6:].permute(0,2,3,1).cpu().numpy(), output_dir=result_path, filename="gt.ply")