
import torch
import numpy as np

import pytorch3d.transforms
from pytorch3d.transforms import so3_log_map, so3_exponential_map
from model import Model
from utils import get_betas
import tqdm
import os
import igso3

I_SAVE = 100
I_VAL = 10
I_PRINT = 1
SAVE_PATH = "./results/model.pth"

@torch.no_grad()
def load_data(path="./test_data/matrices.npy"):
    return torch.from_numpy( np.load(path) ).float().cuda()

@torch.no_grad()
def sample_IGSO3(noise_level):
    """
    noise_level: [b] 
    """
    b = noise_level.shape[0]
    s = torch.randn(b,3).cuda()
    unit_s = s / s.norm(dim=1).view(b,1)
    # [TODO] Modify noise
    # w = torch.cat([igso3.sample(1, eps ,device='cuda') for eps in noise_level])
    w = torch.abs(torch.randn(b).cuda())
    return pytorch3d.transforms.axis_angle_to_matrix( unit_s*w.reshape(-1,1) )


@torch.no_grad()
def inference(model, betas, b, device, T=1000):
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
    x = sample_IGSO3(torch.zeros(b,device=device))
    
    for t in tqdm.trange(T, 0,-1):
        alpha, alpha_bar, sigma = alphas[t-1], alpha_bars[t-1], sigmas[t-1]
        # z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        z = torch.abs(torch.randn(b,device=device)) if t>1 else torch.eye(3,device=device)
        v = model(x.reshape(b,9), torch.tensor([t for _ in range(b)],device=device))
        a1 = so3_exponential_map(so3_log_map(x)*alpha_bar**-0.5)
        a2 = so3_exponential_map(v*(1-alpha_bar)**-0.5)
        import pdb; pdb.set_trace()
        # x = alpha**-0.5 * (x - (1.0 - alpha)/(1.0 - alpha_bar)**0.5 * model(x,torch.tensor([t for _ in range(b)],device=device)) ) + sigma*z
    
    return x

if __name__ == '__main__':
    
    data = load_data()
    data.requires_grad = False
    print("[LOAD DATA] complete")

    model = Model().cuda()
    print("[MODEL CREATE] complete")
    b = 64
    n_iters = 500000
    lr = 0.00001
    grad_clip = 1.0
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    start = 0

    if os.path.exists(SAVE_PATH) and 0:
        print("[LOAD] Checkpoint Detected")
        sdict = torch.load(SAVE_PATH)
        start = sdict.get("train_i",0)
        model_state_dict = sdict["model_state_dict"]
        model.load_state_dict(model_state_dict)

        optimizer_state_dict = sdict.get("optimizer_state_dict", None)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        
        print("[LOAD MODEL] Success.")

    # Noise Schedulers.
    betas = get_betas(beta_start=0.0011, beta_end=0.02, T=1000) 
    betas = betas.cuda()

    # Reparametrizations.
    alphas = 1.0 - betas
    alpha_bars = alphas.cumprod(dim=0)
    sigmas = torch.sqrt(betas)

    
    pbar = tqdm.tqdm(range(start+1, n_iters+1),total=n_iters, initial=start)

    loss_sum = 0.0
    for train_i in pbar:
        
        model.train()
        x = data.clone()
        for i in range(0,x.shape[0],b):
            x_batch = x[i:i+b]
            t = (torch.rand(b, device='cuda')*1000).long()
            alpha_bar = alpha_bars[t]

            R = sample_IGSO3(alpha_bar) # [b x 3 x 3]
            v = so3_log_map(R) / torch.sqrt(1.0-alpha_bar).view(b,1) # [b x 3]
            x_scale = so3_exponential_map( torch.sqrt(alpha_bar).view(b,1) * so3_log_map(x_batch) ) # [b x 3 x 3]
        
            loss = ((v - model((R @ x_scale).reshape(b,9),t) )**2).mean()
            loss.backward()
            loss_sum += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

        # shuffle data
        data = data[torch.randperm(len(data)),:]

        
        # [REST IS LOGGING]

        # if train_i < 10:
        #     print(f"loss : {loss.item():.5f}")
        if train_i % I_SAVE == 0:
            sdict = dict(train_i=train_i,
                         model_state_dict=model.state_dict(),
                         optimizer_state_dict=optimizer.state_dict())
            torch.save(sdict,SAVE_PATH)
            print(f"[Checkpoint Saved] train_i={train_i}")
        
        if train_i % I_VAL == 0:
            print(f"[VALIDATE]train_i: {train_i}")
            inference(model,betas,512,device='cuda')

        if train_i % I_PRINT == 0:
            loss_print = loss_sum / I_PRINT
            loss_sum = 0.0
            pbar.set_description(f"[Training] loss = {loss_print:.4f}")
            pbar.refresh()
            
    