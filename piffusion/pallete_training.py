
import ddim
import os
import shutil
import yaml
import torch
import numpy as np
import cv2

import numpy as np
from scipy.spatial.transform import Rotation
import torchvision

from torch.nn import DataParallel
import tqdm

from tensorboardX import SummaryWriter

import configargparse

import piffusion_data
import piffusion_utils
import pallete_inference

import project

def parse_arg():
    parser = configargparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--data_name", type=str, default="nerf_synthetic_merge")

    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--expname","-e", type=str, default="merge_pallete")

    parser.add_argument("--model_config", type=str, default="ddim_pallete.yml")

    parser.add_argument("--sample_only", action="store_true")
    parser.add_argument("--project_output", action="store_true")

    parser.add_argument("--no_reload", action="store_true")
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_val_ratio", type=float, default=0.95)
    parser.add_argument("--validate_on", type=str, choices=["train", "val"], default="val")
    
    parser.add_argument("--noise_pose", type=float, default=0.0)
    
    parser.add_argument("--i_save", type=int, default=500)
    parser.add_argument("--i_val",  type=int, default=25000)
    parser.add_argument("--i_print", type=int, default=10)

    args = parser.parse_args()
    return args

t_ = None
if __name__ == '__main__':
    
    np.set_printoptions(precision=4, suppress=True)
    args = parse_arg()

    with open(os.path.join("configs", args.model_config), "r") as f:
        config = yaml.safe_load(f)
    config = ddim.dict2namespace(config)

    RESULT_PATH = os.path.join(args.result_dir, args.expname)
    SAVE_PATH = os.path.join(RESULT_PATH, "model.pth")
    DATA_PATH = os.path.join(args.data_dir, args.data_name)
    os.makedirs(RESULT_PATH, exist_ok=True)

    # save argument
    with open( os.path.join(RESULT_PATH, "args.txt"), "w") as fp:
        for k,v in args.__dict__.items():
            print(k, ":", v, file=fp)


    start = 0
    n_iters = 5000000
    B = args.batch_size
    grad_clip = 1.0
    
    # load data.
    
    imgs_arr, _ , extrinsic_arr = piffusion_data.prepare_data_multiple_scenes(data_path=DATA_PATH)
    
    # get model and load checkpoint (download if not exist)
    model = ddim.Model(config)    
    model = model.to('cuda')
    optimizer = torch.optim.Adam(params=model.parameters(),lr=0.0002,amsgrad=False,eps=1e-7)
    
    if not args.no_reload and os.path.exists(SAVE_PATH):
        print("[LOAD] Checkpoint Detected")
        sdict = torch.load(SAVE_PATH)
        start = sdict.get("train_i",0)
        model_state_dict = sdict["model_state_dict"]
        model.load_state_dict(model_state_dict)

        optimizer_state_dict = sdict.get("optimizer_state_dict", None)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        
        print("[LOAD] Success.")

    # Noise Schedulers.
    betas = piffusion_utils.get_betas(beta_start=0.0001, beta_end=0.02, T=1000) 
    betas = betas.cuda()

    # Reparametrizations.
    alphas = 1.0 - betas
    alpha_bars = alphas.cumprod(dim=0)
    sigmas = torch.sqrt(betas)
    
    if args.sample_only:
        print("[SAMPLE_ONLY]")
        pallete_inference.validate(args, imgs_arr, extrinsic_arr, os.path.join(RESULT_PATH,f"sample{start+1}"), model, 64)
        exit()
    
    if args.project_output:
        print("[PRODUCING PROJECT OUTPUT]")
        project.save_result(args, imgs_arr, extrinsic_arr, RESULT_PATH, model, skip=16)
        exit(0)

    tensorboard = SummaryWriter(os.path.join("results","piffusion"))
    model = DataParallel(model)
    
    
   
    I_SAVE = args.i_save
    I_VAL = args.i_val
    I_PRINT = args.i_print

    pbar = tqdm.tqdm(range(start+1,n_iters+1),desc=f"[Training]",initial=start,total=n_iters)
    for train_i in pbar:

        model.train()
        
        # x0 = piffusion_data.sample(B, imgs, extrinsic, bbox=(-4.1,4.1), return_type="tensor") # [B, 13, h, w]
        x0 = piffusion_data.sample_multiple_scenes(B, imgs_arr, extrinsic_arr, bbox=(-4.1,4.1), train_val_ratio=args.train_val_ratio, return_type="tensor", noise=args.noise_pose)
        t = (torch.rand(B,device='cuda')*1000).long() # [B]
        noise = torch.randn_like(x0[:, 6:]) # [B, 7, h, w]
        
        xt = torch.sqrt(alpha_bars[t]).view(-1,1,1,1)*x0[:, 6:]  +  torch.sqrt(1.0 - alpha_bars[t]).view(-1,1,1,1)*noise # [B, 7, h, w]
        noise_estimation = model(torch.cat([x0[:, :6], xt], dim=1), t) # [B, 7, h, w] 
        
        loss = (( noise - noise_estimation )**2 ).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        
        # [REST IS LOGGING]
        if train_i % I_SAVE == 0:
            sdict = dict(train_i=train_i,
                         model_state_dict=model.module.state_dict(),
                         optimizer_state_dict=optimizer.state_dict())
            torch.save(sdict,SAVE_PATH)
            print(f"[Checkpoint Saved] train_i={train_i}")
        
        if train_i % I_VAL == 0:
            print(f"[VALIDATE]train_i: {train_i}")
            pallete_inference.validate(args, imgs_arr, extrinsic_arr, os.path.join(RESULT_PATH,f"res{train_i}"), model, 64)

        if train_i % I_PRINT == 0:
            loss_print = loss.item()
            pbar.set_description(f"[Training] loss = {loss_print:.4f}")
            pbar.refresh()
            tensorboard.add_scalar("loss",loss_print)
            
    