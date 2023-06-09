
import torch
import numpy as np

PI = torch.pi

def pdf(w,sigma,thresh=10000, small=1e-7):
    l = np.arange(thresh).astype(np.float128).reshape((-1,1))
    s = np.sum((2*l+1) * np.exp(-l*(l+1)*sigma**2) * np.sin((l+0.5)*w) / np.sin(w/2 + small), axis=0)
    # summ = np.zeros_like(w)
    # for l in range(thresh+1):
    #     summ = summ + float(2*l+1) * np.exp(-l*(l+1)*sigma**2) * np.sin((l+0.5)*w) / np.sin(w/2)
    return ((1.0-np.cos(w))/np.pi) * s

def f(w, eps, numerical_limit=10000, small=1e-8):
    """
    IG_SO(3)
    PARAMS
    ------
    w : torch.Tensor [n]
        angle between [0, pi]
    eps : float (possibly torch.Tensor [n])
          noise magnitude.
    numerical_limit : int
        value of limit
    RETURNS:
        f(w;0,eps)
    """
    if not torch.is_tensor(eps):
        eps = torch.tensor(eps, device=w.device)
    
    c = (1.0 - torch.cos(w)) / PI #[n]
    l = torch.arange(numerical_limit, device=w.device).float().view(-1,1) #[m,1]
    s = torch.sum((2*l+1) * torch.exp(-l*(l+1)*eps**2)  / torch.sin(w*0.5 + small) * torch.sin((l+0.5)*w) ,dim=0) #[n]

    return c*s
# PDF
# def f(w, eps, small=1e-8):
#     """
#     Numerical approximation of IG_SO(3)
#     PARAMS
#     ------
#     w : torch.Tensor [n]
#         angle between [0, pi]
#     eps : float (possibly torch.Tensor [n])
#           noise magnitude.
    
#     RETURNS
#     -------
#         f(w;0,eps)
#     """
#     if not torch.is_tensor(eps):
#         eps = torch.tensor(eps)
#     eps = eps.to(w.device)
#     return (1 - torch.cos(w))/PI * \
#            PI**0.5 * eps**-1.5 * torch.exp(eps/4) * torch.exp(-(w/2)**2 / eps) * \
#            (w - torch.exp(-PI**2/eps) * ((w - 2*PI) * torch.exp(PI*w/eps) + (w+2*PI) * torch.exp(-PI*w/eps))) / \
#            (2*torch.sin(w*0.5 + small))

# @torch.no_grad()
# def rejection_sampling(n, eps, small=1e-5,device='cuda'):
#     """
#     Rejetion sampling based on pdf f
#     PARAMS
#     ------
#     n : int
#         # of sample
#     eps : float (possibly torch.Tensor [n])
#           noise magnitude.
#     small : float
#         for numeric stability.
#     device : torch.device or str
#         device for return tensor.
#     RETURNS
#     -------
#         f(w;0,eps)
#     """
#     assert isinstance(eps,float) or (torch.is_tensor(eps) and eps.shape[0] == 1 and len(eps.shape) == 1)
#     lb, ub = torch.tensor(small,device=device), torch.tensor(torch.pi-small, device=device)


#     res = torch.empty(0, device=device)
    
#     while res.shape[0] < n:
#         w = torch.rand(n, device=device) * torch.pi
#         u = torch.rand(n, device=device)
#         accept = f(w, eps, small)/M
#         res = torch.cat([res, w[u<accept]])
    
#     return res[:n]

# [CHECK VALIDITY]


torch.set_printoptions(precision=5, sci_mode=False)

@torch.no_grad()
def ternary_search(eps, precision=1e-3):
    lb, ub = 0, PI - 1e-4
    while ub-lb > precision:
        # print(lb, ub)
        x = torch.linspace(lb, ub, 5000)
        fx = f(x, eps)
        maxidx = fx.argmax()
        idx1 = max(0, maxidx-1)
        idx2 = min(maxidx+1,4999)
        lb = x[idx1].item()
        ub = x[idx2].item()

    return (lb+ub) / 2

def bsearch(eps, lb, ub, mode="ascending", precision=1e-3):
    while ub-lb > precision:
        x = (lb+ub)/2
        fx = f(torch.tensor([x]), eps)
        # print(lb,ub,fx)
        if mode == "ascending":
            if fx < 1e-1:
                lb = x
            else:
                ub = x
        elif mode == "descending":
            if fx < 1e-1:
                ub = x
            else:
                lb = x
    return (lb+ub)/2
@torch.no_grad()
def plot_noise(eps, device="cpu"):
    mode_loc = ternary_search(eps)
    print("[MODE LOC] =" , mode_loc)
    
    lb = bsearch(eps, 0, mode_loc)
    ub = bsearch(eps, mode_loc, PI - 1e-4,mode="descending")
    print(f"[BOUND] = ({lb},{ub})")

    N = 5000
    w = torch.linspace(lb+(ub-lb)/N, ub, N, device=device)
    pdf_val = f(w, eps) 

    cdf_val = torch.cat([torch.zeros(1,device=pdf_val.device), pdf_val.cumsum(dim=0)])
    cdf_val = cdf_val / cdf_val[-1]
    w = torch.cat([torch.zeros(1,device=w.device), w])

    N_SAMPLE = 5000
    uni = torch.rand(N_SAMPLE).to(device)

    r_idx = torch.searchsorted(cdf_val, uni)
    l_idx = r_idx-1
    
    wr = cdf_val[r_idx] - uni
    wl = uni - cdf_val[l_idx]
    sample_w = w[l_idx] * (wr/(wr+wl) ) + w[r_idx] * (wl/(wr+wl))
    
    import matplotlib.pyplot as plt

    fig = plt.figure()

    plt.scatter(w[1:].cpu(), pdf_val.cpu(), s=1, c="r")

    # sample_w = rejection_sampling(5000,eps=eps,device='cpu')
    plt.scatter(sample_w.cpu(),torch.rand(N_SAMPLE)*pdf_val.max().cpu(),s=1,c='b')

    import os

    os.makedirs("./plots",exist_ok=True)
    save_path = os.path.join("./plots",f"plt_{eps:.4f}.png")
    fig.savefig(save_path)

betas = torch.linspace(0.001, 0.02, 100)

# betas = torch.linspace(0.0001, 0.02, 10)

# Reparametrizations.
alphas = 1.0 - betas
alpha_bars = alphas.cumprod(dim=0)
sigmas = torch.sqrt(betas)
for eps in betas:
    print("w=", eps.item())
    plot_noise(eps.item(),"cuda")

