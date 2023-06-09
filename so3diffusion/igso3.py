
import torch

PI = torch.pi

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

@torch.no_grad()
def ternary_search(eps, precision=1e-3, device="cpu"):
    lb, ub = 0, PI - 1e-4
    while ub-lb > precision:
        # print(lb, ub)
        x = torch.linspace(lb, ub, 5000, device=device)
        fx = f(x, eps)
        maxidx = fx.argmax()
        idx1 = max(0, maxidx-1)
        idx2 = min(maxidx+1,4999)
        lb = x[idx1].item()
        ub = x[idx2].item()

    return (lb+ub) / 2

def bsearch(eps, lb, ub, mode="ascending", precision=1e-3, device="cpu"):
    while ub-lb > precision:
        x = (lb+ub)/2
        fx = f(torch.tensor([x],device=device), eps)
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

cache = {}
@torch.no_grad()
def sample(n, eps, device="cpu"):
    global cache
    if eps not in cache:
        mode_loc = ternary_search(eps, device=device)
        # print("[MODE LOC] =" , mode_loc)
        
        lb = bsearch(eps, 0, mode_loc, device=device)
        ub = bsearch(eps, mode_loc, PI - 1e-4,mode="descending", device=device)
        # print(f"[BOUND] = ({lb},{ub})")

        N = 5000
        w = torch.linspace(lb+(ub-lb)/N, ub, N, device=device)
        pdf_val = f(w, eps) 

        cdf_val = torch.cat([torch.zeros(1,device=pdf_val.device), pdf_val.cumsum(dim=0)])
        cdf_val = cdf_val / cdf_val[-1]
        w = torch.cat([torch.zeros(1,device=w.device), w])
        cache[eps] = (w,cdf_val)

    w, cdf_val = cache[eps]
    w = w.to(device)
    cdf_val = cdf_val.to(device)

    uni = torch.rand(n).to(device)

    r_idx = torch.searchsorted(cdf_val, uni)
    l_idx = r_idx-1
    
    wr = cdf_val[r_idx] - uni
    wl = uni - cdf_val[l_idx]
    sample_w = w[l_idx] * (wr/(wr+wl) ) + w[r_idx] * (wl/(wr+wl))
    return sample_w

# betas = torch.linspace(0.001, 0.02, 100)

# # betas = torch.linspace(0.0001, 0.02, 10)

# # Reparametrizations.
# alphas = 1.0 - betas
# alpha_bars = alphas.cumprod(dim=0)
# sigmas = torch.sqrt(betas)
# for eps in betas:
#     print("w=", eps.item())
#     plot_noise(eps.item(),"cuda")

