

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SIREN(nn.Module):
    def __init__(self, n_layers, in_channel, out_channel):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_channel,out_channel)] + 
            [nn.Linear(out_channel,out_channel) for _ in range(n_layers - 1)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = torch.sin(layer(x))
        return x

class Temb(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.dim = embedding_dim
    
    def forward(self, timesteps):
        embedding_dim = self.dim
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb

class Transformer_Block(nn.Module):
    def __init__(self, n_tokens):
        super().__init__()
        self.Q = nn.Linear(n_tokens, n_tokens)
        self.K = nn.Linear(n_tokens, n_tokens)
        self.V = nn.Linear(n_tokens, n_tokens)
        self.Proj = nn.Linear(n_tokens, n_tokens)
        self.nonlinear = nn.SiLU()

    def forward(self, x):
        """
        x: [b x n]
        """
        b, n = x.shape
        q = self.Q(x).view(b,n,1)
        k = self.K(x).view(b,1,n)
        v = self.V(x).view(b,n,1)
        
        weight = torch.bmm(q, k) # [b x n x n]
        weight = F.softmax(weight,dim=2)

        attended = torch.bmm(weight, v).view(b,n)
        x = self.nonlinear(x + attended) # [b x n]

        x = self.nonlinear(x + self.Proj(x)) #[b x n]
        return x



class Model(nn.Module):

    def __init__(self, in_channel=9, out_channel=3, r_emb_dim=32, t_emb_dim=128, n_siren_layers=10, n_transformer_blocks=12):
        super().__init__()
        self.siren = SIREN(n_siren_layers, in_channel, r_emb_dim)
        self.temb = Temb(t_emb_dim)
        
        self.transformer_blocks = nn.ModuleList([Transformer_Block(r_emb_dim + t_emb_dim) for _ in range(n_transformer_blocks)])
        self.out_proj = nn.Linear(r_emb_dim + t_emb_dim, out_channel)

    def forward(self, x ,t):
        """
        x: [b x 9]
        t: [b] (.long() tensor)
        """
        r_emb = self.siren(x) # [b, r_emb_dim]
        t_emb = self.temb(t)  # [b, t_emb_dim]
        emb = torch.cat([r_emb,t_emb], dim=1)

        for blk in self.transformer_blocks:
            emb = blk(emb)
    
        return self.out_proj(emb)
    

if __name__ == "__main__":
    print('DEBUG')
    b = 128
    model = Model().cuda()
    x = torch.randn(b,9).cuda()
    t = (torch.rand(b) * 1000).long().cuda()
    print(model(x,t).shape)
