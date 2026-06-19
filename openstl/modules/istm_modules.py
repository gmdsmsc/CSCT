import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
     

def get_rotary_embeddings(seq_len, dim, device):
    freqs = 10000 ** (-torch.arange(0, dim, 2, device=device) / dim)
    pos = torch.arange(seq_len, device=device).unsqueeze(1)
    angles = pos * freqs
    sin, cos = angles.sin(), angles.cos()
    sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, dim)
    cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, dim)
    return sin, cos

def apply_rope(x, sin, cos):
    return x * cos + rotate_half(x) * sin

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)

def get_rope_cache(seq_len, head_dim, device):
    freqs = 10000 ** (-torch.arange(0, head_dim, 2, device=device) / head_dim)
    pos = torch.arange(seq_len, device=device).unsqueeze(1)
    angles = pos * freqs
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, head_dim)
    cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, head_dim)
    return sin, cos




class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # Rotary positional embedding
        sin, cos = get_rope_cache(n, q.shape[-1], x.device)
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

