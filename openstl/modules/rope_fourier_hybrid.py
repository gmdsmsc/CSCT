import torch
from torch import nn, einsum
from einops import rearrange

# --- Rotary Positional Embedding ---
def apply_rope(x, sin, cos):
    return x * cos + rotate_half(x) * sin

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)

# --- Fourier Feature Generator ---
def fourier_features(pos, num_freqs, device):
    freqs = torch.arange(1, num_freqs + 1, device=device).float()
    pos = pos.unsqueeze(1)  # [seq_len, 1]
    return torch.cat([torch.sin(pos * freqs), torch.cos(pos * freqs)], dim=-1)  # [seq_len, 2*num_freqs]

# --- RoPE + Fourier Cache ---
def get_rope_fourier_cache(seq_len, head_dim, device, num_freqs=16):
    freqs = 10000 ** (-torch.arange(0, head_dim, 2, device=device) / head_dim)
    pos = torch.arange(seq_len, device=device).unsqueeze(1)
    angles = pos * freqs
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, head_dim)
    cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, head_dim)

    fourier = fourier_features(torch.arange(seq_len, device=device).float(), num_freqs, device)
    return sin, cos, fourier  # [seq_len, head_dim], [seq_len, head_dim], [seq_len, 2*num_freqs]

# --- Hybrid Attention Module ---
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_freqs=16):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_freqs = num_freqs

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.fourier_proj = nn.Linear(2 * num_freqs, dim_head)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # Positional encodings
        sin, cos, fourier = get_rope_fourier_cache(n, q.shape[-1], x.device, self.num_freqs)
        fourier_encoded = self.fourier_proj(fourier)  # [n, dim_head]
        fourier_encoded = rearrange(fourier_encoded, 'n d -> 1 1 n d')  # [1, 1, n, dim_head]

        # Apply RoPE + Fourier
        q = apply_rope(q, sin, cos) + fourier_encoded
        k = apply_rope(k, sin, cos) + fourier_encoded

        # Scaled dot-product attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
