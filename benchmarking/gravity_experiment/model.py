import torch
import torch.nn as nn
import math

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, emb_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H//p, W//p, C, p, p]
        x = x.contiguous().view(B, -1, C * p * p)  # [B, num_patches, patch_dim]
        x = self.proj(x)  # [B, num_patches, emb_dim]
        return x

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        pos = self.pe[:N].unsqueeze(0).expand(B, -1, -1)
        return x + pos

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class TemporalMemory(nn.Module):
    def __init__(self, emb_dim, memory_size):
        super().__init__()
        self.memory_size = memory_size
        self.temporal_memory = None

        self.gravity_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.gravity_out = nn.Linear(emb_dim, 1)

    def reset(self):
        self.temporal_memory = None

    def update(self, feats):
        # feats: [B, T, N, D]
        if self.temporal_memory is None:
            self.temporal_memory = feats
        else:
            self.temporal_memory = torch.cat([self.temporal_memory, feats], dim=1)
            if self.temporal_memory.shape[1] > self.memory_size:
                self.temporal_memory = self.temporal_memory[:, -self.memory_size:]

    def predict_gravity(self):
        # [B, T_mem, N, D] → [B, D]
        feats_mean = self.temporal_memory.mean(dim=2).mean(dim=1)
        latent_gravity = self.gravity_proj(feats_mean)
        g_hat = self.gravity_out(latent_gravity)
        return g_hat, latent_gravity



class GravityTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.height = config['height']
        self.width = config['width']
        self.patch_size = config['patch_size']
        self.num_channels = config['num_channels']
        self.num_patches = (self.height // self.patch_size) * (self.width // self.patch_size)
        self.emb_dim = config['dim']

        self.patch_embed = PatchEmbed(
            in_channels=self.num_channels,
            patch_size=self.patch_size,
            emb_dim=self.emb_dim,
            img_size=self.height
        )

        self.gravity_token_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.gravity_token_pos = nn.Parameter(torch.zeros(1, 1, self.emb_dim))

        self.pos_encoding = SinusoidalPositionalEncoding(self.emb_dim, max_len=self.num_patches + 1)
        self.transformer = nn.Sequential(*[
            TransformerBlock(self.emb_dim, config['heads'], config.get('dropout', 0.0))
            for _ in range(config['depth'])
        ])

        # Output head (for reconstruction or whatever main output)
        self.head = nn.Sequential(
            nn.LayerNorm(self.emb_dim),
            nn.Linear(self.emb_dim, config['aft_seq'] * self.num_channels * self.patch_size * self.patch_size)
        )

        self.memory = TemporalMemory(self.emb_dim, config.get('memory_size', 10))

            
    def reset_memory(self):
        self.memory.reset()

    def forward(self, x_in):
        B, T, C, H, W = x_in.shape

        frames_flat = x_in.view(B * T, C, H, W)
        feats = self.patch_embed(frames_flat)
        feats = feats.view(B, T, self.num_patches, self.emb_dim)

        self.memory.update(feats)
        g_hat, latent_gravity = self.memory.predict_gravity()  # [B, emb_dim]

        last_frame = x_in[:, -1]
        x = self.patch_embed(last_frame)  # [B, num_patches, emb_dim]

        # Project latent gravity to token embedding and add positional encoding
        gravity_token = self.gravity_token_proj(latent_gravity).unsqueeze(1)  # [B, 1, emb_dim]
        gravity_token = gravity_token + self.gravity_token_pos  # add learned pos emb

        # Concatenate gravity token to patch embeddings
        x = torch.cat([gravity_token, x], dim=1)  # [B, num_patches+1, emb_dim]

        # Add sinusoidal positional encoding (pos_encoding supports longer sequence now)
        x = self.pos_encoding(x)  

        # Pass through transformer
        x = self.transformer(x)  # [B, num_patches+1, emb_dim]

        # Remove gravity token before head (take from index 1 onwards)
        x = x[:, 1:, :]

        x = self.head(x)
        x = x.view(B, self.num_patches, 15, self.num_channels, self.patch_size, self.patch_size)
        x = x.view(B, self.height // self.patch_size, self.width // self.patch_size, 15, self.num_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 4, 1, 5, 2, 6).contiguous()
        out = x.view(B, 15, self.num_channels, self.height, self.width)

        return out, g_hat, latent_gravity

    def get_latent_sequence(self):
        """
        Returns: Tensor of shape [T_mem, D] averaged over patches and batch
        """
        if self.memory.temporal_memory is None:
            return None
        # [B, T_mem, N, D] → [T_mem, D]
        return self.memory.temporal_memory.mean(dim=2).mean(dim=0).detach().cpu()


