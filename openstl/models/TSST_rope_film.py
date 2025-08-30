import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import os
from openstl.utils import measure_throughput
from fvcore.nn import FlopCountAnalysis, flop_count_table
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from openstl.modules.rope_attention import Attention
from openstl.modules import PreNorm, FeedForward
import math

class SwiGLU(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class GatedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)       
            
    def forward(self, x):
        for i, (attn, ff, dp1, dp2) in enumerate(self.layers):
            x_attn = attn(x)
            x = x + dp1(x_attn)
            x = x + dp2(ff(x))
        return self.norm(x)


class PredFormerLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super(PredFormerLayer, self).__init__()
        
        self.ts_temporal_transformer = GatedTransformer(dim, depth, heads, dim_head, 
                                                   mlp_dim, dropout, attn_dropout, drop_path)
        self.ts_space_transformer = GatedTransformer(dim, depth, heads, dim_head, 
                                                mlp_dim, dropout, attn_dropout, drop_path)
        self.st_space_transformer = GatedTransformer(dim, depth, heads, dim_head, 
                                                mlp_dim, dropout, attn_dropout, drop_path)
        self.st_temporal_transformer = GatedTransformer(dim, depth, heads, dim_head, 
                                                   mlp_dim, dropout, attn_dropout, drop_path)

    def forward(self, x):

        b, t, n, _ = x.shape        
        x_ts, x_ori = x, x    
        
        
        # ts-t branch
        x_ts = rearrange(x_ts, 'b t n d -> b n t d')
        x_ts = rearrange(x_ts, 'b n t d -> (b n) t d')
        x_ts = self.ts_temporal_transformer(x_ts)
            
        # ts-s branch
        x_ts = rearrange(x_ts, '(b n) t d -> b n t d', b=b)
        x_ts = rearrange(x_ts, 'b n t d -> b t n d')
        x_ts = rearrange(x_ts, 'b t n d -> (b t) n d') 
        x_ts = self.ts_space_transformer(x_ts)


        # ts output branch     
        x_ts = rearrange(x_ts, '(b t) n d -> b t n d', b=b) 
  
        # add   
        # x_ts += x_ori
        
        x_st, x_ori = x_ts, x_ts     
        
        # st-s branch
        x_st = rearrange(x_st, 'b t n d -> (b t) n d')
        x_st = self.st_space_transformer(x_st)

        # st-t branch
        x_st = rearrange(x_st, '(b t) ... -> b t ...', b=b)  
        x_st = x_st.permute(0, 2, 1, 3) # b n T d        
        x_st = rearrange(x_st, 'b n t d -> (b n) t d')  
        x_st = self.st_temporal_transformer(x_st)

        # st output branch     
        x_st = rearrange(x_st, '(b n) t d -> b n t d', b=b)
        x_st = rearrange(x_st, 'b n t d -> b t n d', b=b) 
        
        return x_st
      

class PredFormer_Model(nn.Module):
    def __init__(self, model_config, **kwargs):
        super().__init__()
        self.image_height = model_config['height']
        self.image_width = model_config['width']
        self.patch_size = model_config['patch_size']
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_frames_in = model_config['pre_seq']
        self.dim = model_config['dim']
        self.num_channels = model_config['num_channels']
        self.num_classes = self.num_channels
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.attn_dropout = model_config['attn_dropout']
        self.drop_path = model_config['drop_path']
        self.scale_dim = model_config['scale_dim']
        self.Ndepth = model_config['Ndepth']  # Ensure this is defined
        self.depth = model_config['depth']  # Ensure this is defined
        
        assert self.image_height % self.patch_size == 0, 'Image height must be divisible by the patch size.'
        assert self.image_width % self.patch_size == 0, 'Image width must be divisible by the patch size.'
        self.patch_dim = self.num_channels * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim),
            )

        self.blocks = nn.ModuleList([
            PredFormerLayer(self.dim, self.depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout, self.attn_dropout, self.drop_path)
            for i in range(self.Ndepth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_channels * self.patch_size ** 2)
            ) 

        self.film_gamma_x = nn.Sequential(
            nn.Embedding(3, self.dim),
            nn.Linear(self.dim, self.dim))
        self.film_beta_x = nn.Sequential(
            nn.Embedding(3, self.dim),
            nn.Linear(self.dim, self.dim))
        self.film_gamma_y = nn.Sequential(
            nn.Embedding(3, self.dim),
            nn.Linear(self.dim, self.dim))
        self.film_beta_y = nn.Sequential(
            nn.Embedding(3, self.dim),
            nn.Linear(self.dim, self.dim))

    def reset_memory(self, device):
        ''' No memory on this model '''
        pass

    def forward(self, x, x_labels=None, y_label=None):
        B, T, C, H, W = x.shape
        
        # Patch Embedding
        x = self.to_patch_embedding(x)
        
        if x_labels is not None:
            x_emb = self.film_gamma_x[0](x_labels.to(x.device))  # [B, L, D]
            x_emb = x_emb.mean(dim=1)  # or use attention pooling
            gamma = self.film_gamma_x[1](x_emb)
            beta = self.film_beta_x[1](x_emb)
            x = x * (1 + gamma) + beta

        if y_label is not None:
            y_emb = self.film_gamma_y[0](y_label.to(x.device))  # [B, L, D]
            y_emb = y_emb.mean(dim=1)  # or use attention pooling
            gamma = self.film_gamma_y[1](y_emb)
            beta = self.film_beta_y[1](y_emb)
            x = x * (1 + gamma) + beta

        
        # PredFormer Encoder
        for blk in self.blocks:
            x = blk(x)
        
        # MLP head        
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)
        
        return x

