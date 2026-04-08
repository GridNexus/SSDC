import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self,
                 dim,  # the dim of input token
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.k = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.v = nn.Linear(dim, dim * 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x, pose):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        q = self.q(pose).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = self.k(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.v(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = q[0], k[0], v[0]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim=1024,
                 num_heads=16,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)

    def forward(self, x, pose):
        x = x + self.attn(self.norm(x), self.norm(pose))
        return x



class ConvExpandReduce(nn.Module):
    def __init__(self, in_channels=3, expand_dim=96, reduce_dim=3):
        super(ConvExpandReduce, self).__init__()

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expand_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(expand_dim),
            nn.ReLU(inplace=True),
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(expand_dim, reduce_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        x = self.expand(x)
        x = self.reduce(x)
        return x

