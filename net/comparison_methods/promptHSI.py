import torch
from torch import nn
from torch.nn import functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import pywt 
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numbers
import math
import numpy as np
from einops import rearrange
import clip

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # 结构为 [B, num_patches, C]
        if self.norm is not None:
            x = self.norm(x)  # 归一化
        return x


    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    输入:
        img_size (int): 图像的大小，默认为 224*224.
        patch_size (int): Patch token 的大小，默认为 4*4.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): 线性 projection 输出的通道数，默认为 96.
        norm_layer (nn.Module, optional): 归一化层， 默认为N None.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  # 图像的大小，默认为 224*224
        patch_size = to_2tuple(patch_size)  # Patch token 的大小，默认为 4*4
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # patch 的分辨率
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # patch 的个数，num_patches

        self.in_chans = in_chans  # 输入图像的通道数
        self.embed_dim = embed_dim  # 线性 projection 输出的通道数

    def forward(self, x, x_size):
        B, HW, C = x.shape  # 输入 x 的结构
        x = x.transpose(1, 2).view(B, -1, x_size[0], x_size[1])  # 输出结构为 [B, Ph*Pw, C]
        return x

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class RDG(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, shift_size, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer, gc, patch_size, img_size):
        super(RDG, self).__init__()

        self.swin1 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                          num_heads=num_heads, window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust1 = nn.Conv2d(dim, gc, 1) 
        
        self.swin2 = SwinTransformerBlock(dim + gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + gc)%num_heads), window_size=window_size,
                                          shift_size=window_size//2,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust2 = nn.Conv2d(dim+gc, gc, 1) 
        
        self.swin3 = SwinTransformerBlock(dim + 2 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 2 * gc)%num_heads), window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust3 = nn.Conv2d(dim+gc*2, gc, 1) 
        
        self.swin4 = SwinTransformerBlock(dim + 3 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 3 * gc)%num_heads), window_size=window_size,
                                          shift_size=window_size//2,  # For first block
                                          mlp_ratio=1,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust4 = nn.Conv2d(dim+gc*3, gc, 1) 
        
        self.swin5 = SwinTransformerBlock(dim + 4 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 4 * gc)%num_heads), window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=1,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust5 = nn.Conv2d(dim+gc*4, dim, 1) 
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.pe = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.pue = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)
        
       

    def forward(self, x, xsize):
        x1 = self.pe(self.lrelu(self.adjust1(self.pue(self.swin1(x,xsize), xsize))))
        x2 = self.pe(self.lrelu(self.adjust2(self.pue(self.swin2(torch.cat((x, x1), -1), xsize), xsize))))
        x3 = self.pe(self.lrelu(self.adjust3(self.pue(self.swin3(torch.cat((x, x1, x2), -1), xsize), xsize))))
        x4 = self.pe(self.lrelu(self.adjust4(self.pue(self.swin4(torch.cat((x, x1, x2, x3), -1), xsize), xsize))))
        x5 = self.pe(           self.adjust5(self.pue(self.swin5(torch.cat((x, x1, x2, x3, x4), -1), xsize), xsize)))
        

        return x5 * 0.2 + x

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class PromptAdapter(nn.Module):
    def __init__(self, in_dim, act=nn.ReLU(), bias=False):
        super(PromptAdapter, self).__init__()
        self.linear_dw = nn.Linear(in_dim, in_dim // 8, bias=bias)
        self.act = act
        self.linear_up = nn.Linear(in_dim // 8, in_dim, bias=bias)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        res = x
        x = self.linear_dw(x)
        x = self.act(x)
        x = self.linear_up(x)
        x = self.act(self.norm(x) + res)
        return x

#################################### Text-IF #######################################
'''
@misc{yi2024textifleveragingsemantictext,
      title={Text-IF: Leveraging Semantic Text Guidance for Degradation-Aware and Interactive Image Fusion}, 
      author={Xunpeng Yi and Han Xu and Hao Zhang and Linfeng Tang and Jiayi Ma},
      year={2024},
      eprint={2403.16387},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.16387}, 
}
'''
## Feature Modulation
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level)),
        )
        self.adapter = PromptAdapter(512, act=nn.LeakyReLU(), bias=True)

    def forward(self, x, text_embed):
        text_embed = self.adapter(text_embed)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x

class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()

        self.fusion_proj = nn.Conv2d(
            embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias
        )

    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1)
        x = self.fusion_proj(x)
        return x
class Attention_spatial(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        qkv = self.qkv(self.norm(input)).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        attn = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        attn = self.out(attn.view(batch, channel, height, width))

        return attn + input
class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        query_A, key_A, value_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width).chunk(3, dim=2)

        x_B = self.norm_B(x_B)
        query_B, key_B, value_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width).chunk(3, dim=2)

        out_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)

        # B, N, C, H, W = query_B.shape
        # _, _, _, Y, X = key_A.shape
        # query_B_reshaped = query_B.permute(0, 1, 3, 4, 2).reshape(B * N * H * W, C)  # (B*N*H*W, C)
        # key_A_reshaped = key_A.permute(0, 1, 3, 4, 2).reshape(B * N, C, Y * X)       # (B*N, C, Y*X)
        # out = torch.matmul(query_B_reshaped, key_A_reshaped.reshape(B * N, C, -1).transpose(1, 2))
        # out_A = out.view(B, N, H, W, Y, X).contiguous() / math.sqrt(channel)

        out_A = out_A.view(batch, n_head, height, width, -1)
        out_A = torch.softmax(out_A, -1)
        out_A = out_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", out_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        out_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        out_B = out_B.view(batch, n_head, height, width, -1)
        out_B = torch.softmax(out_B, -1)
        out_B = out_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", out_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B


#############################################################################
'''
@misc{zamir2022restormerefficienttransformerhighresolution,
      title={Restormer: Efficient Transformer for High-Resolution Image Restoration}, 
      author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat and Fahad Shahbaz Khan and Ming-Hsuan Yang},
      year={2022},
      eprint={2111.09881},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2111.09881}, 
}
'''
## Layer Norm
def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
###############################################################
BatchNorm2d = nn.BatchNorm2d

class emptyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

###########################################
class SpectralWiseAttention(nn.Module):
    def __init__(self, dim, bias=False):
        super(SpectralWiseAttention, self).__init__()
        self.sigma = nn.Parameter(torch.ones(1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.view(b, c, -1).permute(0, 2, 1)  # b, h*w, c
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = torch.nn.functional.normalize(q, dim=1)
        k = torch.nn.functional.normalize(k, dim=1)

        attn = (k.transpose(-2, -1) @ q) * self.sigma
        attn = attn.softmax(dim=-1)
        out = self.linear(v @ attn).permute(0, 2, 1).view(b, c, h, w)

        return out


class SpectralAttentionBlock(nn.Module):
    def __init__(self, dim, bias=False, LayerNorm_type="WithBias"):
        super(SpectralAttentionBlock, self).__init__()
        if LayerNorm_type is None:
            self.norm = emptyModule()
        else:
            self.norm = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.specatt = SpectralWiseAttention(dim, bias)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.specatt(x)
        x = self.conv2(x)
        x = x + res
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_channel, embeding_dim, bias):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, embeding_dim, 5, 1, 2, bias=bias)
        self.conv2 = self.depwiseSepConv(embeding_dim, embeding_dim * 2**1, 5, bias)
        self.conv3 = self.depwiseSepConv(
            embeding_dim * 2**1, embeding_dim * 2**2, 3, bias
        )
        self.conv4 = self.depwiseSepConv(
            embeding_dim * 2**2,
            embeding_dim * 2**3,
            3,
            bias,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x1, x2, x3, x4

    def depwiseSepConv(self, in_dim, out_dim, ker_sz, bias=False):
        depwiseConv = nn.Conv2d(
            in_dim, in_dim, ker_sz, 2, ker_sz // 2, groups=in_dim, bias=bias
        )
        ptwiseConv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=bias)
        bn = BatchNorm2d(out_dim)
        relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        return nn.Sequential(depwiseConv, ptwiseConv, bn, relu)

'''
@misc{hsu2024realtimecompressedsensingjoint,
      title={Real-Time Compressed Sensing for Joint Hyperspectral Image Transmission and Restoration for CubeSat}, 
      author={Chih-Chung Hsu and Chih-Yu Jian and Eng-Shen Tu and Chia-Ming Lee and Guan-Lin Chen},
      year={2024},
      eprint={2404.15781},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.15781}, 
}
'''
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf, gc=32, bias=False, groups=4):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias, groups=groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5*0.2 + x

'''
@misc{hsu2024drctsavingimagesuperresolution,
      title={DRCT: Saving Image Super-resolution away from Information Bottleneck}, 
      author={Chih-Chung Hsu and Chia-Ming Lee and Yi-Shiuan Chou},
      year={2024},
      eprint={2404.00722},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.00722}, 
}
'''
class RDGsBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        img_size,
        num_heads,
        window_size,
        patch_size,
        num_layers=1,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        gc=32,
        mlp_ratio=4.0,
        drop_path=0.0,
    ):
        super(RDGsBlock, self).__init__()

        self.ape = ape
        self.window_size = window_size
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.patch_norm = patch_norm

        self.conv = nn.Conv2d(
            in_dim, in_dim // 4, kernel_size=1, stride=1, bias=False, groups=in_dim // 4
        )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_dim // 4,
            embed_dim=in_dim // 4,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_dim // 4,
            embed_dim=in_dim // 4,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, in_dim // 4)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                RDG(
                    dim=in_dim // 4,
                    input_resolution=(patches_resolution[0], patches_resolution[1]),
                    num_heads=num_heads,
                    window_size=window_size,
                    depth=0,
                    shift_size=window_size // 2,
                    drop_path=drop_path,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    gc=gc,
                    img_size=img_size,
                    patch_size=patch_size,
                )  # with 5 swin layers
            )

        self.norm = norm_layer(in_dim // 4)
        self.conv_up = nn.Conv2d(
            in_dim // 4, in_dim, kernel_size=1, stride=1, bias=False
        )

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])

        x = self.conv(x)
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)
        x = self.conv_up(x)  # + 0.5 * x

        return x  # out_channel equals the same in_chammel of x

class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        patch_size,
        img_size=(224, 224),
        num_layers=(2, 1),
        bias=False,
        upsample=True,
    ):
        super(DecoderBlock, self).__init__()
        self.upsample = upsample
        self.num_layers = num_layers

        if self.num_layers[0] > 0:
            self.conv_spa_1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias, groups=1)
        if self.num_layers[1] > 0:
            self.conv_spe_1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias, groups=1)

        if self.num_layers[0]>0 and self.num_layers[1]>0:
            self.cross_att = Cross_attention(dim, norm_groups=dim // 4)
        if self.num_layers[0] > 0 or self.num_layers[1] > 0:
            self.feature_fusion = Fusion_Embed(embed_dim=dim)
        
        self.attention_spatial = Attention_spatial(
            dim, n_head=num_heads // 2, norm_groups=dim // 4
        )
        self.attention_spectral = SpectralAttentionBlock(
            dim, bias=bias, LayerNorm_type="WithBias"
        )
        self.prompt_guidance = FeatureWiseAffine(in_channels=512, out_channels=dim)

        if self.num_layers[0] > 0:
            self.spatial_branch = RDGsBlock(
                dim,
                num_layers=num_layers[0],
                img_size=img_size[0],
                num_heads=num_heads,
                window_size=window_size,
                patch_size=patch_size,
            )
        if self.num_layers[1] > 0:
            self.spectral_branch = nn.Sequential(
            *[
               ResidualDenseBlock_5C(
                   dim
               )
               for _ in range(self.num_layers[1])
            ]
            )

        self.upconv = nn.Conv2d(dim, dim // 2, 3, 1, 1, bias=bias, groups=dim // 2)
        self.HRconv = nn.Conv2d(dim // 2, dim // 2, 1, 1, 0, bias=False, groups=1)

        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x, text_emb):
        if self.num_layers[0] > 0 and self.num_layers[1] > 0:
            fea1 = self.prompt_guidance(self.conv_spa_1(x), text_emb)
            fea2 = self.prompt_guidance(self.conv_spe_1(x), text_emb)
        elif self.num_layers[0] > 0:
            fea1 = self.prompt_guidance(self.conv_spa_1(x), text_emb)
            fea2 = x
        elif self.num_layers[1] > 0:
            fea1 = x
            fea2 = self.prompt_guidance(self.conv_spe_1(x), text_emb)

        if self.num_layers[0] > 0:
            fea1 = self.spatial_branch(fea1)
        if self.num_layers[1] > 0:
            fea2 = self.spectral_branch(fea2)

        if self.num_layers[0] > 0 and self.num_layers[1] > 0:
            fea1, fea2 = self.cross_att(fea1, fea2)
        
        if self.num_layers[0] > 0 or self.num_layers[1] > 0:
            x = self.feature_fusion(fea1, fea2)
        
        x = self.attention_spectral(x)
        x = self.attention_spatial(x)

        if self.upsample:
            x = self.lrelu(
                self.upconv(F.interpolate(x, scale_factor=2, mode="bilinear"))
            )
            x = self.HRconv(x)

        return x

class Text_Prompt(nn.Module):
    def __init__(self, task_classes = 7):
        super(Text_Prompt,self).__init__()
        if task_classes == 6:
            self.task_text_prompts = [
                "A hyperspectral image with Gaussian noise.",
                "A hyperspectral image with complex noise.",
                "A hyperspectral image with blur.",
                "A hyperspectral image with low resolution.", 
                "A hyperspectral image with occlusion.",
                "A hyperspectral image with bandmissing.",
                # "A hyperspectral image with haze.",
            ]
        elif task_classes == 7:
            self.task_text_prompts = [
                "A hyperspectral image with Gaussian noise.",
                "A hyperspectral image with complex noise.",
                "A hyperspectral image with blur.",
                "A hyperspectral image with low resolution.", 
                "A hyperspectral image with occlusion.",
                "A hyperspectral image with bandmissing.",
                "A hyperspectral image with haze.",
            ]
        else: 
            raise ValueError("Wrong number of tasks: {}".format(task_classes))
        self.task_classes = task_classes

        # self.clip_linear = nn.Linear(512, text_prompt_dim//2)
        clip_model, _ = clip.load("ViT-B/32", device="cpu")
        clip_text_encoder = clip_model.encode_text
        text_token = clip.tokenize(self.task_text_prompts)
        self.clip_prompt = clip_text_encoder(text_token)


    def forward(self, x, de_class = None):
        B,C,H,W = x.shape        
        if de_class.ndimension() > 1:
            mixed_one_hot_labels = torch.stack(
                [torch.mean(torch.stack([F.one_hot(c, num_classes=self.task_classes).float() for c in pair]), dim=0) 
                for pair in de_class])
            prompt_weights = mixed_one_hot_labels
        else: 
            prompt_weights = torch.nn.functional.one_hot(de_class, num_classes = self.task_classes).to(x.device) # .cuda() 
        # prompt_weights = torch.mean(prompt_weights, dim = 0)

        clip_prompt = self.clip_prompt.detach().to(x.device)
        clip_prompt = prompt_weights.unsqueeze(-1) * clip_prompt.unsqueeze(0).repeat(B,1,1)
        clip_prompt = torch.mean(clip_prompt, dim = 1) # (B, 512)

        return clip_prompt

class PromptHSI(nn.Module):
    def __init__(
        self,
        img_size=(64,64,64),
        in_channel=31,
        embeding_dim=64,
        task_classes=6,
        num_blocks_tf=2,
        num_layers=(2, 1),
        num_heads=8,
        window_size=(8, 8, 8),
        patch_size=(1, 1, 1),
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super(PromptHSI, self).__init__()

        self.encoder = Encoder(in_channel, embeding_dim, bias=True)
        self.conv_tail = nn.Conv2d(2 * embeding_dim, in_channel, 1, 1, 0, bias=bias)

        self.decoder4 = DecoderBlock(
            embeding_dim * 2**3,
            num_heads,
            window_size[0],
            patch_size[0],
            to_2tuple(img_size[0] // 8),
            bias=bias,
            num_layers=num_layers,
            upsample=True,
        )
        self.decoder3 = DecoderBlock(
            2 * embeding_dim * 2**1,
            num_heads,
            window_size[1],
            patch_size[1],
            to_2tuple(img_size[0] // 4),
            bias=bias,
            num_layers=num_layers,
        )
        self.decoder2 = DecoderBlock(
            embeding_dim * 2**1,
            num_heads // 2,
            window_size[2],
            patch_size[2],
            to_2tuple(img_size[0] // 2),
            bias=bias,
            num_layers=num_layers,
            upsample=True,
        )

        self.text_propmt = Text_Prompt(task_classes = task_classes)
        # Enhancement block
        self.enhance = nn.Sequential(
            *[
                TransformerBlock(
                    dim=2 * embeding_dim,
                    num_heads=num_heads // 2,
                    ffn_expansion_factor=2,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks_tf)
            ]
        )
        # Enhancement block end

        self.conv_a3 = nn.Conv2d(
            2 * embeding_dim * 2**2, embeding_dim * 2**2, 1, 1, bias=bias
        )
        self.conv_a2 = nn.Conv2d(
            2 * embeding_dim * 2**1, embeding_dim * 2**1, 1, 1, bias=bias
        )
        self.conv_a1 = nn.Conv2d(2 * embeding_dim, 2 * embeding_dim, 1, 1, bias=bias)

        ## Metrics
        self.L1Loss = torch.nn.L1Loss()
        self.sam_loss = SAMLoss()
        self.BandWiseMSE = BandWiseMSE()
        self.Waveletloss = HyperspectralSWTLoss()

    def forward(self, x, x_gt, task_id):
        text_emb = self.text_propmt(x,task_id)#B,512
        x1, x2, x3, x4 = self.encoder(x)  # for 112 size hsi: 112, 56, 28, 14

        x = self.decoder4(x4, text_emb)
        x = self.conv_a3(torch.concat((x, x3), 1))
        x = self.decoder3(x, text_emb)
        x = self.conv_a2(torch.concat((x, x2), 1))
        x = self.decoder2(x, text_emb)
        x = self.conv_a1(torch.concat((x, x1), 1))
        x = self.enhance(x) + x
        x = self.conv_tail(x)

        loss1 = torch.unsqueeze(self.L1Loss(x, x_gt), 0).to(x.device)
        loss2 = torch.unsqueeze(self.BandWiseMSE(x, x_gt), 0).to(x.device)
        loss3 = torch.unsqueeze(self.sam_loss(x, x_gt), 0).to(x.device)
        loss4 = torch.unsqueeze(self.Waveletloss(x, x_gt), 0).to(x.device)

        return x, x1, x2, x3, x4, loss1, loss2, loss3, loss4

def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.
    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)


def mypad(x, pad, mode='constant', value=0):
    """ Function to do numpy like padding on tensors. Only works for 2-D
    padding.
    Inputs:
        x (tensor): tensor to pad
        pad (tuple): tuple of (left, right, top, bottom) pad sizes
        mode (str): 'symmetric', 'wrap', 'constant, 'reflect', 'replicate', or
            'zero'. The padding technique.
    """
    if mode == 'symmetric':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            m1, m2 = pad[2], pad[3]
            l = x.shape[-2]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,xe]
        # horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            m1, m2 = pad[0], pad[1]
            l = x.shape[-1]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,:,xe]
        # Both
        else:
            m1, m2 = pad[0], pad[1]
            l1 = x.shape[-1]
            xe_row = reflect(np.arange(-m1, l1+m2, dtype='int32'), -0.5, l1-0.5)
            m1, m2 = pad[2], pad[3]
            l2 = x.shape[-2]
            xe_col = reflect(np.arange(-m1, l2+m2, dtype='int32'), -0.5, l2-0.5)
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:,:,i,j]
    elif mode == 'periodic':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            xe = np.arange(x.shape[-2])
            xe = np.pad(xe, (pad[2], pad[3]), mode='wrap')
            return x[:,:,xe]
        # Horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            xe = np.arange(x.shape[-1])
            xe = np.pad(xe, (pad[0], pad[1]), mode='wrap')
            return x[:,:,:,xe]
        # Both
        else:
            xe_col = np.arange(x.shape[-2])
            xe_col = np.pad(xe_col, (pad[2], pad[3]), mode='wrap')
            xe_row = np.arange(x.shape[-1])
            xe_row = np.pad(xe_row, (pad[0], pad[1]), mode='wrap')
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:,:,i,j]

    elif mode == 'constant' or mode == 'reflect' or mode == 'replicate':
        return F.pad(x, pad, mode, value)
    elif mode == 'zero':
        return F.pad(x, pad)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.
    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to
    Returns:
        (h0_col, h1_col, h0_row, h1_row)
    """
    h0_col = np.array(h0_col[::-1]).ravel()
    h1_col = np.array(h1_col[::-1]).ravel()
    t = torch.get_default_dtype()
    if h0_row is None:
        h0_row = h0_col
    else:
        h0_row = np.array(h0_row[::-1]).ravel()
    if h1_row is None:
        h1_row = h1_col
    else:
        h1_row = np.array(h1_row[::-1]).ravel()
    h0_col = torch.tensor(h0_col, device=device, dtype=t).reshape((1,1,-1,1))
    h1_col = torch.tensor(h1_col, device=device, dtype=t).reshape((1,1,-1,1))
    h0_row = torch.tensor(h0_row, device=device, dtype=t).reshape((1,1,1,-1))
    h1_row = torch.tensor(h1_row, device=device, dtype=t).reshape((1,1,1,-1))

    return h0_col, h1_col, h0_row, h1_row
        
def prep_filt_sfb2d(g0_col, g1_col, g0_row=None, g1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the sfb2d function.  In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb2d uses conv2d_transpose which acts like normal convolution.
    Inputs:
        g0_col (array-like): low pass column filter bank
        g1_col (array-like): high pass column filter bank
        g0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        g1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to
    Returns:
        (g0_col, g1_col, g0_row, g1_row)
    """
    g0_col = np.array(g0_col).ravel()
    g1_col = np.array(g1_col).ravel()
    t = torch.get_default_dtype()
    if g0_row is None:
        g0_row = g0_col
    if g1_row is None:
        g1_row = g1_col
    g0_col = torch.tensor(g0_col, device=device, dtype=t).reshape((1,1,-1,1))
    g1_col = torch.tensor(g1_col, device=device, dtype=t).reshape((1,1,-1,1))
    g0_row = torch.tensor(g0_row, device=device, dtype=t).reshape((1,1,1,-1))
    g1_row = torch.tensor(g1_row, device=device, dtype=t).reshape((1,1,1,-1))

    return g0_col, g1_col, g0_row, g1_row

def afb1d_atrous(x, h0, h1, mode='symmetric', dim=-1, dilation=1):
    """ 1D analysis filter bank (along one dimension only) of an image without
    downsampling. Does the a trous algorithm.
    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        h0 (tensor): 4D input for the lowpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        mode (str): padding method
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns).
        dilation (int): dilation factor. Should be a power of 2.
    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    shape = [1,1,1,1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    # Calculate the pad size
    L2 = (L * dilation)//2
    pad = (0, 0, L2-dilation, L2) if d == 2 else (L2-dilation, L2, 0, 0)
    #ipdb.set_trace()
    x = mypad(x, pad=pad, mode=mode)
    lohi = F.conv2d(x, h, groups=C, dilation=dilation)

    return lohi

def afb2d_atrous(x, filts, mode='symmetric', dilation=1):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to `afb1d_atrous`
    Inputs:
        x (torch.Tensor): Input to decompose
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by `prep_filt_afb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling `prep_filt_afb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
        dilation (int): dilation factor for the filters. Should be 2**level
    Returns:
        y: Tensor of shape (N, C, 4, H, W)
    """
    tensorize = [not isinstance(f, torch.Tensor) for f in filts]
    if len(filts) == 2:
        h0, h1 = filts
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                h0, h1, device=x.device)
        else:
            h0_col = h0
            h0_row = h0.transpose(2,3)
            h1_col = h1
            h1_row = h1.transpose(2,3)
    elif len(filts) == 4:
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                *filts, device=x.device)
        else:
            h0_col, h1_col, h0_row, h1_row = filts
    else:
        raise ValueError("Unknown form for input filts")

    lohi = afb1d_atrous(x, h0_row, h1_row, mode=mode, dim=3, dilation=dilation)
    y = afb1d_atrous(lohi, h0_col, h1_col, mode=mode, dim=2, dilation=dilation)

    return y

def sfb1d_atrous(lo, hi, g0, g1, mode='symmetric', dim=-1, dilation=1,
                 pad1=None, pad=None):
    """ 1D synthesis filter bank of an image tensor with no upsampling. Used for
    the stationary wavelet transform.
    """
    C = lo.shape[1]
    d = dim % 4
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1,1,1,1]
    shape[d] = L
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)
    g0 = torch.cat([g0]*C,dim=0)
    g1 = torch.cat([g1]*C,dim=0)

    # Calculate the padding size.
    # With dilation, zeros are inserted between the filter taps but not after.
    # that means a filter that is [a b c d] becomes [a 0 b 0 c 0 d].
    centre = L / 2
    fsz = (L-1)*dilation + 1
    newcentre = fsz / 2
    before = newcentre - dilation*centre

    # When conv_transpose2d is done, a filter with k taps expands an input with
    # N samples to be N + k - 1 samples. The 'padding' is really the opposite of
    # that, and is how many samples on the edges you want to cut out.
    # In addition to this, we want the input to be extended before convolving.
    # This means the final output size without the padding option will be
    #   N + k - 1 + k - 1
    # The final thing to worry about is making sure that the output is centred.
    short_offset = dilation - 1
    centre_offset = fsz % 2
    a = fsz//2
    b = fsz//2 + (fsz + 1) % 2
    
    #pad = (0, 0, a, b) if d == 2 else (a, b, 0, 0)
    pad = (0, 0, b, a) if d == 2 else (b, a, 0, 0)
    lo = mypad(lo, pad=pad, mode=mode)
    hi = mypad(hi, pad=pad, mode=mode)

    #unpad = (fsz - 1, 0) if d == 2 else (0, fsz - 1)
    unpad = (fsz, 0) if d == 2 else (0, fsz)
    
    y = F.conv_transpose2d(lo, g0, padding=unpad, groups=C, dilation=dilation) + \
        F.conv_transpose2d(hi, g1, padding=unpad, groups=C, dilation=dilation)

    return y/(2*dilation)


def sfb2d_atrous(ll, lh, hl, hh, filts, mode='symmetric'):
    """ Does a single level 2d wavelet reconstruction of wavelet coefficients.
    Does separate row and column filtering by two calls to `sfb1d_atrous`
    Inputs:
        ll (torch.Tensor): lowpass coefficients
        lh (torch.Tensor): horizontal coefficients
        hl (torch.Tensor): vertical coefficients
        hh (torch.Tensor): diagonal coefficients
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by `prep_filt_sfb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling `prep_filt_sfb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
    """
    tensorize = [not isinstance(x, torch.Tensor) for x in filts]
    if len(filts) == 2:
        g0, g1 = filts
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(g0, g1)
        else:
            g0_col = g0
            g0_row = g0.transpose(2,3)
            g1_col = g1
            g1_row = g1.transpose(2,3)
    elif len(filts) == 4:
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(*filts)
        else:
            g0_col, g1_col, g0_row, g1_row = filts
    else:
        raise ValueError("Unknown form for input filts")

    lo = sfb1d_atrous(ll, lh, g0_col, g1_col, mode=mode, dim=2)
    hi = sfb1d_atrous(hl, hh, g0_col, g1_col, mode=mode, dim=2)
    y = sfb1d_atrous(lo, hi, g0_row, g1_row, mode=mode, dim=3)

    return y

class SWTForward(nn.Module):
    """ Performs a 2d Stationary wavelet transform (or undecimated wavelet
    transform) of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet): Which wavelet to use. Can be a string to
            pass to pywt.Wavelet constructor, can also be a pywt.Wavelet class,
            or can be a two tuple of array-like objects for the analysis low and
            high pass filters.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme. PyWavelets uses only periodization so we use this
            as our default scheme.
        """
    def __init__(self, J=1, wave='db1', mode='symmetric'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.h0_col = nn.Parameter(filts[0], requires_grad=False)
        self.h1_col = nn.Parameter(filts[1], requires_grad=False)
        self.h0_row = nn.Parameter(filts[2], requires_grad=False)
        self.h1_row = nn.Parameter(filts[3], requires_grad=False)

        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the SWT.
        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Returns:
            List of coefficients for each scale. Each coefficient has
            shape :math:`(N, C_{in}, 4, H_{in}, W_{in})` where the extra
            dimension stores the 4 subbands for each scale. The ordering in
            these 4 coefficients is: (A, H, V, D) or (ll, lh, hl, hh).
        """
        ll = x
        coeffs = []
        # Do a multilevel transform
        filts = (self.h0_col, self.h1_col, self.h0_row, self.h1_row)
        for j in range(self.J):
            # Do 1 level of the transform
            y = afb2d_atrous(ll, filts, self.mode)
            coeffs.append(y)
            ll = y[:,0:1,:,:]            
            
        return coeffs

class SWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image
    Args:
        wave (str or pywt.Wavelet): Which wavelet to use
        C: deprecated, will be removed in future
    """
    def __init__(self, wave='db1', mode='symmetric'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
        # Prepare the filters
        
        filts = prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
        self.g0_col = nn.Parameter(filts[0], requires_grad=False)
        self.g1_col = nn.Parameter(filts[1], requires_grad=False)
        self.g0_row = nn.Parameter(filts[2], requires_grad=False)
        self.g1_row = nn.Parameter(filts[3], requires_grad=False)

        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward
        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """

        yl = coeffs[-1][:,0:1,:,:]
        yh = []
        for lohi in coeffs:
            yh.append(lohi[:,None,1:4,:,:])

        ll = yl

        # Do the synthesis filter banks
        for h_ in yh[::-1]:
            lh, hl, hh = torch.unbind(h_, dim=2)        
            filts = (self.g0_col, self.g1_col, self.g0_row, self.g1_row)
            ll = sfb2d_atrous(ll, lh, hl, hh, filts, mode=self.mode)
            
        return ll
    

class HyperspectralSWTLoss(nn.Module):
    def __init__(self, loss_weights=None, reduction='mean'):
        super(HyperspectralSWTLoss, self).__init__()
        self.loss_weights = loss_weights if loss_weights is not None else [0.01] * 4
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        wavelet = pywt.Wavelet('sym19')
        
        dlo = wavelet.dec_lo
        an_lo = np.divide(dlo, sum(dlo))
        an_hi = wavelet.dec_hi
        rlo = wavelet.rec_lo
        syn_lo = 2*np.divide(rlo, sum(rlo))
        syn_hi = wavelet.rec_hi

        filters = pywt.Wavelet('wavelet_normalized', [an_lo, an_hi, syn_lo, syn_hi])
        sfm = SWTForward(1, filters, 'periodic').to(pred.device)

        # 对每个波段进行小波变换
        loss = 0
        for band in range(pred.shape[1]):
            sr_band = pred[:, band:band+1, :, :]
            hr_band = target[:, band:band+1, :, :]

            wavelet_sr = sfm(sr_band)[0]
            wavelet_hr = sfm(hr_band)[0]
            LL_sr, LH_sr, HL_sr, HH_sr = wavelet_sr[:, 0:1, :, :], wavelet_sr[:, 1:2, :, :], wavelet_sr[:, 2:3, :, :], wavelet_sr[:, 3:, :, :]
            LL_hr, LH_hr, HL_hr, HH_hr = wavelet_hr[:, 0:1, :, :], wavelet_hr[:, 1:2, :, :], wavelet_hr[:, 2:3, :, :], wavelet_hr[:, 3:, :, :]

            loss_subband_LL = self.loss_weights[0] * self.criterion(LL_sr, LL_hr)
            loss_subband_LH = self.loss_weights[1] * self.criterion(LH_sr, LH_hr)
            loss_subband_HL = self.loss_weights[2] * self.criterion(HL_sr, HL_hr)
            loss_subband_HH = self.loss_weights[3] * self.criterion(HH_sr, HH_hr)

            loss += loss_subband_LL + loss_subband_LH + loss_subband_HL + loss_subband_HH

        return loss
############################### SAM Loss ###################################
class SAMLoss(nn.Module):

    def forward(self, x, y, factor=0.01):
        num = torch.sum(torch.multiply(x+1e-5, y+1e-5), 1)
        den = torch.sqrt(torch.multiply(torch.sum(x**2+1e-5, 1), torch.sum(y**2+1e-5, 1)))
        sam = torch.clip(torch.divide(num, den), -1, 1)
        sam = torch.mean(torch.rad2deg(torch.arccos(sam)))
        
        return sam*factor

############################### L2 Loss (BandWiseMSE) ###################################
class BandWiseMSE(nn.Module):
    
    def forward(self, x, y, reduce=True, factor=1.0):
        yp = torch.sqrt(torch.sum(y**2, (2,3))) / (y.shape[2]*y.shape[3])+1e-9
        # print(yp)
        yp = torch.nn.functional.normalize(1/yp)
        # print(yp)
        loss = (x - y)**2
        loss = torch.sqrt(torch.mean(loss, (2,3))) * yp 
        if reduce:
            return torch.mean(loss) 

        return torch.mean(loss, dim=1)*factor

if __name__ == '__main__':
    from thop import profile, clever_format
    device = torch.device('cuda:5')
    x = torch.rand((1, 100, 64, 64)).to(device)
    x_gt = torch.rand((1, 100, 64, 64)).to(device)
    text_emb = torch.tensor([0]).to(device)

    t = torch.randint(0, 100, (1,), device=device).long()
    net = PromptHSI(in_channel=100,embeding_dim=96).to(device)

    macs, params = profile(net, inputs=(x,x_gt,text_emb))
    macs, params = clever_format([macs, params], "%.4f")
    print(macs, params)
