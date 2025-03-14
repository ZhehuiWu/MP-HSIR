import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
import time
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import clip

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
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
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

    
class GatedMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features*2)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_fc1,x_gate = self.fc1(x).chunk(2, dim=-1)  # 主分支
        x = x_fc1 * self.act(x_gate)  # 应用门控
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Spectral_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
       
        super(Spectral_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class PG_Spectral_Attention(nn.Module):
    def __init__(self, dim, compress_ratio, num_heads, prompt_len, bias):
        super(PG_Spectral_Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim//compress_ratio) ** -0.5

        self.linear_down = nn.Linear(dim, dim//compress_ratio,bias=bias)#compress_ratio=8，prompt_len=128
        self.linear_up = nn.Linear(dim//compress_ratio, dim, bias=bias)
        self.linear_prompt = nn.Linear(dim, prompt_len, bias=bias)
        self.prompt_param = nn.Parameter(torch.rand(1,1,prompt_len,dim//compress_ratio))#1,1,128,8

        self.q = nn.Linear(dim//compress_ratio, dim//compress_ratio, bias=bias)
        self.kv = nn.Linear(dim//compress_ratio, dim*2//compress_ratio, bias=bias)
        self.proj = nn.Linear(dim//compress_ratio, dim//compress_ratio)


    def forward(self, x_kv):
        shourtcut = x_kv
        B_, N, C = x_kv.shape#B,64,64
        x_kv = x_kv.mean(dim=1).unsqueeze(1)#B,64,64->B,1,64  
        prompt_weights = F.softmax(self.linear_prompt(x_kv),dim=-1)
        x_kv = self.linear_down(x_kv) #B,1,8
        
        spectral_prompt = prompt_weights.unsqueeze(-1) * self.prompt_param.repeat(B_,1,1,1)#B,1,128,8
        spectral_prompt = torch.sum(spectral_prompt,dim=2)#B,1,8

        q = self.q(spectral_prompt)#B,1,8  
        kv = self.kv(x_kv)
        k,v = kv.chunk(2, dim=2)#B,1,8      

        attn_weights = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)

        out = (attn_weights @ v.transpose(-2, -1))#B,8,1
        out = out.transpose(-2, -1).contiguous()#B,1,8
        out = self.proj(out)#B,1,8
        out = self.linear_up(out)#B,1,64
        out = out*shourtcut #B,64,64

        return out


class Spatial_Attention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super(Spatial_Attention,self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

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
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

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

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x_q, x_kv):
        b,c,h,w = x_q.shape
        q = self.q_dwconv(self.q(x_q))
        kv = self.kv_dwconv(self.kv(x_kv))
        k,v = kv.chunk(2, dim=1)  
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class CrossTransformer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                cross_residual = True):
        super(CrossTransformer, self).__init__()

        self.norm11 = LayerNorm(dim, LayerNorm_type)
        self.norm12 = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FFN(dim, ffn_expansion_factor, bias)

        self.cross_residual = cross_residual

    def forward(self, x_q, x_kv):
        if self.cross_residual:
            x_attn = x_q + self.attn(self.norm11(x_q), self.norm12(x_kv)) 
        else:
            x_attn = self.attn(self.norm11(x_q), self.norm12(x_kv)) 
            
        y = x_attn + self.ffn(self.norm2(x_attn))
        return y

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

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
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
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

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


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

class Text_Prompt(nn.Module):
    def __init__(self, task_classes = 7):
        super(Text_Prompt,self).__init__()
        if task_classes == 6:
            self.task_text_prompts = [
                "A hyperspectral image corrupted by Gaussian noise.",
                "A hyperspectral image affected by complex noise patterns.",
                "A hyperspectral image degraded by Gasussian blur.",
                "A hyperspectral image with reduced spatial resolution.",
                "A hyperspectral image compressed to a certain ratio.",
                "A hyperspectral image with missing spectral bands.",
            ]
        elif task_classes == 7:
            self.task_text_prompts = [
                "A hyperspectral image corrupted by Gaussian noise.",
                "A hyperspectral image affected by complex noise patterns.",
                "A hyperspectral image degraded by Gasussian blur.",
                "A hyperspectral image with reduced spatial resolution.",
                "A hyperspectral image compressed to a certain ratio.",
                "A hyperspectral image degraded by atmospheric haze.",
                "A hyperspectral image with missing spectral bands.",
            ]
        elif task_classes == 1:
            self.task_text_prompts = [
                "A hyperspectral image modulated by a coded aperture and compressed into a snapshot measurement.",
            ]
        else: 
            raise ValueError("task_classes must be 6 or 7")
        
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

        return clip_prompt,prompt_weights
    def get_clip_prompt(self):
        """Returns the pre-computed clip_prompt tensor."""
        return self.clip_prompt


class TVSP(nn.Module):
    def __init__(self, 
                task_classes = 6, 
                prompt_size = 64, 
                prompt_dim = 96,
                out_dim = 96,
                clip_prompts = None
                ):
        super(TVSP,self).__init__() 

        self.task_classes = task_classes
        self.prompt_size = prompt_size
        self.prompt_dim = prompt_dim

        self.text_linear = nn.Linear(512, prompt_dim)
        self.visual_prompt = nn.Parameter(torch.randn(1,prompt_dim,prompt_size,prompt_size))  ## 1,96,128,128

        self.clip_linear = nn.Linear(512, prompt_dim)

        if clip_prompts is not None:
            with torch.no_grad():
                encoded_prompts = self.clip_linear(clip_prompts)  
                encoded_prompts = encoded_prompts.view(1, task_classes, prompt_dim, 1, 1)  
        else:
            encoded_prompts = torch.randn(1, task_classes, prompt_dim, 1, 1)
        self.text_prompt_learnable = nn.Parameter(encoded_prompts)

        self.cross_transformer = CrossTransformer(dim = prompt_dim, num_heads = 2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
        self.conv_last = nn.Conv2d(prompt_dim, 
                                 out_dim, 
                                 kernel_size=3, stride=1, padding=1, bias=False)



    def forward(self, x, clip_prompt = None, prompt_weights = None):
        B,C,H,W = x.shape        
        # clip_prompt = self.text_linear(clip_prompt).unsqueeze(-1).unsqueeze(-1)#8,96
        text_prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.text_prompt_learnable.repeat(B,1,1,1,1)#8,6,96,1,1
        text_prompt = torch.mean(text_prompt, dim = 1) * clip_prompt#8,96,1,1
        text_prompt = F.interpolate(text_prompt, size=(self.prompt_size, self.prompt_size))
        prompts = self.cross_transformer(text_prompt,self.visual_prompt.repeat(B, 1, 1, 1))

        output_prompt = F.interpolate(prompts,(H,W),mode="bilinear")
        output_prompt = self.conv_last(output_prompt) #8,96,64,64

        return output_prompt



class PromptFusion(nn.Module):
    def __init__(self, dim = 96, out_dim = 48, head = 6, ffn_expansion_factor = 2.66, bias = False):
        super(PromptFusion, self).__init__()

        self.transformer = TransformerBlock(dim=dim, num_heads=head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type='WithBias')
        self.conv = nn.Conv2d(dim,out_dim,kernel_size=1,bias=bias)

    def forward(self, x, prompt):
        out = torch.cat([x, prompt], 1)#96+96->192
        out = self.transformer(out)
        out = self.conv(out)

        return out

class PGSSTB(nn.Module):

    def __init__(self, dim, num_heads, input_resolution=[64,64],window_size=7,shift_size=0,drop_path=0.0,
                 mlp_ratio=4., compress_ratio=8, prompt_len=128, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,act_layer=nn.GELU,bias=False):
        super(PGSSTB,self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.input_resolution = input_resolution

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 =  nn.LayerNorm(dim) 
        self.norm2 = nn.LayerNorm(dim) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GatedMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Spatial_Attention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.gobal_spectral_attn = Spectral_Attention(dim, num_heads, bias)
        self.local_spectral_attn = PG_Spectral_Attention(dim, compress_ratio, num_heads, prompt_len, bias)


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

    def forward(self, in_put, text_prompt=None):
        x = in_put
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1, 2)
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

        if self.input_resolution == [H,W]:                        
            sa_attns = self.attn(x_windows, mask=self.attn_mask) 
        else:
            sa_attns = self.attn(x_windows, mask=self.calculate_mask([H,W]).to(x.device))

        attn_windows = sa_attns

        x1 = self.local_spectral_attn(sa_attns)#local spectral attention
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x2 = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x2 = shifted_x

        x2 = x2.view(B, H * W, C)
        
        x2 = x2.transpose(1, 2).view(B, C, H, W)
        x2 = self.gobal_spectral_attn(x2)   #global spectral attention
        x2 = x2.flatten(2).transpose(1, 2)

        x1 = x1.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(x1, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x1 = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x1 = shifted_x

        x1 = x1.view(B, H * W, C)

        x = x1 + x2

        # MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).view(B, C, H, W)


        return x



class BaseBlock(nn.Module):
    def __init__(self,
        dim = 96,
        window_size=8,
        input_resolution=[64,64],
        depth=6,
        num_head=6,
        mlp_ratio=2,
        compress_ratio=8, 
        prompt_len=128,
        qkv_bias=True, qk_scale=None,
        drop_path=0.0,
        bias = False,
        use_checkpoint=False):

        super(BaseBlock,self).__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList()

        for i_block in range(depth): 
            block = PGSSTB(dim=dim, num_heads=num_head,input_resolution=input_resolution, window_size=window_size,
                        shift_size=0 if (i_block % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        compress_ratio=compress_ratio, 
                        prompt_len=prompt_len,
                        drop_path = drop_path[i_block],
                        qkv_bias=qkv_bias, qk_scale=qk_scale,bias=bias)
            self.blocks.append(block)

    def forward(self, x, text_prompt=None): 
        shortcut = x
        for blk in self.blocks:
                x= blk(x,text_prompt) 
        out = x+shortcut
        return out

class MP_HSIR_Net(nn.Module):
    def __init__(self, 
        in_channel = 31,
        out_channel = 31, 
        dim = 64,
        num_blocks = [2,4,6], 
        window_size = [8,8,8],
        task_classes = 6,
        num_refinement_blocks = 4,
        heads = [2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
    ):

        super(MP_HSIR_Net, self).__init__()

        self.patch_embed = OverlapPatchEmbed(in_channel, dim)
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(num_blocks))]

        self.text_prompt = Text_Prompt(task_classes=task_classes)
        self.clip_prompts =  self.text_prompt.get_clip_prompt()

        self.prompt1 = TVSP(task_classes=task_classes, prompt_size = 64, prompt_dim = dim, out_dim = dim)
        self.prompt2 = TVSP(task_classes=task_classes, prompt_size = 32, prompt_dim = dim*2, out_dim = dim*2)
        self.fusion1 = PromptFusion(dim = dim*2, out_dim=dim, head = 4, ffn_expansion_factor = 2.66, bias = False)
        self.fusion2 = PromptFusion(dim = dim*4, out_dim=dim*2, head = 8, ffn_expansion_factor = 2.66,bias = False)


        self.encoder_level1 = BaseBlock(dim = dim,window_size=window_size[0],input_resolution=[64,64],depth=num_blocks[0],num_head=heads[0],mlp_ratio=ffn_expansion_factor,compress_ratio=8,prompt_len=128,qkv_bias=True, qk_scale=None,drop_path =dpr[sum(num_blocks[:0]):sum(num_blocks[:1])],bias = bias)
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = BaseBlock(dim = dim*2**1,window_size=window_size[1],input_resolution=[32,32],depth=num_blocks[1],num_head=heads[1],mlp_ratio=ffn_expansion_factor,compress_ratio=16,prompt_len=128,qkv_bias=True, qk_scale=None,drop_path =dpr[sum(num_blocks[:1]):sum(num_blocks[:2])],bias = bias)
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 

        self.latent = BaseBlock(dim = dim*2**2,window_size=window_size[2],input_resolution=[16,16],depth=num_blocks[2],num_head=heads[2],mlp_ratio=ffn_expansion_factor,compress_ratio=32,prompt_len=128,qkv_bias=True, qk_scale=None,drop_path =dpr[sum(num_blocks[:2]):sum(num_blocks[:3])],bias = bias)
  
        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)

        self.decoder_level2 = BaseBlock(dim = dim*2**1,window_size=window_size[1],input_resolution=[32,32],depth=num_blocks[1],num_head=heads[1],mlp_ratio=ffn_expansion_factor,compress_ratio=16,prompt_len=128,qkv_bias=True, qk_scale=None,drop_path =dpr[sum(num_blocks[:1]):sum(num_blocks[:2])],bias = bias)
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = BaseBlock(dim = dim*2**1,window_size=window_size[0],input_resolution=[64,64],depth=num_blocks[0],num_head=heads[0],mlp_ratio=ffn_expansion_factor,compress_ratio=8,prompt_len=128,qkv_bias=True, qk_scale=None,drop_path =dpr[sum(num_blocks[:0]):sum(num_blocks[:1])],bias = bias)
        self.refinement = BaseBlock(dim = dim*2**1,window_size=window_size[0],input_resolution=[64,64],depth=num_refinement_blocks,num_head=heads[0],mlp_ratio=ffn_expansion_factor,compress_ratio=8,prompt_len=128,qkv_bias=True, qk_scale=None,drop_path = dpr[sum(num_blocks[:1]):sum(num_blocks[:2])],bias = bias)
                        
        self.output = nn.Conv2d(int(dim*2**1), out_channel, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prompts = None

    def forward(self, inp_img,task_id = None):
        B,C,H,W = inp_img.shape
        
        text_prompt,prompt_weights = self.text_prompt(inp_img,task_id)#torch.Size([1, 512]) torch.Size([1, 6])
        inp_enc_level1 = self.patch_embed(inp_img)#3->48

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)#48->96

        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)#96->192

        latent = self.latent(inp_enc_level3)

        inp_dec_level2 = self.up3_2(latent)#192->96
        prompt2 = self.prompt2(out_enc_level2,text_prompt,prompt_weights)
        out_enc_level2 = self.fusion2(out_enc_level2,prompt2)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)#96+96->192
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)#192->96
        out_dec_level2 = self.decoder_level2(inp_dec_level2)


        inp_dec_level1 = self.up2_1(out_dec_level2)#96->48
        prompt1 = self.prompt1(out_enc_level1,text_prompt,prompt_weights)
        out_enc_level1 = self.fusion1(out_enc_level1,prompt1)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)#48+48->96


        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img#96->3


        return out_dec_level1


if __name__ == '__main__':
    from thop import profile, clever_format
    device = torch.device('cuda:3')
    x = torch.rand((1, 100, 64, 64)).to(device)
    y = torch.tensor([1]).to(device)
    #y = torch.rand((16, 31, 64, 64)).to(device)
    #t = torch.randint(0, 1000, (1,), device=device).long()
    net = MP_HSIR_Net(100,100,96).to(device)
    macs, params = profile(net, inputs=(x,y))
    macs, params = clever_format([macs, params], "%.4f")
    print(macs, params)