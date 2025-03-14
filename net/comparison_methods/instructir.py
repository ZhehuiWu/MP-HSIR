import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import clip    

class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class ICB(nn.Module):
    """
    Instruction Condition Block (ICB)
    Paper Section 3.3
    """

    def __init__(self, feature_dim, text_dim=512):
        super(ICB, self).__init__()
        self.fc    = nn.Linear(text_dim, feature_dim)
        self.block = NAFBlock(feature_dim)
        self.beta  = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)

    def forward(self, x, text_embedding):
        gating_factors = torch.sigmoid(self.fc(text_embedding))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)

        f = x * self.gamma + self.beta  # 1) learned feature scaling/modulation
        f = f * gating_factors          # 2) (soft) feature routing based on text
        f = self.block(f)               # 3) block feature enhancement
        return f + x

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
        elif task_classes == 1:
            self.task_text_prompts = [
                "A hyperspectral image modulated by a coded aperture and compressed into a snapshot measurement.",
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


class InstructIR(nn.Module):
    """
    InstructIR model using NAFNet (ECCV 2022) as backbone.
    The model takes as input an RGB image and a text embedding (encoded instruction).
    Described in Paper Section 3.3
    """

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], txtdim=768, task_classes=6):
        super().__init__()

        self.intro  = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders    = nn.ModuleList()
        self.decoders    = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups         = nn.ModuleList()
        self.downs       = nn.ModuleList()
        self.enc_cond    = nn.ModuleList()
        self.dec_cond    = nn.ModuleList()
        self.text_propmt = Text_Prompt(task_classes = task_classes)

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            
            self.enc_cond.append(ICB(chan, txtdim))

            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            # Add text embedding as modulation
            self.dec_cond.append(ICB(chan, txtdim))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, task_id):
        B, C, H, W = inp.shape
        txtembd = self.text_propmt(inp,task_id)
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        encs = []

        for encoder, enc_mod, down in zip(self.encoders, self.enc_cond, self.downs):
            x = encoder(x)
            x = enc_mod(x, txtembd)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip, dec_mod in zip(self.decoders, self.ups, encs[::-1], self.dec_cond):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            x = dec_mod(x, txtembd)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


def create_model(input_channels = 3, width = 32, enc_blks = [2, 2, 4, 8], middle_blk_num = 12, dec_blks = [2, 2, 2, 2], txtdim=512):

    net = InstructIR(img_channel=input_channels, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, txtdim=txtdim)

    return net

if __name__ == '__main__':
    from thop import profile, clever_format
    device = torch.device('cuda:0')
    x = torch.rand((1, 100, 64, 64)).to(device)
    y = torch.tensor([1]).to(device)
    #y = torch.rand((16, 31, 64, 64)).to(device)
    #t = torch.randint(0, 1000, (1,), device=device).long()
    net = InstructIR(img_channel = 100, width = 72, enc_blk_nums = [2, 2, 4, 8], middle_blk_num = 12, dec_blk_nums = [2, 2, 2, 2], txtdim=512).to(device)
    macs, params = profile(net, inputs=(x,y))
    macs, params = clever_format([macs, params], "%.4f")
    print(macs, params)