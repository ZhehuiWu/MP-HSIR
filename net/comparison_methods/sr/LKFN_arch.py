import torch
from torch import nn as nn
from torch.nn import functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sr.utils.registry import ARCH_REGISTRY
from sr.arch_util import default_init_weights


class BSConvU(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


class PartialBSConvU(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        padding=2,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        scale=2,
    ):
        super().__init__()

        # pointwise
        self.remaining_channels = in_channels // scale
        self.other_channels = in_channels - self.remaining_channels
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # partialdepthwise
        self.pdw = nn.Conv2d(
            in_channels=self.remaining_channels,
            out_channels=self.remaining_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=self.remaining_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea1, fea2 = torch.split(
            fea, [self.remaining_channels, self.other_channels], dim=1
        )
        fea1 = self.pdw(fea1)
        fea = torch.cat((fea1, fea2), 1)
        fea = self.pw(fea)
        return fea


class Attention(nn.Module):

    def __init__(self, embed_dim, fft_norm="ortho"):
        # bn_layer not used
        super(Attention, self).__init__()
        self.conv_layer1 = torch.nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.conv_layer2 = torch.nn.Conv2d(embed_dim // 2, embed_dim // 2, 1, 1, 0)
        self.conv_layer3 = torch.nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        real = ffted.real + self.conv_layer3(
            self.relu(self.conv_layer2(self.relu(self.conv_layer1(ffted.real))))
        )
        imag = ffted.imag + self.conv_layer3(
            self.relu(self.conv_layer2(self.relu(self.conv_layer1(ffted.imag))))
        )
        ffted = torch.complex(real, imag)

        ifft_shape_slice = x.shape[-2:]

        output = torch.fft.irfftn(
            ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm
        )

        return x * output


class LKFB(nn.Module):

    def __init__(self, in_channels, out_channels, atten_channels=None):
        super().__init__()

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        if atten_channels is None:
            self.atten_channels = in_channels
        else:
            self.atten_channels = atten_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = PartialBSConvU(in_channels, self.rc, kernel_size=5, padding=2)
        self.c2_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c2_r = PartialBSConvU(self.rc, self.rc, kernel_size=5, padding=2)
        self.c3_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c3_r = PartialBSConvU(self.rc, self.rc, kernel_size=5, padding=2)

        self.c4 = BSConvU(self.rc, self.dc, kernel_size=3, padding=1)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, self.atten_channels, 1, 1, 0)
        self.atten = Attention(self.atten_channels)
        self.c6 = nn.Conv2d(self.atten_channels, out_channels, 1, 1, 0)
        self.pixel_norm = nn.LayerNorm(out_channels)  # channel-wise
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.c1_r(input)
        r_c1 = self.act(r_c1)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c2)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c3)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)

        out = self.atten(out)
        out = self.c6(out)
        # (B, H, W, C)
        out = out.permute(0, 2, 3, 1)
        out = self.pixel_norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return out + input


def UpsampleOneStep(in_channels, out_channels, upscale_factor=4):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class Upsampler_rep(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels * 2, 1)
        self.conv3x3 = nn.Conv2d(in_channels * 2, out_channels * (upscale_factor**2), 3)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        v1 = F.conv2d(x, self.conv1x1.weight, self.conv1x1.bias, padding=0)
        v1 = F.pad(v1, (1, 1, 1, 1), "constant", 0)
        b0_pad = self.conv1x1.bias.view(1, -1, 1, 1)
        v1[:, :, 0:1, :] = b0_pad
        v1[:, :, -1:, :] = b0_pad
        v1[:, :, :, 0:1] = b0_pad
        v1[:, :, :, -1:] = b0_pad
        v2 = F.conv2d(v1, self.conv3x3.weight, self.conv3x3.bias, padding=0)
        out = self.conv1(x) + self.conv3(x) + v2
        return self.pixel_shuffle(out)


# @ARCH_REGISTRY.register()
class LKFN(nn.Module):

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=56,
        num_atten=56,
        num_block=8,
        upscale=4,
        num_in=4,
        upsampler="pixelshuffledirect",
        rgb_mean=(0.4488, 0.4371, 0.4040),
    ):
        super().__init__()
        self.num_in = num_in
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.fea_conv = BSConvU(num_in_ch * num_in, num_feat, kernel_size=3, padding=1)

        self.B1 = LKFB(
            in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten
        )
        self.B2 = LKFB(
            in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten
        )
        self.B3 = LKFB(
            in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten
        )
        self.B4 = LKFB(
            in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten
        )
        self.B5 = LKFB(
            in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten
        )
        self.B6 = LKFB(
            in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten
        )
        self.B7 = LKFB(
            in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten
        )
        self.B8 = LKFB(
            in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten
        )

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1, 1, 0)
        self.GELU = nn.GELU()

        self.c2 = BSConvU(num_feat, num_feat, kernel_size=3, padding=1)

        if upsampler == "pixelshuffledirect":
            self.upsampler = UpsampleOneStep(
                num_feat, num_out_ch, upscale_factor=upscale
            )
        elif upsampler == "pixelshuffle_rep":
            self.upsampler = Upsampler_rep(num_feat, num_out_ch, upscale_factor=upscale)
        else:
            raise NotImplementedError("Check the Upsampler. None or not support yet.")

    def forward(self, input):
        self.mean = self.mean.type_as(input)
        input = input - self.mean
        input = torch.cat([input] * self.num_in, dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat(
            [out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1
        )
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea
        output = self.upsampler(out_lr) + self.mean

        return output


if __name__ == '__main__':
    from thop import profile, clever_format
    device = torch.device('cuda:0')
    x = torch.rand((1, 31, 512, 512)).to(device)
    #y = torch.rand((16, 31, 64, 64)).to(device)
    #t = torch.randint(0, 1000, (1,), device=device).long()
    net = LKFN(num_in_ch=31,num_out_ch=31,upscale=1,num_in =1).to(device)
    macs, params = profile(net, inputs=(x,))
    macs, params = clever_format([macs, params], "%.4f")
    print(macs, params)