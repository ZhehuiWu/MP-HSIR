import functools
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import time
import torchvision
import torchvision.models as models
import timm
from packaging import version
    
class BackboneClassifier(nn.Module):
    def __init__(self, backbone='resnet18', feat_dim=31, mid_dim=256, out_dim=9, drop_rate=0.5, size = (256,256)):
        super(BackboneClassifier, self).__init__()

        self.feat_dim = feat_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.drop_rate = drop_rate
        self.size = size

        if backbone == 'resnet18':
            self.encoder = models.resnet18(pretrained=False)
            self.encoder_out_channels = 512  
            self.features_layer = self.encoder.layer4  
            self.encoder.fc = nn.Identity()  
        elif backbone == 'resnet50':
            self.encoder = models.resnet50(pretrained=False)
            self.encoder_out_channels = 2048 
            self.features_layer = self.encoder.layer4 
            self.encoder.fc = nn.Identity()  
        elif backbone == 'vgg16':
            self.encoder = models.vgg16(pretrained=False)
            self.encoder_out_channels = 512  
            self.features_layer = self.encoder.features[28]  
            self.encoder.classifier = nn.Identity()  
        elif backbone == 'densenet121':
            self.encoder = models.densenet121(pretrained=False)
            self.encoder_out_channels = 1024  
            self.features_layer = self.encoder.features  
            self.encoder.classifier = nn.Identity()  
        elif backbone == 'vit':
            self.encoder = timm.create_model('vit_base_patch16_224', pretrained=False)
            self.encoder_out_channels = 768 
            self.features_layer = self.encoder.patch_embed.proj  
            self.encoder.head = nn.Identity() 
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Modify the encoder to accept feat_dim channels
        if backbone.startswith('resnet'):
            self.encoder.conv1 = nn.Conv2d(feat_dim, self.encoder.conv1.out_channels, 
                                           kernel_size=self.encoder.conv1.kernel_size, 
                                           stride=self.encoder.conv1.stride, 
                                           padding=self.encoder.conv1.padding, 
                                           bias=False)
        elif backbone == 'vgg16':
            self.encoder.features[0] = nn.Conv2d(feat_dim, 64, kernel_size=3, stride=1, padding=1)
        elif backbone == 'densenet121':
            self.encoder.features[0] = nn.Conv2d(feat_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif backbone == 'vit':
            self.encoder.patch_embed.proj = nn.Conv2d(feat_dim, self.encoder.patch_embed.proj.out_channels, 
                                                      kernel_size=16, stride=16, padding=0, bias=False)

        if backbone == 'vit':
            self.decoder = nn.Sequential(
                nn.Linear(768, mid_dim),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(),  
                nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity(),
                nn.Linear(mid_dim, out_dim)
            )
        else:

            self.decoder = nn.Sequential(
                nn.Conv2d(self.encoder_out_channels, mid_dim, kernel_size=1, bias=False),  
                nn.BatchNorm2d(mid_dim),
                nn.ReLU(),
                nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity(),
                nn.AdaptiveAvgPool2d((1, 1)),  
                nn.Flatten(),  
                nn.Linear(mid_dim, out_dim)  
            )

        self.features_output = None
        self._register_hooks()

    def _register_hooks(self):
        # Register a forward hook to get features from the desired layer
        def hook_fn(module, input, output):
            self.features_output = output

        if isinstance(self.features_layer, nn.ModuleList):  
            self.features_layer[-1].register_forward_hook(hook_fn)
        else:
            self.features_layer.register_forward_hook(hook_fn)

    def forward(self, x):

        x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)

        x = self.encoder(x)

        if self.features_output is not None:
            x = self.features_output  

        x = self.decoder(x)
        return x

# ffc-resnet from https://github.com/pkumivision/FFC

class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # 自动版本
        current_version = torch.__version__
        if version.parse(current_version) > version.parse("1.7.1"):
            # import torch.fft as fft
            # from torch.fft import *
            # import torch.fft
            self.is_high_torch = True
        else:
            self.is_high_torch = False

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        if self.is_high_torch:
            # (batch, c, h, w/2+1) complex number
            ffted = torch.fft.rfftn(x,s=(h,w),dim=(2,3),norm='ortho')
            ffted = torch.cat([ffted.real,ffted.imag],dim=1)

            ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
            ffted = self.relu(self.bn(ffted))

            ffted = torch.tensor_split(ffted,2,dim=1)
            ffted = torch.complex(ffted[0],ffted[1])
            output = torch.fft.irfftn(ffted,s=(h,w),dim=(2,3),norm='ortho')
        else:
            (batch, c, h, w/2+1, 2)
            ffted = torch.rfft(x, signal_ndim=2, normalized=True)
            # (batch, c, 2, h, w/2+1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
            ffted = ffted.view((batch, -1,) + ffted.size()[3:])

            ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
            ffted = self.relu(self.bn(ffted))

            ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
                0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

            output = torch.irfft(ffted, signal_ndim=2,
                                 signal_sizes=r_size[2:], normalized=True)


        return output


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        return x

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output

class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 enable_lfu=True):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g
    



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, ratio_gin=0.5, ratio_gout=0.5, lfu=True, use_se=False, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = FFC_BN_ACT(inplanes, width, kernel_size=3, padding=1, stride=stride,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout, norm_layer=norm_layer, activation_layer=nn.ReLU, enable_lfu=lfu)
        self.conv2 = FFC_BN_ACT(width, planes * self.expansion, kernel_size=3, padding=1,
                                ratio_gin=ratio_gout, ratio_gout=ratio_gout, norm_layer=norm_layer, enable_lfu=lfu)
        self.se_block = FFCSE_block(
            planes * self.expansion, ratio_gout) if use_se else nn.Identity()
        self.relu_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.relu_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x if self.downsample is None else self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x_l, x_g = self.se_block(x)

        x_l = self.relu_l(x_l + id_l)
        x_g = self.relu_g(x_g + id_g)

        return x_l, x_g

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, ratio_gin=0.5, ratio_gout=0.5, lfu=True, use_se=False):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = FFC_BN_ACT(inplanes, width, kernel_size=1,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout,
                                activation_layer=nn.ReLU, enable_lfu=lfu)
        self.conv2 = FFC_BN_ACT(width, width, kernel_size=3,
                                ratio_gin=ratio_gout, ratio_gout=ratio_gout,
                                stride=stride, padding=1, groups=groups,
                                activation_layer=nn.ReLU, enable_lfu=lfu)
        self.conv3 = FFC_BN_ACT(width, planes * self.expansion, kernel_size=1,
                                ratio_gin=ratio_gout, ratio_gout=ratio_gout, enable_lfu=lfu)
        self.se_block = FFCSE_block(
            planes * self.expansion, ratio_gout) if use_se else nn.Identity()
        self.relu_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.relu_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x if self.downsample is None else self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_l, x_g = self.se_block(x)

        x_l = self.relu_l(x_l + id_l)
        x_g = self.relu_g(x_g + id_g)

        return x_l, x_g


class FFCResNet(nn.Module):

    def __init__(self, layers, in_channel = 31, inplanes=64, size = (256,256), num_classes=8, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, ratio=0.5, lfu=True, use_se=False):
        super(FFCResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.size = size
        block = BasicBlock
        # inplanes = 64
        # TODO add ratio-inplanes-groups assertion

        self.inplanes = inplanes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.lfu = lfu
        self.use_se = use_se
        self.conv1 = nn.Conv2d(in_channel, inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, inplanes * 1, layers[0], stride=1, ratio_gin=0, ratio_gout=ratio)
        self.layer2 = self._make_layer(
            block, inplanes * 2, layers[1], stride=2, ratio_gin=ratio, ratio_gout=ratio)
        self.layer3 = self._make_layer(
            block, inplanes * 4, layers[2], stride=2, ratio_gin=ratio, ratio_gout=ratio)
        self.layer4 = self._make_layer(
            block, inplanes * 8, layers[3], stride=2, ratio_gin=ratio, ratio_gout=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # TODO 临时 使用global average pooling
        self.gobal_average_pooling = GlobalAvgPool2d()


    def _make_layer(self, block, planes, blocks, stride=1, ratio_gin=0.5, ratio_gout=0.5):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or ratio_gin == 0:
            downsample = FFC_BN_ACT(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                                    ratio_gin=ratio_gin, ratio_gout=ratio_gout, enable_lfu=self.lfu)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                            self.dilation, ratio_gin, ratio_gout, lfu=self.lfu, use_se=self.use_se))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                                ratio_gin=ratio_gout, ratio_gout=ratio_gout, lfu=self.lfu, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x[0])
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



if __name__ == '__main__':
    from thop import profile, clever_format
    device = torch.device('cuda:1')
    x = torch.rand((8, 31, 256, 256)).to(device)
    y = torch.randint(low=0, high=3, size=(1, 1)).to(device)

    #t = torch.randint(0, 1000, (1,), device=device).long()
    # net = BackboneClassifier(backbone='vit', feat_dim=31, mid_dim=256, out_dim=9, drop_rate=0.5, size=(256,256)).to(device)
    net = FFCResNet(BasicBlock, [2, 2, 2, 2],in_channel=31,size = (256,256)).to(device)
    #net = Classifier().to(device)

    macs, params = profile(net, inputs=(x,))
    macs, params = clever_format([macs, params], "%.4f")
    print(macs, params)