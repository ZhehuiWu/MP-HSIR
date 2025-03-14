import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary



class Basic_Residual_Block(nn.Module):
    def __init__(self,BCB_channels=100//2,BCB_feature=100//2):
        super( Basic_Residual_Block, self).__init__()
        self.BCB_channels = BCB_channels
        self.input_channel =BCB_feature
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=self.BCB_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=self.BCB_channels, out_channels=self.BCB_channels, kernel_size=1),
            nn.ReLU()
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=self.BCB_channels, out_channels=self.BCB_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=self.BCB_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, BCB_feature):
        feature = self.conv_layer1(BCB_feature)
        feature = self.conv_layer2(feature)
        feature = self.conv_layer3(feature)
        # identity_feature = self.conv_layer4(BCB_feature)
        identity_feature = self.conv_layer4(BCB_feature)  # 减少显存分配
        output = torch.add(feature, identity_feature)
        return output
# 融合模块
class Fusion_Block(nn.Module):
    def __init__(self, input_channel=100):
        super(Fusion_Block, self).__init__()
        self.input_channel = input_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channel, out_channels=self.input_channel, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.conv2 = nn.Conv2d(in_channels=self.input_channel, out_channels=100//2, kernel_size=3, padding=1)

    def forward(self, FFB_input_feature):
        feature = self.conv1(FFB_input_feature)
        output = self.conv2(feature)
        return output
class SPCA_Block(nn.Module):
    def __init__(self, input_channel=100//2):
        super(SPCA_Block, self).__init__()
        self.reduction = 4
        self.input_channel = input_channel
        # Spatial attention
        self.conv_sp_atten_1 = nn.Conv2d(in_channels=self.input_channel, out_channels=100//2, kernel_size=3,
                                         padding=1)  # kernel_size =3*3, padding =1, stride =1 ,不改变feature_map的分辨率
        self.conv_sp_atten_2_1 = nn.Conv2d(in_channels=100//2, out_channels=1, kernel_size=3, padding=1)
        self.conv_sp_atten_2_2 = nn.Conv2d(in_channels=100//2, out_channels=1, kernel_size=5, padding=2)
        self.conv_sp_atten_3_layer = nn.Sequential(  # 7*7卷积+Sigmoid函数
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.conv_sp_atten_4_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=100//2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_sp_atten_5 = nn.Conv2d(in_channels=100//2, out_channels=100//2, kernel_size=3, padding=1)

        # channel attention
        self.conv_ca_atten_1 = nn.Conv2d(in_channels=100//2, out_channels=(100//2 // self.reduction-1), kernel_size=(1, 1))
        self.leakyReLU = torch.nn.LeakyReLU(negative_slope=0.3)
        self.conv_ca_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=100//2, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

        self.conv_ca_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=100//2, out_channels=100//2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # pixel attention
        self.conv_pa_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=100//2, out_channels=100//2, kernel_size=(1, 1)),
            nn.Sigmoid()
        )
        # end of pixel attention

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=100//2, out_channels=100//2, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, Input_features):
        # start Spatial attention
        follow_feature = self.conv_sp_atten_1(Input_features)    # 37
        spatial_att_feature1 = self.conv_sp_atten_2_1(follow_feature)   # 1
        spatial_att_feature2 = self.conv_sp_atten_2_2(follow_feature)    # 1


        temp_spatial_att_feature = torch.cat((spatial_att_feature1, spatial_att_feature2),
                                             dim=1)      # 拼接    2
        spatial_att_feature = self.conv_sp_atten_3_layer(temp_spatial_att_feature)  # 7*7卷积    1
        final_spatial_attention_feature = torch.mul(spatial_att_feature,
                                                    Input_features)              # 广播  37
        output_spatial_feature = self.conv_sp_atten_4_layer(
            final_spatial_attention_feature)                                   # 37
        conv_feature_sa = self.conv_sp_atten_5(output_spatial_feature)        # 37
        # end spatial attention

        # start channel attention
        max_pool_feature = F.max_pool2d(conv_feature_sa, (conv_feature_sa.shape[2], conv_feature_sa.shape[3]))    # 全局最大池化     # 37
        max_pool_feature_1 = self.conv_ca_atten_1(max_pool_feature)   # 8
        max_pool_feature_2 = self.leakyReLU(max_pool_feature_1)  #
        max_pool_feature_3 = self.conv_ca_layer_1(max_pool_feature_2)  #  37

        average_pool_feature = F.avg_pool2d(conv_feature_sa, (conv_feature_sa.shape[2], conv_feature_sa.shape[3]))  # 全局平均池化  # 37
        average_pool_feature_1 = self.conv_ca_atten_1(average_pool_feature)  # 8
        average_pool_feature_2 = self.leakyReLU(average_pool_feature_1)  #
        average_pool_feature_3 = self.conv_ca_layer_1(average_pool_feature_2)  # 37

        channel_attention = torch.add(max_pool_feature_3,average_pool_feature_3)  # 对应元素相加   37

        final_channel_attention_feature = torch.mul(channel_attention,
                                                    conv_feature_sa)  # 对应元素相乘    # 37   此处触发广播机制
        # end of channel attention

        output_channel_feature = self.conv_ca_layer_2(final_channel_attention_feature)  # 37

        ###  Start pixel attention
        PA_features = self.conv_pa_layer_1(output_channel_feature)   # 37
        final_PA_feature = torch.mul(PA_features, output_channel_feature)  # 37
        ###  End pixel attention

        output_channel_feature = self.conv_layer(final_PA_feature)  # 37

        return output_channel_feature

def fusion_block_f(input_channel, FFB_input_feature):
    fusion_block = Fusion_Block(input_channel=input_channel)
    output = fusion_block(FFB_input_feature)
    return output

def spca_block_f(input_channel, Input_features):
    spca_block = SPCA_Block(input_channel=input_channel)
    output = spca_block(Input_features)
    return output
def Spectral_integrate(rest_feature, upper_feature_integrate):
    spca_block = SPCA_Block(input_channel=rest_feature.shape[1])    # 下面波段的特征经过SPCA模块
    output = spca_block_f(input_channel=rest_feature.shape[1], Input_features=rest_feature)

    concate_SPCA_feature_rest_feature = torch.cat((upper_feature_integrate, output), dim=1)

    fusion_final_feature = fusion_block_f(input_channel=concate_SPCA_feature_rest_feature.shape[1],
                                          FFB_input_feature=concate_SPCA_feature_rest_feature)
    return fusion_final_feature

class SGNet(nn.Module):

    # 深度卷积
    # def conv_dw(self, in_channel, deep_multiplier):  # block块
    #     # Depthwise卷积, 分组卷积的特例(group = in_channel)
    #     return nn.Conv2d(in_channels=in_channel,
    #                      out_channels=in_channel * deep_multiplier,
    #                      kernel_size=3,
    #                      stride=1,
    #                      padding=1,
    #                      bias=False,
    #                      groups=in_channel)

    # 实现全局残差
    def __init__(self, input_image_channel=100):
        super(SGNet, self).__init__()
        self.input_image_channel = input_image_channel
        self.upper_branch_channel = 100//2
        self.conv1_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.upper_branch_channel, out_channels=100//2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv1_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=100//2, out_channels=100//2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv1_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=(self.input_image_channel - self.upper_branch_channel), out_channels=100//2,
                      kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv1_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=100//2, out_channels=100//2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_image_channel, out_channels=100//2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_image_channel, out_channels=self.input_image_channel, kernel_size=3,
                      padding=1),
            nn.ReLU()
        )

        self.conv4 = nn.Conv2d(in_channels=100,out_channels=100,kernel_size=1)
        # self.conv5 = nn.Conv2d(in_channels=100//2,out_channels=100,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv6 = nn.Conv2d(in_channels=100//2,out_channels=100,kernel_size=1)
        self.conv_dw_1=nn.Conv2d(in_channels=100//2,out_channels=100//2,kernel_size=3,padding=1,stride=1,groups=100//2)
        self.conv_dw_2=nn.Conv2d(in_channels=100//2,out_channels=100,kernel_size=1)
        self.SPCA_Block = SPCA_Block()
        self.Fusion_Block = Fusion_Block()
        self.Basic_Residual = Basic_Residual_Block(BCB_feature=100//2)

    def global_residual(self, integrate_feature, Input_HSI):
        global_residual_feature = self.conv2(Input_HSI)
        input_GhostNet_feature_num = global_residual_feature.shape[1]
        output_band = Input_HSI.shape[1]  # 输入高光谱图像的通道数
        depth_ratio = output_band // input_GhostNet_feature_num  # 深度比
        global_residual_feature = self.conv_dw_1(global_residual_feature)  # 深度卷积   depth_multiply
        global_residual_feature = self.conv_dw_2(global_residual_feature)
        # global_residual_feature_2 = conv_2.forward(global_residual_feature)
        # global_residual_feature_2 = self.conv5(global_residual_feature)
        # global_residual_feature_2 = self.conv4(global_residual_feature_2)     # 1*1生维，将128--->100
        tail_feature_HSI = torch.add(integrate_feature, global_residual_feature)

        return tail_feature_HSI

    def forward(self, Input_HSI):
        # _, band, _, _ = Input_HSI.shape     # 获得HSI的通道数
        upper_branch_channel = self.upper_branch_channel  # Spectral grouping boundary   分组边界 37      # 分为两组，两个分支
        upper_branch_input = Input_HSI[:, 0:upper_branch_channel, :, :]        #  37
        rest_input = Input_HSI[:, upper_branch_channel:, :, :]     #  100 - 37 = 113


        temp_shallow_feature_upper = self.conv1_layer1(upper_branch_input)         #  37
        shallow_feature_upper = self.conv1_layer2(temp_shallow_feature_upper)           #  37


        temp_shallow_feature_rest = self.conv1_layer3(rest_input)                   #  37
        shallow_feature_rest = self.conv1_layer4(temp_shallow_feature_rest)             #  37

        # Inject 1
        upper_feature = self.SPCA_Block(shallow_feature_rest)
        # print("upper_feature:",upper_feature.shape)    # upper_feature: torch.Size([2, 32, 256, 256])
        concate_SPCA_feature_rest_feature = torch.cat((shallow_feature_upper, upper_feature), dim=1)
        # print("concate_SPCA_feature_rest_feature:",concate_SPCA_feature_rest_feature.shape)  # concate_SPCA_feature_rest_feature: torch.Size([2, 64, 256, 256])
        upper_feature1 = self.Fusion_Block(concate_SPCA_feature_rest_feature)
        # print("upper_feature1:",upper_feature1.shape)  #  upper_feature1: torch.Size([2, 32, 256, 256])
        upper_feature2 = self.Basic_Residual(upper_feature1)
        # print("upper_feature2:",upper_feature2.shape)  # upper_feature2: torch.Size([2, 32, 256, 256])

        # upper_feature1 = Spectral_integrate(shallow_feature_rest, shallow_feature_upper)
        # upper_feature2 = basic_residual_f(BCB_feature=upper_feature1, BCB_channels=32)

        ## Inject 2
        rest_feature = self.Basic_Residual(shallow_feature_rest)
        rest_feature1 = self.SPCA_Block(rest_feature)
        concate_SPCA_feature_rest_feature1 = torch.cat((upper_feature2, rest_feature1), dim=1)
        upper_feature3 = self.Fusion_Block(concate_SPCA_feature_rest_feature1)

        # rest_feature = basic_residual_f(BCB_feature=shallow_feature_rest, BCB_channels=32)
        # upper_feature3 = Spectral_integrate(rest_feature, upper_feature2)



        ## Inject 3
        rest_feature_2 = self.Basic_Residual(rest_feature)
        rest_feature_3 = self.SPCA_Block(rest_feature_2)
        concate_SPCA_feature_rest_feature2 = torch.cat((upper_feature3, rest_feature_3), dim=1)
        upper_feature5 = self.Fusion_Block(concate_SPCA_feature_rest_feature2)

        # rest_feature_2 = basic_residual_f(BCB_feature=rest_feature, BCB_channels=32)
        # upper_feature5 = Spectral_integrate(rest_feature_2, upper_feature3)
        # print("upper_feature5.shape[1]",upper_feature5.shape[1])   #upper_feature5.shape[1] 32
        # integrate_feature_2 = spca_block_f(input_channel=upper_feature5.shape[1], Input_features=upper_feature5)
        integrate_feature_2 = self.SPCA_Block(upper_feature5)
#        print("integrate_feature_2:", integrate_feature_2.shape)
        integrate_feature = self.Basic_Residual(BCB_feature=integrate_feature_2)
        integrate_feature = self.conv6(integrate_feature)
#        print("integrate_feature:", integrate_feature.shape)  # integrate_feature: torch.Size([1, 128, 512, 512])

        ##   residual learning   #全局残差

        integrate_feature = self.global_residual(integrate_feature, Input_HSI)
        # print("integrate_feature:",integrate_feature.shape)
        integrate_feature = self.conv3(integrate_feature)
        # tail_feature_HSI = self.global_residual(integrate_feature, Input_HSI)
        # dehaze_result_HSI = self.conv3(tail_feature_HSI)
        return integrate_feature

if __name__ == '__main__':
    from thop import profile, clever_format
    device = torch.device('cuda:0')
    x = torch.rand((1, 100, 64, 64)).to(device)
    #y = torch.rand((16, 31, 64, 64)).to(device)
    #t = torch.randint(0, 1000, (1,), device=device).long()
    net = SGNet(input_image_channel=100).to(device)
    macs, params = profile(net, inputs=(x,))
    macs, params = clever_format([macs, params], "%.4f")
    print(macs, params)