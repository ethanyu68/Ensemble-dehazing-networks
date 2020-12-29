import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable


def conv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
                         nn.AvgPool2d(kernel_size=2, stride=2))


def deconv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.UpsamplingNearest2d(scale_factor=2))
						 
						 
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, n):
        super(UNet, self).__init__()
        self.dconv_down1 = double_conv(n, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.SELayer = SELayer(512,reduction=16)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, 3, 1)
        

        self.enhance_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        x = self.SELayer(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out_u = self.conv_last(x)

        out = out_u
        #for b in range(2):
            #out = self.enhance_block(out)

        return out


def blockUNet1(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s.relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False))
    else:
        block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, 3, 1, 1, bias=False))
    if bn:
        block.add_module('%s.bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s.dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s.relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s.bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s.dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block

class G2(nn.Module):
    def __init__(self, input_nc, output_nc, nf):
        super(G2, self).__init__()
        # input is 256 x 256
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
        # input is 128 x 128
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf * 2, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 64 x 64
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf * 2, nf * 4, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 32
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf * 4, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 16
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf * 8, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 8
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf * 8, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 4
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer7 = blockUNet(nf * 8, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 2 x  2
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer8 = blockUNet(nf * 8, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)

        ## NOTE: decoder
        # input is 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8
        dlayer8 = blockUNet(d_inc, nf * 8, name, transposed=True, bn=False, relu=True, dropout=True)

        # import pdb; pdb.set_trace()
        # input is 2
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8 * 2
        dlayer7 = blockUNet(d_inc, nf * 8, name, transposed=True, bn=True, relu=True, dropout=True)
        # input is 4
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8 * 2
        dlayer6 = blockUNet(d_inc, nf * 8, name, transposed=True, bn=True, relu=True, dropout=True)
        # input is 8
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8 * 2
        dlayer5 = blockUNet(d_inc, nf * 8, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 16
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8 * 2
        dlayer4 = blockUNet(d_inc, nf * 4, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 32
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 4 * 2
        dlayer3 = blockUNet(d_inc, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 64
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 2 * 2
        dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 128
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = nn.Sequential()
        d_inc = nf * 2
        dlayer1.add_module('%s.relu' % name, nn.ReLU(inplace=True))
        dlayer1.add_module('%s.tconv' % name, nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False))
        dlayer1.add_module('%s.tanh' % name, nn.LeakyReLU(0.2, inplace=True))

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.layer7 = layer7
        self.layer8 = layer8
        self.dlayer8 = dlayer8
        self.dlayer7 = dlayer7
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        dout8 = self.dlayer8(out8)
        dout8_out7 = torch.cat([dout8, out7], 1)
        dout7 = self.dlayer7(dout8_out7)
        dout7_out6 = torch.cat([dout7, out6], 1)
        dout6 = self.dlayer6(dout7_out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1


class BottleneckDecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckDecoderBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(in_planes + 32)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(in_planes + 2*32)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(in_planes + 3*32)
        self.relu4 = nn.ReLU(inplace=True)
        self.bn5 = nn.BatchNorm2d(in_planes + 4*32)
        self.relu5 = nn.ReLU(inplace=True)
        self.bn6 = nn.BatchNorm2d(in_planes + 5*32)
        self.relu6= nn.ReLU(inplace=True)
        self.bn7 = nn.BatchNorm2d(inter_planes)
        self.relu7= nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_planes + 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_planes + 2*32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_planes + 3*32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_planes + 4*32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)        
        self.conv6 = nn.Conv2d(in_planes + 5*32, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv7 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)        
        self.droprate = dropRate

    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out1 = torch.cat([x, out1], 1)
        out2 = self.conv2(self.relu2(self.bn2(out1)))
        out2 = torch.cat([out1, out2], 1)
        out3 = self.conv3(self.relu3(self.bn3(out2)))
        out3 = torch.cat([out2, out3], 1)
        out4 = self.conv4(self.relu4(self.bn4(out3)))
        out4 = torch.cat([out3, out4], 1)
        out5 = self.conv5(self.relu5(self.bn5(out4)))
        out5 = torch.cat([out4, out5], 1)
        out6 = self.conv6(self.relu6(self.bn6(out5)))
        out = self.conv7(self.relu7(self.bn7(out6)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        #out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, dropRate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)					   
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        out = x + x2
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)


class Dense_decoder(nn.Module):
    def __init__(self):
        super(Dense_decoder, self).__init__()
        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckDecoderBlock(384, 256)
        self.trans_block5 = TransitionBlock(640, 128)
        self.residual_block51 = ResidualBlock(128)
        self.residual_block52 = ResidualBlock(128)
        self.residual_block53 = ResidualBlock(128)
        self.residual_block54 = ResidualBlock(128)
        self.residual_block55 = ResidualBlock(128)
        self.residual_block56 = ResidualBlock(128)

        ############# Block6-up 32-32   ##############
        self.dense_block6 = BottleneckDecoderBlock(256, 128)
        self.trans_block6 = TransitionBlock(384, 64)
        self.residual_block61 = ResidualBlock(64)
        self.residual_block62 = ResidualBlock(64)
        self.residual_block63 = ResidualBlock(64)
        self.residual_block64 = ResidualBlock(64)
        self.residual_block65 = ResidualBlock(64)
        self.residual_block66 = ResidualBlock(64)

        ############# Block7-up 64-64   ##############
        self.dense_block7 = BottleneckDecoderBlock(64, 64)
        self.trans_block7 = TransitionBlock(128, 32)
        self.residual_block71 = ResidualBlock(32)
        self.residual_block72 = ResidualBlock(32)
        self.residual_block73 = ResidualBlock(32)
        self.residual_block74 = ResidualBlock(32)
        self.residual_block75 = ResidualBlock(32)
        self.residual_block76 = ResidualBlock(32)
        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8 = BottleneckDecoderBlock(32, 32)
        self.trans_block8 = TransitionBlock(64, 16)
        self.residual_block81 = ResidualBlock(16)
        self.residual_block82 = ResidualBlock(16)
        self.residual_block83 = ResidualBlock(16)
        self.residual_block84 = ResidualBlock(16)
        self.residual_block85 = ResidualBlock(16)
        self.residual_block86 = ResidualBlock(16)
        self.conv_refin = nn.Conv2d(19, 20, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.refine3 = nn.Conv2d(20 + 4, 3, kernel_size=3, stride=1, padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)
        self.upsample = F.upsample_nearest
        # self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x1, x2, x4, opt):
        x42 = torch.cat([x4, x2], 1)
        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        #x5 = self.residual_block51(x5)
        #x5 = self.residual_block52(x5)
        #x5 = self.residual_block53(x5)
        #x5 = self.residual_block54(x5)
        #x5 = self.residual_block55(x5)
        #x5 = self.residual_block56(x5)
        x52 = torch.cat([x5, x1], 1)
        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        #x6 = self.residual_block61(x6)
        #x6 = self.residual_block62(x6)
        #x6 = self.residual_block63(x6)
        #x6 = self.residual_block64(x6)
        #x6 = self.residual_block65(x6)
        #x6 = self.residual_block66(x6)		
        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        #x7 = self.residual_block71(x7)
        #x7 = self.residual_block72(x7)
        #x7 = self.residual_block73(x7)
        #x7 = self.residual_block74(x7)
        #x7 = self.residual_block75(x7)
        #x7 = self.residual_block76(x7)
        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))
        #x8 = self.residual_block81(x8)
        #x8 = self.residual_block82(x8)
        #x8 = self.residual_block83(x8)
        #x8 = self.residual_block84(x8)
        #x8 = self.residual_block85(x8)
        #x8 = self.residual_block86(x8)
        x8 = torch.cat([x8, x], 1)
        # print x8.size()
        x9 = self.relu(self.conv_refin(x8))
        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)
        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = self.tanh(self.refine3(dehaze))
        if opt.activation == 'no_relu':
            # print(">>no activation")
            dehaze = self.refine3(dehaze)
        else:
            dehaze = self.tanh(self.refine3(dehaze))
        return dehaze

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Dense_SEdecoder_W(nn.Module):
    def __init__(self):
        super(Dense_SEdecoder_W, self).__init__()
        ############# Block5-up  16-16 ##############
        self.se42 = SELayer(384, reduction=16)
        self.dense_block5 = BottleneckDecoderBlock(384, 256)
        self.trans_block5 = TransitionBlock(640, 128)
        self.residual_block51 = ResidualBlock(128)
        self.residual_block52 = ResidualBlock(128)
        # self.residual_block53 = ResidualBlock(128)
        # self.residual_block54 = ResidualBlock(128)
        # self.residual_block55 = ResidualBlock(128)
        # self.residual_block56 = ResidualBlock(128)

        ############# Block6-up 32-32   ##############
        self.se51 = SELayer(256, reduction=16)
        self.dense_block6 = BottleneckDecoderBlock(256, 128)
        self.trans_block6 = TransitionBlock(384, 64)
        self.residual_block61 = ResidualBlock(64)
        self.residual_block62 = ResidualBlock(64)
        # self.residual_block63 = ResidualBlock(64)
        # self.residual_block64 = ResidualBlock(64)
        # self.residual_block65 = ResidualBlock(64)
        # self.residual_block66 = ResidualBlock(64)

        ############# Block7-up 64-64   ##############
        self.dense_block7 = BottleneckDecoderBlock(64, 64)
        self.trans_block7 = TransitionBlock(128, 32)
        self.residual_block71 = ResidualBlock(32)
        self.residual_block72 = ResidualBlock(32)
        # self.residual_block73 = ResidualBlock(32)
        # self.residual_block74 = ResidualBlock(32)
        # self.residual_block75 = ResidualBlock(32)
        # self.residual_block76 = ResidualBlock(32)
        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8 = BottleneckDecoderBlock(32, 32)
        self.trans_block8 = TransitionBlock(64, 16)
        self.residual_block81 = ResidualBlock(16)
        self.residual_block82 = ResidualBlock(16)
        
        self.dense_blocky8 = BottleneckDecoderBlock(32, 32)
        self.trans_blocky8 = TransitionBlock(64, 16)
        self.residual_blocky81 = ResidualBlock(16)
        self.residual_blocky82 = ResidualBlock(16)
        
        self.dense_blockz8 = BottleneckDecoderBlock(32, 32)
        self.trans_blockz8 = TransitionBlock(64, 16)
        self.residual_blockz81 = ResidualBlock(16)
        self.residual_blockz82 = ResidualBlock(16)
        # self.residual_block83 = ResidualBlock(16)
        # self.residual_block84 = ResidualBlock(16)
        # self.residual_block85 = ResidualBlock(16)
        # self.residual_block86 = ResidualBlock(16)

        
        self.tanh = nn.Tanh()
        
        
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.refine3 = nn.Conv2d(20 + 4, 1, kernel_size=3, stride=1, padding=1)
        self.se80 = SELayer(19, reduction=3)
        self.conv_refin = nn.Conv2d(19, 20, 3, 1, 1)
        
        self.convy1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.convy1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.convy1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.convy1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.refiney3 = nn.Conv2d(20 + 4, 1, kernel_size=3, stride=1, padding=1)
        self.sey80 = SELayer(19, reduction=3)
        self.conv_refiny = nn.Conv2d(19, 20, 3, 1, 1)
        
        self.convz1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.convz1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.convz1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.convz1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.refinez3 = nn.Conv2d(20 + 4, 1, kernel_size=3, stride=1, padding=1)
        self.sez80 = SELayer(19, reduction=3)
        self.conv_refinz = nn.Conv2d(19, 20, 3, 1, 1)
        
        self.upsample = F.upsample_nearest
        # self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigx = nn.Sigmoid()
        self.sigy = nn.Sigmoid()
        self.sigz = nn.Sigmoid()

    def forward(self, x, x1, x2, x4, opt):
        x42 = self.se42(torch.cat([x4, x2], 1))
        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x5 = self.residual_block51(x5)
        #x5 = self.residual_block52(x5)
        #x5 = self.residual_block53(x5)
        #x5 = self.residual_block54(x5)
        #x5 = self.residual_block55(x5)
        #x5 = self.residual_block56(x5)
        x52 = self.se51(torch.cat([x5, x1], 1))
        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x6 = self.residual_block61(x6)
        #x6 = self.residual_block62(x6)
        #x6 = self.residual_block63(x6)
        #x6 = self.residual_block64(x6)
        #x6 = self.residual_block65(x6)
        #x6 = self.residual_block66(x6)
        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x7 = self.residual_block71(x7)
        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))
        x8 = self.residual_block81(x8)
        x8 = self.se80(torch.cat([x8, x], 1))
        # print x8.size()
        x9 = self.relu(self.conv_refin(x8))
        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        
        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)
        dehaze1 = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = self.tanh(self.refine3(dehaze))
        if opt.activation == 'no_relu':
            # print(">>no activation")
            p1 = self.sigx(self.refine3(dehaze1))
        else:
            p1 = self.tanh(self.refine3(dehaze1))
        
        
        y8 = self.trans_blocky8(self.dense_blocky8(x7))
        y8 = self.residual_blocky81(y8)
        y8 = self.sey80(torch.cat([y8, x], 1))
        # print x8.size()
        y9 = self.relu(self.conv_refiny(y8))
        y101 = F.avg_pool2d(y9, 32)
        y102 = F.avg_pool2d(y9, 16)
        y103 = F.avg_pool2d(y9, 8)
        y104 = F.avg_pool2d(y9, 4)
        y1010 = self.upsample(self.relu(self.convz1010(y101)), size=shape_out)
        y1020 = self.upsample(self.relu(self.convz1020(y102)), size=shape_out)
        y1030 = self.upsample(self.relu(self.convz1030(y103)), size=shape_out)
        y1040 = self.upsample(self.relu(self.convz1040(y104)), size=shape_out)
        dehaze2 = torch.cat((y1010, y1020, y1030, y1040, y9), 1)
        # dehaze = self.tanh(self.refine3(dehaze))
        if opt.activation == 'no_relu':
            # print(">>no activation")
            p2 = self.sigy(self.refiney3(dehaze2))
        else:
            p2 = self.tanh(self.refiney3(dehaze2))
        
        
        z8 = self.trans_blockz8(self.dense_blockz8(x7))
        z8 = self.residual_blockz81(z8)
        z8 = self.sez80(torch.cat([z8, x], 1))
        # print x8.size()
        z9 = self.relu(self.conv_refinz(z8))
        z101 = F.avg_pool2d(z9, 32)
        z102 = F.avg_pool2d(z9, 16)
        z103 = F.avg_pool2d(z9, 8)
        z104 = F.avg_pool2d(z9, 4)
        z1010 = self.upsample(self.relu(self.convz1010(z101)), size=shape_out)
        z1020 = self.upsample(self.relu(self.convz1020(z102)), size=shape_out)
        z1030 = self.upsample(self.relu(self.convz1030(z103)), size=shape_out)
        z1040 = self.upsample(self.relu(self.convz1040(z104)), size=shape_out)
        dehaze3 = torch.cat((z1010, z1020, z1030, z1040, z9), 1)
        # dehaze = self.tanh(self.refine3(dehaze))
        if opt.activation == 'no_relu':
            # print(">>no activation")
            p3 = self.sigz(self.refinez3(dehaze3))
        else:
            p3 = self.tanh(self.refinez3(dehaze3))
        
        w1 = p1/(p1 + p2 + p3)
        w2 = p2/(p1 + p2 + p3)
        w3 = p3/(p1 + p2 + p3)
        return p1,p2,p3

class Dense_SEdecoder(nn.Module):
    def __init__(self):
        super(Dense_SEdecoder, self).__init__()
        ############# Block5-up  16-16 ##############
        self.se42 = SELayer(384, reduction=16)
        self.dense_block5 = BottleneckDecoderBlock(384, 256)
        self.trans_block5 = TransitionBlock(640, 128)
        self.residual_block51 = ResidualBlock(128)
        self.residual_block52 = ResidualBlock(128)
        # self.residual_block53 = ResidualBlock(128)
        # self.residual_block54 = ResidualBlock(128)
        # self.residual_block55 = ResidualBlock(128)
        # self.residual_block56 = ResidualBlock(128)

        ############# Block6-up 32-32   ##############
        self.se51 = SELayer(256, reduction=16)
        self.dense_block6 = BottleneckDecoderBlock(256, 128)
        self.trans_block6 = TransitionBlock(384, 64)
        self.residual_block61 = ResidualBlock(64)
        self.residual_block62 = ResidualBlock(64)
        # self.residual_block63 = ResidualBlock(64)
        # self.residual_block64 = ResidualBlock(64)
        # self.residual_block65 = ResidualBlock(64)
        # self.residual_block66 = ResidualBlock(64)

        ############# Block7-up 64-64   ##############
        self.dense_block7 = BottleneckDecoderBlock(64, 64)
        self.trans_block7 = TransitionBlock(128, 32)
        self.residual_block71 = ResidualBlock(32)
        self.residual_block72 = ResidualBlock(32)
        # self.residual_block73 = ResidualBlock(32)
        # self.residual_block74 = ResidualBlock(32)
        # self.residual_block75 = ResidualBlock(32)
        # self.residual_block76 = ResidualBlock(32)
        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8 = BottleneckDecoderBlock(32, 32)
        self.trans_block8 = TransitionBlock(64, 16)
        self.residual_block81 = ResidualBlock(16)
        self.residual_block82 = ResidualBlock(16)
        # self.residual_block83 = ResidualBlock(16)
        # self.residual_block84 = ResidualBlock(16)
        # self.residual_block85 = ResidualBlock(16)
        # self.residual_block86 = ResidualBlock(16)

        self.se80 = SELayer(19, reduction=3)
        self.conv_refin = nn.Conv2d(19, 20, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.refine3 = nn.Conv2d(20 + 4, 3, kernel_size=3, stride=1, padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)
        self.upsample = F.upsample_nearest
        # self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x1, x2, x4, opt):
        x42 = self.se42(torch.cat([x4, x2], 1))
        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x5 = self.residual_block51(x5)
        #x5 = self.residual_block52(x5)
        #x5 = self.residual_block53(x5)
        #x5 = self.residual_block54(x5)
        #x5 = self.residual_block55(x5)
        #x5 = self.residual_block56(x5)
        x52 = self.se51(torch.cat([x5, x1], 1))
        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x6 = self.residual_block61(x6)
        #x6 = self.residual_block62(x6)
        #x6 = self.residual_block63(x6)
        #x6 = self.residual_block64(x6)
        #x6 = self.residual_block65(x6)
        #x6 = self.residual_block66(x6)
        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x7 = self.residual_block71(x7)
        #x7 = self.residual_block72(x7)
        #x7 = self.residual_block73(x7)
        #x7 = self.residual_block74(x7)
        #x7 = self.residual_block75(x7)
        #x7 = self.residual_block76(x7)
        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))
        x8 = self.residual_block81(x8)
        #x8 = self.residual_block82(x8)
        #x8 = self.residual_block83(x8)
        #x8 = self.residual_block84(x8)
        #x8 = self.residual_block85(x8)
        #x8 = self.residual_block86(x8)
        x8 = self.se80(torch.cat([x8, x], 1))
        # print x8.size()
        x9 = self.relu(self.conv_refin(x8))
        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)
        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = self.tanh(self.refine3(dehaze))
        if opt.activation == 'no_relu':
            # print(">>no activation")
            dehaze = self.refine3(dehaze)
        else:
            dehaze = self.tanh(self.refine3(dehaze))
        return dehaze




class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
		
		############# Mask Unet #################
 
      

        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0 = haze_class.features.conv0
        self.norm0 = haze_class.features.norm0
        self.relu0 = haze_class.features.relu0
        self.pool0 = haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1 = haze_class.features.denseblock1
        self.trans_block1 = haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2 = haze_class.features.denseblock2
        self.trans_block2 = haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3 = haze_class.features.denseblock3
        self.trans_block3 = haze_class.features.transition3

        ############# Block4-up  8-8  ##############
        self.dense_block4 = BottleneckDecoderBlock(512, 256)#512
        self.trans_block4 = TransitionBlock(768, 128)#768
        self.residual_block41 = ResidualBlock(128)
        self.residual_block42 = ResidualBlock(128)
        self.residual_block43 = ResidualBlock(128)
        self.residual_block44 = ResidualBlock(128)
        self.residual_block45 = ResidualBlock(128)
        self.residual_block46 = ResidualBlock(128)

        self.decoder_MSE = Dense_decoder()
        self.decoder_L1 = Dense_decoder()
        self.decoder_SSIM = Dense_decoder()
        self.decoder_VGG = Dense_decoder()
        self.decoder_W = Dense_decoder()
        self.decoder_SE1 = Dense_SEdecoder()
        self.decoder_SE2 = Dense_SEdecoder()
        self.decoder_SE3 = Dense_SEdecoder()
        self.decoder_SE_W = Dense_SEdecoder_W()
        self.unet1 = UNet(3)
        self.unet2 = UNet(3)
        self.unet3 = UNet(3)
        self.convout = conv_block(12,64)
        self.deconvout = deconv_block(64,3)
        #self.decoder_At = Dense_decoder_At()
        self.convW1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.ResT = ResidualBlock(32)
        self.convW = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.sigT = nn.Sigmoid()

        self.refine1 = nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1)
        self.bn_refine1 = nn.BatchNorm2d(20)		
        self.refine2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)
        self.bn_refine2 = nn.BatchNorm2d(20)
        self.refine3 = nn.Conv2d(20 + 4, 3, kernel_size=3, stride=1, padding=1)
        self.threshold = nn.Threshold(0.1, 0.1)
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.bn_conv1010 = nn.BatchNorm2d(1)
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.bn_conv1020 = nn.BatchNorm2d(1)
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.bn_conv1030 = nn.BatchNorm2d(1)
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.bn_conv1040 = nn.BatchNorm2d(1)
        self.upsample = F.upsample_nearest
        self.relu = nn.ReLU(inplace=True)
        self.sigW = nn.Sigmoid()

    def split_params(self):
        pretrained_params = []
        rest_params = []
        for name, module in self.named_children():
            if (name == "conv0") or (name == "norm0") or (name == "relu0") or (name == "pool0") or \
                    (name == "dense_block1") or (name == "dense_block2") or (name == "dense_block3") or \
                    (name == "trans_block1") or (name == "trans_block2") or (name == "trans_block3"):
                for p in module.parameters():
                    pretrained_params.append(p)
            else:
                for p in module.parameters():
                    rest_params.append(p)

        return pretrained_params, rest_params

    def freeze_pretrained(self):
        for name, module in self.named_children():
            if (name=="conv0") or (name=="norm0") or (name=="relu0") or (name=="pool0") or \
                (name=="dense_block1") or (name=="dense_block2") or (name=="dense_block3") or \
                (name=="trans_block1") or (name=="trans_block2") or (name=="trans_block3"):
                for p in module.parameters():
                    p.requires_grad = False
            else:
                for p in module.parameters():
                    p.requires_grad = True

    def forward(self, x, opt):
        ## 256x256
        x0 = self.pool0(self.relu0(self.norm0(self.conv0(x))))
        ## 64 X 64
        x1 = self.dense_block1(x0)
        # print x1.size()
        x1 = self.trans_block1(x1)
        ###  32x32
        x2 = self.trans_block2(self.dense_block2(x1))
        # print  x2.size()
        ### 16 X 16
        x3 = self.trans_block3(self.dense_block3(x2))
        # x3=Variable(x3.data,requires_grad=True)
        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        x4 = self.residual_block41(x4)
        x4 = self.residual_block42(x4)

        ######################################
        J1 = self.decoder_SE1(x, x1, x2, x4, opt)
        #J1 = self.unet1(D1)
        J2 = self.decoder_SE2(x, x1, x2, x4, opt)
        #J2 = self.unet2(D2)
        J3 = self.decoder_SE3(x, x1, x2, x4, opt)
        #J3 = self.unet3(D3)
        w1,w2,w3  = self.decoder_SE_W(x, x1, x2, x4, opt)
        #w2 = 1 - w1
        J = J1 * w1 + J2*w2 + J3*w3
        
        #D2 = self.decoder_SE(x, x1, x2, x4, opt)
        #A = self.decoder_At(x,x1,x2,x4,opt)
        #T = self.decoder_At(x,x1,x2,x4,opt)
        #T = self.sigT(self.convT(self.convT1(T)))
        #T = torch.cat([T,T,T], 1)
        #invT = 1.0 - T
        #At = x - A*invT
        #At = At/T
        
        #SSIM = self.unet(D1)
        #L1 = self.unet(D2)
        #EDGE = self.unet(D2)
        
        #out1 = MSE*w1 + SSIM*(1-w1)
        #out2 = L1*w2 + EDGE*(1-w2)
        
        return J,J1,J2,J3,w1,w2,w3 #,w1,w2,out1,out2
