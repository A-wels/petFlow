import torch
import torch.nn as nn
from torch.nn import init

# https://github.com/NVIDIA/flownet2-pytorch/issues/60
# padding for input data
class padding(nn.Module):
    def __init__(self):
        super(padding,self).__init__()
        self.wpad = nn.ReplicationPad2d((0, -1, 0, 0))
        self.hpad = nn.ReplicationPad2d((0, 0, 0, -1))
    def forward(self, input, targetsize):
        if input.size()[2] != targetsize[2]:
            input = self.hpad(input)
        if input.size()[3] != targetsize[3]:
            input = self.wpad(input)
        return input
# Based on FlowNet2S: https://arxiv.org/abs/1612.01925
# https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py
'''
Portions of this code copyright 2017, Clement Pinard
'''
def conv(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True)
    )

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)


class FlowNetS(nn.Module):
    def __init__(self):
        super(FlowNetS, self).__init__()
        print("Building model...")
        # padding for input data: divisible by 64
        self.pad = padding()
        self.conv1   = conv( 2,   64, kernel_size=7, stride=2)
        self.conv2   = conv( 64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(256,  256)
        self.conv4   = conv(256,  512, stride=2)
        self.conv4_1 = conv(512,  512)
        self.conv5   = conv(512,  512, stride=2)
        self.conv5_1 = conv(512,  512)
        self.conv6   = conv(512, 1024, stride=2)
        self.conv6_1 = conv(1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')


    def forward(self, x):

        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        # pad
        flow6_up = self.pad(flow6_up, out_conv5.size())
        out_deconv5 = self.pad(out_deconv5, out_conv5.size())

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        # pad
        flow5_up = self.pad(flow5_up, out_conv4.size())
        out_deconv4 = self.pad(out_deconv4, out_conv4.size())

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        
        # pad
        flow4_up = self.pad(flow4_up, out_conv3.size())
        out_deconv3 = self.pad(out_deconv3, out_conv3.size())

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        # pad
        flow3_up = self.pad(flow3_up, out_conv2.size())
        out_deconv2 = self.pad(out_deconv2, out_conv2.size())

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            # upsample output with a factor of 4
            flow2 = self.upsample1(flow2)[:,:,0:344,0:127]
            flow3 = self.upsample1(flow3)[:,:,0:344,0:127]
            flow4 = self.upsample1(flow4)[:,:,0:344,0:127]
            flow5 = self.upsample1(flow5)[:,:,0:344,0:127]
            flow6 = self.upsample1(flow6)[:,:,0:344,0:127]
            return flow2,flow3,flow4,flow5,flow6
        else:
            flow2 = self.upsample1(flow2)[:,:,0:344,0:127]
            return flow2



