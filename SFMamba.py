import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

import einops
from einops import rearrange
# from visualizer import get_local
import numpy as np

# from model import blocks
from mamba_ssm import Mamba

import math
from torch.nn import Module
# import pywt
from torch.autograd import Function

m = None


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2   # 100 96 12 12 -> 100 96 6 12          切片操作 ：表示保留该维度的所有元素  0::2 表示从第0个元素开始 每隔两个取一个元素 会取 [0, 2, 4, 6, 8, 10]
    x02 = x[:, :, 1::2, :] / 2    #  100 96 6 12                                     1::2 表示从索引 1 开始，每隔 2 个元素取一个 取 [1, 3, 5, 7, 9]
    x1 = x01[:, :, :, 0::2]    # 100 96 6 12 -> 100 96 6 6
    x2 = x02[:, :, :, 0::2]   # 100 96 6 12 -> 100 96 6 6
    x3 = x01[:, :, :, 1::2]   # 100 96 6 12  -> 100 96 6 6
    x4 = x02[:, :, :, 1::2]   # 100 96 6 12 -> 100 96 6 6
    x_LL = x1 + x2 + x3 + x4    # 100 96 6 6
    x_HL = -x1 - x2 + x3 + x4   # 100 96 6 6
    x_LH = -x1 + x2 - x3 + x4    # 100 96 6 6
    x_HH = x1 - x2 - x3 + x4     # 100 96 6 6

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x): # 传入x.shape: 100 96 11 11
        return dwt_init(x)

def iwt_init(x):    # 输入x.shape:(400 96 6 6)
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()   # in_batch: 400 in_channel: 96 in_height: 6 in_width: 6
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)), in_channel, r * in_height, r * in_width    # out_batch: 100 out_channel: 96 out_height: 12 out_width: 12
    x1 = x[0:out_batch, :, :, :] / 2     # x1.shape: (100, 96, 6, 6)
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2   # x2.shape: (100, 96, 6, 6)
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2    # # x3.shape: (100, 96, 6, 6)
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2   # x4.shape: (100, 96, 6, 6)

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

## Layer Norm

def to_3d(x):    # 输入的x.shape: 
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):   # 传入的x.shape: 
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


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
        sigma = x.var(-1, keepdim=True, unbiased=False)  ##返回所有元素的方差
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

    def forward(self, x):   # 传入的x.shape: 100 121 96
        mu = x.mean(-1, keepdim=True)   # mu.shape: 100 121 96 -> 100 121 1     keepdim=True: 压缩的维度保留为1 （即[A,B,C]->[A,B,1])   False: 直接移除该维度 （即[A,B,C] ->[A, B])
        sigma = x.var(-1, keepdim=True, unbiased=False)  # sigma.shape: 100 121 1       x.var()计算张量x沿着最后一个维度的方差
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):     
        h, w = x.shape[-2:]  
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.rep_conv1 = Conv2d_BN(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.rep_conv2 = Conv2d_BN(hidden_features, hidden_features, 1, 1, 0, groups=hidden_features)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        identity = x
        x = self.project_in(x)
        x1 = x + self.rep_conv1(x) + self.rep_conv2(x)
        x2 = self.dwconv(x)
        x = F.gelu(x2) * x1 + F.gelu(x1) * x2
        x = self.project_out(x)
        return x + identity

    @torch.no_grad()
    def fuse(self):
        conv = self.rep_conv1.fuse()  ##Conv_BN
        conv1 = self.rep_conv2.fuse()  ##Conv_BN

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv

class SpaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )

    def forward(self, x):

        # x = x + self.conv(x)

        x = self.conv(x)
        return x

class LocalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
            )

    def forward(self, x):

        x = self.conv(x)    # -> 100 96 6 6 

        return x

class GlobalBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            # d_state=32,  # SSM state expansion factor
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        
    def forward(self, x):

        b, c, h, w = x.shape  # 300 96 6 6
        x = rearrange(x, 'b c h w -> b (h w) c',  b=b, c=c, h=h, w=w)  # (300 96 6 6) -> (300 36 96)
        x = self.mamba(x) # 300 36 96
        x = rearrange(x, 'b (h w) c -> b c h w',  b=b, c=c, h=h, w=w)   # (300 36 96) -> (300 96 6 6)

        return x

class WME(nn.Module):
    def __init__(self, dim, num_heads=1, mlp_ratio=4, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(WME, self).__init__()
        self.DWT = DWT()
        self.IWT = IWT()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.conv = SpaBlock(in_channels=dim, out_channels=dim)
        self.localBlock_L = LocalBlock(in_channels=dim, out_channels=dim)
        self.globalBlock_L = GlobalBlock(dim)

        self.localBlock_H = LocalBlock(in_channels=dim, out_channels=dim)
        self.globalBlock_H = GlobalBlock(dim)

        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, input_):

        global m
        x = input_    # (100 96 12 12)
        # 填充到偶数尺寸
        if x.size(2) % 2 != 0 or x.size(3) % 2 != 0:
            x = torch.nn.functional.pad(x, (0, 1, 0, 1), mode='reflect')
        n, c, h, w = x.shape   # n = 100
        x1 = self.norm1(x)    # (100 96 12 12)

        x_ConvBlock = self.conv(x1)   # 卷积模块分支 用于提取空间信息

##########################################################################

        input_dwt = self.DWT(x1)    # (400 96 6 6)
        input_LL, input_high = input_dwt[:n, ...], input_dwt[n:, ...]    # input_LL.shape:(100 96 6 6)    input_high.shape: (300 96 6 6)

        # 低频分量
        # output_LL = self.localBlock_L(input_LL) + self.globalBlock_L(input_LL)   # 100 96 6 6
        # output_LL = self.globalBlock_L(input_LL)   # 100 96 6 6    # 低频全局
        output_LL = self.localBlock_L(input_LL)   # 100 96 6 6       # 低频局部

        # 高频分量
        # output_high = self.localBlock_H(input_high) + self.globalBlock_H(input_high)   # 300 96 6 6
        output_high = self.globalBlock_H(input_high)   # 300 96 6 6   #  高频全局
        # output_high = self.localBlock_H(input_high)       #  高频局部

        output = self.IWT(torch.cat((output_LL, output_high), dim=0))   # (100 96 12 12)

##########################################################################

        # x = output + x   # 100 96 12 12    不添加空间卷积块分支
        # x = x_ConvBlock + x   # 100 96 12 12    不添加频域分支

        x = output + x + x_ConvBlock   # 100 96 12 12    #  添加卷积模块提取空间信息

        x = x + self.ffn(self.norm2(x))   # 100 96 12 12
        return x


class SFMamba(nn.Module):
    def __init__(self, 
        image_size=12,
        in_chans=3, 
        num_classes=16, 
        embed_dims=96,
        mlp_ratios=8, 
        depth=1,
        drop_path_rate=0.
    ):
        super().__init__()

        self.depth = depth
        self.image_size = image_size
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU()
            )
        
        self.block = nn.ModuleList([WME(dim=embed_dims)
                                    for i in range(self.depth)])

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(in_features=embed_dims, out_features=num_classes)
            )
        
    def forward(self, x):

        # x = x.squeeze(1)
        # B, C, H, W = x.shape    # B=100 C=200 H=12 W=12
        x = self.conv(x)      # 100 200 12 12 -> 100 96 12 12

        for i in range(self.depth):
            x = self.block[i](x)

        x = x.flatten(2).transpose(1, 2)     # 100 96 12 12 -> 100 96 144 -> 100 144 96
        x = x.mean(dim=1)    # 100 144 96  -> 100 96
        x = self.classifier(x)   # 100 96 -> 100 16
        return x



if __name__ == '__main__':
    input = torch.randn(size=(100, 200, 11, 11))
    input = input.cuda()
    print("input shape:", input.shape)
    model = SFMamba(in_chans=200, num_classes=16)
    model = model.cuda()
    print("output shape:", model(input).shape)