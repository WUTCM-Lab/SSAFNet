import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule, NonLocal2d
from .base_decoder import resize
from tools.heatmap_fun import draw_features

up_kwargs = {'mode': 'bilinear', 'align_corners': False}

def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y

def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input


class SCSEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x)  # + x * self.sSE(x)

class ReverseEdge(nn.Module):
    def __init__(self, channel):
        super().__init__()

        # self.edge_pred = nn.Conv2d(
        #     channel, channel,
        #     kernel_size=3, stride=1,
        #     padding=1, bias=False)
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, c_fuse):
        avg_high = torch.mean(c_fuse, dim=1, keepdim=True)
        max_high, _ = torch.max(c_fuse, dim=1, keepdim=True)
        x = torch.cat([avg_high, max_high], dim=1)
        x = self.conv1(x).sigmoid()

        return c_fuse * x

class EdgeHead(nn.Module):
    """Edge awareness module"""

    def __init__(self, in_channels=[96, 192], channels=96, out_fea=2, in_index=[0, 1]):
        super(EdgeHead, self).__init__()
        self.in_index = in_index
        self.in_channels = in_channels
        self.channels = channels
        self.norm_cfg = dict(type='BN', requires_grad=True)
        self.act_cfg = dict(type='ReLU')
        self.conv_base1 = ConvModule(
            in_channels=self.in_channels[0],  # self.in_channels[0]  256
            out_channels=self.channels,  # 128
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv_base01 = ConvModule(
            in_channels=self.in_channels[0],  # self.in_channels[0]  256
            out_channels=self.channels,  # 128
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv_fuse1 = nn.Sequential(
            ConvModule(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,  # p = k/2 * d
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        )
        self.conv_base02 = ConvModule(
            in_channels=self.in_channels[1],  # self.in_channels[0]  256
            out_channels=self.channels,  # 128
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv_base2 = ConvModule(
            in_channels=self.in_channels[1],  # self.in_channels[0]  256
            out_channels=self.channels,  # 128
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv_fuse2 = nn.Sequential(
            ConvModule(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,  # p = k/2 * d
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        )
        self.low_fuse = ConvModule(
            in_channels=self.channels * 2,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.sobel_x1, self.sobel_y1 = get_sobel(256, 1)
        self.sobel_x2, self.sobel_y2 = get_sobel(512, 1)
        self.conv_edge = nn.Conv2d(self.channels, out_fea, 1, 1, 0)
        # self.RevEA = ReverseEdge(self.channels)
        self.scse = SCSEAttention(in_channels=self.channels)
        self.global_embedding11 = ConvModule(
            in_channels=512,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.global_embedding12 = ConvModule(
            in_channels=512,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.global_embedding21 = ConvModule(
            in_channels=512,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.global_embedding22 = ConvModule(
            in_channels=512,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.act = nn.Sigmoid()

    def _transform_inputs(self, inputs):
        if isinstance(self.in_index, (list, tuple)):
            inputs = [inputs[i] for i in self.in_index]
        elif isinstance(self.in_index, int):
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs, x_semantic):
        inputs = self._transform_inputs(inputs)
        c1, c2 = inputs  # 1/4 + 1/8
        # v5plus  sobel base + x_semantic
        c1_b = run_sobel(self.sobel_x1, self.sobel_y1, c1)
        c1_t = self.conv_base1(c1_b)
        c1_b = self.conv_fuse1(c1_t)
        c1_b = c1_t + c1_b
        # c1_b (1/4 x 1/4 x 128)
        # x_semantic (1/8 x 1/8 x 512)
        # draw_features(c1_b, "c1_b")
        x_semantic1 = self.global_embedding11(x_semantic)
        x_semantic_feat1 = self.global_embedding12(x_semantic)
        x_semantic_feat1 = resize(x_semantic_feat1, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # sig_act1 = resize(self.act(x_semantic1), size=c1.size()[2:], mode='bilinear', align_corners=False)

        c1_b = c1_b * x_semantic_feat1  +  x_semantic_feat1  # c1_b (1/4 x 1/4 x 128)
        # draw_features(c1_b, "c1_ba")
        c2_b = run_sobel(self.sobel_x2, self.sobel_y2, c2)
        c2_t = self.conv_base2(c2_b)
        c2_b = self.conv_fuse2(c2_t)
        c2_b = c2_t + c2_b
        # draw_features(c2_b, "c2_b")
        x_semantic2 = self.global_embedding21(x_semantic)
        x_semantic_feat2 = self.global_embedding22(x_semantic)
        # sig_act2 = self.act(x_semantic2)
        c2_b = c2_b * x_semantic_feat2 + x_semantic_feat2  # c2_b (1/8 x 1/8 x 128)
        c2_b = resize(c2_b, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # draw_features(c2_b, "c2_ba")

        c_fuse = self.low_fuse(torch.cat([c1_b, c2_b], dim=1))
        # draw_features(c_fuse, "c_bfuse")
        c_fuse = self.scse(c_fuse)
        edge = self.conv_edge(c_fuse)
        return edge, c_fuse
