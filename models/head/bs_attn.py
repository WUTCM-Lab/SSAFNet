# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, NonLocal2d

from .base_decoder import resize
from .base_decoder import BaseDecodeHead
from einops import rearrange
from tools.heatmap_fun import draw_features


up_kwargs = {'mode': 'bilinear', 'align_corners': False}
norm_cfg = dict(type='BN', requires_grad=True)


class ContAttn(NonLocal2d):
    def __init__(self, *arg, head=1,
                 patch_size=None, **kwargs):
        super().__init__(*arg, **kwargs)
        self.head = head
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(self.in_channels)
        self.position_mixing = nn.ParameterDict({
            'weight': nn.Parameter(torch.zeros(head, patch_size ** 2, patch_size ** 2)),
            'bias': nn.Parameter(torch.zeros(1, head, 1, patch_size ** 2))
        })

    def forward(self, query, context):
        # x: [N, C, H, W]
        n, c, h, w = context.shape  # [128, 512, 8, 8]
        context = context.reshape(n, c, -1)
        context = rearrange(self.norm(context.transpose(1, -1)), 'b n (h d) -> b h d n', h=self.head)
        context_mlp = torch.einsum('bhdn, hnm -> bhdm', context, self.position_mixing['weight']) + self.position_mixing[
            'bias']
        context = context + context_mlp
        context = context.reshape(n, c, h, w)

        # g_x: [N, HxW, C]
        g_x = self.g(context).view(n, self.inter_channels, -1)
        g_x = rearrange(g_x, 'b (h dim) n -> (b h) dim n', h=self.head)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        if self.mode == 'gaussian':
            theta_x = query.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(context).view(n, self.in_channels, -1)
            else:
                phi_x = context.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(query).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(query).view(n, self.inter_channels, -1)
            theta_x = rearrange(theta_x, 'b (h dim) n -> (b h) dim n', h=self.head)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, -1)
            phi_x = rearrange(phi_x, 'b (h dim) n -> (b h) dim n', h=self.head)

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        y = rearrange(y, '(b h) n dim -> b n (h dim)', h=self.head)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *query.size()[2:])

        output = query + self.conv_out(y)
        return output


class BSHead(BaseDecodeHead):
    def __init__(self, in_channels=[256, 512, 1024, 2048], in_index=[0, 1, 2, 3], num_classes=2, channels=512):
        super().__init__(input_transform='multiple_select',
                         in_channels=in_channels, in_index=[0, 1, 2, 3], num_classes=num_classes,
                         channels=512, dropout_ratio=0.1, norm_cfg=norm_cfg, align_corners=False)
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        self.squeeze = ConvModule(  # stage2 3 4
            sum(self.in_channels) - self.in_channels[0],
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.channels = channels
        self.in_index = in_index
        self.conv00 = nn.Conv2d(self.in_channels[-1], self.channels, 1)
        self.conv0 = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=5,
            padding=4,
            dilation=2,
            # groups=self.channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv_spatial = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=7,
            padding=9,
            dilation=3,
            # groups=self.channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv11 = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,  # // 2
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.conv12 = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,  # // 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        # ============================
        #  Context Attention Module  #
        # ============================
        self.patch_size = 4
        self.conta = ContAttn(
            in_channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode='embedded_gaussian',
            head=16,
            patch_size=self.patch_size)

        self.conv13 = ConvModule(
            in_channels=self.channels * 2,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.attn_fuse = ConvModule(
            in_channels=self.channels * 2,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.low_level_fuse = ConvModule(
            in_channels=self.channels + self.in_channels[0],
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Conv2d(in_channels, self.channels, 1)
            self.lateral_convs.append(l_conv)

    def get_patch(self, x, patch_size):
        n, _, h, w = x.shape
        # get patch
        patch = F.unfold(x, kernel_size=patch_size, stride=patch_size)
        patch = rearrange(patch, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw',
                          ph=patch_size, pw=patch_size,
                          nh=h // patch_size, nw=w // patch_size)
        return patch

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        c1, c2, c3, c4 = inputs
        #  1/4  [1/8, 1/16, 1/32] ---> 1/8
        #  in_channels=[256, 512, 1024, 2048]

        # for i, out in enumerate(inputs):
        #     draw_features(out, f'C{i}')

        # x_fuse = [
        #     resize(
        #         level,
        #         size=inputs[1].shape[2:],  # 1/8
        #         mode='bilinear',
        #         align_corners=self.align_corners) for level in inputs[1:]
        # ]
        # x_fuse = self.squeeze(torch.cat(x_fuse, dim=1))  # [2, 512, 64, 64]
        # ============================
        #  Context Attention Module  #
        # ============================
        x_fuse = self.conv00(c4)
        attn1 = self.conv0(x_fuse)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv11(attn1)  # middle field  [2, 512, 64, 64]
        attn2 = self.conv12(attn2)  # large field   [2, 512, 64, 64]
        query = self.get_patch(attn1, self.patch_size)  # [128, 512, 8, 8]
        context = self.get_patch(attn2, self.patch_size)  # [128, 512, 8, 8]
        x_attn = self.conta(query, context)  # [128, 512, 8, 8]  --> [2, 512, 64, 64]
        _, _, h, w = inputs[1].shape
        x_attn = rearrange(x_attn, '(b nh nw) c ph pw -> b c (nh ph) (nw pw)',
                           ph=self.patch_size, pw=self.patch_size,
                           nh=h // self.patch_size, nw=w // self.patch_size)  # [2, 512, 64, 64]

        x_attn = x_attn + x_fuse  # v2(+)
        out_semantic = x_attn
        # # ========= output ==============
        # out = resize(x_attn, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # out = self.low_level_fuse(torch.cat([out, c1], dim=1))
        # out = self.cls_seg(out)
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(x_attn)
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
        out = self.cls_seg(laterals[0])
        out_ori = laterals[0]
        return out, out_semantic, out_ori
