# -*- coding: utf-8 -*-
# @Time    : 2021/7/20 21:14
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import numpy as np
import torch
import torch.nn as nn

from models.backbone.darknet53_C5 import BaseConv, DWConv

__all__ = ['YOLOF_Head']


class YOLOF_Head(nn.Module):
    def __init__(self, num_classes=80, reid_dim=0, width=1.0, in_channels=512, act="silu",
                 depthwise=False):
        super().__init__()

        self.n_anchors = 1
        self.mid_channels = in_channels / 2
        self.num_classes = num_classes
        self.reid_dim = reid_dim
        Conv = DWConv if depthwise else BaseConv

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        if self.reid_dim > 0:
            self.reid_preds = nn.ModuleList()

        for i in range(1):
            self.stems.append(
                BaseConv(in_channels=int(in_channels * width),
                         out_channels=int(self.mid_channels * width),
                         ksize=1,       # 1*1
                         stride=1,
                         act=act))
            self.cls_convs.append(
                nn.Sequential(      # 两个 256通道的3*3卷积
                    *[
                        Conv(
                            in_channels=int(self.mid_channels * width),
                            out_channels=int(self.mid_channels * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(self.mid_channels * width),
                            out_channels=int(self.mid_channels * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(           # 两个 256通道的3*3卷积
                    *[
                        Conv(
                            in_channels=int(self.mid_channels * width),
                            out_channels=int(self.mid_channels * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(self.mid_channels * width),
                            out_channels=int(self.mid_channels * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(self.mid_channels * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,      # 1*1
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(self.mid_channels * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(self.mid_channels * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            if self.reid_dim > 0:
                self.reid_preds.append(
                    nn.Conv2d(
                        in_channels=int(self.mid_channels * width),
                        out_channels=self.reid_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )

    def init_weights(self, prior_prob=1e-2):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-np.math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-np.math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, feats):
        outputs = []
        for k, (cls_conv, reg_conv) in enumerate(zip(self.cls_convs, self.reg_convs)):

            x = self.stems[k](feats)    # 512 -> 256

            # classify
            cls_feat = cls_conv(x)  # 两个3*3的
            cls_output = self.cls_preds[k](cls_feat)  # 1*1

            # regress, object, (reid)，reg_output是中心坐标和宽高
            reg_feat = reg_conv(x)  # 两个3*3的
            reg_output = self.reg_preds[k](reg_feat)  # 1*1
            obj_output = self.obj_preds[k](reg_feat)  # 1*1
            if self.reid_dim > 0:
                reid_output = self.reid_preds[k](reg_feat)
                output = torch.cat([reg_output, obj_output, cls_output, reid_output], 1)
            else:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)

        return outputs


if __name__ == "__main__":
    from thop import profile

    # in_channel = [256, 512, 1024]
    # feats = [torch.rand([1, in_channel[0], 64, 64]), torch.rand([1, in_channel[1], 32, 32]),
    #          torch.rand([1, in_channel[2], 16, 16])]

    feats = torch.zeros([1,512,20,20])
    head = YOLOF_Head(reid_dim=0)
    head.init_weights()
    head.eval()
    total_ops, total_params = profile(head, (feats,))
    print("total_ops {:.2f}G, total_params {:.2f}M".format(total_ops / 1e9, total_params / 1e6))
    out = head(feats)
    # for o in out:
    #     print(o.size())

    # 输出结果为
    # total_ops 13.28G, total_params 7.61M
    # torch.Size([1, 85, 64, 64])
    # torch.Size([1, 85, 32, 32])
    # torch.Size([1, 85, 16, 16])
